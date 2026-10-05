"""ASR: frozen upstream + learned weighted sum over layers + 2-layer BiLSTM +
CTC over characters (SUPERB's ASR downstream). Greedy decoding, no language
model. Early-stopped on dev CER; reports test WER and CER.

Every layer's frames are extracted once (slsb/features.py) and the head trains
on those, so the upstream runs once per task instead of once per epoch. Audio
is never cropped -- that would desync the transcript.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slsb.features import FrameStore, extract_frames, length_batches
from slsb.metrics import compute as metrics
from slsb.tasks._common import WeightedSum, fit_with_dev, learning_rates, peak_gpu_gb, reset_peak_memory
from slsb.utils.datasets import load_split, load_vocab, read_transcripts


class BLSTMCTCHead(nn.Module):
    def __init__(self, num_layers, input_size, vocab_size, hidden_size, rnn_layers, dropout):
        super().__init__()
        self.weighted_sum = WeightedSum(num_layers)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=rnn_layers, dropout=dropout,
                           bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, hidden_states, lengths):  # (L,B,T,H), (B,) -> log-probs (B,T,V)
        x = self.weighted_sum(hidden_states, dim=0)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        return nn.functional.log_softmax(self.out(self.dropout(y)), dim=-1)


@dataclass
class ASRFeatures:
    store: FrameStore
    files: list
    transcripts: list
    vocab: dict
    split: dict
    device: torch.device

    def rows(self, part):
        position = {f: i for i, f in enumerate(self.files)}
        return [position[f] for f in self.split[part]]

    def cleanup(self):
        self.store.cleanup()


def prepare(upstream, spec, params, work_dir, shared):
    split = load_split(spec)
    transcript_of = read_transcripts(spec)
    files = split["train"] + split["dev"] + split["test"]
    vocab = load_vocab(spec, list(transcript_of.values()))
    store = extract_frames(upstream, [spec.audio_dir / f for f in files], work_dir / "frames")
    return ASRFeatures(store, files, [transcript_of[f] for f in files], vocab, split, upstream.device)


def _load_batch(store: FrameStore, rows):
    arrays = [store.load(r) for r in rows]
    num_layers, _, hidden = arrays[0].shape
    lengths = torch.tensor([a.shape[1] for a in arrays])
    x = torch.zeros(num_layers, len(rows), int(lengths.max()), hidden, dtype=torch.float16)
    for j, a in enumerate(arrays):
        x[:, j, :a.shape[1]] = torch.from_numpy(a)
    return rows, x, lengths


def _batches(features, rows, cfg, rng=None):
    """Yields (rows, (L,B,T,H) fp16 CPU, lengths); the next batch is read from
    disk while the current one is on the GPU."""
    lengths = [features.store.lengths[r] for r in rows]
    batches = [[rows[i] for i in b] for b in length_batches(lengths, cfg["max_items"], cfg["max_frames"])]
    if rng is not None:
        rng.shuffle(batches)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_load_batch, features.store, batches[0]) if batches else None
        for nxt in batches[1:] + [None]:
            batch = pending.result()
            pending = pool.submit(_load_batch, features.store, nxt) if nxt is not None else None
            yield batch


def _encode(transcripts, vocab):
    ids = [[vocab.get(c, vocab["<unk>"]) for c in t] for t in transcripts]
    return torch.tensor([i for t in ids for i in t], dtype=torch.long), torch.tensor([len(t) for t in ids])


def greedy_decode(log_probs, lengths, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    blank = vocab["<blank>"]
    preds = log_probs.argmax(dim=-1).cpu()
    results = []
    for b in range(preds.shape[0]):
        collapsed, prev = [], None
        for t in preds[b, :int(lengths[b])].tolist():
            if t != prev and t != blank:
                collapsed.append(t)
            prev = t
        results.append("".join(inv_vocab.get(t, "") for t in collapsed))
    return results


@torch.no_grad()
def transcribe(model, features, rows, cfg):
    model.eval()
    hyps = {}
    for batch_rows, x, lengths in _batches(features, rows, cfg):
        log_probs = model(x.to(features.device).float(), lengths)
        for r, h in zip(batch_rows, greedy_decode(log_probs, lengths, features.vocab)):
            hyps[r] = h
    refs = [features.transcripts[r] for r in rows]
    # jiwer chokes on empty strings; substitute a single space (counts as one error).
    return ([r if r.strip() else " " for r in refs],
            [hyps[r] if hyps[r].strip() else " " for r in rows])


def run(features, params, seed, tuned):
    cfg = params["asr"]
    device = features.device
    train, dev, test = features.rows("train"), features.rows("dev"), features.rows("test")
    num_layers, _, hidden = features.store.load(0, mmap=True).shape
    ctc_loss = nn.CTCLoss(blank=features.vocab["<blank>"], zero_infinity=True)

    def build():
        return BLSTMCTCHead(num_layers, hidden, len(features.vocab), cfg["hidden_size"], cfg["rnn_layers"],
                            cfg["dropout"]).to(device)

    def train_epoch(model, optimizer, rng):
        for batch_rows, x, lengths in _batches(features, train, cfg, rng):
            targets, target_lengths = _encode([features.transcripts[r] for r in batch_rows], features.vocab)
            log_probs = model(x.to(device).float(), lengths)
            loss = ctc_loss(log_probs.transpose(0, 1), targets.to(device), lengths.to(device),
                            target_lengths.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

    def dev_cer(model):
        return metrics.cer(*transcribe(model, features, dev, cfg))

    reset_peak_memory(device)
    fit = fit_with_dev(build, train_epoch, dev_cer, learning_rates(cfg, tuned), cfg["max_epochs"],
                       cfg["patience"], seed, maximize=False, verbose=True)
    tuned.setdefault("lr", fit.lr)
    refs, hyps = transcribe(fit.model, features, test, cfg)
    metrics_out = {"wer": metrics.wer(refs, hyps), "cer": metrics.cer(refs, hyps)}
    perf = {"train_seconds": fit.train_seconds, "peak_gpu_gb": peak_gpu_gb(device)}
    details = {"lr": fit.lr, "best_epoch": fit.best_epoch, "dev_cer": fit.dev_score}
    return metrics_out, perf, features.split["split_type"], details
