"""CTC ASR probe: frozen upstream + weighted-sum + linear CTC head. Reports WER/CER.

ASR audio is never truncated (that would desync the transcript from the audio),
so a batch can contain very long clips. To avoid OOM on those without changing
the data, an over-budget batch is split into smaller sub-batches that are each
forward/backward-passed separately, with gradients accumulated (scaled by each
sub-batch's share) before a single optimizer step -- same effective batch,
bounded peak memory.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from slsb.metrics import compute as metrics
from slsb.tasks._common import WeightedSum, iterate_batches, reset_peak_memory, peak_gpu_gb
from slsb.utils.datasets import load_asr_task

MAX_BATCH_SAMPLE_BUDGET = 8_000_000  # ~500s of total padded audio per forward/backward pass


class CTCHead(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size):
        super().__init__()
        self.weighted_sum = WeightedSum(num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):  # (L,B,T,H) -> log-probs (B,T,V)
        logits = self.linear(self.weighted_sum(hidden_states))
        return nn.functional.log_softmax(logits, dim=-1)


def _encode(transcript, vocab):
    return [vocab.get(c, vocab["<unk>"]) for c in transcript]


def load_items(dataset, indices):
    waveforms, transcripts = [], []
    for i in indices:
        wav, transcript = dataset[i]
        waveforms.append(wav)
        transcripts.append(transcript)
    return waveforms, transcripts


def encode_targets(transcripts, vocab):
    targets, target_lengths = [], []
    for t in transcripts:
        ids = _encode(t, vocab)
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(target_lengths, dtype=torch.long)


def split_by_budget(waveforms, max_items, max_budget):
    """Greedily group local indices into [waveforms] so max_len_in_group * group_size
    never exceeds max_budget (and never exceeds max_items)."""
    chunks, current, current_max = [], [], 0
    for i, w in enumerate(waveforms):
        length = len(w)
        prospective_max = max(current_max, length)
        if current and (len(current) >= max_items or prospective_max * (len(current) + 1) > max_budget):
            chunks.append(current)
            current, current_max = [i], length
        else:
            current.append(i)
            current_max = prospective_max
    if current:
        chunks.append(current)
    return chunks


def train_ctc_probe(upstream, train_ds, vocab, params, seed=42):
    head = CTCHead(upstream.num_hidden_states, upstream.hidden_size, len(vocab)).to(upstream.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=params["lr"])
    ctc_loss = nn.CTCLoss(blank=vocab["<blank>"], zero_infinity=True)
    rng = np.random.RandomState(seed)

    head.train()
    for _ in range(params["epochs"]):
        for batch_idx in iterate_batches(len(train_ds), params["batch_size"], shuffle=True, rng=rng):
            waveforms, transcripts = load_items(train_ds, batch_idx)
            optimizer.zero_grad()
            for sub in split_by_budget(waveforms, params["batch_size"], MAX_BATCH_SAMPLE_BUDGET):
                sub_waveforms = [waveforms[j] for j in sub]
                targets, target_lengths = encode_targets([transcripts[j] for j in sub], vocab)
                hidden_states, frame_mask = upstream.extract(sub_waveforms)
                log_probs = head(hidden_states)
                input_lengths = frame_mask.sum(dim=1)
                loss = ctc_loss(
                    log_probs.transpose(0, 1),  # CTCLoss wants (T,B,V)
                    targets.to(upstream.device),
                    input_lengths.to(upstream.device),
                    target_lengths.to(upstream.device),
                )
                (loss * len(sub) / len(batch_idx)).backward()
            optimizer.step()
    return head


def greedy_decode(log_probs, frame_mask, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    blank = vocab["<blank>"]
    preds = log_probs.argmax(dim=-1)  # (B,T)
    results = []
    for b in range(preds.shape[0]):
        length = int(frame_mask[b].sum().item())
        collapsed, prev = [], None
        for t in preds[b, :length].cpu().tolist():
            if t != prev and t != blank:
                collapsed.append(t)
            prev = t
        results.append("".join(inv_vocab.get(t, "") for t in collapsed))
    return results


@torch.no_grad()
def evaluate_ctc(upstream, head, test_ds, vocab, params):
    head.eval()
    all_refs, all_hyps = [], []
    for batch_idx in iterate_batches(len(test_ds), params["batch_size"], shuffle=False, rng=None):
        waveforms, transcripts = load_items(test_ds, batch_idx)
        for sub in split_by_budget(waveforms, params["batch_size"], MAX_BATCH_SAMPLE_BUDGET):
            sub_waveforms = [waveforms[j] for j in sub]
            hidden_states, frame_mask = upstream.extract(sub_waveforms)
            log_probs = head(hidden_states)
            all_hyps.extend(greedy_decode(log_probs, frame_mask, vocab))
            all_refs.extend(transcripts[j] for j in sub)
    # jiwer chokes on empty strings; substitute a single space (counts as one error).
    all_refs = [r if r.strip() else " " for r in all_refs]
    all_hyps = [h if h.strip() else " " for h in all_hyps]
    return metrics.wer(all_refs, all_hyps), metrics.cer(all_refs, all_hyps)


def run(upstream, spec, params, seed=42):
    train_ds, test_ds, vocab, split_type = load_asr_task(spec, seed=seed)
    reset_peak_memory(upstream.device)
    start = time.time()
    head = train_ctc_probe(upstream, train_ds, vocab, params, seed=seed)
    train_seconds = time.time() - start
    wer, cer = evaluate_ctc(upstream, head, test_ds, vocab, params)
    n_steps = params["epochs"] * max(1, -(-len(train_ds) // params["batch_size"]))
    metrics_out = {"wer": wer, "cer": cer}
    perf = {"seconds_per_step": train_seconds / n_steps, "peak_gpu_gb": peak_gpu_gb(upstream.device)}
    return metrics_out, perf, head, split_type
