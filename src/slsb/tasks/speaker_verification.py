"""Automatic speaker verification (ASV): a speaker-embedding head trained with
AM-softmax on speakers that never appear in the test trials (SLCeleb's dev
split, see data_prep/prep_slceleb_train.py), selected on dev trials drawn from
held-out training speakers, scored by cosine-similarity EER on
data/asv/trials_*.csv.

The default head is statistics pooling (mean + std over frames -> linear
embedding). It was chosen over SUPERB's x-vector on dev EER, on the
80-speaker Tamil training set: dev EER 0.123 +- 0.003 over 3 seeds, against
0.142-0.186 for four x-vector variants, which peak within 3-6 epochs while
statistics pooling keeps improving for ~13.
`head: xvector` in params.yaml restores the x-vector.

Storing every layer's frames for ~45 h of training audio would take hundreds of
GB, so the layer mix is fixed first: a weighted-sum + mean-pool + linear speaker
classifier is fit on a subset of the training speakers, and its layer weights
are used to store one mixed frame sequence per clip. This is the one place v0.2
departs from SUPERB, which learns the layer weights jointly with the x-vector.

Both trial lists (asv_sinhala, asv_tamil) share the training data, the layer
mix and the trained model of each seed (`shared`); only the test clips differ.
"""
import csv
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slsb.features import FrameStore, extract_frames, layer_means
from slsb.metrics import compute as metrics
from slsb.tasks._common import fit_utterance_head, fit_with_dev, learning_rates, peak_gpu_gb, reset_peak_memory
from slsb.utils.audio import MAX_PROBE_DURATION_SECONDS
from slsb.utils.datasets import load_split

MIN_EVAL_FRAMES = 20  # the x-vector's TDNN sees 15 frames of context; tile anything shorter


class XVector(nn.Module):
    """TDNN x-vector: five frame-level layers, mean+std statistics pooling, and
    a segment-level embedding layer."""

    def __init__(self, input_size, channels=512, stats_channels=1500, embedding_size=512):
        super().__init__()
        spec = [(input_size, channels, 5, 1), (channels, channels, 3, 2), (channels, channels, 3, 3),
                (channels, channels, 1, 1), (channels, stats_channels, 1, 1)]
        self.frames = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(i, o, k, dilation=d), nn.ReLU(), nn.BatchNorm1d(o)) for i, o, k, d in spec
        ])
        self.embedding = nn.Linear(2 * stats_channels, embedding_size)

    def forward(self, x):  # (B,T,H) -> (B,E)
        h = self.frames(x.transpose(1, 2))
        return self.embedding(torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1))


class StatsPooling(nn.Module):
    """Mean + standard deviation over frames -> linear embedding."""

    def __init__(self, input_size, embedding_size=256):
        super().__init__()
        self.embedding = nn.Linear(2 * input_size, embedding_size)

    def forward(self, x):  # (B,T,H) -> (B,E)
        return self.embedding(torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1))


class AMSoftmax(nn.Module):
    """Additive-margin softmax: logits = scale * (cos(theta) - margin on the target class)."""

    def __init__(self, embedding_size, num_classes, margin, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size) * 0.01)
        self.margin, self.scale = margin, scale

    def forward(self, embeddings, labels):
        cos = F.normalize(embeddings) @ F.normalize(self.weight).T
        logits = self.scale * (cos - self.margin * F.one_hot(labels, cos.shape[1]))
        return F.cross_entropy(logits, labels)


HEADS = {"stats": StatsPooling, "xvector": lambda h, e: XVector(h, embedding_size=e)}


class SpeakerModel(nn.Module):
    def __init__(self, input_size, num_speakers, cfg):
        super().__init__()
        self.encoder = HEADS[cfg["head"]](input_size, cfg["embedding_size"])
        self.loss = AMSoftmax(cfg["embedding_size"], num_speakers, cfg["margin"], cfg["scale"])


@dataclass
class ASVFeatures:
    train: FrameStore      # training speakers' clips, then the dev speakers' clips
    train_labels: list     # speaker index per training clip (dev clips: -1)
    num_train: int
    dev_trials: list       # (label, row, row) into `train`
    test: FrameStore       # this trial list's clips
    test_trials: list      # (label, row, row) into `test`
    layer_weights: list
    shared: dict
    device: torch.device

    def cleanup(self):
        self.test.cleanup()  # the training store is shared; the runner cleans it up via `shared`


def _layer_weights(upstream, asv_dir, split, label_of, cfg):
    """Layer mix from a mean-pool speaker classifier on a subset of training clips."""
    rng = np.random.RandomState(0)
    by_speaker = defaultdict(list)
    for f in split["train"]:
        by_speaker[label_of[f]].append(f)
    subset_train, subset_dev = [], []
    for speaker in sorted(by_speaker):
        clips = sorted(by_speaker[speaker])
        rng.shuffle(clips)
        clips = clips[:cfg["layer_mix_clips_per_speaker"]]
        cut = max(1, len(clips) // 5)
        subset_dev += clips[:cut]
        subset_train += clips[cut:]
    speakers = sorted(by_speaker)
    index = {s: i for i, s in enumerate(speakers)}
    files = subset_train + subset_dev
    x = layer_means(upstream, [asv_dir / "train_audio" / f for f in files],
                    max_seconds=MAX_PROBE_DURATION_SECONDS).to(upstream.device)
    y = torch.tensor([index[label_of[f]] for f in files], device=upstream.device)
    n = len(subset_train)
    fit = fit_utterance_head(x[:n], y[:n], x[n:], y[n:], len(speakers), cfg["layer_mix"],
                             cfg["layer_mix"]["lr_grid"], seed=0)
    return fit.model.weighted_sum.layer_weights().detach().cpu()


def _shared_training(upstream, spec, cfg, work_dir, shared):
    if "train" in shared:
        return shared
    asv_dir = spec.task_dir
    split = load_split(asv_dir)
    label_of = _read_csv(asv_dir / "train_labels.csv", "filename", "label")
    weights = _layer_weights(upstream, asv_dir, split, label_of, cfg)
    files = split["train"] + split["dev"]
    store = extract_frames(upstream, [asv_dir / "train_audio" / f for f in files], work_dir.parent / "asv_train",
                           max_seconds=MAX_PROBE_DURATION_SECONDS, layer_weights=weights)
    speakers = sorted({label_of[f] for f in split["train"]})
    index = {s: i for i, s in enumerate(speakers)}
    row = {f: i for i, f in enumerate(files)}
    shared.update(
        train=store, num_train=len(split["train"]), num_speakers=len(speakers),
        train_labels=[index[label_of[f]] for f in split["train"]] + [-1] * len(split["dev"]),
        dev_trials=[(int(lab), row[a], row[b]) for lab, a, b in split["dev_trials"]],
        layer_weights=[round(float(w), 4) for w in weights], weights=weights, models={},
    )
    shared.setdefault("cleanup", []).append(store.cleanup)
    return shared


def prepare(upstream, spec, params, work_dir, shared):
    cfg = params["asv"]
    shared = _shared_training(upstream, spec, cfg, work_dir, shared)
    with open(spec.label_path, newline="", encoding="utf-8") as f:
        trials = [(int(r["label"]), r["wav1"], r["wav2"]) for r in csv.DictReader(f)]
    files = sorted({w for _, a, b in trials for w in (a, b)})
    row = {f: i for i, f in enumerate(files)}
    test = extract_frames(upstream, [spec.audio_dir / f for f in files], work_dir / "test",
                          max_seconds=MAX_PROBE_DURATION_SECONDS, layer_weights=shared["weights"])
    return ASVFeatures(shared["train"], shared["train_labels"], shared["num_train"], shared["dev_trials"], test,
                       [(lab, row[a], row[b]) for lab, a, b in trials], shared["layer_weights"], shared,
                       upstream.device)


def _read_csv(path, key, value):
    with open(path, newline="", encoding="utf-8") as f:
        return {r[key]: r[value] for r in csv.DictReader(f)}


def _tile(a, frames):
    return a if len(a) >= frames else np.tile(a, (int(np.ceil(frames / len(a))), 1))


def _crop_batch(store, rows, crop, rng):
    """Random `crop`-frame windows; shorter clips are tiled up to `crop`."""
    out = []
    for r in rows:
        a = _tile(store.load(r, mmap=True), crop)
        start = rng.randint(0, len(a) - crop + 1)
        out.append(torch.from_numpy(np.array(a[start:start + crop])))
    return torch.stack(out)


@torch.no_grad()
def _embed(model, store, rows, device):
    """Unit-length embedding of each whole clip (up to the 20 s crop)."""
    model.eval()
    out = {}
    for r in rows:
        x = torch.from_numpy(_tile(store.load(r), MIN_EVAL_FRAMES))[None]
        out[r] = F.normalize(model.encoder(x.to(device).float()), dim=1)[0]
    return out


def _eer(model, store, trials, device):
    emb = _embed(model, store, sorted({x for _, a, b in trials for x in (a, b)}), device)
    scores = [float(emb[a] @ emb[b]) for _, a, b in trials]
    return metrics.eer(scores, [lab for lab, _, _ in trials])


def _train(features, cfg, seed, lrs):
    device = features.device
    rows = list(range(features.num_train))
    labels = torch.tensor(features.train_labels[:features.num_train])
    input_size = features.train.load(0, mmap=True).shape[1]

    def build():
        return SpeakerModel(input_size, features.shared["num_speakers"], cfg).to(device)

    def train_epoch(model, optimizer, rng):
        order = np.array(rows)
        rng.shuffle(order)
        for start in range(0, len(order), cfg["batch_size"]):
            batch = order[start:start + cfg["batch_size"]]
            if len(batch) < 2:  # BatchNorm needs more than one example
                continue
            x = _crop_batch(features.train, batch, cfg["crop_frames"], rng).to(device).float()
            loss = model.loss(model.encoder(x), labels[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return fit_with_dev(build, train_epoch, lambda m: _eer(m, features.train, features.dev_trials, device),
                        lrs, cfg["max_epochs"], cfg["patience"], seed, maximize=False, verbose=True)


def run(features, params, seed, tuned):
    cfg = params["asv"]
    shared = features.shared
    reset_peak_memory(features.device)
    if seed not in shared["models"]:  # one model per seed serves both trial lists
        fit = _train(features, cfg, seed, learning_rates(cfg, shared.setdefault("tuned", {})))
        shared["tuned"].setdefault("lr", fit.lr)
        shared["models"][seed] = fit
    fit = shared["models"][seed]
    eer = _eer(fit.model, features.test, features.test_trials, features.device)
    perf = {"train_seconds": fit.train_seconds, "peak_gpu_gb": peak_gpu_gb(features.device)}
    details = {"head": cfg["head"], "lr": fit.lr, "best_epoch": fit.best_epoch, "dev_eer": fit.dev_score,
               "train_speakers": shared["num_speakers"], "layer_weights": features.layer_weights}
    return {"eer": eer}, perf, "speaker_disjoint", details
