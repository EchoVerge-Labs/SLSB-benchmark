"""Shared probe infrastructure: seeding, the learnable weighted sum over upstream
layers, dev-set model selection, and the utterance-level head (SUPERB's
mean-pool + linear) used by SID and ER.

The upstream itself stays frozen -- the heads here train on features that
slsb/features.py extracted once per task.
"""
import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeightedSum(nn.Module):
    """Learnable softmax-weighted combination over the upstream's hidden-state layers."""

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights, dim=0)

    def forward(self, x, dim: int = 0):  # contracts the layer axis `dim` of x
        return torch.tensordot(self.layer_weights(), x, dims=([0], [dim]))


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return None


@dataclass
class FitResult:
    model: nn.Module
    lr: float
    best_epoch: int
    dev_score: float
    train_seconds: float


def fit_with_dev(build, train_epoch, dev_score, lrs, max_epochs: int, patience: int, seed: int,
                 maximize: bool, verbose: bool = False) -> FitResult:
    """Train a fresh `build()` model per learning rate, one epoch at a time,
    keeping the weights of the epoch with the best dev score and stopping after
    `patience` epochs without improvement. Returns the best model over all lrs.

    train_epoch(model, optimizer, rng) runs one epoch; dev_score(model) -> float.
    verbose prints every epoch's dev score (for heads whose epochs take minutes).
    """
    better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)
    best = None
    start = time.time()
    for lr in lrs:
        set_seed(seed)
        model = build()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.RandomState(seed)
        best_score, best_state, best_epoch, stale = None, None, 0, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_epoch(model, optimizer, rng)
            model.eval()
            with torch.no_grad():
                score = dev_score(model)
            if verbose:
                print(f"        lr={lr:g} epoch {epoch}: dev {score:.4f} ({time.time() - start:.0f}s)", flush=True)
            if best_score is None or better(score, best_score):
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        print(f"      lr={lr:g}: best dev {best_score:.4f} at epoch {best_epoch} (ran {epoch})", flush=True)
        if best is None or better(best_score, best.dev_score):
            model.load_state_dict(best_state)
            best = FitResult(model, lr, best_epoch, best_score, 0.0)
    best.train_seconds = time.time() - start
    return best


def learning_rates(cfg: dict, tuned: dict, key: str = "lr") -> list[float]:
    """The lr grid on the first seed; the lr it picked on every later seed.
    Learning rates are chosen on dev once, then held fixed across seeds."""
    return [tuned[key]] if key in tuned else list(cfg["lr_grid"])


class UtteranceHead(nn.Module):
    """Weighted sum over layers + mean-pool + linear (SUPERB's utterance-level
    head). Takes per-layer mean-pooled features (B, L, H) -- equivalent to
    pooling after the weighted sum, since both are linear."""

    def __init__(self, num_layers, hidden_size, num_classes):
        super().__init__()
        self.weighted_sum = WeightedSum(num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)

    def embed(self, layer_means):
        return self.weighted_sum(layer_means, dim=1)

    def forward(self, layer_means):
        return self.linear(self.embed(layer_means))


def batch_indices(n: int, batch_size: int, rng=None):
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


@torch.no_grad()
def predict_utterance(model, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = [model(x[b].float()).argmax(dim=-1).cpu() for b in batch_indices(len(x), batch_size)]
    return torch.cat(preds).numpy()


def fit_utterance_head(train_x, train_y, dev_x, dev_y, num_classes, cfg, lrs, seed) -> FitResult:
    """train_x/dev_x: (N, L, H) layer means on the training device; *_y: (N,) long.
    Selects on dev accuracy."""
    device = train_x.device

    def build():
        return UtteranceHead(train_x.shape[1], train_x.shape[2], num_classes).to(device)

    def train_epoch(model, optimizer, rng):
        for b in batch_indices(len(train_x), cfg["batch_size"], rng):
            loss = nn.functional.cross_entropy(model(train_x[b].float()), train_y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def dev_accuracy(model):
        return float((predict_utterance(model, dev_x) == dev_y.cpu().numpy()).mean())

    return fit_with_dev(build, train_epoch, dev_accuracy, lrs, cfg["max_epochs"], cfg["patience"],
                        seed, maximize=True)


@dataclass
class UtteranceFeatures:
    """Layer-mean features of every clip a classification task uses."""
    files: list
    x: torch.Tensor  # (N, L, H) float16, CPU
    y: torch.Tensor  # (N,) long
    classes: list
    device: torch.device

    def rows(self, files) -> list[int]:
        position = {f: i for i, f in enumerate(self.files)}
        return [position[f] for f in files]

    def cleanup(self):
        pass


def prepare_utterance_features(upstream, spec, files) -> UtteranceFeatures:
    from slsb.features import layer_means
    from slsb.utils.audio import MAX_PROBE_DURATION_SECONDS
    from slsb.utils.datasets import read_labels

    label_of = read_labels(spec)
    classes = sorted(set(label_of.values()))
    index = {c: i for i, c in enumerate(classes)}
    # Utterance labels still hold for a cropped clip (standard for speaker/emotion tasks).
    x = layer_means(upstream, [spec.audio_dir / f for f in files], max_seconds=MAX_PROBE_DURATION_SECONDS)
    y = torch.tensor([index[label_of[f]] for f in files], dtype=torch.long)
    return UtteranceFeatures(list(files), x, y, classes, upstream.device)


def score_utterance_fold(features: UtteranceFeatures, train, dev, test, cfg, lrs, seed):
    """Fit on train, select on dev, predict test. Returns (fit, accuracy, macro_f1)."""
    from slsb.metrics import compute as metrics

    x = features.x.to(features.device)
    y = features.y.to(features.device)
    tr, dv, te = (torch.tensor(features.rows(part), device=features.device) for part in (train, dev, test))
    fit = fit_utterance_head(x[tr], y[tr], x[dv], y[dv], len(features.classes), cfg, lrs, seed)
    preds = predict_utterance(fit.model, x[te])
    labels = y[te].cpu().numpy()
    return fit, metrics.accuracy(labels, preds), metrics.macro_f1(labels, preds)
