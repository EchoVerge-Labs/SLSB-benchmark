"""Shared probe infra used by more than one task module: the learnable
weighted-sum over upstream layers, the linear classification head, and small
batching/GPU-memory helpers. The upstream encoder itself stays frozen (see
slsb/upstream/loader.py); only what's defined here trains.
"""
import numpy as np
import torch
import torch.nn as nn


class WeightedSum(nn.Module):
    """Learnable softmax-weighted combination over the upstream's hidden-state layers."""

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states):  # (L, B, T, H) -> (B, T, H)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (hidden_states * w).sum(dim=0)


def masked_mean_pool(features, frame_mask):  # (B,T,H), (B,T) -> (B,H)
    mask = frame_mask.unsqueeze(-1).to(features.dtype)
    summed = (features * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)
    return summed / count


class ClassificationHead(nn.Module):
    def __init__(self, num_layers, hidden_size, num_classes):
        super().__init__()
        self.weighted_sum = WeightedSum(num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)

    def embed(self, hidden_states, frame_mask):
        return masked_mean_pool(self.weighted_sum(hidden_states), frame_mask)

    def forward(self, hidden_states, frame_mask):
        return self.linear(self.embed(hidden_states, frame_mask))


def iterate_batches(n, batch_size, shuffle, rng):
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return None


def train_classification_probe(upstream, train_ds, num_classes, params, seed=42):
    head = ClassificationHead(upstream.num_hidden_states, upstream.hidden_size, num_classes).to(upstream.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=params["lr"])
    rng = np.random.RandomState(seed)

    head.train()
    for _ in range(params["epochs"]):
        for batch_idx in iterate_batches(len(train_ds), params["batch_size"], shuffle=True, rng=rng):
            waveforms, labels = zip(*(train_ds[i] for i in batch_idx))
            hidden_states, frame_mask = upstream.extract(list(waveforms))
            logits = head(hidden_states, frame_mask)
            loss = nn.functional.cross_entropy(logits, torch.tensor(labels, device=upstream.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head


@torch.no_grad()
def evaluate_classification(upstream, head, test_ds, params):
    from slsb.metrics import compute as metrics

    head.eval()
    all_preds, all_labels = [], []
    for batch_idx in iterate_batches(len(test_ds), params["batch_size"], shuffle=False, rng=None):
        waveforms, labels = zip(*(test_ds[i] for i in batch_idx))
        hidden_states, frame_mask = upstream.extract(list(waveforms))
        logits = head(hidden_states, frame_mask)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels)
    return metrics.accuracy(all_labels, all_preds), metrics.macro_f1(all_labels, all_preds)


def run_classification_task(upstream, spec, params, seed=42, leakage_free=False):
    from slsb.utils.datasets import load_classification_task, load_classification_task_leakage_free

    diagnostics = None
    if leakage_free:
        train_ds, test_ds, classes, split_type, diagnostics = load_classification_task_leakage_free(spec, seed=seed)
    else:
        train_ds, test_ds, classes = load_classification_task(spec, seed=seed)
        split_type = "random"
    reset_peak_memory(upstream.device)
    import time
    start = time.time()
    head = train_classification_probe(upstream, train_ds, len(classes), params, seed=seed)
    train_seconds = time.time() - start
    acc, f1 = evaluate_classification(upstream, head, test_ds, params)
    n_steps = params["epochs"] * max(1, -(-len(train_ds) // params["batch_size"]))
    metrics_out = {"accuracy": acc, "macro_f1": f1}
    perf = {"seconds_per_step": train_seconds / n_steps, "peak_gpu_gb": peak_gpu_gb(upstream.device)}
    if leakage_free:
        return metrics_out, perf, head, split_type, diagnostics
    return metrics_out, perf, head, split_type


@torch.no_grad()
def embed_wavs(upstream, head, wav_paths, params):
    from slsb.utils.audio import load_wav_16k_mono, MAX_PROBE_DURATION_SECONDS

    head.eval()
    embeddings = {}
    for batch_idx in iterate_batches(len(wav_paths), params["batch_size"], shuffle=False, rng=None):
        batch_paths = [wav_paths[i] for i in batch_idx]
        waveforms = [load_wav_16k_mono(p, max_seconds=MAX_PROBE_DURATION_SECONDS) for p in batch_paths]
        hidden_states, frame_mask = upstream.extract(waveforms)
        emb = head.embed(hidden_states, frame_mask)
        for p, e in zip(batch_paths, emb.cpu().numpy()):
            embeddings[p] = e
    return embeddings
