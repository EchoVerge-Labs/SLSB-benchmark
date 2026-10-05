"""Emotion recognition (ER): weighted sum + mean-pool + linear, scored by
speaker-disjoint k-fold cross-validation (SUPERB's protocol for small emotion
corpora). For fold k: test = fold k, dev = fold k+1, train = the rest; the
reported accuracy / macro-F1 are means over folds.

Speakers are the leak that matters here: each sentence is recorded in most
emotions, so sentence overlap doesn't reveal the label.

er_sinhala is EXCLUDED from this benchmark's data/ (known dataset-quality
issues -- see KNOWN_ISSUES.md).
"""
import numpy as np

from slsb.tasks._common import (
    learning_rates, peak_gpu_gb, prepare_utterance_features, reset_peak_memory, score_utterance_fold,
)
from slsb.utils.datasets import load_split


def prepare(upstream, spec, params, work_dir, shared):
    split = load_split(spec)
    features = prepare_utterance_features(upstream, spec, [f for fold in split["folds"] for f in fold])
    features.split = split
    return features


def run(features, params, seed, tuned):
    cfg = params["utterance"]
    folds = features.split["folds"]
    reset_peak_memory(features.device)
    accs, f1s, lrs, train_seconds = [], [], [], 0.0
    for k in range(len(folds)):
        dev_k = (k + 1) % len(folds)
        train = [f for j, fold in enumerate(folds) if j not in (k, dev_k) for f in fold]
        fit, acc, f1 = score_utterance_fold(features, train, folds[dev_k], folds[k], cfg,
                                            learning_rates(cfg, tuned, key=f"lr_fold{k}"), seed)
        tuned.setdefault(f"lr_fold{k}", fit.lr)
        accs.append(acc)
        f1s.append(f1)
        lrs.append(fit.lr)
        train_seconds += fit.train_seconds
    perf = {"train_seconds": train_seconds, "peak_gpu_gb": peak_gpu_gb(features.device)}
    details = {"fold_accuracy": [round(a, 4) for a in accs], "fold_accuracy_std": float(np.std(accs)),
               "fold_lr": lrs}
    metrics_out = {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
    return metrics_out, perf, features.split["split_type"], details
