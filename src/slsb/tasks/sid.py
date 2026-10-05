"""Speaker identification: closed-set classification over data/sid/'s speakers
(every speaker is a class, so train and test share speakers by definition).
Weighted sum + mean-pool + linear, selected on the dev split.
"""
from slsb.tasks._common import (
    learning_rates, peak_gpu_gb, prepare_utterance_features, reset_peak_memory, score_utterance_fold,
)
from slsb.utils.datasets import load_split


def prepare(upstream, spec, params, work_dir, shared):
    split = load_split(spec)
    features = prepare_utterance_features(upstream, spec, split["train"] + split["dev"] + split["test"])
    features.split = split
    return features


def run(features, params, seed, tuned):
    cfg = params["utterance"]
    split = features.split
    reset_peak_memory(features.device)
    fit, acc, f1 = score_utterance_fold(features, split["train"], split["dev"], split["test"], cfg,
                                        learning_rates(cfg, tuned), seed)
    tuned.setdefault("lr", fit.lr)
    perf = {"train_seconds": fit.train_seconds, "peak_gpu_gb": peak_gpu_gb(features.device)}
    details = {"lr": fit.lr, "best_epoch": fit.best_epoch, "dev_accuracy": fit.dev_score}
    return {"accuracy": acc, "macro_f1": f1}, perf, split["split_type"], details
