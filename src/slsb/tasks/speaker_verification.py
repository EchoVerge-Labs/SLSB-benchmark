"""Automatic speaker verification (ASV): cosine-similarity EER on held-out trial
pairs, using SID's trained weighted-sum + head as the embedding extractor.

There is no separate ASV training set -- data/asv/ is trial pairs only -- so
this task must run after sid for the same upstream (runner.py enforces the
ordering and passes the trained SID head in as `embedding_head`).
"""
import time

import numpy as np

from slsb.metrics import compute as metrics
from slsb.tasks._common import embed_wavs, reset_peak_memory, peak_gpu_gb
from slsb.utils.datasets import VerificationTrials


def run(upstream, trials: VerificationTrials, embedding_head, params):
    reset_peak_memory(upstream.device)
    start = time.time()
    embeddings = embed_wavs(upstream, embedding_head, trials.unique_wavs, params)
    embed_seconds = time.time() - start

    scores, labels = [], []
    for label, wav1, wav2 in trials.pairs:
        e1, e2 = embeddings[str(wav1)], embeddings[str(wav2)]
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        scores.append(cos)
        labels.append(label)

    metrics_out = {"eer": metrics.eer(scores, labels)}
    perf = {
        "seconds_per_step": embed_seconds / max(1, len(trials.unique_wavs)),
        "peak_gpu_gb": peak_gpu_gb(upstream.device),
    }
    return metrics_out, perf
