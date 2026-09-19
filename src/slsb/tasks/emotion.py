"""Emotion recognition (ER): classification probe, same head architecture as SID.

Supports both the plain random split (run(leakage_free=False), kept for
backward comparison) and the leakage-free split (run(leakage_free=True)) --
see slsb.utils.datasets.load_classification_task_leakage_free's docstring for
why a random split leaks speaker/sentence identity for this task family.

er_sinhala is EXCLUDED from this benchmark's data/ (known dataset-quality
issues -- see README.md / KNOWN_ISSUES.md). This module still supports it if
a caller supplies data for it directly; only the data/ folder is withheld.
"""
from slsb.tasks._common import run_classification_task

# What to tag results with, per language, when running the leakage-free split.
# er_sinhala is a single-speaker corpus -> sentence-disjoint is the only option;
# er_tamil has 22 speakers -> speaker-disjoint (see load_classification_task_leakage_free).
LEAKAGE_FREE_SPLIT_TAGS = {"sinhala": "sentence_disjoint", "tamil": "speaker_sentence_disjoint"}


def run(upstream, spec, params, seed=42, leakage_free=True):
    """Defaults to the leakage-free split -- the plain random split is known to
    leak speaker/sentence identity between train/test for this task family
    (see docs/known_issues.md)."""
    if leakage_free:
        metrics_out, perf, head, split_type, diagnostics = run_classification_task(
            upstream, spec, params, seed=seed, leakage_free=True
        )
        return metrics_out, perf, head, split_type, diagnostics
    metrics_out, perf, head, split_type = run_classification_task(upstream, spec, params, seed=seed)
    return metrics_out, perf, head, split_type, None
