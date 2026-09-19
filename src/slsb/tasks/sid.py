"""Speaker identification: classification probe over data/sid/.

The trained head is also handed to tasks/speaker_verification.py -- there is
no separate ASV training set, so ASV reuses SID's weighted-sum + head as the
embedding extractor (see runner.py, which runs sid before asv_* for this reason).
"""
from slsb.tasks._common import run_classification_task


def run(upstream, spec, params, seed=42):
    metrics_out, perf, head, split_type = run_classification_task(upstream, spec, params, seed=seed)
    return metrics_out, perf, head, split_type
