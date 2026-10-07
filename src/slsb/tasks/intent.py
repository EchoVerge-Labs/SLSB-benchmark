"""Intent classification (IC): weighted sum + mean-pool + linear (SUPERB's IC
head). Tasks with speakers are scored by speaker-disjoint k-fold
cross-validation, like ER; ic_banking_sinhala, which has no speaker
information, uses a stratified train/dev/test split instead.
"""
from slsb.tasks import emotion, sid


def prepare(upstream, spec, params, work_dir, shared):
    return emotion.prepare(upstream, spec, params, work_dir, shared)


def run(features, params, seed, tuned):
    module = emotion if "folds" in features.split else sid
    return module.run(features, params, seed, tuned)
