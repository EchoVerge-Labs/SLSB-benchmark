"""Intent classification (IC): weighted sum + mean-pool + linear (SUPERB's IC
head), scored by k-fold cross-validation like ER. Since v0.5 the folds are
sentence-disjoint: the prompts read in a test fold are never heard in
training (data_prep/make_splits.py).
"""
from slsb.tasks import emotion, sid


def prepare(upstream, spec, params, work_dir, shared):
    return emotion.prepare(upstream, spec, params, work_dir, shared)


def run(features, params, seed, tuned):
    module = emotion if "folds" in features.split else sid
    return module.run(features, params, seed, tuned)
