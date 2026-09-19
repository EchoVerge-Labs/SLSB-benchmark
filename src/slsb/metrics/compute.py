"""Metrics shared across probing runners: classification, verification, ASR."""
import jiwer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def eer(scores, labels):
    """Equal error rate. scores: higher = more likely same speaker. labels: 1=genuine, 0=impostor."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def wer(references, hypotheses):
    return jiwer.wer(references, hypotheses)


def cer(references, hypotheses):
    return jiwer.cer(references, hypotheses)
