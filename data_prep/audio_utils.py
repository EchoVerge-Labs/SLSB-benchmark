"""Shared audio helpers for data_prep scripts."""
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16000


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        return data.mean(axis=1)
    return data


def resample_to_16k(data: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return data
    frac = Fraction(TARGET_SR, orig_sr).limit_denominator(1000)
    return resample_poly(data, frac.numerator, frac.denominator)


def to_pcm16_16k_mono(data: np.ndarray, orig_sr: int) -> np.ndarray:
    return resample_to_16k(to_mono(np.asarray(data, dtype=np.float32)), orig_sr)
