"""Audio loading for the benchmark runners: always returns 16kHz mono float32."""
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
# Classification/verification labels still hold for a cropped segment (standard
# practice for speaker/emotion tasks); ASR must never be capped -- it would
# desync the transcript from the audio.
MAX_PROBE_DURATION_SECONDS = 20.0


def load_wav_16k_mono(path, max_seconds: float = None) -> np.ndarray:
    path = str(path)
    if path.lower().endswith(".mp3"):
        import librosa
        data, sr = librosa.load(path, sr=None, mono=True)
    else:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

    if sr != TARGET_SR:
        frac = Fraction(TARGET_SR, sr).limit_denominator(1000)
        data = resample_poly(data, frac.numerator, frac.denominator)

    if max_seconds is not None:
        max_samples = int(max_seconds * TARGET_SR)
        data = data[:max_samples]

    return np.asarray(data, dtype=np.float32)
