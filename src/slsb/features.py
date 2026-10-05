"""Frozen-upstream feature extraction, done once per (upstream, task).

The upstream never trains, so a clip's hidden states are identical in every
epoch, run seed and learning-rate trial. Extracting them once and training the
heads on stored features is what makes long, dev-early-stopped training of
SUPERB-sized heads affordable.

- layer_means(): each clip's per-layer mean over frames, (N, L, H). Exact for
  the mean-pool heads, because mean-pooling commutes with the learned weighted
  sum over layers.
- extract_frames(): frame-level features on disk, one fp16 .npy per clip --
  every layer (L, T, H) when the head learns its own layer weights (ASR), or a
  single fixed layer mix (T, H) where every layer would be too large (ASV).
"""
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from slsb.utils.audio import TARGET_SR, load_wav_16k_mono

# One forward pass holds at most this many clips and this much padded audio.
MAX_CLIPS_PER_PASS = 16
MAX_SAMPLES_PER_PASS = 8_000_000  # ~500 s


def audio_samples(path, max_seconds: float = None) -> int:
    """Length in 16 kHz samples, from the file header when soundfile can read it."""
    try:
        info = sf.info(str(path))
        n = int(info.frames * TARGET_SR / info.samplerate)
    except RuntimeError:  # e.g. mp3
        n = len(load_wav_16k_mono(path))
    return min(n, int(max_seconds * TARGET_SR)) if max_seconds else n


def length_batches(lengths, max_items: int, max_padded: int) -> list[list[int]]:
    """Index batches, longest items first, each holding at most max_items and at
    most max_padded padded units (items x longest item)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, current = [], []
    for i in order:
        # descending order: a batch's first item is its longest
        if current and (len(current) >= max_items or lengths[current[0]] * (len(current) + 1) > max_padded):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def _forward(upstream, paths, max_seconds):
    lengths = [audio_samples(p, max_seconds) for p in paths]
    for batch in length_batches(lengths, MAX_CLIPS_PER_PASS, MAX_SAMPLES_PER_PASS):
        waveforms = [load_wav_16k_mono(paths[i], max_seconds=max_seconds) for i in batch]
        hidden_states, frame_mask = upstream.extract(waveforms)  # (L,B,T,H), (B,T)
        yield batch, hidden_states, frame_mask


@torch.no_grad()
def layer_means(upstream, paths, max_seconds: float = None) -> torch.Tensor:
    """(N, L, H) float16 on CPU: each clip's mean over its real frames, per layer."""
    out = torch.empty(len(paths), upstream.num_hidden_states, upstream.hidden_size, dtype=torch.float16)
    for batch, hidden_states, frame_mask in _forward(upstream, paths, max_seconds):
        mask = frame_mask.to(hidden_states.dtype)[None, :, :, None]  # (1,B,T,1)
        means = (hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (L,B,H)
        out[batch] = means.transpose(0, 1).half().cpu()
    return out


class FrameStore:
    """Frame-level features of a list of clips, one fp16 .npy per clip under root."""

    def __init__(self, root: Path, lengths: list[int]):
        self.root = Path(root)
        self.lengths = lengths  # frames per clip

    def __len__(self):
        return len(self.lengths)

    def load(self, i: int, mmap: bool = False) -> np.ndarray:
        """The clip's frames; mmap=True maps the file read-only instead of reading
        it, for when only a slice is needed."""
        return np.load(self.root / f"{i:06d}.npy", mmap_mode="r" if mmap else None)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


@torch.no_grad()
def extract_frames(upstream, paths, root: Path, max_seconds: float = None,
                   layer_weights: torch.Tensor = None) -> FrameStore:
    """Writes each clip's frames to root/<index>.npy: (L, T, H), or (T, H) mixed
    with the given (L,) layer_weights. Audio longer than max_seconds is cropped."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    weights = None if layer_weights is None else layer_weights.to(upstream.device).view(-1, 1, 1, 1)
    lengths = [0] * len(paths)
    for batch, hidden_states, frame_mask in _forward(upstream, paths, max_seconds):
        if weights is not None:
            hidden_states = (hidden_states * weights).sum(dim=0, keepdim=True)
        frames = frame_mask.sum(dim=1).tolist()
        for j, i in enumerate(batch):
            x = hidden_states[:, j, :frames[j]].half().cpu().numpy()
            np.save(root / f"{i:06d}.npy", x[0] if weights is not None else x)
            lengths[i] = frames[j]
    return FrameStore(root, lengths)
