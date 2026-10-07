"""Speaker diarization (SD) on SiTa (data/sd_<lang>/: <name>.wav + <name>.rttm):
a clustering pipeline over frozen-upstream speaker embeddings, scored by DER.

Recordings are split train / dev / test (data_prep/make_splits.py). Per run:

1. Speech regions come from the reference RTTM (oracle speech activity), so the
   score measures speaker discrimination, not a speech detector.
2. A speaker-embedding head -- weighted sum over layers + statistics pooling +
   AM-softmax, as for ASV -- is trained on single-speaker stretches of the
   training recordings (every language's), each recording's speakers being
   separate classes. Like ASV, the layer mix is fixed first from a mean-pool
   speaker classifier, because every layer's frames would not fit on disk.
3. Each recording's speech is cut into overlapping windows, embedded, and
   clustered (agglomerative, average linkage, cosine distance). The number of
   speakers is never given: the distance threshold is tuned on the language's
   dev recordings, and the head is early-stopped on dev DER.
4. DER with a 0.25 s collar, overlapped speech included (pyannote.metrics);
   missed speech, false alarm and speaker confusion are reported as details.

Long recordings go through the upstream in CHUNK_SECONDS chunks: self-attention
over a whole 15-minute file would need tens of GB.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.cluster.hierarchy import fcluster, linkage

from slsb.tasks._common import fit_utterance_head, fit_with_dev, learning_rates, peak_gpu_gb, reset_peak_memory
from slsb.tasks.speaker_verification import SpeakerModel
from slsb.utils.audio import load_wav_16k_mono
from slsb.utils.datasets import TaskSpec, load_split

CHUNK_SECONDS = 30.0
FRAME_HOP = 320 / 16000          # upstream frame rate (20 ms)
FRAME_CENTER = 200 / 16000       # centre of frame 0's 400-sample receptive field
MIN_EMBED_FRAMES = 20            # shorter windows are tiled up to this


def discover(data_dir: Path, lang: str) -> list[str]:
    """Returns the RTTM basenames staged under data_dir/sd_<lang>/, if any."""
    task_dir = Path(data_dir) / f"sd_{lang}"
    if not task_dir.exists():
        return []
    return sorted(p.stem for p in task_dir.glob("*.rttm"))


def diarization_specs(data_dir: Path, dirnames) -> list[TaskSpec]:
    """sd_<lang> folders hold wav/rttm pairs, which discover_tasks() doesn't
    recognize, so their specs are built here."""
    specs = []
    for name in dirnames:
        task_dir = Path(data_dir) / name
        if discover(data_dir, name.removeprefix("sd_")):
            specs.append(TaskSpec(name=name, task="sd", lang=name.removeprefix("sd_"), kind="diarization",
                                  task_dir=task_dir, audio_dir=task_dir, label_path=task_dir))
    return specs


def read_rttm(path: Path, duration: float = None) -> list[tuple[float, float, str]]:
    """Speaker turns (start, end, speaker), clipped to the audio's duration when
    given -- SiTa's YT_49_BS has turns up to 680 s on 600 s of audio, and only
    the stretch the audio covers can be scored."""
    with open(path, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.startswith("SPEAKER")]
    turns = [(float(r[3]), float(r[3]) + float(r[4]), r[7]) for r in rows]
    if duration is not None:
        turns = [(s, min(e, duration), spk) for s, e, spk in turns if s < duration]
    return turns


def speech_regions(turns, min_gap: float = 0.0) -> list[tuple[float, float]]:
    """Union of the reference turns (oracle speech activity)."""
    regions = []
    for start, end, _ in sorted(turns):
        if regions and start <= regions[-1][1] + min_gap:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    return [tuple(r) for r in regions]


def single_speaker_spans(turns) -> list[tuple[float, float, str]]:
    """Stretches where exactly one reference speaker talks (for training)."""
    events = sorted({t for s, e, _ in turns for t in (s, e)})
    spans = []
    for a, b in zip(events, events[1:]):
        active = {spk for s, e, spk in turns if s < b and e > a}
        if len(active) == 1:
            spk = active.pop()
            if spans and spans[-1][2] == spk and abs(spans[-1][1] - a) < 1e-6:
                spans[-1] = (spans[-1][0], b, spk)
            else:
                spans.append((a, b, spk))
    return spans


def windows(start: float, end: float, length: float, hop: float) -> list[tuple[float, float]]:
    """Windows covering [start, end]; one window if the span is shorter than `length`."""
    if end - start <= length:
        return [(start, end)]
    starts = list(np.arange(start, end - length, hop)) + [end - length]
    return [(s, s + length) for s in starts]


@torch.no_grad()
def recording_frames(upstream, wav_path, layer_weights=None):
    """Frame features of a whole recording, chunk by chunk: (L, T, H) float16 on
    the upstream's device -- or (T, H) mixed with layer_weights -- plus each
    frame's centre time in seconds."""
    wav = load_wav_16k_mono(wav_path)
    chunk = int(CHUNK_SECONDS * 16000)
    pieces = [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pieces = [p for p in pieces if len(p) >= 400]
    feats, times = [], []
    for i in range(0, len(pieces), 4):
        batch = pieces[i:i + 4]
        hidden, mask = upstream.extract(batch)  # (L,B,T,H), (B,T)
        if layer_weights is not None:
            hidden = (hidden * layer_weights.to(hidden.device).view(-1, 1, 1, 1)).sum(0, keepdim=True)
        for j, n in enumerate(mask.sum(1).tolist()):
            feats.append(hidden[:, j, :n].half())
            times.append((i + j) * CHUNK_SECONDS + FRAME_CENTER + FRAME_HOP * np.arange(n))
    x = torch.cat(feats, dim=1)
    return (x[0] if layer_weights is not None else x), np.concatenate(times)


def frame_slice(times, start, end):
    a, b = np.searchsorted(times, [start, end])
    return a, max(b, a + 1)


@dataclass
class Recording:
    name: str
    lang: str
    turns: list
    frames: Path = None   # (T, H) float16 .npy of the mixed frames
    times: np.ndarray = None


def _recordings(task_dir: Path, names, lang):
    return [Recording(n, lang, read_rttm(task_dir / f"{n}.rttm", sf.info(str(task_dir / f"{n}.wav")).duration))
            for n in names]


def _layer_weights(upstream, train_recs, data_dir, cfg):
    """Layer mix from a mean-pool speaker classifier over training windows."""
    means, labels = [], []
    for rec in train_recs:
        x, times = recording_frames(upstream, data_dir / f"sd_{rec.lang}" / f"{rec.name}.wav")
        for start, end, spk in single_speaker_spans(rec.turns):
            for ws, we in windows(start, end, cfg["window_seconds"], cfg["window_seconds"]):
                if we - ws < cfg["min_train_seconds"]:
                    continue
                a, b = frame_slice(times, ws, we)
                means.append(x[:, a:b].float().mean(dim=1).cpu())
                labels.append(f"{rec.lang}/{rec.name}/{spk}")
        del x
    classes = sorted(set(labels))
    y = torch.tensor([classes.index(lab) for lab in labels])
    x = torch.stack(means).half()
    order = np.random.RandomState(0).permutation(len(y))
    cut = len(order) // 5
    dv, tr = order[:cut], order[cut:]
    device = upstream.device
    fit = fit_utterance_head(x[tr].to(device), y[tr].to(device), x[dv].to(device), y[dv].to(device), len(classes),
                             cfg["layer_mix"], cfg["layer_mix"]["lr_grid"], seed=0)
    return fit.model.weighted_sum.layer_weights().detach().cpu()


def _prepare_shared(upstream, spec, cfg, work_dir, shared):
    if "recordings" in shared:
        return shared
    data_dir = spec.task_dir.parent
    recordings = {}
    for lang_dir in sorted(data_dir.glob("sd_*")):
        lang = lang_dir.name.removeprefix("sd_")
        split = load_split(lang_dir)
        for part in ("train", "dev", "test"):
            recordings[(lang, part)] = _recordings(lang_dir, split[part], lang)
    train = [r for (lang, part), recs in recordings.items() if part == "train" for r in recs]
    weights = _layer_weights(upstream, train, data_dir, cfg)

    store = work_dir.parent / "sd_frames"
    store.mkdir(parents=True, exist_ok=True)
    for recs in recordings.values():
        for rec in recs:
            x, rec.times = recording_frames(upstream, data_dir / f"sd_{rec.lang}" / f"{rec.name}.wav", weights)
            rec.frames = store / f"{rec.lang}_{rec.name}.npy"
            np.save(rec.frames, x.cpu().numpy())

    # Training windows: (recording, start, end, class index), single-speaker only.
    items, classes = [], {}
    for rec in train:
        for start, end, spk in single_speaker_spans(rec.turns):
            for ws, we in windows(start, end, cfg["window_seconds"], cfg["window_seconds"]):
                if we - ws >= cfg["min_train_seconds"]:
                    key = f"{rec.lang}/{rec.name}/{spk}"
                    items.append((rec, ws, we, classes.setdefault(key, len(classes))))
    shared.update(recordings=recordings, train_items=items, num_classes=len(classes),
                  layer_weights=[round(float(w), 4) for w in weights], models={}, store=store)
    shared.setdefault("cleanup", []).append(lambda: [p.unlink() for p in store.glob("*.npy")])
    return shared


@dataclass
class SDFeatures:
    lang: str
    shared: dict
    device: torch.device

    def cleanup(self):
        pass  # frames are shared by every language; the runner cleans them up via `shared`


def prepare(upstream, spec, params, work_dir, shared):
    _prepare_shared(upstream, spec, params["sd"], work_dir, shared)
    return SDFeatures(spec.lang, shared, upstream.device)


def _load(rec: Recording):
    return np.load(rec.frames)


@torch.no_grad()
def embed_recording(model, rec, cfg, device):
    """Embeddings of overlapping windows over the recording's speech regions,
    with the span of the timeline each window is responsible for."""
    frames = torch.from_numpy(_load(rec)).to(device)
    crops, spans = [], []
    for start, end in speech_regions(rec.turns):
        wins = windows(start, end, cfg["window_seconds"], cfg["hop_seconds"])
        centres = [(a + b) / 2 for a, b in wins]
        cuts = [start] + [(c1 + c2) / 2 for c1, c2 in zip(centres, centres[1:])] + [end]
        for (ws, we), a, b in zip(wins, cuts, cuts[1:]):
            i, j = frame_slice(rec.times, ws, we)
            x = frames[i:j]
            if len(x) == 0:  # past the last frame (the audio's final ~20 ms)
                continue
            if len(x) < MIN_EMBED_FRAMES:
                x = x.repeat(int(np.ceil(MIN_EMBED_FRAMES / len(x))), 1)
            crops.append(x)
            spans.append((a, b))
    # Most windows have the same length: embed those in batches.
    embs = [None] * len(crops)
    by_length = {}
    for k, x in enumerate(crops):
        by_length.setdefault(len(x), []).append(k)
    for ks in by_length.values():
        for s in range(0, len(ks), 256):
            part = ks[s:s + 256]
            out = F.normalize(model.encoder(torch.stack([crops[k] for k in part]).float()), dim=1).cpu()
            for k, e in zip(part, out):
                embs[k] = e
    return torch.stack(embs).numpy(), spans


def cluster(embeddings, threshold):
    if len(embeddings) == 1:
        return np.ones(1, dtype=int)
    return fcluster(linkage(embeddings, method="average", metric="cosine"), t=threshold, criterion="distance")


def annotation(segments, uri):
    """Annotation from (start, end, label), merging touching turns of one label."""
    merged = []
    for start, end, label in sorted(segments):
        if merged and merged[-1][2] == label and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])
    ann = Annotation(uri=uri)
    for start, end, label in merged:
        ann[Segment(start, end)] = label
    return ann


def der_components(recs, embedded, threshold, collar):
    """Aggregate DER over recordings for one clustering threshold."""
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    for rec in recs:
        embs, spans = embedded[rec.name]
        labels = cluster(embs, threshold)
        metric(annotation(rec.turns, rec.name),
               annotation([(a, b, int(lab)) for (a, b), lab in zip(spans, labels)], rec.name))
    return metric


def best_threshold(recs, embedded, cfg):
    grid = np.linspace(cfg["thresholds"]["start"], cfg["thresholds"]["stop"], cfg["thresholds"]["num"])
    scored = [(float(abs(der_components(recs, embedded, t, cfg["collar"]))), float(t)) for t in grid]
    return min(scored)  # (der, threshold)


def _train(features, cfg, seed, lrs):
    shared, device = features.shared, features.device
    items = shared["train_items"]
    dev = {lang: recs for (lang, part), recs in shared["recordings"].items() if part == "dev"}
    input_size = np.load(items[0][0].frames, mmap_mode="r").shape[1]
    cache = {}

    def frames_of(rec):
        if rec.name not in cache:
            cache[rec.name] = torch.from_numpy(_load(rec))
        return cache[rec.name]

    def build():
        return SpeakerModel(input_size, shared["num_classes"], cfg).to(device)

    def train_epoch(model, optimizer, rng):
        order = rng.permutation(len(items))
        span = int(cfg["window_seconds"] / FRAME_HOP)
        for s in range(0, len(order), cfg["batch_size"]):
            batch = [items[k] for k in order[s:s + cfg["batch_size"]]]
            if len(batch) < 2:
                continue
            xs = []
            for rec, ws, we, _ in batch:
                i, j = frame_slice(rec.times, ws, we)
                x = frames_of(rec)[i:j]
                x = x.repeat(int(np.ceil(span / len(x))), 1)[:span] if len(x) < span else x[:span]
                xs.append(x)
            x = torch.stack(xs).to(device).float()
            y = torch.tensor([c for *_, c in batch], device=device)
            loss = model.loss(model.encoder(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def dev_der(model):
        model.eval()
        errors = total = 0.0
        for lang, recs in dev.items():
            embedded = {r.name: embed_recording(model, r, cfg, device) for r in recs}
            der, t = best_threshold(recs, embedded, cfg)
            metric = der_components(recs, embedded, t, cfg["collar"])
            errors += metric["confusion"] + metric["missed detection"] + metric["false alarm"]
            total += metric["total"]
        return float(errors / total)

    return fit_with_dev(build, train_epoch, dev_der, lrs, cfg["max_epochs"], cfg["patience"], seed,
                        maximize=False, verbose=True)


def run(features, params, seed, tuned):
    cfg = params["sd"]
    shared = features.shared
    reset_peak_memory(features.device)
    if seed not in shared["models"]:  # one head per seed serves every language
        fit = _train(features, cfg, seed, learning_rates(cfg, shared.setdefault("tuned", {})))
        shared["tuned"].setdefault("lr", fit.lr)
        shared["models"][seed] = fit
    fit = shared["models"][seed]
    model = fit.model.eval()

    recs = shared["recordings"]
    dev_embedded = {r.name: embed_recording(model, r, cfg, features.device) for r in recs[(features.lang, "dev")]}
    dev_der, threshold = best_threshold(recs[(features.lang, "dev")], dev_embedded, cfg)
    test = recs[(features.lang, "test")]
    test_embedded = {r.name: embed_recording(model, r, cfg, features.device) for r in test}
    metric = der_components(test, test_embedded, threshold, cfg["collar"])
    total = metric["total"]
    metrics_out = {"der": float(abs(metric))}
    perf = {"train_seconds": fit.train_seconds, "peak_gpu_gb": peak_gpu_gb(features.device)}
    details = {
        "head": cfg["head"], "lr": fit.lr, "best_epoch": fit.best_epoch, "threshold": threshold,
        "dev_der": dev_der, "missed": float(metric["missed detection"] / total),
        "false_alarm": float(metric["false alarm"] / total), "confusion": float(metric["confusion"] / total),
        "test_recordings": len(test),
        "train_classes": shared["num_classes"], "layer_weights": shared["layer_weights"],
    }
    return metrics_out, perf, "recording_disjoint", details

