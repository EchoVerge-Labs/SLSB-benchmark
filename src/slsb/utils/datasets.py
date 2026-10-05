"""Task auto-discovery and dataset classes for the frozen-upstream benchmark.

<data_dir>/<task>_<lang>/ folders are auto-detected by which label file they contain:
  labels.csv       -> classification (ER/SID/LID/KS/IC)
  transcripts.csv  -> ASR
  trials_<lang>.csv (inside a folder with no <lang> suffix, e.g. <data_dir>/asv/) -> verification

Never hardcode a task list: adding a new <data_dir>/<task>_<lang>/ folder with one
of these three files is picked up automatically, no code changes needed.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional



@dataclass
class TaskSpec:
    name: str          # e.g. "er_sinhala", "asr_sinhala", "asv_tamil", "sid"
    task: str          # e.g. "er", "asr", "asv", "sid"
    lang: str          # e.g. "sinhala", "tamil", or "multilingual"
    kind: str          # "classification" | "asr" | "verification"
    task_dir: Path
    audio_dir: Path
    label_path: Path


def _split_task_lang(dirname: str) -> tuple[str, str]:
    if "_" not in dirname:
        return dirname, "multilingual"
    task, lang = dirname.rsplit("_", 1)
    return task, lang


def _resolve_audio_dir(entry: Path) -> Path:
    """Normally <task_dir>/audio/, unless audio_dir.txt points elsewhere (e.g. data/sid/
    sharing data/asv/audio/ instead of duplicating files) -- lets a task share another
    task's audio without any code change here."""
    marker = entry / "audio_dir.txt"
    if marker.exists():
        return (entry / marker.read_text().strip()).resolve()
    return entry / "audio"


def discover_tasks(data_dir: Path) -> list[TaskSpec]:
    data_dir = Path(data_dir)
    specs = []
    if not data_dir.exists():
        return specs

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue

        labels_csv = entry / "labels.csv"
        transcripts_csv = entry / "transcripts.csv"
        trial_files = sorted(entry.glob("trials_*.csv"))
        audio_dir = _resolve_audio_dir(entry)

        if trial_files:
            for trial_path in trial_files:
                lang = trial_path.stem.removeprefix("trials_")
                specs.append(TaskSpec(
                    name=f"{entry.name}_{lang}", task=entry.name, lang=lang,
                    kind="verification", task_dir=entry, audio_dir=audio_dir,
                    label_path=trial_path,
                ))
        elif labels_csv.exists():
            task, lang = _split_task_lang(entry.name)
            specs.append(TaskSpec(
                name=entry.name, task=task, lang=lang, kind="classification",
                task_dir=entry, audio_dir=audio_dir, label_path=labels_csv,
            ))
        elif transcripts_csv.exists():
            task, lang = _split_task_lang(entry.name)
            specs.append(TaskSpec(
                name=entry.name, task=task, lang=lang, kind="asr",
                task_dir=entry, audio_dir=audio_dir, label_path=transcripts_csv,
            ))
    return specs


def find_unrecognized_dirs(data_dir: Path) -> list[str]:
    """Folders under data_dir with none of labels.csv / transcripts.csv / trials_*.csv --
    e.g. sd_sinhala/sd_tamil (RTTM-only, needs the separate DER runner in tasks/speaker_diarization.py)."""
    data_dir = Path(data_dir)
    unrecognized = []
    if not data_dir.exists():
        return unrecognized
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        has_labels = (entry / "labels.csv").exists()
        has_transcripts = (entry / "transcripts.csv").exists()
        has_trials = bool(list(entry.glob("trials_*.csv")))
        if not (has_labels or has_transcripts or has_trials):
            unrecognized.append(entry.name)
    return unrecognized


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def derive_speaker_group(filename: str) -> str:
    """Speaker/group id = everything before the last underscore in the stem
    (e.g. "<hash>_428.wav" -> "<hash>"). Degrades gracefully: a filename with
    no underscore becomes its own singleton group, so ungroupable tasks (e.g.
    asr_sinhala's plain hex ids) fall through to an effectively random split."""
    return Path(filename).stem.rsplit("_", 1)[0]


def parse_speaker_sentence(filename: str) -> tuple[str, Optional[str]]:
    """For '<speaker>_<sentence>_<...>.wav' filenames (the ER convention): returns
    (speaker, sentence). Distinct from derive_speaker_group (which is for ASR's
    '<hash>_<NNN>.wav' convention and splits on the *last* underscore instead)."""
    parts = Path(filename).stem.split("_")
    return parts[0], (parts[1] if len(parts) > 1 else None)


def build_char_vocab(transcripts: list[str]) -> dict:
    chars = sorted(set("".join(transcripts)))
    vocab = {"<blank>": 0, "<unk>": 1, " ": 2}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def load_split(spec_or_dir) -> dict:
    """The task's fixed v0.2 partition (<task_dir>/split_v2.json, written by
    data_prep/make_splits.py): train/dev/test lists, or ER's speaker folds, or
    ASV's training speakers plus dev trials."""
    task_dir = Path(getattr(spec_or_dir, "task_dir", spec_or_dir))
    path = task_dir / "split_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python data_prep/make_splits.py` first")
    return json.loads(path.read_text())


def read_labels(spec: TaskSpec) -> dict:
    """filename -> class label, from labels.csv."""
    return {r["filename"]: r["label"] for r in _read_csv_rows(spec.label_path)}


def read_transcripts(spec: TaskSpec) -> dict:
    """filename -> transcript, from transcripts.csv."""
    return {r["filename"]: r["transcript"] for r in _read_csv_rows(spec.label_path)}


def load_vocab(spec: TaskSpec, transcripts: list[str]) -> dict:
    vocab_path = spec.task_dir / "vocab.json"
    if vocab_path.exists():
        return json.loads(vocab_path.read_text())
    vocab = build_char_vocab(transcripts)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    return vocab


@dataclass
class VerificationTrials:
    spec: TaskSpec
    pairs: list  # (label:int, wav1_path:Path, wav2_path:Path)
    unique_wavs: list = field(default_factory=list)


def load_verification_task(spec: TaskSpec) -> VerificationTrials:
    rows = _read_csv_rows(spec.label_path)
    pairs = [(int(r["label"]), spec.audio_dir / r["wav1"], spec.audio_dir / r["wav2"]) for r in rows]
    unique = sorted({str(p) for _, w1, w2 in pairs for p in (w1, w2)})
    return VerificationTrials(spec=spec, pairs=pairs, unique_wavs=unique)
