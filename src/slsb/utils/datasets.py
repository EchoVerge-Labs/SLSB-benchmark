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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from slsb.utils.audio import load_wav_16k_mono, MAX_PROBE_DURATION_SECONDS

TEST_SIZE = 0.2


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


def get_or_create_split(spec: TaskSpec, ids: list[str], stratify_labels: Optional[list] = None,
                         groups: Optional[list] = None, seed: int = 42):
    """Reproducible train/test split, cached to task_dir/split_<task>.json (keyed by label file name).

    If `groups` is given and there are >=2 unique groups, splits are group-disjoint
    (GroupShuffleSplit) -- e.g. no speaker in both train and test for ASR. Falls back
    to a stratified/random split if grouping isn't meaningful (a single group, or no
    groups given at all)."""
    split_path = spec.task_dir / f"split_{spec.label_path.stem}.json"
    if split_path.exists():
        cached = json.loads(split_path.read_text())
        if set(cached["train"]) | set(cached["test"]) == set(ids):
            return cached["train"], cached["test"], cached.get("split_type", "random")

    can_group = groups is not None and len(set(groups)) >= 2
    if can_group:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(splitter.split(ids, groups=groups))
        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]
        split_type = "speaker_disjoint"
    else:
        if groups is not None:
            print(f"  NOTE: only {len(set(groups))} unique group(s) in {spec.name} -- "
                  f"can't make a group-disjoint split, falling back to a random split")
        can_stratify = stratify_labels is not None and min(
            [list(stratify_labels).count(v) for v in set(stratify_labels)]
        ) >= 2
        train_ids, test_ids = train_test_split(
            ids, test_size=TEST_SIZE, random_state=seed,
            stratify=stratify_labels if can_stratify else None,
        )
        split_type = "random"
    split_path.write_text(json.dumps({"train": train_ids, "test": test_ids, "split_type": split_type}, indent=2))
    return train_ids, test_ids, split_type


class ClassificationDataset:
    def __init__(self, spec: TaskSpec, ids: list[str], id_to_label: dict, label_to_idx: dict):
        self.spec = spec
        self.ids = ids
        self.id_to_label = id_to_label
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        filename = self.ids[i]
        wav = load_wav_16k_mono(self.spec.audio_dir / filename, max_seconds=MAX_PROBE_DURATION_SECONDS)
        label = self.label_to_idx[self.id_to_label[filename]]
        return wav, label


def load_classification_task(spec: TaskSpec, seed: int = 42):
    rows = _read_csv_rows(spec.label_path)
    id_to_label = {r["filename"]: r["label"] for r in rows}
    ids = list(id_to_label.keys())
    labels = [id_to_label[i] for i in ids]
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}

    train_ids, test_ids, _split_type = get_or_create_split(spec, ids, stratify_labels=labels, seed=seed)
    train_ds = ClassificationDataset(spec, train_ids, id_to_label, label_to_idx)
    test_ds = ClassificationDataset(spec, test_ids, id_to_label, label_to_idx)
    return train_ds, test_ds, classes


def parse_speaker_sentence(filename: str) -> tuple[str, Optional[str]]:
    """For '<speaker>_<sentence>_<...>.wav' filenames (the ER convention): returns
    (speaker, sentence). Distinct from derive_speaker_group (which is for ASR's
    '<hash>_<NNN>.wav' convention and splits on the *last* underscore instead)."""
    parts = Path(filename).stem.split("_")
    return parts[0], (parts[1] if len(parts) > 1 else None)


def find_connected_blocks(speaker_of: dict, sentence_of: dict) -> dict:
    """Union-find over speakers: two speakers land in the same block iff they share
    at least one sentence id. Discovers latent block structure (e.g. a corpus split
    into disjoint speaker/sentence pools) without hardcoding it for any one dataset."""
    sentence_to_speakers = defaultdict(set)
    for filename, spk in speaker_of.items():
        sent = sentence_of[filename]
        if sent is not None:
            sentence_to_speakers[sent].add(spk)

    parent = {s: s for s in set(speaker_of.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for sset in sentence_to_speakers.values():
        sset = list(sset)
        for s in sset[1:]:
            union(sset[0], s)

    return {s: find(s) for s in parent}


def load_classification_task_leakage_free(spec: TaskSpec, seed: int = 42, test_size: float = 0.3):
    """Leakage-free variant of load_classification_task for ER-style
    '<speaker>_<sentence>_<...>.wav' data. Cached separately from the plain random
    split (split_<label>_leakagefree.json) so both old (leaky) and new (clean)
    results stay reproducible side by side.

    - Single-speaker corpus (speaker id unusable): falls back to a pure
      sentence-disjoint split (GroupShuffleSplit on sentence id).
    - Multi-speaker, fully crossed (every speaker shares sentences with every
      other): plain speaker-disjoint split (sentence overlap is then
      unavoidable without discarding data -- reported honestly, not hidden).
    - Multi-speaker with distinct speaker/sentence blocks: blocks that alone
      cover every label are held out whole for test/train (zero sentence
      leakage there); a block missing a label is speaker-split internally so
      that label stays evaluable on both sides (the only sentence overlap
      that can occur, confined to that block, and reported).

    Returns (train_ds, test_ds, classes, split_type, diagnostics).
    """
    rows = _read_csv_rows(spec.label_path)
    id_to_label = {r["filename"]: r["label"] for r in rows}
    ids = list(id_to_label.keys())
    classes = sorted(set(id_to_label.values()))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    all_labels = set(id_to_label.values())

    cache_path = spec.task_dir / f"split_{spec.label_path.stem}_leakagefree.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if set(cached["train"]) | set(cached["test"]) == set(ids):
            train_ids, test_ids, split_type = cached["train"], cached["test"], cached["split_type"]
            train_ds = ClassificationDataset(spec, train_ids, id_to_label, label_to_idx)
            test_ds = ClassificationDataset(spec, test_ids, id_to_label, label_to_idx)
            return train_ds, test_ds, classes, split_type, cached["diagnostics"]

    speaker_of, sentence_of = {}, {}
    for i in ids:
        spk, sent = parse_speaker_sentence(i)
        speaker_of[i] = spk
        sentence_of[i] = sent
    unique_speakers = set(speaker_of.values())

    if len(unique_speakers) < 2:
        groups = [sentence_of[i] for i in ids]
        if len(set(groups)) < 2:
            raise ValueError(f"{spec.name}: no usable speaker or sentence id for a leakage-free split")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(ids, groups=groups))
        train_ids, test_ids = [ids[i] for i in train_idx], [ids[i] for i in test_idx]
        split_type = "sentence_disjoint"
    else:
        blocks = find_connected_blocks(speaker_of, sentence_of)
        block_ids = sorted(set(blocks.values()))
        block_labels = {b: set() for b in block_ids}
        for i in ids:
            block_labels[blocks[speaker_of[i]]].add(id_to_label[i])

        partial_blocks = [b for b in block_ids if block_labels[b] != all_labels]
        full_blocks = [b for b in block_ids if block_labels[b] == all_labels]

        if len(block_ids) == 1 or not partial_blocks or not full_blocks:
            groups = [speaker_of[i] for i in ids]
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(ids, groups=groups))
            train_ids, test_ids = [ids[i] for i in train_idx], [ids[i] for i in test_idx]
            split_type = "speaker_sentence_disjoint"
        else:
            held_ids = [i for i in ids if blocks[speaker_of[i]] in partial_blocks]
            splittable_ids = [i for i in ids if blocks[speaker_of[i]] in full_blocks]
            target_test_total = test_size * len(ids)
            inner_test_size = min(0.9, max(0.1, target_test_total / len(splittable_ids)))

            groups = [speaker_of[i] for i in splittable_ids]
            splitter = GroupShuffleSplit(n_splits=1, test_size=inner_test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(splittable_ids, groups=groups))
            train_ids = held_ids + [splittable_ids[i] for i in train_idx]
            test_ids = [splittable_ids[i] for i in test_idx]
            split_type = "speaker_sentence_disjoint_partial"

    train_speakers = {speaker_of[i] for i in train_ids}
    test_speakers = {speaker_of[i] for i in test_ids}
    train_sentences = {sentence_of[i] for i in train_ids if sentence_of[i] is not None}
    test_sentences = {sentence_of[i] for i in test_ids if sentence_of[i] is not None}
    diagnostics = {
        "train_size": len(train_ids), "test_size": len(test_ids),
        "train_speakers": len(train_speakers), "test_speakers": len(test_speakers),
        "speaker_overlap": len(train_speakers & test_speakers),
        "train_sentences": len(train_sentences), "test_sentences": len(test_sentences),
        "sentence_overlap": len(train_sentences & test_sentences),
        "train_label_counts": {c: sum(1 for i in train_ids if id_to_label[i] == c) for c in classes},
        "test_label_counts": {c: sum(1 for i in test_ids if id_to_label[i] == c) for c in classes},
    }
    cache_path.write_text(json.dumps(
        {"train": train_ids, "test": test_ids, "split_type": split_type, "diagnostics": diagnostics}, indent=2))

    train_ds = ClassificationDataset(spec, train_ids, id_to_label, label_to_idx)
    test_ds = ClassificationDataset(spec, test_ids, id_to_label, label_to_idx)
    return train_ds, test_ds, classes, split_type, diagnostics


class ASRDataset:
    def __init__(self, spec: TaskSpec, ids: list[str], id_to_transcript: dict):
        self.spec = spec
        self.ids = ids
        self.id_to_transcript = id_to_transcript

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        filename = self.ids[i]
        wav = load_wav_16k_mono(self.spec.audio_dir / filename)
        return wav, self.id_to_transcript[filename]


def build_char_vocab(transcripts: list[str]) -> dict:
    chars = sorted(set("".join(transcripts)))
    vocab = {"<blank>": 0, "<unk>": 1, " ": 2}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def load_asr_task(spec: TaskSpec, seed: int = 42):
    rows = _read_csv_rows(spec.label_path)
    id_to_transcript = {r["filename"]: r["transcript"] for r in rows}
    ids = list(id_to_transcript.keys())

    vocab_path = spec.task_dir / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
    else:
        vocab = build_char_vocab(list(id_to_transcript.values()))
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))

    groups = [derive_speaker_group(i) for i in ids]
    train_ids, test_ids, split_type = get_or_create_split(spec, ids, stratify_labels=None, groups=groups, seed=seed)
    train_ds = ASRDataset(spec, train_ids, id_to_transcript)
    test_ds = ASRDataset(spec, test_ids, id_to_transcript)
    return train_ds, test_ds, vocab, split_type


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
