#!/usr/bin/env python3
"""Write <task_dir>/split_v2.json for every benchmark task: the fixed
train/dev/test partition the v0.2 protocol trains and scores on.

Splits are data, not run-time state: they are made once here with a fixed
seed, versioned with data/, and every upstream and every run seed sees the
same partition. The dev set is what learning rates and stopping epochs are
chosen on; the test set is scored once at the end.

  ASR (transcripts.csv)  speaker-disjoint train/dev/test (70/10/20). Speakers
                         come from speakers.csv when the filenames don't carry
                         them (asr_sinhala). A v0.1 test set that already is
                         speaker-disjoint is kept, so its scores stay comparable.
                         A single-speaker task (asr_omni_sinhala) can only be
                         split at random: split_type single_speaker_random.
  ER  (er_*)             speaker-disjoint k-fold cross-validation. Each sentence
                         is recorded in most emotions, so sentence overlap
                         doesn't give the label away -- speakers are what leak.
  Other classification   closed-set (e.g. SID: every speaker is a class), so a
                         stratified split; the v0.1 test set is kept.
  ASV                    the trial lists are the test set. Training speakers
                         (train_labels.csv, see prep_slceleb_train.py) are split
                         into train / dev speakers, and dev trials are drawn
                         from the dev speakers for model selection.

Re-running overwrites split_v2.json with identical content.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from slsb.utils.datasets import derive_speaker_group, discover_tasks, parse_speaker_sentence

SPLIT_SEED = 1234
TEST_FRACTION = 0.2
DEV_FRACTION = 0.1  # of the whole task
ER_FOLDS = 5
ASV_DEV_SPEAKER_FRACTION = 0.1
ASV_DEV_GENUINE_PER_SPEAKER = 300


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_split(task_dir: Path, split: dict):
    (task_dir / "split_v2.json").write_text(json.dumps(split, indent=2, ensure_ascii=False))


def group_split(ids, groups, fraction, seed):
    """(kept, held_out) with whole groups held out, held_out ~ fraction of ids."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    keep_idx, held_idx = next(splitter.split(ids, groups=groups))
    return [ids[i] for i in keep_idx], [ids[i] for i in held_idx]


def overlap(a, b, group_of):
    return len({group_of[i] for i in a} & {group_of[i] for i in b})


def asr_split(spec) -> dict:
    ids = [r["filename"] for r in read_rows(spec.label_path)]
    speakers_csv = spec.task_dir / "speakers.csv"
    if speakers_csv.exists():
        group_of = {r["filename"]: r["speaker"] for r in read_rows(speakers_csv)}
        group_source = "speakers.csv"
    else:
        group_of = {i: derive_speaker_group(i) for i in ids}
        group_source = "filename"

    if len(set(group_of.values())) < 2:
        # One speaker (asr_omni_sinhala): a speaker-dependent test, labelled as such.
        rest, test = train_test_split(ids, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
        train, dev = train_test_split(rest, test_size=DEV_FRACTION / (1 - TEST_FRACTION), random_state=SPLIT_SEED)
        return {
            "protocol": "v0.2", "split_type": "single_speaker_random", "train": train, "dev": dev, "test": test,
            "diagnostics": {"speakers_from": group_source, "train_clips": len(train), "dev_clips": len(dev),
                            "test_clips": len(test), "speakers": 1},
        }

    v1_path = spec.task_dir / f"split_{spec.label_path.stem}.json"
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else None
    if v1 and set(v1["train"]) | set(v1["test"]) == set(ids) and overlap(v1["train"], v1["test"], group_of) == 0:
        rest, test = v1["train"], v1["test"]
        test_source = "kept from v0.1"
    else:
        rest, test = group_split(ids, [group_of[i] for i in ids], TEST_FRACTION, SPLIT_SEED)
        test_source = "new (v0.1 test set was not speaker-disjoint)" if v1 else "new"

    dev_fraction = DEV_FRACTION / (1 - len(test) / len(ids))
    train, dev = group_split(rest, [group_of[i] for i in rest], dev_fraction, SPLIT_SEED)
    parts = {"train": train, "dev": dev, "test": test}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert overlap(parts[a], parts[b], group_of) == 0, f"{spec.name}: speakers shared by {a} and {b}"
    return {
        "protocol": "v0.2", "split_type": "speaker_disjoint", **parts,
        "diagnostics": {
            "speakers_from": group_source, "test_set": test_source,
            **{f"{k}_clips": len(v) for k, v in parts.items()},
            **{f"{k}_speakers": len({group_of[i] for i in v}) for k, v in parts.items()},
        },
    }


def closed_set_split(spec) -> dict:
    rows = read_rows(spec.label_path)
    label_of = {r["filename"]: r["label"] for r in rows}
    ids = list(label_of)

    v1_path = spec.task_dir / f"split_{spec.label_path.stem}.json"
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else None
    if v1 and set(v1["train"]) | set(v1["test"]) == set(ids):
        rest, test = v1["train"], v1["test"]
        test_source = "kept from v0.1"
    else:
        rest, test = train_test_split(ids, test_size=TEST_FRACTION, random_state=SPLIT_SEED,
                                      stratify=[label_of[i] for i in ids])
        test_source = "new"
    dev_fraction = DEV_FRACTION / (1 - len(test) / len(ids))
    train, dev = train_test_split(rest, test_size=dev_fraction, random_state=SPLIT_SEED,
                                  stratify=[label_of[i] for i in rest])
    parts = {"train": train, "dev": dev, "test": test}
    return {
        "protocol": "v0.2", "split_type": "stratified_closed_set", **parts,
        "diagnostics": {"test_set": test_source, **{f"{k}_clips": len(v) for k, v in parts.items()},
                        "classes": len(set(label_of.values()))},
    }


def kfold_split(spec) -> dict:
    rows = read_rows(spec.label_path)
    label_of = {r["filename"]: r["label"] for r in rows}
    ids = sorted(label_of)
    speaker_of = {i: parse_speaker_sentence(i)[0] for i in ids}
    splitter = GroupKFold(n_splits=ER_FOLDS)
    folds = [[ids[i] for i in fold_idx]
             for _, fold_idx in splitter.split(ids, groups=[speaker_of[i] for i in ids])]
    for a in range(ER_FOLDS):
        for b in range(a + 1, ER_FOLDS):
            assert overlap(folds[a], folds[b], speaker_of) == 0
    return {
        "protocol": "v0.2", "split_type": f"speaker_disjoint_{ER_FOLDS}fold", "folds": folds,
        "diagnostics": {
            "fold_clips": [len(f) for f in folds],
            "fold_speakers": [len({speaker_of[i] for i in f}) for f in folds],
            "fold_label_counts": [dict(Counter(label_of[i] for i in f)) for f in folds],
        },
    }


def asv_split(asv_dir: Path) -> dict:
    rows = read_rows(asv_dir / "train_labels.csv")
    speaker_of = {r["filename"]: r["label"] for r in rows}
    by_lang = defaultdict(set)
    for filename, speaker in speaker_of.items():
        by_lang[filename.split("/")[0]].add(speaker)

    rng = random.Random(SPLIT_SEED)
    dev_speakers = set()
    for lang in sorted(by_lang):
        speakers = sorted(by_lang[lang])
        rng.shuffle(speakers)
        dev_speakers.update(speakers[:max(2, round(ASV_DEV_SPEAKER_FRACTION * len(speakers)))])

    train = sorted(f for f, s in speaker_of.items() if s not in dev_speakers)
    dev = sorted(f for f, s in speaker_of.items() if s in dev_speakers)
    clips_of = defaultdict(list)
    for f in dev:
        clips_of[speaker_of[f]].append(f)

    # Balanced dev trials: same-speaker pairs, and as many different-speaker pairs
    # whose second clip comes from another dev speaker of the same language.
    trials = []
    dev_by_lang = {lang: sorted(s for s in by_lang[lang] if s in dev_speakers) for lang in by_lang}
    for speaker in sorted(clips_of):
        clips = clips_of[speaker]
        lang = clips[0].split("/")[0]
        others = [s for s in dev_by_lang[lang] if s != speaker] or [s for s in clips_of if s != speaker]
        for _ in range(ASV_DEV_GENUINE_PER_SPEAKER):
            a, b = rng.sample(clips, 2)
            trials.append([1, a, b])
            trials.append([0, a, rng.choice(clips_of[rng.choice(others)])])
    return {
        "protocol": "v0.2", "split_type": "speaker_disjoint_from_trials",
        "train": train, "dev": dev, "dev_trials": trials,
        "diagnostics": {
            "train_clips": len(train), "dev_clips": len(dev),
            "train_speakers": len({speaker_of[f] for f in train}), "dev_speakers": len(dev_speakers),
            "dev_trials": len(trials),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    for spec in discover_tasks(args.data_dir):
        if spec.kind == "asr":
            split = asr_split(spec)
        elif spec.kind == "classification" and spec.task == "er":
            split = kfold_split(spec)
        elif spec.kind == "classification":
            split = closed_set_split(spec)
        else:
            continue
        write_split(spec.task_dir, split)
        print(f"{spec.name}: {split['split_type']}  {split['diagnostics']}")

    asv_dir = args.data_dir / "asv"
    if (asv_dir / "train_labels.csv").exists():
        split = asv_split(asv_dir)
        write_split(asv_dir, split)
        print(f"asv: {split['split_type']}  {split['diagnostics']}")
    else:
        print(f"WARNING: {asv_dir / 'train_labels.csv'} missing -- run data_prep/prep_slceleb_train.py")


if __name__ == "__main__":
    main()
