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
  ER, IC                 speaker-disjoint k-fold cross-validation, speakers from
                         speakers.csv or (ER) the filename. For ER each sentence
                         is recorded in most emotions, so sentence overlap
                         doesn't give the label away -- speakers are what leak.
  IC without speakers    (ic_banking_sinhala) a stratified random split, labelled
                         random_stratified_no_speaker_ids: speakers may be shared.
  SID                    closed set (every speaker is a class), split by source
                         VIDEO: a speaker's test videos are never seen in
                         training (v0.4). Identical audio is kept to one copy,
                         and audio filed under more than one label is dropped
                         (SLCeleb repeats ~1/3 of its clips, some under
                         different ids).
  SD (sd_*)              recordings split train / dev / test (40/20/40): train
                         for the speaker-embedding head, dev for the clustering
                         threshold and early stopping, test for DER.
  ASV                    the trial lists are the test set. Training speakers
                         (train_labels.csv, see prep_slceleb_train.py) are split
                         into train / dev speakers, and dev trials are drawn
                         from the dev speakers for model selection.

Every split is checked so no audio (by decoded samples) is in more than one
part, and no ASV training clip is identical to a clip the trials use.

Re-running overwrites split_v2.json with identical content.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
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
SD_FRACTIONS = {"train": 0.4, "dev": 0.2}  # the rest is test


def audio_digest(path: Path) -> str:
    """md5 of the decoded 16-bit samples: equal for the same audio in any file."""
    data, _ = sf.read(str(path), dtype="int16")
    return hashlib.md5(data.tobytes()).hexdigest()


def assert_no_shared_audio(name: str, parts: dict, audio_dir: Path):
    """No audio may sit in more than one part (train/dev/test, or folds)."""
    owner = {}
    for part, files in parts.items():
        for f in files:
            d = audio_digest(audio_dir / f)
            if owner.setdefault(d, part) != part:
                raise AssertionError(f"{name}: {f} ({part}) has the same audio as a clip in {owner[d]}")


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


def deduplicated_labels(spec):
    """(label_of, ids, diagnostics): one copy of repeated audio is kept, and audio
    filed under more than one label is dropped."""
    label_of = {r["filename"]: r["label"] for r in read_rows(spec.label_path)}
    by_audio = defaultdict(list)
    for f in sorted(label_of):
        by_audio[audio_digest(spec.audio_dir / f)].append(f)
    ambiguous = [fs for fs in by_audio.values() if len({label_of[f] for f in fs}) > 1]
    ids = [fs[0] for fs in by_audio.values() if len({label_of[f] for f in fs}) == 1]
    return label_of, ids, {
        "classes": len({label_of[i] for i in ids}), "files": len(label_of),
        "duplicate_copies_dropped": len(label_of) - len(ids) - sum(len(fs) for fs in ambiguous),
        "ambiguous_label_files_dropped": sum(len(fs) for fs in ambiguous),
    }


def slceleb_video(filename: str) -> str:
    """'id10001/interview/interview-03-012.wav' -> 'id10001/interview-03': the
    source video a SLCeleb clip was cut from."""
    speaker, _genre, name = filename.split("/")
    return f"{speaker}/{Path(name).stem.rsplit('-', 1)[0]}"


def video_disjoint_split(spec) -> dict:
    """Closed-set speaker ID where test videos are never seen in training: per
    speaker, ~20% of its videos (at least one) are test, one more is dev, the
    rest train. Clips of one video share microphone, room and session, so a
    random clip split lets a model recognise the recording instead of the voice
    (v0.3: 100% of test clips had sibling clips from the same video in training)."""
    label_of, ids, diagnostics = deduplicated_labels(spec)
    videos = defaultdict(lambda: defaultdict(list))
    for f in ids:
        videos[label_of[f]][slceleb_video(f)].append(f)
    rng = random.Random(SPLIT_SEED)
    parts = {"train": [], "dev": [], "test": []}
    for speaker in sorted(videos):
        names = sorted(videos[speaker])
        rng.shuffle(names)
        n_test = max(1, round(TEST_FRACTION * len(names)))
        if len(names) - n_test < 2:
            raise ValueError(f"{spec.name}: speaker {speaker} has only {len(names)} videos")
        assignment = {"test": names[:n_test], "dev": names[n_test:n_test + 1], "train": names[n_test + 1:]}
        for part, chosen in assignment.items():
            for name in chosen:
                parts[part].extend(sorted(videos[speaker][name]))
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not {slceleb_video(f) for f in parts[a]} & {slceleb_video(f) for f in parts[b]}, \
            f"{spec.name}: a video is in both {a} and {b}"
    for part, files in parts.items():
        assert {label_of[f] for f in files} == {label_of[f] for f in ids}, f"{spec.name}: a speaker is missing from {part}"
    return {
        "protocol": "v0.4", "split_type": "video_disjoint_closed_set", **parts,
        "diagnostics": {**diagnostics, **{f"{k}_clips": len(v) for k, v in parts.items()},
                        **{f"{k}_videos": len({slceleb_video(f) for f in v}) for k, v in parts.items()}},
    }


def closed_set_split(spec, split_type: str = "stratified_closed_set") -> dict:
    label_of, ids, diagnostics = deduplicated_labels(spec)
    rest, test = train_test_split(ids, test_size=TEST_FRACTION, random_state=SPLIT_SEED,
                                  stratify=[label_of[i] for i in ids])
    test_source = "new"
    dev_fraction = DEV_FRACTION / (1 - len(test) / len(ids))
    train, dev = train_test_split(rest, test_size=dev_fraction, random_state=SPLIT_SEED,
                                  stratify=[label_of[i] for i in rest])
    parts = {"train": train, "dev": dev, "test": test}
    return {
        "protocol": "v0.2", "split_type": split_type, **parts,
        "diagnostics": {"test_set": test_source, **{f"{k}_clips": len(v) for k, v in parts.items()},
                        **diagnostics},
    }


def kfold_split(spec) -> dict:
    rows = read_rows(spec.label_path)
    label_of = {r["filename"]: r["label"] for r in rows}
    ids = sorted(label_of)
    speakers_csv = spec.task_dir / "speakers.csv"
    if speakers_csv.exists():
        speaker_of = {r["filename"]: r["speaker"] for r in read_rows(speakers_csv)}
    else:
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
    test_audio = {audio_digest(p) for p in (asv_dir / "audio").rglob("*.wav")}
    leaked = [f for f in train + dev if test_audio and audio_digest(asv_dir / "train_audio" / f) in test_audio]
    assert not leaked, f"asv: {len(leaked)} training clips are identical to trial clips, e.g. {leaked[:3]}"
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


def sd_split(task_dir: Path) -> dict:
    names = sorted(p.stem for p in task_dir.glob("*.rttm"))
    order = names[:]
    random.Random(SPLIT_SEED).shuffle(order)
    n_train = round(SD_FRACTIONS["train"] * len(order))
    n_dev = max(2, round(SD_FRACTIONS["dev"] * len(order)))
    parts = {"train": sorted(order[:n_train]), "dev": sorted(order[n_train:n_train + n_dev]),
             "test": sorted(order[n_train + n_dev:])}

    def hours(recs):
        return round(sum(sf.info(str(task_dir / f"{r}.wav")).duration for r in recs) / 3600, 2)

    return {
        "protocol": "v0.2", "split_type": "recording_disjoint", **parts,
        "diagnostics": {**{f"{k}_recordings": len(v) for k, v in parts.items()},
                        **{f"{k}_hours": hours(v) for k, v in parts.items()}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    for spec in discover_tasks(args.data_dir):
        if spec.kind == "asr":
            split = asr_split(spec)
        elif spec.kind == "classification" and spec.task == "sid":
            split = video_disjoint_split(spec)
        elif spec.kind == "classification" and (spec.task == "er" or (spec.task_dir / "speakers.csv").exists()):
            split = kfold_split(spec)
        elif spec.kind == "classification":
            split = closed_set_split(spec, split_type="random_stratified_no_speaker_ids")
        else:
            continue
        parts = ({f"fold{i}": fold for i, fold in enumerate(split["folds"])} if "folds" in split
                 else {k: split[k] for k in ("train", "dev", "test")})
        assert_no_shared_audio(spec.name, parts, spec.audio_dir)
        write_split(spec.task_dir, split)
        print(f"{spec.name}: {split['split_type']}  {split['diagnostics']}")

    for sd_dir in sorted(args.data_dir.glob("sd_*")):
        if any(sd_dir.glob("*.rttm")):
            split = sd_split(sd_dir)
            write_split(sd_dir, split)
            print(f"{sd_dir.name}: {split['split_type']}  {split['diagnostics']}")

    asv_dir = args.data_dir / "asv"
    if (asv_dir / "train_labels.csv").exists():
        split = asv_split(asv_dir)
        write_split(asv_dir, split)
        print(f"asv: {split['split_type']}  {split['diagnostics']}")
    else:
        print(f"WARNING: {asv_dir / 'train_labels.csv'} missing -- run data_prep/prep_slceleb_train.py")


if __name__ == "__main__":
    main()
