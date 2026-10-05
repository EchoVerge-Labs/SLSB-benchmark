#!/usr/bin/env python3
"""Prepare data/asv/train_audio/ + data/asv/train_labels.csv: speaker-verification
TRAINING data from the SLCeleb archive's dev split.

The ASV trial lists (data/asv/trials_*.csv) cover every clip and speaker in
data/asv/audio/, so nothing there can train a speaker-embedding model without
testing on speakers it has seen. SLCeleb's dev split holds other speakers;
any dev speaker id that also appears in the trials is dropped, so training and
test speakers are disjoint.

Speakers are capped at MAX_CLIPS_PER_SPEAKER clips (a fixed random subset) so a
few very prolific Tamil speakers don't dominate training, and to bound the
per-upstream feature-extraction cost.

Idempotent: wavs already converted on disk are not reconverted.
"""
import csv
import io
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data_prep" / "downloads" / "slceleb.zip"
LANG_ZIP_PREFIX = {"sinhala": "SLCeleb/sinhala", "tamil": "SLCeleb/Tamil"}

ASV_DIR = REPO_ROOT / "data" / "asv"
TRAIN_AUDIO_DIR = ASV_DIR / "train_audio"
TRAIN_LABELS_CSV = ASV_DIR / "train_labels.csv"

MAX_CLIPS_PER_SPEAKER = 150
SEED = 1234


def trial_speakers() -> set[str]:
    speakers = set()
    for path in sorted(ASV_DIR.glob("trials_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                speakers.add(row["wav1"].split("/")[0])
                speakers.add(row["wav2"].split("/")[0])
    if not speakers:
        sys.exit(f"ERROR: no trial lists in {ASV_DIR} -- run prep_slceleb.py first")
    return speakers


def main():
    if not ZIP_PATH.exists():
        sys.exit(f"ERROR: {ZIP_PATH} not found (the SLCeleb audio archive)")
    excluded = trial_speakers()
    zf = zipfile.ZipFile(ZIP_PATH)

    rows = []  # (filename relative to train_audio/, speaker)
    for lang, prefix in LANG_ZIP_PREFIX.items():
        dev_prefix = f"{prefix}/dev/"
        by_speaker = defaultdict(list)
        for member in zf.namelist():
            if member.startswith(dev_prefix) and member.endswith(".wav"):
                by_speaker[member[len(dev_prefix):].split("/")[0]].append(member)

        kept = {s: m for s, m in by_speaker.items() if s not in excluded}
        print(f"--- {lang}: {len(by_speaker)} dev speakers, "
              f"{len(by_speaker) - len(kept)} dropped (also in the trials), {len(kept)} kept")
        for speaker in sorted(kept):
            members = sorted(kept[speaker])
            random.Random(f"{SEED}/{lang}/{speaker}").shuffle(members)
            for member in members[:MAX_CLIPS_PER_SPEAKER]:
                filename = f"{lang}/{member[len(dev_prefix):]}"
                out = TRAIN_AUDIO_DIR / filename
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data, orig_sr = sf.read(io.BytesIO(zf.read(member)), dtype="float32")
                    sf.write(str(out), to_pcm16_16k_mono(data, orig_sr), TARGET_SR, subtype="PCM_16")
                rows.append((filename, speaker))
        print(f"  {sum(1 for f, _ in rows if f.startswith(lang + '/'))} clips")

    with open(TRAIN_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)
    print(f"wrote {TRAIN_LABELS_CSV}: {len(rows)} clips, {len({s for _, s in rows})} speakers")


if __name__ == "__main__":
    main()
