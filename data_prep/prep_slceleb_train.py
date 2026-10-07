#!/usr/bin/env python3
"""Prepare data/asv/train_audio/ + data/asv/train_labels.csv: speaker-verification
TRAINING data from the SLCeleb archive's dev split.

The ASV trial lists (data/asv/trials_*.csv) cover every clip and speaker in
data/asv/audio/, so nothing there can train a speaker-embedding model without
testing on speakers it has seen. SLCeleb's dev split holds other speaker ids --
but not always other speakers: the archive's sinhala/dev clips are byte-for-byte
copies of test clips filed under different ids. So a dev speaker is dropped if
its id appears in the trials OR any of its clips is identical to a test clip,
and clips whose audio is filed under more than one speaker are dropped too. In
practice this keeps the 89 Tamil dev speakers and none of the Sinhala ones.

Speakers are capped at MAX_CLIPS_PER_SPEAKER clips (a fixed random subset) so a
few very prolific Tamil speakers don't dominate training, and to bound the
per-upstream feature-extraction cost.

Idempotent: wavs already converted on disk are not reconverted.
"""
import csv
import hashlib
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


def audio_digest(source) -> str:
    """md5 of the decoded 16-bit samples, so a re-written file with a different
    header still matches its original."""
    data, _ = sf.read(source, dtype="int16")
    return hashlib.md5(data.tobytes()).hexdigest()


def test_audio_hashes() -> set[str]:
    """Digests of every clip the trials can use (data/asv/audio/)."""
    return {audio_digest(str(p)) for p in (ASV_DIR / "audio").rglob("*.wav")}


def main():
    if not ZIP_PATH.exists():
        sys.exit(f"ERROR: {ZIP_PATH} not found (the SLCeleb audio archive)")
    excluded = trial_speakers()
    test_hashes = test_audio_hashes()
    zf = zipfile.ZipFile(ZIP_PATH)

    rows = []  # (filename relative to train_audio/, speaker)
    for lang, prefix in LANG_ZIP_PREFIX.items():
        dev_prefix = f"{prefix}/dev/"
        by_speaker = defaultdict(list)
        for member in zf.namelist():
            if member.startswith(dev_prefix) and member.endswith(".wav"):
                by_speaker[member[len(dev_prefix):].split("/")[0]].append(member)

        digest = {m: audio_digest(io.BytesIO(zf.read(m))) for ms in by_speaker.values() for m in ms}
        speakers_of = defaultdict(set)
        for speaker, members in by_speaker.items():
            for m in members:
                speakers_of[digest[m]].add(speaker)
        in_test = {s for s, ms in by_speaker.items() if any(digest[m] in test_hashes for m in ms)}
        kept = {s: [m for m in ms if len(speakers_of[digest[m]]) == 1]
                for s, ms in by_speaker.items() if s not in excluded and s not in in_test}
        print(f"--- {lang}: {len(by_speaker)} dev speakers; dropped {len(excluded & set(by_speaker))} "
              f"(id in the trials) and {len(in_test - excluded)} (audio identical to test clips); "
              f"{len(kept)} kept")
        for speaker in sorted(kept):
            members = list({digest[m]: m for m in sorted(kept[speaker])}.values())  # one copy of repeated audio
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
