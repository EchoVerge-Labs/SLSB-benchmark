#!/usr/bin/env python3
"""Prepare data/er_sinhala/ from jithara/sinhala-emotional-tts-dataset.

The dataset is an HF "audiofolder": top-level folder name is the emotion
label (its own metadata.csv is broken -- empty "emotion" column, bogus
0.04s durations for every row -- so the folder structure is the source of
truth, not metadata.csv).

Idempotent: utterances whose wav already exists on disk are not reconverted.
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "er_sinhala"
AUDIO_DIR = OUT_DIR / "audio"
LABELS_CSV = OUT_DIR / "labels.csv"

DATASET_REPO = "jithara/sinhala-emotional-tts-dataset"

LABEL_MAP = {
    "anger": "anger",
    "angry": "anger",
    "happy": "happiness",
    "happiness": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "fear": "fear",
    "neutral": "neutral",
}
VALID_LABELS = {"anger", "happiness", "sadness", "fear", "neutral"}


def output_filename(basename: str) -> str:
    if basename.endswith(".wav.wav"):
        return basename[:-4]
    return basename


def download(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["wget", "-q", "-c", "--tries=1", "--timeout=30", "-O", str(dest), url]
        )
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return
        print(f"    retry {attempt}/{retries} for {url}")
    raise RuntimeError(f"failed to download {url}")


def already_converted():
    if not LABELS_CSV.exists():
        return set()
    done = set()
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (AUDIO_DIR / row["filename"]).exists():
                done.add(row["filename"])
    return done


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    print(f"listing files in {DATASET_REPO} ...")
    all_files = list_repo_files(DATASET_REPO, repo_type="dataset", token=token)
    wav_files = [f for f in all_files if f.endswith(".wav")]
    print(f"  {len(wav_files)} wav files found")

    done = already_converted()
    rows = []
    label_counts = {}
    converted, skipped = 0, 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="er_sinhala_"))

    try:
        for i, repo_path in enumerate(sorted(wav_files), 1):
            folder = repo_path.split("/")[0]
            if folder not in LABEL_MAP:
                print(f"  WARNING: unrecognized folder '{folder}' for {repo_path}, skipping")
                continue
            label = LABEL_MAP[folder]

            filename = output_filename(Path(repo_path).name)
            wav_path = AUDIO_DIR / filename

            if filename in done and wav_path.exists():
                rows.append((filename, label))
                label_counts[label] = label_counts.get(label, 0) + 1
                skipped += 1
                continue

            url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{repo_path}"
            tmp_path = tmp_dir / "tmp.wav"
            download(url, tmp_path)
            data, orig_sr = sf.read(str(tmp_path), dtype="float32")
            data = to_pcm16_16k_mono(data, orig_sr)
            sf.write(str(wav_path), data, TARGET_SR, subtype="PCM_16")
            tmp_path.unlink(missing_ok=True)

            rows.append((filename, label))
            label_counts[label] = label_counts.get(label, 0) + 1
            converted += 1
            if i % 100 == 0 or i == len(wav_files):
                print(f"  {i}/{len(wav_files)} processed ({converted} converted, {skipped} already done)")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    rows.sort()
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    total_seconds = sum(sf.info(str(AUDIO_DIR / fname)).duration for fname, _ in rows)

    print()
    print("task: er_sinhala")
    print("language: sinhala")
    print(f"clips: {len(rows)}")
    print(f"total audio: {total_seconds / 3600:.3f} hours")
    print("label distribution:")
    for label in sorted(VALID_LABELS):
        print(f"  {label}: {label_counts.get(label, 0)}")
    print("example rows:")
    for fname, label in rows[:3]:
        print(f"  {fname}\t{label}")


if __name__ == "__main__":
    main()
