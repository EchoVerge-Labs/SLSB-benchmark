#!/usr/bin/env python3
"""Prepare data/er_tamil/ from aaivu-labs/EmoTa (gated; requires HF_TOKEN with accepted license).

Idempotent: utterances whose wav already exists on disk are not reconverted.
"""
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "er_tamil"
AUDIO_DIR = OUT_DIR / "audio"
LABELS_CSV = OUT_DIR / "labels.csv"

DATASET_REPO = "aaivu-labs/EmoTa"
METADATA_PATH = "data/metadata.csv"

LABEL_MAP = {
    "angry": "anger",
    "anger": "anger",
    "happy": "happiness",
    "happiness": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "fear": "fear",
    "neutral": "neutral",
}
VALID_LABELS = {"anger", "happiness", "sadness", "fear", "neutral"}


def already_converted():
    if not LABELS_CSV.exists():
        return set()
    done = set()
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (AUDIO_DIR / row["filename"]).exists():
                done.add(row["filename"])
    return done


def download_with_retry(repo_path, cache_dir, token, retries=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(
                DATASET_REPO, repo_path, repo_type="dataset", token=token, cache_dir=cache_dir
            )
        except Exception as e:
            last_err = e
            print(f"    retry {attempt}/{retries} for {repo_path}: {e}")
    raise last_err


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    if not token:
        print("ERROR: HF_TOKEN is not set. EmoTa is gated and requires a token with an "
              "accepted license. export HF_TOKEN=... and retry.")
        sys.exit(1)

    print(f"authenticating and checking access to {DATASET_REPO} ...")
    try:
        api = HfApi()
        info = api.dataset_info(DATASET_REPO, token=token)
    except HfHubHTTPError as e:
        print(f"ERROR: authentication/access to {DATASET_REPO} failed: {e}")
        print("Check that HF_TOKEN is valid and that the EmoTa license has been accepted "
              "on huggingface.co for this account. STOPPING -- no files were written.")
        sys.exit(1)
    print(f"  access OK, gated={info.gated}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="er_tamil_"))
    try:
        print("downloading metadata.csv ...")
        meta_path = download_with_retry(METADATA_PATH, tmp_dir, token)
        with open(meta_path, encoding="utf-8-sig") as f:
            meta_rows = list(csv.DictReader(f))
        print(f"  {len(meta_rows)} rows in metadata")

        done = already_converted()
        rows = []
        label_counts = {}
        converted, skipped = 0, 0

        for i, m in enumerate(meta_rows, 1):
            filename = m["file_name"]
            emotion = m["emotion"].strip().lower()
            if emotion not in LABEL_MAP:
                print(f"  WARNING: unrecognized emotion '{emotion}' for {filename}, skipping")
                continue
            label = LABEL_MAP[emotion]
            wav_path = AUDIO_DIR / filename

            if filename in done and wav_path.exists():
                rows.append((filename, label))
                label_counts[label] = label_counts.get(label, 0) + 1
                skipped += 1
                continue

            local_path = download_with_retry(f"data/{filename}", tmp_dir, token)
            data, orig_sr = sf.read(local_path, dtype="float32")
            data = to_pcm16_16k_mono(data, orig_sr)
            sf.write(str(wav_path), data, TARGET_SR, subtype="PCM_16")

            rows.append((filename, label))
            label_counts[label] = label_counts.get(label, 0) + 1
            converted += 1
            if i % 100 == 0 or i == len(meta_rows):
                print(f"  {i}/{len(meta_rows)} processed ({converted} converted, {skipped} already done)")

        rows.sort()
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)

        total_seconds = sum(sf.info(str(AUDIO_DIR / fname)).duration for fname, _ in rows)

        print()
        print("task: er_tamil")
        print("language: tamil")
        print(f"clips: {len(rows)}")
        print(f"total audio: {total_seconds / 3600:.3f} hours")
        print("label distribution:")
        for label in sorted(VALID_LABELS):
            print(f"  {label}: {label_counts.get(label, 0)}")
        print("example rows:")
        for fname, label in rows[:3]:
            print(f"  {fname}\t{label}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
