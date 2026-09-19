#!/usr/bin/env python3
"""Prepare data/asr_omni_sinhala/ from facebook/omnilingual-asr-corpus (sin_Sinh, test split).

Idempotent: utterances whose wav already exists on disk are not reconverted.
"""
import argparse
import csv
import io
import os
import re
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "asr_omni_sinhala"
AUDIO_DIR = OUT_DIR / "audio"
TRANSCRIPTS_CSV = OUT_DIR / "transcripts.csv"

DATASET_NAME = "facebook/omnilingual-asr-corpus"
CONFIG = "sin_Sinh"
TARGET_N = 2000
SEED = 42
ANNOTATION_TAG_RE = re.compile(r"<[^>]*>")


def clean_transcript(raw_text: str) -> str:
    return re.sub(r"\s+", " ", ANNOTATION_TAG_RE.sub(" ", raw_text)).strip()


def already_converted():
    if not TRANSCRIPTS_CSV.exists():
        return set()
    done = set()
    with open(TRANSCRIPTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (AUDIO_DIR / row["filename"]).exists():
                done.add(row["filename"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-n", type=int, default=TARGET_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    print(f"loading {DATASET_NAME} config={CONFIG} split=test ...")
    ds = load_dataset(DATASET_NAME, CONFIG, split="test", token=token)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  {len(ds)} examples in test split")

    if len(ds) > args.target_n:
        ds = ds.shuffle(seed=args.seed).select(range(args.target_n))
        print(f"  capped to {len(ds)} (seed={args.seed})")

    done = already_converted()
    rows = []
    total_seconds = 0.0
    converted, skipped = 0, 0

    for i, row in enumerate(ds, 1):
        # segment_id is a session id, not per-utterance unique; prompt_id is.
        filename = f"{row['speaker_id']}_{row['prompt_id']}.wav"
        transcript = clean_transcript(row["raw_text"])
        wav_path = AUDIO_DIR / filename

        if filename in done and wav_path.exists():
            total_seconds += sf.info(str(wav_path)).duration
            rows.append((filename, transcript))
            skipped += 1
            continue

        audio = row["audio"]
        raw_array, orig_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        data = to_pcm16_16k_mono(raw_array, orig_sr)
        sf.write(str(wav_path), data, TARGET_SR, subtype="PCM_16")

        total_seconds += len(data) / TARGET_SR
        rows.append((filename, transcript))
        converted += 1
        if i % 250 == 0 or i == len(ds):
            print(f"  {i}/{len(ds)} processed ({converted} converted, {skipped} already done)")

    rows.sort()
    with open(TRANSCRIPTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcript"])
        writer.writerows(rows)

    print()
    print("task: asr_omni_sinhala")
    print("language: sinhala")
    print(f"clips: {len(rows)}")
    print(f"total audio: {total_seconds / 3600:.3f} hours")
    print("label distribution: n/a (ASR task)")
    print("example rows:")
    for fname, transcript in rows[:3]:
        print(f"  {fname}\t{transcript}")


if __name__ == "__main__":
    main()
