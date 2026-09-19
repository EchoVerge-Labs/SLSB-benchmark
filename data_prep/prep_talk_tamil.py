#!/usr/bin/env python3
"""Prepare data/asr_tamil/ from the TaLk Sri Lankan Tamil ASR dataset.

Input: data_prep/downloads/talk_audio/ containing the wav files plus
"TaLk Transcription.csv" (columns: File name, Transcription). The CSV is
correct UTF-8 -- read/write it as such, never re-encode.

Idempotent: wavs already converted on disk are not reconverted.
"""
import csv
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "data_prep" / "downloads" / "talk_audio"
INPUT_CSV = INPUT_DIR / "TaLk Transcription.csv"

OUT_DIR = REPO_ROOT / "data" / "asr_tamil"
AUDIO_DIR = OUT_DIR / "audio"
TRANSCRIPTS_CSV = OUT_DIR / "transcripts.csv"

SEED = 42


def check_inputs():
    missing = []
    if not INPUT_CSV.exists():
        missing.append(f"  {INPUT_CSV}")
    if not INPUT_DIR.exists():
        missing.append(f"  {INPUT_DIR}/ (audio folder)")
    if missing:
        print("ERROR: missing required input(s). STOPPING.")
        for m in missing:
            print(m)
        sys.exit(1)


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
    check_inputs()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} rows in source CSV")

    done = already_converted()
    out_rows = []
    total_seconds = 0.0
    converted, skipped, missing_audio = 0, 0, 0

    for i, row in enumerate(rows, 1):
        filename = row["File name"]
        transcript = row["Transcription"]
        src_path = INPUT_DIR / filename
        dst_path = AUDIO_DIR / filename

        if filename in done and dst_path.exists():
            total_seconds += sf.info(str(dst_path)).duration
            out_rows.append((filename, transcript))
            skipped += 1
            continue

        if not src_path.exists():
            print(f"  WARNING: no source audio for {filename}, skipping")
            missing_audio += 1
            continue

        data, orig_sr = sf.read(str(src_path), dtype="float32")
        data = to_pcm16_16k_mono(data, orig_sr)
        sf.write(str(dst_path), data, TARGET_SR, subtype="PCM_16")

        total_seconds += len(data) / TARGET_SR
        out_rows.append((filename, transcript))
        converted += 1
        if i % 250 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} processed ({converted} converted, {skipped} already done)")

    out_rows.sort()
    with open(TRANSCRIPTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["filename", "transcript"])
        writer.writerows(out_rows)

    speakers = {fn.rsplit("_", 1)[0] for fn, _ in out_rows}

    print()
    print("task: asr_tamil")
    print("language: tamil")
    print(f"clips: {len(out_rows)}")
    print(f"speakers: {len(speakers)}")
    print(f"total audio: {total_seconds / 3600:.3f} hours")
    if missing_audio:
        print(f"WARNING: {missing_audio} transcript row(s) had no matching audio file")
    print("example rows:")
    for fname, transcript in out_rows[:3]:
        print(f"  {fname}\t{transcript}")


if __name__ == "__main__":
    main()
