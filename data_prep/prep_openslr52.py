#!/usr/bin/env python3
"""Prepare a balanced Sinhala ASR benchmark subset from OpenSLR-52.

Downloads utt_spk_text.tsv and the first 3 audio archives from OpenSLR-52,
takes a speaker-balanced random subset of utterances we have audio for,
converts them to 16 kHz mono wav, and writes data/asr_sinhala/{audio/,transcripts.csv}.

Idempotent: utterances whose wav already exists on disk are not reconverted.
"""
import argparse
import csv
import hashlib
import io
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fractions import Fraction

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "asr_sinhala"
AUDIO_DIR = OUT_DIR / "audio"
TRANSCRIPTS_CSV = OUT_DIR / "transcripts.csv"

MIRRORS = [
    "https://openslr.org/resources/52",
    "https://openslr.elda.org/resources/52",
    "https://openslr.magicdatatech.com/resources/52",
]
ARCHIVES = ["asr_sinhala_0.zip", "asr_sinhala_1.zip", "asr_sinhala_2.zip"]
TSV_NAME = "utt_spk_text.tsv"
CHECKSUM_NAME = "checksum.md5"
TARGET_SR = 16000


def download(fname, dest_dir, retries=3):
    dest = dest_dir / fname
    for mirror in MIRRORS:
        url = f"{mirror}/{fname}"
        for attempt in range(1, retries + 1):
            print(f"  downloading {fname} from {mirror} (attempt {attempt}/{retries})...")
            result = subprocess.run(
                ["wget", "-q", "-c", "--tries=1", "--timeout=60", "-O", str(dest), url]
            )
            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                return dest
            print(f"    failed (rc={result.returncode})")
    raise RuntimeError(f"failed to download {fname} from all mirrors")


def parse_checksums(path):
    checksums = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            checksums[parts[1]] = parts[0]
    return checksums


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resample_to_16k(data, orig_sr):
    if orig_sr == TARGET_SR:
        return data
    frac = Fraction(TARGET_SR, orig_sr).limit_denominator(1000)
    return resample_poly(data, frac.numerator, frac.denominator)


def to_mono(data):
    if data.ndim > 1:
        return data.mean(axis=1)
    return data


def balanced_sample(pool_by_speaker, n, seed):
    rng = random.Random(seed)
    speakers = list(pool_by_speaker.keys())
    rng.shuffle(speakers)
    for s in speakers:
        rng.shuffle(pool_by_speaker[s])
    selected = []
    idx = 0
    while len(selected) < n:
        added = False
        for s in speakers:
            lst = pool_by_speaker[s]
            if idx < len(lst):
                selected.append(lst[idx])
                added = True
                if len(selected) >= n:
                    break
        if not added:
            break
        idx += 1
    return selected


def already_converted():
    """Return dict utt_id -> transcript for entries whose wav already exists."""
    if not TRANSCRIPTS_CSV.exists():
        return {}
    done = {}
    with open(TRANSCRIPTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wav_path = AUDIO_DIR / row["filename"]
            if wav_path.exists():
                done[Path(row["filename"]).stem] = row["transcript"]
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="openslr52_"))
    print(f"temp dir: {tmp_dir}")

    zip_handles = []
    try:
        print("downloading checksum + transcript list...")
        checksum_path = download(CHECKSUM_NAME, tmp_dir)
        checksums = parse_checksums(checksum_path)
        tsv_path = download(TSV_NAME, tmp_dir)
        if TSV_NAME in checksums and md5sum(tsv_path) != checksums[TSV_NAME]:
            raise RuntimeError(f"checksum mismatch for {TSV_NAME}")

        print("downloading audio archives (this is ~2.7 GB total)...")
        zip_paths = []
        for arc in ARCHIVES:
            p = download(arc, tmp_dir)
            if arc in checksums and md5sum(p) != checksums[arc]:
                raise RuntimeError(f"checksum mismatch for {arc}")
            zip_paths.append(p)

        print("indexing archive contents...")
        flac_index = {}  # utt_id -> (zip_index, member_name)
        for zi, zp in enumerate(zip_paths):
            zh = zipfile.ZipFile(zp)
            zip_handles.append(zh)
            for member in zh.namelist():
                if member.endswith(".flac"):
                    utt_id = Path(member).stem
                    flac_index[utt_id] = (zi, member)
        print(f"  {len(flac_index)} flac clips available across {len(ARCHIVES)} archives")

        print("parsing utt_spk_text.tsv...")
        pool_by_speaker = {}
        transcripts = {}
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) != 3:
                    continue
                utt_id, speaker_id, transcript = row
                if utt_id not in flac_index:
                    continue
                pool_by_speaker.setdefault(speaker_id, []).append(utt_id)
                transcripts[utt_id] = transcript
        total_pool = sum(len(v) for v in pool_by_speaker.values())
        print(f"  {total_pool} utterances with both transcript + audio, across {len(pool_by_speaker)} speakers")

        n = min(args.target_n, total_pool)
        selected = balanced_sample(pool_by_speaker, n, args.seed)
        print(f"selected {len(selected)} utterances (speaker-balanced, seed={args.seed})")

        done = already_converted()
        rows = []
        total_seconds = 0.0
        converted, skipped = 0, 0
        for i, utt_id in enumerate(selected, 1):
            wav_path = AUDIO_DIR / f"{utt_id}.wav"
            transcript = transcripts[utt_id]
            if utt_id in done and wav_path.exists():
                total_seconds += sf.info(str(wav_path)).duration
                rows.append((f"{utt_id}.wav", transcript))
                skipped += 1
                continue

            zi, member = flac_index[utt_id]
            raw = zip_handles[zi].read(member)
            data, orig_sr = sf.read(io.BytesIO(raw), dtype="float32")
            data = to_mono(data)
            duration = len(data) / orig_sr
            data = resample_to_16k(data, orig_sr)
            sf.write(str(wav_path), data, TARGET_SR, subtype="PCM_16")

            total_seconds += duration
            rows.append((f"{utt_id}.wav", transcript))
            converted += 1
            if i % 250 == 0 or i == len(selected):
                print(f"  {i}/{len(selected)} processed ({converted} converted, {skipped} already done)")

        rows.sort()
        with open(TRANSCRIPTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "transcript"])
            writer.writerows(rows)

        print()
        print(f"clips: {len(rows)}")
        print(f"total audio: {total_seconds / 3600:.3f} hours")
        print("example rows:")
        for fname, transcript in rows[:3]:
            print(f"  {fname}\t{transcript}")

    finally:
        for zh in zip_handles:
            zh.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"cleaned up temp dir: {tmp_dir}")


if __name__ == "__main__":
    main()
