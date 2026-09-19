#!/usr/bin/env python3
"""Stage SiTa (speaker diarization) into data/sd_sinhala/ and data/sd_tamil/.

SiTa audio is already 16 kHz mono, so this only extracts and pairs up
wav/rttm files -- no resampling. Diarization is evaluated with DER, which
needs its own runner (not the linear-probe/ASR runner in src/).

Idempotent: files already staged on disk are not re-extracted.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "data_prep" / "downloads" / "SiTa.zip"

LANGUAGES = {
    "sinhala": "SiTa_dataset/sinhala",
    "tamil": "SiTa_dataset/tamil",
}


def check_inputs():
    if not ZIP_PATH.exists():
        print(f"ERROR: missing {ZIP_PATH}. STOPPING.")
        print("Please place the SiTa dataset archive there and retry.")
        sys.exit(1)


def main():
    check_inputs()
    zf = zipfile.ZipFile(ZIP_PATH)
    namelist = zf.namelist()

    summary = {}
    for lang, prefix in LANGUAGES.items():
        out_dir = REPO_ROOT / "data" / f"sd_{lang}"
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_members = {
            Path(m).stem: m for m in namelist
            if m.startswith(f"{prefix}/wav_files/") and m.endswith(".wav")
        }
        rttm_members = {
            Path(m).stem: m for m in namelist
            if m.startswith(f"{prefix}/rttm/") and m.endswith(".rttm")
        }
        stems = sorted(set(wav_members) & set(rttm_members))
        unmatched = (set(wav_members) | set(rttm_members)) - set(stems)
        if unmatched:
            print(f"  WARNING [{lang}]: {len(unmatched)} file(s) without a wav+rttm pair, skipping: {sorted(unmatched)}")

        total_seconds = 0.0
        staged = 0
        for stem in stems:
            wav_dest = out_dir / f"{stem}.wav"
            rttm_dest = out_dir / f"{stem}.rttm"
            if not wav_dest.exists():
                with zf.open(wav_members[stem]) as src, open(wav_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if not rttm_dest.exists():
                with zf.open(rttm_members[stem]) as src, open(rttm_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            total_seconds += sf.info(str(wav_dest)).duration
            staged += 1

        summary[lang] = (staged, total_seconds)
        print(f"task: sd_{lang}")
        print(f"language: {lang}")
        print(f"files: {staged}")
        print(f"total audio: {total_seconds / 3600:.3f} hours")
        print("label distribution: n/a (SD task; DER computed from .rttm, not a classification label)")
        print("example files:")
        for stem in stems[:3]:
            print(f"  {stem}.wav / {stem}.rttm")
        print()

    zf.close()
    print("NOTE: SD (speaker diarization) requires a separate DER evaluation runner, "
          "not the linear-probe (classification) or CTC (ASR) runner.")


if __name__ == "__main__":
    main()
