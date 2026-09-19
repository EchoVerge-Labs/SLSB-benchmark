#!/usr/bin/env python3
"""Prepare data/asv/ and data/sid/ from a manually-placed SLCeleb archive + trial lists.

Only converts the wavs actually referenced by the trial lists (not the full
~55k-file corpus) to keep this tractable. SID labels.csv points at the
shared data/asv/audio/ folder rather than duplicating files.

Idempotent: wavs already converted on disk are not reconverted.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = REPO_ROOT / "data_prep" / "downloads"
ZIP_PATH = DOWNLOADS_DIR / "slceleb.zip"
TRIAL_LISTS = {
    "sinhala": DOWNLOADS_DIR / "test_list_sinhala.txt",
    "tamil": DOWNLOADS_DIR / "test_list_tamil.txt",
}
LANG_ZIP_PREFIX = {"sinhala": "SLCeleb/sinhala", "tamil": "SLCeleb/Tamil"}
SPLIT_PRIORITY = ["dev", "test"]  # later entries win on relpath collision

ASV_DIR = REPO_ROOT / "data" / "asv"
ASV_AUDIO_DIR = ASV_DIR / "audio"
SID_DIR = REPO_ROOT / "data" / "sid"
SID_LABELS_CSV = SID_DIR / "labels.csv"


def check_inputs():
    missing = []
    if not ZIP_PATH.exists():
        missing.append(f"  {ZIP_PATH}  (the SLCeleb audio archive)")
    for lang, path in TRIAL_LISTS.items():
        if not path.exists():
            missing.append(f"  {path}  (the {lang} trial list)")
    if missing:
        print("ERROR: missing required input file(s). STOPPING.")
        print("Please place:")
        for m in missing:
            print(m)
        sys.exit(1)


def build_relpath_index(zf, lang):
    prefix = LANG_ZIP_PREFIX[lang] + "/"
    index = {}
    for split in SPLIT_PRIORITY:
        split_prefix = f"{prefix}{split}/"
        for member in zf.namelist():
            if member.startswith(split_prefix) and member.endswith(".wav"):
                relpath = member[len(split_prefix):]
                index[relpath] = member  # later split (test) overwrites dev
    return index


def parse_trials(path):
    trials = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            label, wav1, wav2 = parts
            trials.append((label, wav1, wav2))
    return trials


def ensure_converted(zf, index, relpath, cache):
    if relpath in cache:
        return cache[relpath]
    wav_path = ASV_AUDIO_DIR / relpath
    if wav_path.exists():
        cache[relpath] = True
        return True
    member = index.get(relpath)
    if member is None:
        cache[relpath] = False
        return False
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    raw = zf.read(member)
    data, orig_sr = sf.read(io.BytesIO(raw), dtype="float32")
    data = to_pcm16_16k_mono(data, orig_sr)
    sf.write(str(wav_path), data, TARGET_SR, subtype="PCM_16")
    cache[relpath] = True
    return True


def main():
    check_inputs()
    ASV_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SID_DIR.mkdir(parents=True, exist_ok=True)

    print(f"opening {ZIP_PATH} ...")
    zf = zipfile.ZipFile(ZIP_PATH)

    sid_rows = []  # (relpath, speaker_label)
    seen_relpaths = set()

    for lang in ("sinhala", "tamil"):
        print(f"\n--- {lang} ---")
        print("indexing archive paths...")
        index = build_relpath_index(zf, lang)
        print(f"  {len(index)} wavs available for {lang}")

        trials = parse_trials(TRIAL_LISTS[lang])
        print(f"  {len(trials)} trial pairs in list")

        referenced = set()
        for _, wav1, wav2 in trials:
            referenced.add(wav1)
            referenced.add(wav2)
        print(f"  {len(referenced)} unique wavs referenced")

        cache = {}
        converted, already_done, missing = 0, 0, 0
        for i, relpath in enumerate(sorted(referenced), 1):
            existed_before = (ASV_AUDIO_DIR / relpath).exists()
            ok = ensure_converted(zf, index, relpath, cache)
            if ok:
                if existed_before:
                    already_done += 1
                else:
                    converted += 1
            else:
                missing += 1
            if i % 1000 == 0 or i == len(referenced):
                print(f"  {i}/{len(referenced)} resolved ({converted} converted, "
                      f"{already_done} already done, {missing} missing)")

        if missing:
            missing_paths = [r for r in referenced if not cache.get(r, False)]
            print(f"  WARNING: {missing} referenced wav(s) not found in archive, e.g.:")
            for m in missing_paths[:5]:
                print(f"    {m}")

        kept_trials = [
            (label, w1, w2) for label, w1, w2 in trials
            if cache.get(w1, False) and cache.get(w2, False)
        ]
        dropped = len(trials) - len(kept_trials)
        trials_csv = ASV_DIR / f"trials_{lang}.csv"
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "wav1", "wav2"])
            writer.writerows(kept_trials)

        verified_wavs = {r for r in referenced if cache.get(r, False)}
        print(f"  wrote {trials_csv.name}: {len(kept_trials)} pairs kept"
              f" ({dropped} dropped due to missing audio), {len(verified_wavs)} unique wavs verified")

        for relpath in verified_wavs:
            if relpath not in seen_relpaths:
                seen_relpaths.add(relpath)
                speaker = relpath.split("/")[0]
                sid_rows.append((relpath, speaker, lang))

    zf.close()

    sid_rows.sort()
    with open(SID_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows((relpath, speaker) for relpath, speaker, _ in sid_rows)

    print()
    print("=== SID summary ===")
    for lang in ("sinhala", "tamil"):
        lang_rows = [r for r in sid_rows if r[2] == lang]
        speakers = {r[1] for r in lang_rows}
        print(f"  {lang}: {len(speakers)} speakers, {len(lang_rows)} clips")
    print(f"  total: {len(sid_rows)} clips -> {SID_LABELS_CSV}")


if __name__ == "__main__":
    main()
