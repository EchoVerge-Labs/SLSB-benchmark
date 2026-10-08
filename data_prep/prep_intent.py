#!/usr/bin/env python3
"""Prepare the intent classification (IC) tasks from the three archives in
data_prep/downloads/intent/ (the team's "Intent Classification" Drive folder):

  Banking/Sinhala_Datset.zip       -> data/ic_banking_sinhala/  6 banking intents, 39 sentences
  Banking/Tamil_Dataset.zip        -> data/ic_banking_tamil/    6 banking intents, 40 speakers
  Health/drive-download-*.zip      -> data/ic_health_tamil/     16 symptom intents, 100 speakers

Each task gets audio/ (16 kHz mono PCM16), labels.csv (filename,label),
sentences.csv (filename,sentence: which fixed prompt was read) and, where the
source names speakers, speakers.csv (filename,speaker). data_prep/make_splits.py
splits intent tasks by sentence, so test prompts are never heard in training.

- Banking-Sinhala has no speaker information (filenames are recording
  timestamps), so it can only be split at random; see KNOWN_ISSUES.md.
- Clips with no label in the source CSVs are skipped (74 Banking-Sinhala,
  187 Health).
- Identical audio is kept once; audio labelled with two different intents is
  dropped.

Licence (both banking archives): academic / research use only, no commercial use.

Idempotent: wavs already converted on disk are not reconverted.
"""
import csv
import hashlib
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import to_pcm16_16k_mono, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "data_prep" / "downloads" / "intent"
DATA_DIR = REPO_ROOT / "data"


def read_csv(zf, member, **kwargs):
    return list(csv.DictReader(io.StringIO(zf.read(member).decode("utf-8")), **kwargs))


def banking_sinhala(zf):
    root = "Sinhala_Datset/"
    names = {r["intent"]: r["intent_details"].strip() for r in read_csv(zf, root + "Sinhala_Sentences.csv")}
    return [(root + "audio_files/" + r["audio_file"], r["audio_file"], names[r["intent"]], None,
             f"{r['intent']}_{r['inflection']}")
            for r in read_csv(zf, root + "Sinhala_Data.csv")]


def banking_tamil(zf):
    root = "Tamil_Dataset/"
    names = {r["intent"]: r["intent_details"].strip() for r in read_csv(zf, root + "Tamil_Sentences.csv")}
    items = []
    for r in read_csv(zf, root + "Tamil_Data.csv"):
        speaker, name = r["audio_file"].split("/")
        items.append((root + "audio_files/" + r["audio_file"], f"{speaker}/{name.replace('..wav', '.wav')}",
                      names[r["intent"]], speaker, f"{r['intent']}_{r['inflection']}"))
    return items


def health_tamil(zf):
    # voiceFile_tamil.csv has no header: id, speaker, gender, age, intent, phrase, timestamp, flag, note.
    # The audio file is named after the timestamp, with ':' replaced by '_'.
    rows = csv.reader(io.StringIO(zf.read("voiceFile_tamil.csv").decode("utf-8")))
    members = {Path(m).stem: m for m in zf.namelist() if m.startswith("audio_files/") and m.endswith(".wav")}
    items = []
    for row in rows:
        stem = row[6].replace(":", "_")
        if stem in members:
            # The prompt's English text (column 5) identifies which of the 160 phrases was read.
            items.append((members[stem], f"{stem}.wav", row[4].strip().lower(), row[1].strip(),
                          " ".join(row[5].lower().split())))
    return items


TASKS = {
    "ic_banking_sinhala": ("Banking/Sinhala_Datset.zip", banking_sinhala),
    "ic_banking_tamil": ("Banking/Tamil_Dataset.zip", banking_tamil),
    "ic_health_tamil": (None, health_tamil),  # the Health archive's name is Drive's download name
}


def prepare(task, zip_path, reader):
    zf = zipfile.ZipFile(zip_path)
    items = reader(zf)
    task_dir = DATA_DIR / task
    audio_dir = task_dir / "audio"

    # Decode once: resample to 16 kHz mono, and find identical audio.
    decoded, labels_of = {}, defaultdict(set)
    for member, filename, label, speaker, _sentence in items:
        data, sr = sf.read(io.BytesIO(zf.read(member)), dtype="float32")
        audio = to_pcm16_16k_mono(data, sr)
        digest = hashlib.md5((audio * 32767).round().astype("int16").tobytes()).hexdigest()
        decoded[filename] = (audio, digest)
        labels_of[digest].add(label)

    kept, seen = [], set()
    for member, filename, label, speaker, sentence in items:
        audio, digest = decoded[filename]
        if len(labels_of[digest]) > 1 or digest in seen:
            continue
        seen.add(digest)
        out = audio_dir / filename
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), audio, TARGET_SR, subtype="PCM_16")
        kept.append((filename, label, speaker, sentence))

    with open(task_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows((fn, label) for fn, label, _, _ in kept)
    with open(task_dir / "sentences.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "sentence"])
        writer.writerows((fn, sentence) for fn, _, _, sentence in kept)
    if all(speaker is not None for _, _, speaker, _ in kept):
        with open(task_dir / "speakers.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "speaker"])
            writer.writerows((fn, speaker) for fn, _, speaker, _ in kept)

    conflicting = sum(1 for _, filename, _, _, _ in items if len(labels_of[decoded[filename][1]]) > 1)
    speakers = {s for _, _, s, _ in kept if s is not None}
    print(f"{task}: {len(items)} labelled clips -> {len(kept)} kept "
          f"({conflicting} with conflicting labels dropped, "
          f"{len(items) - len(kept) - conflicting} duplicate copies dropped); "
          f"{len({label for _, label, _, _ in kept})} intents, {len({s for *_, s in kept})} sentences; "
          f"{len(speakers) if speakers else 'no'} speakers")


def main():
    for task, (zip_name, reader) in TASKS.items():
        zip_path = SOURCE_DIR / zip_name if zip_name else next((SOURCE_DIR / "Health").glob("*.zip"), None)
        if zip_path is None or not zip_path.exists():
            sys.exit(f"ERROR: archive for {task} not found under {SOURCE_DIR}")
        prepare(task, zip_path, reader)


if __name__ == "__main__":
    main()
