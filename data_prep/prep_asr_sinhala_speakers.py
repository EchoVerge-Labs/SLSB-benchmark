#!/usr/bin/env python3
"""Write data/asr_sinhala/speakers.csv (filename,speaker) from OpenSLR-52's
utt_spk_text.tsv.

asr_sinhala's filenames are bare utterance ids with no speaker in them, so
without this map its train/test split can't be speaker-disjoint (in v0.1 every
test speaker also appeared in training). Any ASR task folder with a
speakers.csv is split by those speakers -- see data_prep/make_splits.py.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_openslr52 import TSV_NAME, download

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = REPO_ROOT / "data_prep" / "downloads"
TASK_DIR = REPO_ROOT / "data" / "asr_sinhala"


def main():
    tsv_path = DOWNLOADS_DIR / TSV_NAME
    if not tsv_path.exists():
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        download(TSV_NAME, DOWNLOADS_DIR)
    speaker_of = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                speaker_of[parts[0]] = parts[1]

    with open(TASK_DIR / "transcripts.csv", newline="", encoding="utf-8") as f:
        filenames = [row["filename"] for row in csv.DictReader(f)]
    missing = [name for name in filenames if Path(name).stem not in speaker_of]
    if missing:
        sys.exit(f"ERROR: {len(missing)} utterance(s) not in {TSV_NAME}, e.g. {missing[:3]}")

    with open(TASK_DIR / "speakers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker"])
        writer.writerows((name, speaker_of[Path(name).stem]) for name in filenames)
    print(f"wrote {TASK_DIR / 'speakers.csv'}: {len(filenames)} clips, "
          f"{len({speaker_of[Path(n).stem] for n in filenames})} speakers")


if __name__ == "__main__":
    main()
