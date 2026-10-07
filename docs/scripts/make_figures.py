#!/usr/bin/env python
"""Regenerate the documentation figures and the tables they are drawn from.

    python docs/scripts/make_figures.py            # figures from the versioned CSVs
    python docs/scripts/make_figures.py --print    # ...and print the README tables
    python docs/scripts/make_figures.py --refresh  # first rebuild the CSVs from data/
                                                   # and a results directory (needs `dvc pull`)

The CSVs in docs/results/ are versioned; every number in the figures and tables
is read from them, nothing is typed in by hand. Each figure is written twice,
for GitHub's light and dark themes, and the pages pick one with <picture>.

Colour follows the language, never the rank: Sinhala and Tamil keep the same hue
in every chart (the palette of the XLS-R codebook-experiment repository), and
every mark is also labelled, so colour is never the only cue.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets"
TABLES = REPO / "docs" / "results"
TASKS_CSV = TABLES / "tasks.csv"
RESULTS_CSV = TABLES / "reference_xlsr300m_norm_v0.2.csv"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", Sinhala="#eb6834", Tamil="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", Sinhala="#d95926", Tamil="#199e70"),
}

# Display order and names. ASR omni / ASV Sinhala are excluded from v0.2 runs.
TASKS = [
    ("asr_sinhala", "ASR", "Sinhala", "OpenSLR-52"),
    ("asr_tamil", "ASR", "Tamil", "TaLk"),
    ("er_tamil", "Emotion recognition", "Tamil", "EmoTa"),
    ("sid", "Speaker identification", "Tamil", "SLCeleb"),
    ("asv_tamil", "Speaker verification", "Tamil", "SLCeleb"),
    ("sd_sinhala", "Speaker diarization", "Sinhala", "SiTa"),
    ("sd_tamil", "Speaker diarization", "Tamil", "SiTa"),
    ("ic_banking_sinhala", "Intent · banking", "Sinhala", "Banking intents"),
    ("ic_banking_tamil", "Intent · banking", "Tamil", "Banking intents"),
    ("ic_health_tamil", "Intent · health", "Tamil", "Health intents"),
]
PRIMARY = {"asr": ["wer", "cer"], "asv": ["eer"], "sd": ["der"], "er": ["accuracy", "macro_f1"],
           "sid": ["accuracy", "macro_f1"], "ic": ["accuracy", "macro_f1"]}


# ----------------------------------------------------------------------- refresh
def refresh(results_dirs):
    import soundfile as sf

    data = REPO / "data"

    def hours(paths):
        return sum(sf.info(str(p)).duration for p in paths) / 3600

    def read(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows = []
    for task, family, lang, source in TASKS:
        split_dir = data / ("asv" if task == "asv_tamil" else task)
        split = json.loads((split_dir / "split_v2.json").read_text())
        if task.startswith("sd_"):
            names = split["train"] + split["dev"] + split["test"]
            clips, hrs = len(names), hours(split_dir / f"{n}.wav" for n in names)
            speakers = ""
            scheme = f"{len(split['train'])}/{len(split['dev'])}/{len(split['test'])} recordings"
        elif task == "asv_tamil":
            labels = {r["filename"]: r["label"] for r in read(split_dir / "train_labels.csv")}
            trials = read(split_dir / "trials_tamil.csv")
            test = sorted({t[k] for t in trials for k in ("wav1", "wav2")})
            train = split["train"] + split["dev"]
            clips = len(train) + len(test)
            hrs = hours([split_dir / "train_audio" / f for f in train] + [split_dir / "audio" / f for f in test])
            speakers = len({labels[f] for f in train}) + len({f.split("/")[0] for f in test})
            scheme = f"{len(trials):,} test trials"
        else:
            marker = split_dir / "audio_dir.txt"
            audio = (split_dir / marker.read_text().strip()).resolve() if marker.exists() else split_dir / "audio"
            files = ([f for fold in split["folds"] for f in fold] if "folds" in split
                     else split["train"] + split["dev"] + split["test"])
            clips, hrs = len(files), hours(audio / f for f in files)
            spk_csv = split_dir / "speakers.csv"
            if spk_csv.exists():
                spk = {r["filename"]: r["speaker"] for r in read(spk_csv)}
                speakers = len({spk[f] for f in files})
            elif task == "er_tamil":
                speakers = len({f.split("_")[0] for f in files})
            elif task == "sid":
                speakers = len({f.split("/")[0] for f in files})
            elif task.startswith("asr_"):  # speakers inferred from the filename prefix
                speakers = len({Path(f).stem.rsplit("_", 1)[0] for f in files})
            else:
                speakers = ""
            scheme = "5 folds" if "folds" in split else "train/dev/test"
        rows.append(dict(task=task, family=family, language=lang, source=source, split_type=split["split_type"],
                         scheme=scheme, clips=clips, hours=round(hrs, 2), speakers=speakers))
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TASKS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Later results directories replace earlier ones task by task (e.g. a re-run of sd).
    runs = {}
    for d in results_dirs:
        summary = json.loads(next(Path(d).glob("results_*.json")).read_text())
        by_task = defaultdict(list)
        for r in summary["results"]:
            if r["status"] == "ok":
                by_task[r["name"]].append(r)
        runs.update(by_task)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "metric", "seed", "value"])
        for task, *_ in TASKS:
            for r in sorted(runs.get(task, []), key=lambda r: r["seed"]):
                for metric, value in r["metrics"].items():
                    writer.writerow([task, metric, r["seed"], f"{value:.6f}"])
    print(f"wrote {TASKS_CSV.relative_to(REPO)} and {RESULTS_CSV.relative_to(REPO)}")


# ----------------------------------------------------------------------- figures
def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])


def title(fig, t, text, sub):
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=12.5, weight="bold", color=t["ink"])
    fig.text(0.012, 0.925, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_tasks():
    with open(TASKS_CSV, newline="", encoding="utf-8") as f:
        return {r["task"]: r for r in csv.DictReader(f)}


def load_results():
    out = defaultdict(list)
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["task"], r["metric"])].append(float(r["value"]))
    return out


def fig_tasks(t, theme):
    tasks = load_tasks()
    order = [task for task, *_ in TASKS]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ys = list(range(len(order)))[::-1]
    for y, task in zip(ys, order):
        row = tasks[task]
        h = float(row["hours"])
        ax.barh(y, h, height=0.56, color=t[row["language"]], linewidth=0)
        unit = "recordings" if task.startswith("sd_") else "clips"
        ax.text(h + 0.6, y, f"{h:.1f} h · {int(row['clips']):,} {unit}", va="center", fontsize=8.5, color=t["ink2"])
    labels = [f"{tasks[k]['family']} · {tasks[k]['language']}" for k in order]
    ax.set_yticks(ys, labels, color=t["ink"], fontsize=9)
    ax.set_xlabel("hours of audio")
    ax.set_xlim(0, max(float(r["hours"]) for r in tasks.values()) * 1.32)
    title(fig, t, "Ten tasks across Sinhala and Tamil",
          "Audio per task; bar colour marks the language. Speaker ID and verification share SLCeleb's Tamil test audio.")
    fig.subplots_adjust(left=0.28, right=0.97, top=0.84, bottom=0.12)
    save(fig, "fig1-tasks", theme)


def fig_results(t, theme):
    tasks = load_tasks()
    res = load_results()
    panels = [
        ("Error rates  ·  lower is better", [
            ("asr_sinhala", "wer", "ASR · WER"), ("asr_sinhala", "cer", "ASR · CER"),
            ("asr_tamil", "wer", "ASR · WER"), ("asr_tamil", "cer", "ASR · CER"),
            ("asv_tamil", "eer", "Verification · EER"),
            ("sd_sinhala", "der", "Diarization · DER"), ("sd_tamil", "der", "Diarization · DER")]),
        ("Accuracy  ·  higher is better", [
            ("er_tamil", "accuracy", "Emotion"), ("sid", "accuracy", "Speaker ID"),
            ("ic_banking_sinhala", "accuracy", "Intent · banking*"), ("ic_banking_tamil", "accuracy", "Intent · banking"),
            ("ic_health_tamil", "accuracy", "Intent · health")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw=dict(wspace=0.62))
    fig.patch.set_facecolor(t["surface"])
    for ax, (heading, rows) in zip(axes, panels):
        style(ax, t)
        ys = list(range(len(rows)))[::-1]
        for y, (task, metric, _) in zip(ys, rows):
            vals = res[(task, metric)]
            mean = statistics.mean(vals)
            c = t[tasks[task]["language"]]
            ax.plot([min(vals), max(vals)], [y, y], color=c, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([mean], [y], s=64, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
            ax.text(max(vals) + 0.025, y, f"{mean:.3f}", va="center", fontsize=8.5, color=t["ink2"])
        ax.set_yticks(ys, [f"{name} · {tasks[task]['language']}" for task, _, name in rows], color=t["ink"],
                      fontsize=9)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(heading, loc="left", fontsize=9.5, color=t["ink2"], pad=8)
    title(fig, t, "Reference run: XLS-R 300M after Sinhala/Tamil continued pre-training",
          "Mean of 3 seeds (dot) and their range (line).  * no speaker ids: the same speakers can be in train "
          "and test, so it is optimistic.")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.78, bottom=0.08)
    save(fig, "fig2-reference-results", theme)


def print_tables():
    tasks = load_tasks()
    res = load_results()
    print("| Task | Language | Data | Clips | Hours | Speakers | Evaluation |")
    print("|---|---|---|---|---|---|---|")
    for task, *_ in TASKS:
        r = tasks[task]
        unit = " recordings" if task.startswith("sd_") else ""
        print(f"| {r['family']} | {r['language']} | {r['source']} | {int(r['clips']):,}{unit} | {float(r['hours']):.1f} "
              f"| {r['speakers'] or '–'} | {r['scheme']} |")
    print()
    print("| Task | Language | Metric | Mean ± s.d. (3 seeds) |")
    print("|---|---|---|---|")
    arrows = {"wer": "↓", "cer": "↓", "eer": "↓", "der": "↓", "accuracy": "↑", "macro_f1": "↑"}
    for task, family, lang, _ in TASKS:
        family_key = task.split("_")[0]
        for metric in PRIMARY[family_key]:
            vals = res.get((task, metric))
            if vals:
                name = {"wer": "WER", "cer": "CER", "eer": "EER", "der": "DER", "accuracy": "Accuracy",
                        "macro_f1": "Macro-F1"}[metric]
                print(f"| {family} | {lang} | {name} {arrows[metric]} | {statistics.mean(vals):.3f} ± "
                      f"{statistics.stdev(vals):.3f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", nargs="*", metavar="RESULTS_DIR",
                        help="rebuild docs/results/*.csv from data/ and these results directories")
    parser.add_argument("--print", action="store_true", help="print the README tables")
    args = parser.parse_args()
    if args.refresh is not None:
        refresh(args.refresh)
    for theme, t in THEMES.items():
        fig_tasks(t, theme)
        fig_results(t, theme)
    print(f"wrote figures to {OUT.relative_to(REPO)}/")
    if args.print:
        print_tables()


if __name__ == "__main__":
    main()
