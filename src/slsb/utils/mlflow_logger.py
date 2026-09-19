"""Per-run results logging: <out_dir>/benchmark_table.csv (always) + MLflow
(only if a tracking URI is configured and DAGSHUB_TOKEN is set). Never fabricates
a row -- callers pass real metrics or call log_skipped() for absent-data tasks.
"""
import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CSV_COLUMNS = ["upstream", "task", "language", "metric", "value", "status", "git_commit", "timestamp", "split"]
DEFAULT_EXPERIMENT = "SLSB-benchmark"

_warned_no_token = False
_mlflow_ready = False
_mlflow_experiment_set = None


def get_git_commit(repo_root: Path = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root or Path.cwd(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _mlflow_enabled(mlflow_uri: str) -> bool:
    global _warned_no_token
    if not mlflow_uri:
        return False
    if os.environ.get("DAGSHUB_TOKEN"):
        return True
    if not _warned_no_token:
        print("WARNING: DAGSHUB_TOKEN is not set -- skipping remote MLflow logging. "
              "Results are still being written to the local benchmark_table.csv.")
        _warned_no_token = True
    return False


def _ensure_mlflow_configured(mlflow_uri: str, experiment: str):
    global _mlflow_ready, _mlflow_experiment_set
    if _mlflow_ready and _mlflow_experiment_set == (mlflow_uri, experiment):
        return
    import mlflow
    token = os.environ["DAGSHUB_TOKEN"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    _mlflow_ready = True
    _mlflow_experiment_set = (mlflow_uri, experiment)


def _append_csv_rows(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def log_run(csv_path: Path, upstream: str, task: str, language: str, seed: int, metrics: dict,
            perf: dict = None, split: str = "random", mlflow_uri: str = None,
            experiment: str = DEFAULT_EXPERIMENT, repo_root: Path = None):
    """metrics: the benchmark score(s) for this run, e.g. {"accuracy": 0.98, "macro_f1": 0.97}
    (or {"wer":.., "cer":..} / {"eer":..}) -- each becomes its own row in
    <csv_path> (the master score table).

    perf: run performance context, e.g. {"seconds_per_step": 0.3, "peak_gpu_gb": 1.5} --
    logged to MLflow only, not written to the CSV (keeps the master table to scores).

    split: how train/test was split for this run -- "random" (default) or
    "speaker_disjoint" for tasks where GroupShuffleSplit grouped by speaker.

    Both are logged together as a single MLflow run per (upstream, task, language).
    """
    git_commit = get_git_commit(repo_root)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = [
        {"upstream": upstream, "task": task, "language": language, "metric": name,
         "value": value, "status": "ok", "git_commit": git_commit, "timestamp": timestamp,
         "split": split}
        for name, value in metrics.items()
    ]
    _append_csv_rows(csv_path, rows)

    if not _mlflow_enabled(mlflow_uri):
        return

    import mlflow
    _ensure_mlflow_configured(mlflow_uri, experiment)
    with mlflow.start_run(run_name=f"{upstream}_{task}_{language}"):
        mlflow.log_params({
            "upstream": upstream, "task": task, "language": language,
            "seed": seed, "git_commit": git_commit, "split": split,
        })
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        for name, value in (perf or {}).items():
            if value is not None:
                mlflow.log_metric(name, value)


def already_logged_combos(csv_path: Path) -> set:
    """(upstream, task, language) combos with at least one status=ok row --
    lets the runner resume without redoing completed work."""
    if not csv_path.exists():
        return set()
    combos = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                combos.add((row["upstream"], row["task"], row["language"]))
    return combos


def log_skipped(csv_path: Path, upstream: str, task: str, language: str):
    """For a (upstream, task, language) combo that has no data or failed -- one
    blank-value row, never a fabricated number."""
    row = {
        "upstream": upstream, "task": task, "language": language, "metric": "",
        "value": "", "status": "skipped", "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_csv_rows(csv_path, [row])
