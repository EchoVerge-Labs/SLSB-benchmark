"""Benchmark orchestration: load one frozen upstream, run every requested
(task family x seed) combination against it, log each result, and return a
structured summary. Used by slsb.cli; also usable directly as a library.

Never fabricates a value: a task that errors (including speaker diarization,
which has no runner yet -- see tasks/speaker_diarization.py) is recorded with
status="skipped"/"error" and a reason, not a guessed score.
"""
import shutil
import time
from pathlib import Path

import torch

from slsb.tasks import EXCLUDED_TASK_DIRS, FAMILY_DIR_PREFIXES, FAMILY_MODULES
from slsb.tasks._common import set_seed
from slsb.tasks.speaker_diarization import discover as discover_diarization
from slsb.upstream.loader import load_upstream
from slsb.utils import mlflow_logger
from slsb.utils.datasets import discover_tasks, find_unrecognized_dirs
from slsb.utils.params import load_params

DIARIZATION_LANGS = ("sinhala", "tamil")


def matching_dirs(data_dir: Path, family: str) -> list[str]:
    """Directory names under data_dir that belong to the given task family,
    found by name prefix (data/<prefix>*) -- works even for families like
    "sd" that discover_tasks() doesn't recognize (RTTM-only, no label file).
    Folders in EXCLUDED_TASK_DIRS are left out."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    prefixes = FAMILY_DIR_PREFIXES[family]
    return sorted(
        entry.name for entry in data_dir.iterdir()
        if entry.is_dir() and entry.name.startswith(prefixes) and entry.name not in EXCLUDED_TASK_DIRS
    )


def validate_tasks_exist(data_dir: Path, families: list[str]) -> dict:
    """Returns {family: [missing families with no matching data dir]}. Empty dict
    means every requested family has at least one matching directory."""
    missing = [f for f in families if not matching_dirs(data_dir, f)]
    return missing


def family_of(spec) -> str:
    if spec.kind == "asr":
        return "asr"
    if spec.kind == "verification":
        return "asv"
    if spec.kind == "classification" and spec.task in ("sid", "er"):
        return spec.task
    raise ValueError(f"no runner for task {spec.name} (kind {spec.kind})")


def run_benchmark(upstream_name: str, families: list[str], data_dir: Path, seeds: list[int],
                   out_dir: Path, mlflow_uri: str = None, params_file: Path = None,
                   device: torch.device = None) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_table.csv"

    params = load_params(params_file)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_specs = discover_tasks(data_dir)
    wanted_dirnames = {d for family in families for d in matching_dirs(data_dir, family)}
    specs = sorted((s for s in all_specs if s.task_dir.name in wanted_dirnames), key=lambda s: s.name)

    unrecognized = find_unrecognized_dirs(data_dir)
    run_diarization = "sd" in families

    print(f"upstream: {upstream_name}  device: {device}")
    print(f"discovered {len(specs)} runnable task(s): {[s.name for s in specs]}")
    if unrecognized and not run_diarization:
        print(f"NOTE: {len(unrecognized)} data folder(s) with no recognized label file "
              f"were not requested (not in --tasks): {unrecognized}")

    upstream = load_upstream(upstream_name, device)
    results = []
    feature_root = out_dir / ".features"
    shared_by_family = {}

    # Task-major: each task's features are extracted once and reused by every
    # seed (the upstream is frozen, so they are identical), then deleted.
    for spec in specs:
        family = family_of(spec)
        module = FAMILY_MODULES[family]
        print(f"\n--- {spec.name} ({spec.kind}) ---", flush=True)
        start = time.time()
        try:
            prepared = module.prepare(upstream, spec, params, feature_root / spec.name,
                                      shared_by_family.setdefault(family, {}))
            feature_seconds = time.time() - start
            print(f"    features ready ({feature_seconds:.0f}s)", flush=True)
        except Exception as e:
            print(f"    FAILED preparing features: {e}")
            prepared, feature_seconds = None, None
            error = str(e)

        tuned = {}  # learning rates picked on dev by the first seed
        for seed in seeds:
            entry = {"upstream": upstream_name, "task": spec.task, "language": spec.lang,
                     "name": spec.name, "kind": spec.kind, "seed": seed}
            try:
                if prepared is None:
                    raise RuntimeError(error)
                print(f"  seed {seed}", flush=True)
                set_seed(seed)
                metrics_out, perf, split_type, details = module.run(prepared, params, seed, tuned)
                perf["feature_seconds"] = feature_seconds
                print(f"    {metrics_out}  split={split_type}  {details}", flush=True)
                mlflow_logger.log_run(csv_path, upstream_name, spec.task, spec.lang, seed=seed,
                                      metrics=metrics_out, perf=perf, split=split_type, details=details,
                                      mlflow_uri=mlflow_uri, repo_root=Path.cwd())
                entry.update(status="ok", split=split_type, metrics=metrics_out, perf=perf, details=details)
            except Exception as e:
                print(f"    FAILED: {e}")
                mlflow_logger.log_skipped(csv_path, upstream_name, spec.task, spec.lang)
                entry.update(status="error", error=str(e))
            results.append(entry)
        if prepared is not None:
            prepared.cleanup()

    for shared in shared_by_family.values():
        for cleanup in shared.get("cleanup", []):
            cleanup()
    shutil.rmtree(feature_root, ignore_errors=True)

    for seed in seeds:
        if run_diarization:
            for lang in DIARIZATION_LANGS:
                if f"sd_{lang}" not in wanted_dirnames:
                    continue
                basenames = discover_diarization(data_dir, lang)
                entry = {"upstream": upstream_name, "task": "sd", "language": lang,
                          "name": f"sd_{lang}", "kind": "diarization", "seed": seed}
                if not basenames:
                    print(f"--- sd_{lang} (diarization) --- no wav/rttm pairs found, skipping")
                    entry.update(status="skipped", error="no data")
                else:
                    print(f"--- sd_{lang} (diarization) ---", flush=True)
                    try:
                        FAMILY_MODULES["sd"].run(upstream, data_dir, lang, params, seed=seed)
                        raise AssertionError("unreachable: speaker_diarization.run should always raise")
                    except NotImplementedError as e:
                        print(f"    SKIPPED: {e}")
                        mlflow_logger.log_skipped(csv_path, upstream_name, "sd", lang)
                        entry.update(status="skipped", error=str(e))
                results.append(entry)

    del upstream
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"upstream": upstream_name, "families": families, "seeds": seeds, "results": results}
