"""Benchmark orchestration: load one frozen upstream, run every requested
(task family x seed) combination against it, log each result, and return a
structured summary. Used by slsb.cli; also usable directly as a library.

Never fabricates a value: a task that errors (including speaker diarization,
which has no runner yet -- see tasks/speaker_diarization.py) is recorded with
status="skipped"/"error" and a reason, not a guessed score.
"""
import time
from pathlib import Path

import torch

from slsb.tasks import FAMILY_DIR_PREFIXES, FAMILY_MODULES
from slsb.tasks.speaker_diarization import discover as discover_diarization
from slsb.upstream.loader import load_upstream
from slsb.utils import mlflow_logger
from slsb.utils.datasets import discover_tasks, find_unrecognized_dirs, load_verification_task
from slsb.utils.params import load_params

DIARIZATION_LANGS = ("sinhala", "tamil")


def matching_dirs(data_dir: Path, family: str) -> list[str]:
    """Directory names under data_dir that belong to the given task family,
    found by name prefix (data/<prefix>*) -- works even for families like
    "sd" that discover_tasks() doesn't recognize (RTTM-only, no label file)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    prefixes = FAMILY_DIR_PREFIXES[family]
    return sorted(
        entry.name for entry in data_dir.iterdir()
        if entry.is_dir() and entry.name.startswith(prefixes)
    )


def validate_tasks_exist(data_dir: Path, families: list[str]) -> dict:
    """Returns {family: [missing families with no matching data dir]}. Empty dict
    means every requested family has at least one matching directory."""
    missing = [f for f in families if not matching_dirs(data_dir, f)]
    return missing


def _sort_key(spec):
    # sid must run before asv_* within the same upstream/seed: ASV reuses SID's trained head.
    is_sid = spec.kind == "classification" and spec.task == "sid"
    return (0 if is_sid else 1, spec.name)


def _run_one(upstream, spec, params, seed, sid_head_cache):
    if spec.kind == "asr":
        metrics_out, perf, _head, split_type = FAMILY_MODULES["asr"].run(upstream, spec, params, seed=seed)
        return metrics_out, perf, split_type

    if spec.kind == "classification" and spec.task == "sid":
        metrics_out, perf, head, split_type = FAMILY_MODULES["sid"].run(upstream, spec, params, seed=seed)
        sid_head_cache[seed] = head
        return metrics_out, perf, split_type

    if spec.kind == "classification" and spec.task == "er":
        metrics_out, perf, _head, split_type, _diag = FAMILY_MODULES["er"].run(upstream, spec, params, seed=seed)
        return metrics_out, perf, split_type

    if spec.kind == "verification":
        sid_head = sid_head_cache.get(seed)
        if sid_head is None:
            raise RuntimeError("no trained SID head available for this seed yet (sid task must run first)")
        trials = load_verification_task(spec)
        metrics_out, perf = FAMILY_MODULES["asv"].run(upstream, trials, sid_head, params)
        return metrics_out, perf, "random"

    raise ValueError(f"unknown task kind: {spec.kind}")


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
    specs = sorted((s for s in all_specs if s.task_dir.name in wanted_dirnames), key=_sort_key)

    unrecognized = find_unrecognized_dirs(data_dir)
    run_diarization = "sd" in families

    print(f"upstream: {upstream_name}  device: {device}")
    print(f"discovered {len(specs)} runnable task(s): {[s.name for s in specs]}")
    if unrecognized and not run_diarization:
        print(f"NOTE: {len(unrecognized)} data folder(s) with no recognized label file "
              f"were not requested (not in --tasks): {unrecognized}")

    upstream = load_upstream(upstream_name, device)
    results = []

    for seed in seeds:
        print(f"\n=== seed: {seed} ===")
        sid_head_cache = {}

        for spec in specs:
            print(f"--- {spec.name} ({spec.kind}) ---", flush=True)
            entry = {"upstream": upstream_name, "task": spec.task, "language": spec.lang,
                      "name": spec.name, "kind": spec.kind, "seed": seed}
            try:
                start = time.time()
                metrics_out, perf, split_type = _run_one(upstream, spec, params, seed, sid_head_cache)
                elapsed = time.time() - start
                print(f"    {metrics_out}  perf={perf}  split={split_type}  (wall {elapsed:.1f}s)")
                mlflow_logger.log_run(csv_path, upstream_name, spec.task, spec.lang, seed=seed,
                                       metrics=metrics_out, perf=perf, split=split_type,
                                       mlflow_uri=mlflow_uri, repo_root=Path.cwd())
                entry.update(status="ok", split=split_type, metrics=metrics_out, perf=perf)
            except Exception as e:
                print(f"    FAILED: {e}")
                mlflow_logger.log_skipped(csv_path, upstream_name, spec.task, spec.lang)
                entry.update(status="error", error=str(e))
            results.append(entry)

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
