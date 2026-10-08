#!/usr/bin/env python
"""Log a finished benchmark run to MLflow (DagsHub) as ONE run per model.

`slsb run` writes a results folder (results_<upstream>.json, benchmark_table.csv,
run.log). This uploads it as a single "leaderboard row" in the experiment
slsb-<protocol>: every task's metric as `<task>/<metric>` (mean over seeds) and
`<task>/<metric>_std`, the model described in params and tags, and the folder's
files as artifacts (per-seed values, the chosen learning rates, logs).

Logging after the fact, from the folder, keeps every server's results in one
format and never leaves a half-logged run behind.

    python scripts/log_results.py results/v0.3/wavlm_large \\
        --name wavlm-large --kind frozen --family wavlm --base-model microsoft/wavlm-large

    python scripts/log_results.py results/v0.3/xlsr300m_copt200h_norm \\
        --name xlsr300m-copt200h-norm --kind adapted --family xlsr \\
        --base-model facebook/wav2vec2-xls-r-300m --checkpoint 9000 --pretrain-hours 200

Several folders may be given; a later one replaces an earlier one task by task
(e.g. a re-run of one task). When a new protocol changes only some tasks, re-run
those and pass --carry-over: the other tasks are copied from the model's run under
the previous protocol (only for tasks listed as unchanged in CARRY_OVER). Needs DAGSHUB_TOKEN (and DAGSHUB_USER) in the
environment. A model already logged for the protocol is left alone unless
--replace is given.
"""
import argparse
import csv
import io
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_URI = "https://dagshub.com/EchoVerge-LABS/SLSB-benchmark.mlflow"

# Protocol -> (the protocol it can carry results over from, tasks it changed). A task
# a protocol did not change scores identically under the previous one, so with
# --carry-over its metrics are copied from the model's run in slsb-<previous>.
CARRY_OVER = {"v0.4": ("v0.3", {"sid"})}


def load_folders(folders):
    """Merge the results JSONs of one or more run folders; later folders win per task."""
    summary, by_task = {}, {}
    for folder in folders:
        paths = sorted(Path(folder).glob("results_*.json"))
        if not paths:
            sys.exit(f"ERROR: no results_*.json in {folder}")
        for path in paths:
            data = json.loads(path.read_text())
            summary.update({k: v for k, v in data.items() if k != "results"})
            tasks = defaultdict(list)
            for entry in data["results"]:
                tasks[entry["name"]].append(entry)
            by_task.update(tasks)
    return summary, by_task


def summarize(by_task):
    """{task: [entries]} -> (metrics {"<task>/<metric>": mean, "..._std": s.d.},
    per-seed rows, problems). Mean and sample s.d. over seeds."""
    metrics, rows, problems = {}, [], []
    for task in sorted(by_task):
        entries = by_task[task]
        failed = [e for e in entries if e.get("status") != "ok"]
        if failed:
            problems.append(f"{task}: {len(failed)} of {len(entries)} seed(s) not ok")
        values = defaultdict(list)
        for e in entries:
            if e.get("status") == "ok":
                for name, value in e["metrics"].items():
                    values[name].append(value)
                    rows.append({"task": task, "seed": e["seed"], "metric": name, "value": value})
        for name, vals in values.items():
            metrics[f"{task}/{name}"] = statistics.mean(vals)
            metrics[f"{task}/{name}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return metrics, rows, problems


def layer_norm_of(summary, folders):
    """From the results JSON, or else the run log's "per-layer layer norm: ..." line."""
    if "layer_norm" in summary:
        return str(summary["layer_norm"])
    for folder in folders:
        log = Path(folder) / "run.log"
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("per-layer layer norm:"):
                    return line.split(":", 1)[1].strip()
    return "unknown"


def fetch_previous(uri, protocol, name):
    """Metrics of `name`'s run in slsb-<protocol>."""
    import mlflow
    from mlflow.tracking import MlflowClient

    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        sys.exit("ERROR: DAGSHUB_TOKEN is not set")
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ.get("DAGSHUB_USER", token))
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"slsb-{protocol}")
    runs = client.search_runs([experiment.experiment_id], filter_string=f"tags.`slsb.model` = '{name}'") \
        if experiment else []
    if not runs:
        sys.exit(f"ERROR: no run named {name} in slsb-{protocol} to carry results over from")
    return {"run_id": runs[0].info.run_id, "metrics": dict(runs[0].data.metrics)}


def do_normalize_of(upstream):
    config = Path(upstream) / "preprocessor_config.json"
    if config.is_file():
        return str(json.loads(config.read_text()).get("do_normalize"))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="results folder(s) written by `slsb run`")
    parser.add_argument("--name", required=True, help="run name, e.g. wavlm-large, xlsr300m-copt200h-norm")
    parser.add_argument("--kind", required=True, choices=["frozen", "adapted"])
    parser.add_argument("--family", required=True, help="e.g. xlsr, wavlm, hubert, mhubert, wav2vec2")
    parser.add_argument("--base-model", required=True, help="the pre-trained model it starts from (HF id)")
    parser.add_argument("--checkpoint", default="", help="adapted models: training step of the checkpoint")
    parser.add_argument("--pretrain-hours", default="", help="adapted models: hours of continued pre-training")
    parser.add_argument("--note", default="", help="free-text description shown on the run page")
    parser.add_argument("--protocol", default=None, help="only needed if the results JSON predates the field")
    parser.add_argument("--slsb-version", default=None, help="only needed if the results JSON predates the field")
    parser.add_argument("--slsb-commit", default=None, help="only needed if the results JSON predates the field")
    parser.add_argument("--mlflow-uri", default=DEFAULT_URI)
    parser.add_argument("--replace", action="store_true", help="delete (soft) an existing run of this model first")
    parser.add_argument("--carry-over", action="store_true",
                        help="take tasks missing from the folder(s) from this model's run under the previous "
                             "protocol, for tasks the new protocol did not change (see CARRY_OVER)")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary, by_task = load_folders(args.folders)
    protocol = summary.get("protocol") or args.protocol
    if not protocol:
        sys.exit("ERROR: the results JSON has no protocol field -- pass --protocol (e.g. v0.3)")
    if args.protocol and summary.get("protocol") and args.protocol != summary["protocol"]:
        sys.exit(f"ERROR: --protocol {args.protocol} but the results say {summary['protocol']}")
    metrics, rows, problems = summarize(by_task)
    seeds = sorted({e["seed"] for entries in by_task.values() for e in entries})
    print(f"{args.name}: protocol {protocol}, {len(by_task)} tasks, seeds {seeds}, {len(metrics) // 2} metrics")
    for p in problems:
        print(f"  ! {p}")
    if problems and not args.allow_incomplete:
        sys.exit("ERROR: incomplete results -- fix or pass --allow-incomplete")

    carried = {}
    if args.carry_over:
        if protocol not in CARRY_OVER:
            sys.exit(f"ERROR: protocol {protocol} has no previous protocol to carry results over from")
        previous, changed = CARRY_OVER[protocol]
        missing_changed = changed - set(by_task)
        if missing_changed:
            sys.exit(f"ERROR: {protocol} changed {sorted(changed)}; these must be re-run, not carried over: "
                     f"{sorted(missing_changed)}")
        carried = fetch_previous(args.mlflow_uri, previous, args.name)
        old_tasks = sorted({k.split("/")[0] for k in carried["metrics"]} - set(by_task))
        for key, value in carried["metrics"].items():
            if key.split("/")[0] in old_tasks:
                metrics[key] = value
        carried["tasks"] = old_tasks
        print(f"  carried over from slsb-{previous} run {carried['run_id']}: {', '.join(old_tasks)}")

    params = {
        "upstream": summary.get("upstream", ""), "kind": args.kind, "family": args.family,
        "base_model": args.base_model, "checkpoint": args.checkpoint, "pretrain_hours": args.pretrain_hours,
        "protocol": protocol, "slsb_version": summary.get("slsb_version") or args.slsb_version or "unknown",
        "slsb_commit": summary.get("slsb_commit") or args.slsb_commit or "unknown",
        "seeds": ",".join(map(str, seeds)),
        "tasks": ",".join(sorted(set(by_task) | set(carried.get("tasks", [])))),
        "layer_norm": layer_norm_of(summary, args.folders),
        "do_normalize": do_normalize_of(summary.get("upstream", "")) or "model default",
    }
    if carried:
        params["carried_over"] = f"slsb-{CARRY_OVER[protocol][0]} run {carried['run_id']}: {','.join(carried['tasks'])}"
    tags = {"kind": args.kind, "family": args.family, "protocol": protocol, "slsb.model": args.name}
    if args.dry_run:
        print(json.dumps({"params": params, "tags": tags}, indent=1))
        for key in sorted(metrics):
            if not key.endswith("_std"):
                print(f"  {key:32s} {metrics[key]:.4f} ± {metrics[key + '_std']:.4f}")
        return

    import mlflow
    from mlflow.tracking import MlflowClient

    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        sys.exit("ERROR: DAGSHUB_TOKEN is not set")
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ.get("DAGSHUB_USER", token))
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()
    experiment = mlflow.set_experiment(f"slsb-{protocol}")

    existing = client.search_runs([experiment.experiment_id], filter_string=f"tags.`slsb.model` = '{args.name}'")
    if existing and not args.replace:
        sys.exit(f"ERROR: {args.name} is already logged in slsb-{protocol} (run {existing[0].info.run_id}); "
                 f"pass --replace to overwrite it")
    for run in existing:
        client.delete_run(run.info.run_id)
        print(f"  replaced run {run.info.run_id} (soft-deleted)")

    with mlflow.start_run(run_name=args.name) as run:
        mlflow.set_tags(tags)
        if args.note:
            mlflow.set_tag("mlflow.note.content", args.note)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for folder in args.folders:
            for f in sorted(Path(folder).iterdir()):
                if f.is_file() and f.suffix in (".json", ".csv", ".log", ".txt"):
                    mlflow.log_artifact(str(f), artifact_path=Path(folder).name)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["task", "seed", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
        mlflow.log_text(buffer.getvalue(), "per_seed.csv")
    print(f"  logged {run.info.run_id} to slsb-{protocol} at {args.mlflow_uri}")


if __name__ == "__main__":
    main()
