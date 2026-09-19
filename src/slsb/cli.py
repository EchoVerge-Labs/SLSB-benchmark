"""slsb: CLI entry point for the SLSB frozen-upstream benchmark.

    slsb run --upstream facebook/wav2vec2-xls-r-300m \\
             --tasks asr,sid,er,sd \\
             --data-dir ./data \\
             --seeds 0,1,2 \\
             --out results/ \\
             --mlflow-uri https://dagshub.com/EchoVerge-Labs/SLSB-benchmark.mlflow
"""
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from slsb import __version__
from slsb.runner import run_benchmark, validate_tasks_exist
from slsb.tasks import FAMILY_DIR_PREFIXES

console = Console()


def _parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_seeds(value: str) -> list[int]:
    return [int(v) for v in _parse_csv_list(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slsb", description="SLSB frozen-upstream speech benchmark")
    parser.add_argument("--version", action="version", version=f"slsb {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark for one upstream")
    run_parser.add_argument("--upstream", required=True,
                             help="HF repo id (e.g. facebook/wav2vec2-xls-r-300m) or a short alias "
                                  "(xlsr, mhubert147, wavlm_large)")
    run_parser.add_argument("--tasks", required=True,
                             help=f"Comma-separated task families: {', '.join(FAMILY_DIR_PREFIXES)}")
    run_parser.add_argument("--data-dir", default="./data", help="Path to the benchmark data/ directory")
    run_parser.add_argument("--seeds", default="0", help="Comma-separated random seeds, e.g. 0,1,2")
    run_parser.add_argument("--out", default="results/", help="Directory to write results JSON + CSV table into")
    run_parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI (skipped if not set)")
    run_parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    run_parser.add_argument("--device", default=None, help="Force a device (e.g. cpu, cuda); default: auto-detect")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    families = _parse_csv_list(args.tasks)
    unknown = [f for f in families if f not in FAMILY_DIR_PREFIXES]
    if unknown:
        console.print(f"[red]error:[/red] unknown task family(ies): {unknown}. "
                       f"Valid: {list(FAMILY_DIR_PREFIXES)}")
        return 2

    data_dir = Path(args.data_dir)
    missing = validate_tasks_exist(data_dir, families)
    if missing:
        console.print(f"[red]error:[/red] no data directory found under {data_dir} for task family(ies): "
                       f"{missing} (expected a folder starting with one of "
                       f"{[FAMILY_DIR_PREFIXES[f] for f in missing]})")
        return 2

    seeds = _parse_seeds(args.seeds)
    device = torch.device(args.device) if args.device else None
    out_dir = Path(args.out)

    console.print(f"[bold]slsb run[/bold]  upstream={args.upstream}  tasks={families}  seeds={seeds}")
    summary = run_benchmark(
        upstream_name=args.upstream, families=families, data_dir=data_dir, seeds=seeds,
        out_dir=out_dir, mlflow_uri=args.mlflow_uri, params_file=Path(args.params), device=device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_upstream = args.upstream.replace("/", "__")
    results_path = out_dir / f"results_{safe_upstream}.json"
    results_path.write_text(json.dumps(summary, indent=2))
    console.print(f"[green]done.[/green] results written to {results_path}")

    n_ok = sum(1 for r in summary["results"] if r["status"] == "ok")
    n_other = len(summary["results"]) - n_ok
    console.print(f"{n_ok} ok, {n_other} skipped/error (see {results_path})")
    return 0


def main(argv: list[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
