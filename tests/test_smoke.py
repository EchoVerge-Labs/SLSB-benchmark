"""Basic import + config validation smoke tests. No GPU, no network, no data
needed -- these just confirm the package is installed and wired together."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_slsb():
    import slsb
    assert slsb.__version__


def test_cli_main_callable():
    from slsb.cli import main
    assert callable(main)


def test_cli_run_requires_args():
    from slsb.cli import main
    # `slsb run` with no args should exit non-zero (argparse) rather than raise.
    try:
        main(["run"])
        assert False, "expected SystemExit for missing required args"
    except SystemExit as e:
        assert e.code != 0


def test_defaults_yaml_valid():
    defaults = yaml.safe_load((REPO_ROOT / "configs" / "defaults.yaml").read_text())
    assert "downstream_head" in defaults
    assert "training" in defaults
    assert "evaluation" in defaults
    assert defaults["evaluation"]["frozen_upstream"] is True
    assert defaults["evaluation"]["weighted_sum"] is True
    assert defaults["downstream_head"]["pooling"] == "mean"


def test_params_yaml_valid():
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    for key in ("seed", "batch_size", "lr", "epochs"):
        assert key in params, f"params.yaml missing required key: {key}"
    assert isinstance(params["epochs"], int) and params["epochs"] > 0
    assert isinstance(params["batch_size"], int) and params["batch_size"] > 0
    assert isinstance(params["lr"], float) and params["lr"] > 0


def test_task_family_registry_matches_configs():
    from slsb.tasks import FAMILY_DIR_PREFIXES
    task_config_files = {p.stem for p in (REPO_ROOT / "configs" / "tasks").glob("*.yaml")}
    assert set(FAMILY_DIR_PREFIXES) == task_config_files
