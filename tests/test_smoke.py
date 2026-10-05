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


def test_params_yaml_valid():
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    for section in ("utterance", "asr", "asv"):
        cfg = params[section]
        assert cfg["lr_grid"] and all(isinstance(lr, float) and lr > 0 for lr in cfg["lr_grid"])
        assert isinstance(cfg["max_epochs"], int) and cfg["max_epochs"] > 0
        assert isinstance(cfg["patience"], int) and cfg["patience"] > 0


def test_task_family_registry_matches_configs():
    from slsb.tasks import FAMILY_DIR_PREFIXES
    task_config_files = {p.stem for p in (REPO_ROOT / "configs" / "tasks").glob("*.yaml")}
    assert set(FAMILY_DIR_PREFIXES) == task_config_files
