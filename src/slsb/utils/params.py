"""Loads params.yaml (training hyperparams for the linear/CTC probes)."""
import os
from pathlib import Path

import yaml

DEFAULT_PARAMS_FILE = Path("params.yaml")


def load_params(params_file: Path = None) -> dict:
    """Reads params.yaml from the given path (default: params.yaml in the current
    working directory -- run slsb from the repo root, or pass an explicit path).

    epochs can be overridden per-process via SLSB_EPOCHS_OVERRIDE (e.g. for a fast
    sanity-check run) without touching params.yaml on disk.
    """
    path = Path(params_file) if params_file is not None else DEFAULT_PARAMS_FILE
    params = yaml.safe_load(path.read_text())
    override = os.environ.get("SLSB_EPOCHS_OVERRIDE")
    if override:
        params["epochs"] = int(override)
    return params
