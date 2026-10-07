"""Loads params.yaml (upstream and downstream-head settings for the v0.3 protocol)."""
import os
from pathlib import Path

import yaml

DEFAULT_PARAMS_FILE = Path("params.yaml")


def load_params(params_file: Path = None) -> dict:
    """Reads params.yaml from the given path (default: params.yaml in the current
    working directory -- run slsb from the repo root, or pass an explicit path).

    SLSB_EPOCHS_OVERRIDE caps every head's max_epochs for this process (e.g. a
    fast sanity-check run) without touching params.yaml on disk.
    """
    path = Path(params_file) if params_file is not None else DEFAULT_PARAMS_FILE
    params = yaml.safe_load(path.read_text())
    override = os.environ.get("SLSB_EPOCHS_OVERRIDE")
    if override:
        for section in _sections(params):
            if "max_epochs" in section:
                section["max_epochs"] = min(section["max_epochs"], int(override))
    return params


def _sections(params: dict):
    for value in params.values():
        if isinstance(value, dict):
            yield value
            yield from _sections(value)
