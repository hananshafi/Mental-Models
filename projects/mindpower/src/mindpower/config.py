from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def default_paths() -> dict[str, Path]:
    return {
        "project_root": PROJECT_ROOT,
        "raw_dir": PROJECT_ROOT / "data" / "raw",
        "intermediate_dir": PROJECT_ROOT / "data" / "intermediate",
        "training_dir": PROJECT_ROOT / "training_data",
        "checkpoints_dir": PROJECT_ROOT / "checkpoints",
        "logs_dir": PROJECT_ROOT / "logs",
    }


def load_paths_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        return default_paths()

    path = Path(config_path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    paths = default_paths()
    data = raw.get("data", {})
    for key, rel in data.items():
        if key.endswith("_dir"):
            paths[key] = PROJECT_ROOT / rel
    return paths
