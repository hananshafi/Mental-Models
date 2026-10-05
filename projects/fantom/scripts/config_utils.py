from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def load_project_config(path: Path) -> dict[str, Any]:
    config = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    for key in ("benchmark_root", "split_path", "official_scorer_path"):
        if config["dataset"].get(key):
            config["dataset"][key] = resolve_repo_path(config["dataset"][key])
    for key in ("bigtom_root", "bigtom_scripts"):
        if config["shared"].get(key):
            config["shared"][key] = resolve_repo_path(config["shared"][key])
    for model in config["models"].values():
        for key in ("stage1_ckpt", "policy_ckpt", "adapter_ckpt", "prior_summary"):
            if model.get(key):
                model[key] = resolve_repo_path(model[key])
    return config
