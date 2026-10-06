#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "artifacts", "__pycache__"}
BINARY_SUFFIXES = {".gif", ".jpeg", ".jpg", ".pdf", ".png", ".webp"}
REQUIRED = (
    "README.md",
    "environment.yml",
    "docs/.nojekyll",
    "docs/index.html",
    "docs/styles.css",
    "docs/script.js",
    "docs/assets/favicon.svg",
    "docs/assets/mental-model-teaser.png",
    "docs/assets/method-overview.png",
    "docs/assets/sotopia-results.png",
    "third_party/sources.lock.json",
    "projects/sotopia/README.md",
    "projects/bigtom/README.md",
    "projects/mmrole/README.md",
    "projects/fantom/README.md",
    "projects/tomi/README.md",
    "projects/craigslist_bargain/README.md",
)
EXPECTED_PROJECTS = {
    "bigtom",
    "craigslist_bargain",
    "fantom",
    "mmrole",
    "sotopia",
    "tomi",
}
MACHINE_PATHS = (
    "/" + "bigdata/",
    "/" + "bigdata1/",
    "/" + "home/hanan/",
)
SECRET_PATTERNS = (
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{25,}"),
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"olp_[A-Za-z0-9]{20,}"),
)


def tracked_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == "third_party" and "src" in relative.parts:
            continue
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        files.append(path)
    return files


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    actual_projects = {
        path.name for path in (ROOT / "projects").iterdir() if path.is_dir()
    }
    if actual_projects != EXPECTED_PROJECTS:
        missing = sorted(EXPECTED_PROJECTS - actual_projects)
        unexpected = sorted(actual_projects - EXPECTED_PROJECTS)
        if missing:
            errors.append(f"missing paper project directories: {missing}")
        if unexpected:
            errors.append(f"unexpected non-paper project directories: {unexpected}")

    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"file exceeds 5 MiB: {relative}")
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 file: {relative}")
            continue
        for marker in MACHINE_PATHS:
            if marker in text:
                errors.append(f"machine-specific path {marker!r}: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential: {relative}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                errors.append(f"invalid JSON {relative}: {error}")

    canonical = ROOT / "projects/sotopia/scripts/stage3_evaluate_sotopia.py"
    ablation = ROOT / (
        "projects/sotopia/ablations/supervision_fraction/evaluation/"
        "stage3_evaluate_sotopia_utf8.py"
    )
    if canonical.is_file() and ablation.is_file():
        if canonical.read_text(encoding="utf-8") != ablation.read_text(encoding="utf-8"):
            errors.append("SOTOPIA ablation evaluator drifted from the canonical evaluator")

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
