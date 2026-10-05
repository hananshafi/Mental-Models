#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "third_party" / "sources.lock.json"
PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "openai",
    "numpy",
    "scipy",
    "scikit-learn",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_head(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Mental-Models runtime.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing optional assets.")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    print(f"Repository: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        warnings.append("The consolidated environment is tested with Python 3.10.")

    print("\nPackages:")
    for package in PACKAGES:
        version = package_version(package)
        print(f"  {package:20} {version or 'MISSING'}")
        if version is None:
            failures.append(f"missing package: {package}")

    print("\nCommands:")
    for command in ("git", "conda", "nvidia-smi", "tmux"):
        location = shutil.which(command)
        print(f"  {command:20} {location or 'MISSING'}")
        if command in {"git", "nvidia-smi"} and location is None:
            warnings.append(f"missing command: {command}")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    print("\nPinned sources:")
    for name, source in lock["sources"].items():
        checkout = ROOT / "third_party" / "src" / name
        head = git_head(checkout)
        expected = source["revision"]
        if head is None:
            status = "MISSING"
            warnings.append(f"bootstrap third-party source: {name}")
        elif head != expected:
            status = f"MISMATCH ({head[:12]})"
            failures.append(f"wrong revision for {name}")
        else:
            status = expected[:12]
        print(f"  {name:20} {status}")

    print("\nCredentials:")
    for variable in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"):
        print(f"  {variable:20} {'set' if os.environ.get(variable) else 'not set'}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")

    if failures or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
