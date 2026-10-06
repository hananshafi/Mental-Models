#!/usr/bin/env python3
"""Build the upload directory for the Mental Models Hugging Face dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


REPOSITORY_FILES = ("README.md", "DATA_SOURCES.md", "LICENSE")
STATIC_DATA_FILES = (
    (
        "sotopia/sotopia_turn_rewards_v3.jsonl",
        "sotopia/sotopia_turn_rewards_v3.jsonl",
    ),
    (
        "sotopia/generated_dataset/mental_model_persona_dataset.jsonl",
        "sotopia/mental_model_persona_dataset.jsonl",
    ),
    (
        "bigtom/data/bigtom_qwen_5k_annotated.jsonl",
        "bigtom/bigtom_qwen_5k_annotated.jsonl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("MENTAL_MODELS_DATA_ROOT", "..")),
        help="Directory containing the sotopia, bigtom, and mmrole projects.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("MENTAL_MODELS_HF_OUTPUT", "../Mental-Models-HF")),
        help="Upload-ready output directory.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks to avoid duplicating local data, or copy every file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def iter_selected_files(source_root: Path):
    for source_relative, destination_relative in STATIC_DATA_FILES:
        yield source_root / source_relative, Path("data") / destination_relative

    mmrole_root = source_root / "mmrole" / "training_data"
    yield mmrole_root / "dataset_stats.json", Path("data/mmrole/dataset_stats.json")
    for source_path in sorted(mmrole_root.rglob("*.jsonl")):
        if "smoketest" in source_path.name:
            continue
        yield source_path, Path("data/mmrole") / source_path.relative_to(mmrole_root)


def transfer_file(source_path: Path, destination_path: Path, copy_mode: str) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source_path, destination_path)
        return
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def inspect_file(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    rows = None
    if path.suffix == ".jsonl":
        rows = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
                rows += 1
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    template_root = repository_root / "hub"
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    staged_paths = []
    for filename in REPOSITORY_FILES:
        source_path = template_root / filename
        destination_path = output_dir / filename
        transfer_file(source_path, destination_path, "copy")
        staged_paths.append(destination_path)

    for source_path, destination_relative in iter_selected_files(args.source_root.resolve()):
        if not source_path.is_file():
            raise FileNotFoundError(f"Required data file is missing: {source_path}")
        destination_path = output_dir / destination_relative
        transfer_file(source_path, destination_path, args.copy_mode)
        staged_paths.append(destination_path)

    manifest_entries = []
    for staged_path in sorted(staged_paths):
        entry = inspect_file(staged_path)
        entry["path"] = staged_path.relative_to(output_dir).as_posix()
        manifest_entries.append(entry)

    manifest = {
        "schema_version": 1,
        "total_bytes": sum(int(entry["bytes"]) for entry in manifest_entries),
        "files": manifest_entries,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared {len(manifest_entries)} files in {output_dir}")
    print(f"Validated {manifest['total_bytes']:,} bytes")


if __name__ == "__main__":
    main()
