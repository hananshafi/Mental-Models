#!/usr/bin/env python3
"""Download the released annotations and link them into project data paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO = "hanangani/Mental-Model-Annotation-Dataset"
DATASET_REVISION = "74a58651c4153f7f34e668778565251ed90ee3b3"
LINKS = (
    (
        "data/sotopia/sotopia_turn_rewards_v3.jsonl",
        "projects/sotopia/data/sotopia_turn_rewards_v3.jsonl",
    ),
    (
        "data/sotopia/mental_model_persona_dataset.jsonl",
        "projects/sotopia/data/mental_model_persona_dataset.jsonl",
    ),
    (
        "data/bigtom/bigtom_qwen_5k_annotated.jsonl",
        "projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl",
    ),
    ("data/mmrole", "projects/mmrole/training_data"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("artifacts/datasets/Mental-Model-Annotation-Dataset"),
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force-links", action="store_true")
    return parser.parse_args()


def create_link(source: Path, destination: Path, force: bool) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        if not force:
            raise FileExistsError(f"Link already exists: {destination}")
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing data at {destination}. "
            "Move it first or use --download-only."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative_source, target_is_directory=source.is_dir())


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    local_dir = args.local_dir
    if not local_dir.is_absolute():
        local_dir = repository_root / local_dir

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=local_dir,
        )
    ).resolve()
    print(f"Downloaded {args.repo_id} to {snapshot_path}")

    if args.download_only:
        return

    for source_relative, destination_relative in LINKS:
        source = snapshot_path / source_relative
        destination = repository_root / destination_relative
        if not source.exists():
            raise FileNotFoundError(f"Release file is missing: {source}")
        create_link(source, destination, args.force_links)
        print(f"Linked {destination_relative} -> {source_relative}")

    print(
        "MMRole images are not included. Run "
        "python tools/download_mmrole_images.py before MMRole training or evaluation."
    )


if __name__ == "__main__":
    main()
