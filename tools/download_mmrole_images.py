#!/usr/bin/env python3
"""Fetch every image referenced by the released MMRole annotations.

The annotation release does not redistribute images. MMRole examples use two
sources, and both are required:

- character images from the upstream MMRole dataset (YanqiDai/MMRole_dataset),
  stored as projects/mmrole/images/<Collection>/<file>;
- COCO train2017 images, stored as projects/mmrole/images/coco/<file>.

Only the images referenced by projects/mmrole/training_data are fetched. Run
tools/download_data.py first. The script is resumable and finishes by checking
that every annotated example resolves to an image.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download


MMROLE_REPO = "YanqiDai/MMRole_dataset"
MMROLE_REVISION = "bb03411315609c3e8fc08216612d257d91b8f5db"
COCO_URL = "http://images.cocodataset.org/train2017/{}"
COCO_PREFIX = "COCO/train2017/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path,
                        default=Path("projects/mmrole/training_data"))
    parser.add_argument("--image-dir", type=Path, default=Path("projects/mmrole/images"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/datasets/MMRole_dataset"),
                        help="Download location for the upstream MMRole images.")
    parser.add_argument("--coco-dir", type=Path, default=None,
                        help="Existing COCO train2017 directory to copy from "
                             "instead of downloading.")
    parser.add_argument("--revision", default=MMROLE_REVISION)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--check-only", action="store_true",
                        help="Only report whether every referenced image resolves.")
    return parser.parse_args()


def iter_image_records(annotations_dir: Path):
    """Yield the dict holding image/image_local for every annotated example."""
    for path in sorted(annotations_dir.glob("*/*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("image") or record.get("image_local"):
                    yield record
                elif isinstance(record.get("scene"), dict):
                    # raw_annotated.jsonl keeps the image under "scene".
                    yield record["scene"]


def collect_references(annotations_dir: Path) -> tuple[set[str], set[str]]:
    coco, character = set(), set()
    for record in iter_image_records(annotations_dir):
        image = record.get("image", "")
        if image.startswith(COCO_PREFIX):
            coco.add(image[len(COCO_PREFIX):])
        elif image:
            character.add(image)
    return coco, character


def fetch_character_images(images: set[str], image_dir: Path, cache_dir: Path,
                           revision: str) -> None:
    missing = sorted(name for name in images if not (image_dir / name).is_file())
    if not missing:
        print(f"Character images: all {len(images)} present")
        return
    print(f"Character images: downloading {len(missing)} from {MMROLE_REPO}")
    snapshot = Path(snapshot_download(
        repo_id=MMROLE_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"images/{name}" for name in missing],
        local_dir=cache_dir,
    ))
    for name in missing:
        destination = image_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / "images" / name, destination)


def fetch_one_coco(name: str, destination: Path, coco_dir: Path | None) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    if coco_dir is not None:
        shutil.copy2(coco_dir / name, partial)
    else:
        with urllib.request.urlopen(COCO_URL.format(name), timeout=60) as response:
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
    partial.replace(destination)


def fetch_coco_images(names: set[str], image_dir: Path, coco_dir: Path | None,
                      workers: int) -> None:
    coco_root = image_dir / "coco"
    coco_root.mkdir(parents=True, exist_ok=True)
    missing = sorted(name for name in names if not (coco_root / name).is_file())
    if not missing:
        print(f"COCO images: all {len(names)} present")
        return
    source = coco_dir if coco_dir is not None else "images.cocodataset.org"
    print(f"COCO images: fetching {len(missing)} of {len(names)} from {source}")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one_coco, name, coco_root / name, coco_dir): name
            for name in missing
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - report and continue
                failures.append(f"{futures[future]}: {error}")
            if done % 250 == 0 or done == len(missing):
                print(f"  {done}/{len(missing)}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} COCO images failed; rerun to resume. First: {failures[0]}"
        )


def check_resolution(annotations_dir: Path, image_dir: Path, repository_root: Path) -> int:
    """Count examples whose image the training code cannot resolve."""
    sys.path.insert(0, str(repository_root / "projects" / "mmrole" / "scripts"))
    from model_utils import resolve_image  # noqa: E402

    total = unresolved = 0
    for record in iter_image_records(annotations_dir):
        total += 1
        unresolved += resolve_image(record, str(image_dir)) is None
    print(f"Resolved {total - unresolved}/{total} annotated examples to images in {image_dir}")
    return unresolved


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]

    def absolute(path: Path | None) -> Path | None:
        if path is None or path.is_absolute():
            return path
        return repository_root / path

    annotations_dir = absolute(args.annotations_dir)
    image_dir = absolute(args.image_dir)
    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"{annotations_dir} not found. Run python tools/download_data.py first."
        )

    if not args.check_only:
        coco, character = collect_references(annotations_dir)
        print(f"Referenced images: {len(coco)} COCO train2017, {len(character)} MMRole character")
        fetch_character_images(character, image_dir, absolute(args.cache_dir), args.revision)
        fetch_coco_images(coco, image_dir, absolute(args.coco_dir), args.workers)

    unresolved = check_resolution(annotations_dir, image_dir, repository_root)
    if unresolved:
        print(f"{unresolved} examples have no image; rerun this script to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
