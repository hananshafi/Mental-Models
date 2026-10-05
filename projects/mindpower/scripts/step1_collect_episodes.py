#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.config import default_paths
from mindpower.io_utils import ensure_dir, write_jsonl
from mindpower.schemas import EpisodeRequest
from mindpower.simulators import get_simulator_adapter


def load_requests(path: Path, simulator: str, dry_run: bool, limit: int) -> list[EpisodeRequest]:
    with path.open() as f:
        raw = json.load(f)
    requests = [
        EpisodeRequest(
            request_id=item["request_id"],
            simulator=simulator,
            scene_id=item["scene_id"],
            task_name=item["task_name"],
            natural_language_goal=item["natural_language_goal"],
            scripted_actions=item["scripted_actions"],
            dry_run=dry_run,
        )
        for item in raw
    ]
    if limit > 0:
        requests = requests[:limit]
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect simulator episodes for MindPower scaffolding.")
    parser.add_argument("--simulator", default="virtualhome", choices=["virtualhome", "tdw"])
    parser.add_argument(
        "--tasks_path",
        default=str(PROJECT_ROOT / "configs" / "virtualhome_seed_tasks.json"),
    )
    parser.add_argument(
        "--virtualhome_dataset_root",
        default=None,
        help="Path to a VirtualHome export root or repo directory containing executable_programs/.",
    )
    parser.add_argument(
        "--output_path",
        default=str(default_paths()["raw_dir"] / "virtualhome_episodes.jsonl"),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--scene_filter", default=None)
    parser.add_argument("--virtualhome_root", default=None)
    parser.add_argument("--virtualhome_binary", default=None)
    parser.add_argument("--tdw_build_path", default=None)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    adapter = get_simulator_adapter(
        args.simulator,
        virtualhome_root=args.virtualhome_root,
        virtualhome_binary=args.virtualhome_binary,
        tdw_build_path=args.tdw_build_path,
    )
    if args.virtualhome_dataset_root:
        if args.simulator != "virtualhome":
            raise ValueError("--virtualhome_dataset_root can only be used with --simulator virtualhome")
        if not hasattr(adapter, "build_requests_from_export_root"):
            raise RuntimeError("Selected adapter does not support dataset-root ingestion")
        requests = adapter.build_requests_from_export_root(
            args.virtualhome_dataset_root,
            limit=args.limit,
            scene_filter=args.scene_filter,
        )
    else:
        requests = load_requests(Path(args.tasks_path), args.simulator, args.dry_run, args.limit)
    episodes = adapter.collect_batch(requests, output_path.parent)
    write_jsonl(output_path, [episode.to_dict() for episode in episodes])

    print(f"Collected {len(episodes)} episodes -> {output_path}")
    print(f"Simulator available: {adapter.is_available()} | dry_run={args.dry_run}")
    if args.virtualhome_dataset_root:
        print(f"VirtualHome export root: {args.virtualhome_dataset_root}")


if __name__ == "__main__":
    main()
