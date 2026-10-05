#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.builders import build_intervention_points, simulator_episode_from_dict
from mindpower.config import default_paths
from mindpower.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intervention-point examples from simulator episodes.")
    parser.add_argument(
        "--input_path",
        default=str(default_paths()["raw_dir"] / "virtualhome_episodes.jsonl"),
    )
    parser.add_argument(
        "--output_path",
        default=str(default_paths()["intermediate_dir"] / "intervention_points.jsonl"),
    )
    parser.add_argument("--context_window", type=int, default=4)
    parser.add_argument(
        "--keep_all_frames",
        action="store_true",
        help=(
            "Keep every event as an intervention point instead of filtering to "
            "ToM-worthy moments (focus shift, critical assist verb, or object "
            "state change). Off by default so we don't waste LLM calls on frames "
            "with no ToM signal."
        ),
    )
    parser.add_argument(
        "--no_implicit_robot",
        action="store_true",
        help=(
            "Do not synthesize a 'robot_observer' partner for single-agent "
            "episodes. Off by default because the prompt in step3 assumes a "
            "robot observer exists."
        ),
    )
    args = parser.parse_args()

    episodes = [simulator_episode_from_dict(row) for row in read_jsonl(args.input_path)]
    examples = []
    total_frames = 0
    for episode in episodes:
        total_frames += len(episode.events)
        examples.extend(
            build_intervention_points(
                episode,
                context_window=args.context_window,
                tom_worthy_only=not args.keep_all_frames,
                synthesize_implicit_robot=not args.no_implicit_robot,
            )
        )

    write_jsonl(args.output_path, [example.to_dict() for example in examples])
    print(
        f"Built {len(examples)}/{total_frames} intervention examples -> {args.output_path}"
        f" (tom_worthy_only={not args.keep_all_frames}, "
        f"implicit_robot={not args.no_implicit_robot})"
    )


if __name__ == "__main__":
    main()
