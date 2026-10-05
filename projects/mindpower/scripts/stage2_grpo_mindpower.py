#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.io_utils import ensure_dir, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO scaffold for MindPower.")
    parser.add_argument(
        "--train_path",
        default=str(PROJECT_ROOT / "training_data" / "train" / "hierarchy_prediction.jsonl"),
    )
    parser.add_argument(
        "--reward_config",
        default=str(PROJECT_ROOT / "checkpoints" / "stage0_reward" / "reward_config.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "checkpoints" / "stage2_grpo"),
    )
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--grpo_epochs", type=int, default=1)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_path)
    reward_config = {}
    reward_path = Path(args.reward_config)
    if reward_path.exists():
        with reward_path.open() as f:
            reward_config = json.load(f)

    manifest = {
        "status": "scaffold_only",
        "num_examples": len(train_rows),
        "group_size": args.group_size,
        "grpo_epochs": args.grpo_epochs,
        "reward_dims": reward_config.get("reward_dims", []),
        "reward_formula": (
            "R_total = lambda_1 * R_learned + lambda_2 * R_atomic + "
            "lambda_3 * R_format + lambda_4 * R_exec"
        ),
        "next_step": (
            "Wire in policy generation, frozen reward scoring, group-normalized advantages, "
            "and PPO-style clipped updates."
        ),
    }

    out_dir = ensure_dir(args.output_dir)
    with (out_dir / "grpo_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
