#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.builders import derive_reward_vector, make_manifest_id
from mindpower.io_utils import ensure_dir, read_jsonl
from mindpower.schemas import BDIAnnotation


MINDPOWER_REWARD_DIMS = [
    "belief_match",
    "desire_match",
    "intention_match",
    "decision_match",
    "atomic_local",
    "atomic_global",
    "executability",
]


def annotation_from_record(record: dict) -> BDIAnnotation:
    raw = record.get("annotation", record)
    return BDIAnnotation(
        example_id=raw["example_id"],
        simulator=raw["simulator"],
        task_name=raw["task_name"],
        perception=raw["perception"],
        belief=raw.get("belief", {}),
        desire=raw.get("desire", {}),
        intention=raw["intention"],
        decision=raw["decision"],
        action_plan=raw.get("action_plan", []),
        rationale=raw.get("rationale", ""),
        metadata=raw.get("metadata", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-model scaffold for MindPower.")
    parser.add_argument(
        "--train_path",
        default=str(PROJECT_ROOT / "training_data" / "train" / "hierarchy_prediction.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "checkpoints" / "stage0_reward"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.train_path)
    annotations = [annotation_from_record(row) for row in rows]
    reward_vectors = [derive_reward_vector(annotation) for annotation in annotations]

    out_dir = ensure_dir(args.output_dir)
    manifest = {
        "run_id": make_manifest_id("stage0", args.train_path),
        "status": "scaffold_only",
        "num_examples": len(annotations),
        "reward_dims": MINDPOWER_REWARD_DIMS,
        "mean_reward_vector": {
            dim: (
                sum(vector[dim] for vector in reward_vectors) / len(reward_vectors)
                if reward_vectors else 0.0
            )
            for dim in MINDPOWER_REWARD_DIMS
        },
        "next_step": (
            "Replace this scaffold with a multimodal latent reward model that predicts these "
            "dimensions from observation window + candidate action/response."
        ),
    }

    with (out_dir / "reward_config.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
