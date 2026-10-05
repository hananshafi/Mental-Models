#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.io_utils import ensure_dir, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT scaffold for MindPower hierarchy generation.")
    parser.add_argument(
        "--train_path",
        default=str(PROJECT_ROOT / "training_data" / "train" / "hierarchy_prediction.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "checkpoints" / "stage1_sft"),
    )
    parser.add_argument("--preview_examples", type=int, default=5)
    args = parser.parse_args()

    rows = read_jsonl(args.train_path)
    preview = rows[: args.preview_examples]
    out_dir = ensure_dir(args.output_dir)
    write_jsonl(out_dir / "sft_preview.jsonl", preview)

    manifest = {
        "status": "scaffold_only",
        "num_examples": len(rows),
        "preview_examples": len(preview),
        "training_target": "Generate <perception><belief><desire><intention><decision><action> from embodied context.",
        "next_step": "Plug this dataset into a Qwen2.5-VL or video-VLM LoRA trainer.",
    }
    with (out_dir / "sft_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
