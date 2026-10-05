#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.io_utils import read_jsonl


TAG_NAMES = ["perception", "belief", "desire", "intention", "decision", "action"]


def has_all_tags(text: str) -> bool:
    return all(f"<{tag}>" in text and f"</{tag}>" in text for tag in TAG_NAMES)


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight evaluator scaffold for MindPower outputs.")
    parser.add_argument("--responses_path", required=True)
    parser.add_argument(
        "--reference_path",
        default=str(PROJECT_ROOT / "training_data" / "val" / "hierarchy_prediction.jsonl"),
    )
    args = parser.parse_args()

    responses = {row["example_id"]: row for row in read_jsonl(args.responses_path)}
    references = read_jsonl(args.reference_path)

    total = 0
    format_ok = 0
    decision_match = 0
    action_nonempty = 0

    for ref in references:
        example_id = ref["example_id"]
        pred = responses.get(example_id)
        if pred is None:
            continue
        total += 1
        pred_text = pred.get("prediction", pred.get("target_text", ""))
        ref_text = ref.get("target_text", "")
        if has_all_tags(pred_text):
            format_ok += 1
        if extract_tag(pred_text, "decision") == extract_tag(ref_text, "decision"):
            decision_match += 1
        if extract_tag(pred_text, "action"):
            action_nonempty += 1

    metrics = {
        "evaluated_examples": total,
        "format_rate": (format_ok / total) if total else 0.0,
        "decision_exact_match": (decision_match / total) if total else 0.0,
        "action_nonempty_rate": (action_nonempty / total) if total else 0.0,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
