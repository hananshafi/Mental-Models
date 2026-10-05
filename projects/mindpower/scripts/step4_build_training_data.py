#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.builders import (
    build_action_target,
    build_hierarchy_prediction_record,
    build_preference_pairs,
    build_probe_qa,
    build_tom_prediction_record,
    build_tom_preference_pairs,
    build_tom_probe_qa,
    partition_example_ids,
    split_records,
)
from mindpower.io_utils import ensure_dir, read_jsonl, write_jsonl
from mindpower.schemas import BDIAnnotation, ToMAnnotation


def annotation_from_dict(raw: dict) -> BDIAnnotation:
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


def tom_annotation_from_dict(raw: dict) -> ToMAnnotation:
    # Preserve the full metadata blob (annotator, provider, model, image_path,
    # heuristic flag) so downstream consumers can distinguish LLM- vs
    # heuristic-sourced annotations after step4 builds training artifacts.
    return ToMAnnotation(
        example_id=raw["example_id"],
        simulator=raw["simulator"],
        task_name=raw["task_name"],
        perspective=raw.get("perspective", {}),
        first_order_belief=raw.get("first_order_belief", {}),
        second_order_belief=raw.get("second_order_belief", {}),
        hidden_goal=raw.get("hidden_goal", ""),
        false_belief_risk=raw.get("false_belief_risk", ""),
        intervention_reason=raw.get("intervention_reason", ""),
        intervention_criticality=raw.get("intervention_criticality", ""),
        belief_divergence=raw.get("belief_divergence", "moderate"),
        visual_asymmetry_present=raw.get("visual_asymmetry_present", True),
        tom_relevance=raw.get("tom_relevance", "high"),
        belief_probes=raw.get("belief_probes", []),
        rationale=raw.get("rationale", ""),
        metadata=raw.get("metadata", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training data products from BDI annotations.")
    parser.add_argument(
        "--input_path",
        default=str(PROJECT_ROOT / "data" / "intermediate" / "bdi_annotations.jsonl"),
    )
    parser.add_argument(
        "--tom_input_path",
        default=str(PROJECT_ROOT / "data" / "intermediate" / "tom_annotations.jsonl"),
        help=(
            "Path to ToM annotations. Both step3_annotate_bdi.py (heuristic) "
            "and step3_annotate_tom_openai.py (LLM) now write to this same "
            "canonical filename by default."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "training_data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    annotations = [annotation_from_dict(row) for row in read_jsonl(args.input_path)]
    tom_annotations = []
    tom_input_path = Path(args.tom_input_path)
    if tom_input_path.exists():
        tom_annotations = [tom_annotation_from_dict(row) for row in read_jsonl(tom_input_path)]
    hierarchy = [build_hierarchy_prediction_record(annotation) for annotation in annotations]
    preference_pairs = [pair for annotation in annotations for pair in build_preference_pairs(annotation)]
    probes = [probe for annotation in annotations for probe in build_probe_qa(annotation)]
    actions = [build_action_target(annotation) for annotation in annotations]
    tom_prediction = [build_tom_prediction_record(annotation) for annotation in tom_annotations]
    tom_preference_pairs = [
        pair for annotation in tom_annotations for pair in build_tom_preference_pairs(annotation)
    ]
    tom_probes = [probe for annotation in tom_annotations for probe in build_tom_probe_qa(annotation)]

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Build ONE deterministic train/val partition over the union of all
    # example_ids from BDI and ToM annotations. Every downstream product
    # (hierarchy, preference pairs, probes, ToM records) is routed through the
    # same partition so:
    #   (a) a single example cannot appear in both train and val (fixes the
    #       per-product shuffle leak where 4 preference pairs of the same
    #       example could split 3/1 across train/val),
    #   (b) a given example lands in the SAME split across BDI and ToM
    #       products, which is required for any joint training/eval.
    all_example_ids: list[str] = []
    for annotation in annotations:
        all_example_ids.append(annotation.example_id)
    for annotation in tom_annotations:
        all_example_ids.append(annotation.example_id)
    split_sets = partition_example_ids(
        all_example_ids, train_ratio=args.train_ratio, seed=args.seed
    )

    products = {
        "hierarchy_prediction": hierarchy,
        "preference_pairs": preference_pairs,
        "probe_qa": probes,
        "action_targets": actions,
        "tom_prediction": tom_prediction,
        "tom_preference_pairs": tom_preference_pairs,
        "tom_probe_qa": tom_probes,
    }
    for name, records in products.items():
        if not records:
            continue
        split = split_records(
            records,
            train_ratio=args.train_ratio,
            seed=args.seed,
            split_sets=split_sets,
        )
        for split_name, rows in split.items():
            write_jsonl(output_dir / split_name / f"{name}.jsonl", rows)

    stats = {
        "annotations": len(annotations),
        "hierarchy_prediction": len(hierarchy),
        "preference_pairs": len(preference_pairs),
        "probe_qa": len(probes),
        "action_targets": len(actions),
        "tom_annotations": len(tom_annotations),
        "tom_prediction": len(tom_prediction),
        "tom_preference_pairs": len(tom_preference_pairs),
        "tom_probe_qa": len(tom_probes),
        "split": {
            "train_example_ids": len(split_sets["train"]),
            "val_example_ids": len(split_sets["val"]),
        },
    }
    with (output_dir / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Built training data -> {output_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
