#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.heuristics import derive_bdi_annotation, derive_tom_annotation
from mindpower.io_utils import read_jsonl, write_jsonl
from mindpower.schemas import InterventionExample


def intervention_from_dict(raw: dict) -> InterventionExample:
    return InterventionExample(
        example_id=raw["example_id"],
        episode_id=raw["episode_id"],
        simulator=raw["simulator"],
        scene_id=raw["scene_id"],
        task_name=raw["task_name"],
        natural_language_goal=raw["natural_language_goal"],
        history_actions=raw.get("history_actions", []),
        current_action=raw["current_action"],
        next_action=raw.get("next_action"),
        acting_agent=raw["acting_agent"],
        partner_agent=raw.get("partner_agent"),
        visible_objects=raw.get("visible_objects", []),
        agent_states=raw.get("agent_states", []),
        metadata=raw.get("metadata", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate intervention points with BDI-style targets.")
    parser.add_argument(
        "--input_path",
        default=str(PROJECT_ROOT / "data" / "intermediate" / "intervention_points.jsonl"),
    )
    parser.add_argument(
        "--output_path",
        default=str(PROJECT_ROOT / "data" / "intermediate" / "bdi_annotations.jsonl"),
    )
    parser.add_argument(
        "--tom_output_path",
        # The canonical tom_annotations.jsonl filename is reserved for the LLM
        # annotator (step3_annotate_tom_openai.py). The heuristic fallback
        # writes to a disambiguated filename so the two can coexist on disk
        # and step4 by default consumes the LLM version.
        default=str(PROJECT_ROOT / "data" / "intermediate" / "tom_annotations_heuristic.jsonl"),
    )
    parser.add_argument("--mode", default="heuristic", choices=["heuristic"])
    args = parser.parse_args()

    examples = [intervention_from_dict(row) for row in read_jsonl(args.input_path)]
    bdi_annotations = [derive_bdi_annotation(example) for example in examples]
    tom_annotations = [derive_tom_annotation(example) for example in examples]
    write_jsonl(args.output_path, [annotation.to_dict() for annotation in bdi_annotations])
    write_jsonl(args.tom_output_path, [annotation.to_dict() for annotation in tom_annotations])

    print(f"Annotated {len(bdi_annotations)} BDI examples -> {args.output_path}")
    print(f"Annotated {len(tom_annotations)} ToM examples -> {args.tom_output_path}")


if __name__ == "__main__":
    main()
