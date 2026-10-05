#!/usr/bin/env python3
"""Generate and filter balanced counterfactual flip stimuli.

This script has two modes:

1. Generate a larger candidate pool from an existing counterfactual JSONL by
   applying neutral prefixes/suffixes to both candidate responses.
2. Filter a scored pool to keep pairs where observed-only scoring is close to
   indifferent, then optionally keep only pairs where the mental-z scorer flips
   correctly across hidden states.

The filtering step uses scores produced by run_counterfactual_flip_experiment.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from itertools import product
from pathlib import Path
from typing import Any


PREFIXES = [
    "",
    "I hear you. ",
    "That makes sense. ",
    "Thanks for saying that. ",
    "I want to handle this thoughtfully. ",
    "I appreciate you being direct. ",
    "I want to be fair about this. ",
    "I understand the concern. ",
]

SUFFIXES = [
    "",
    " I want this to feel fair.",
    " I want to keep trust between us.",
    " This feels like the fairest next step.",
    " I think that respects what matters here.",
    " We can adjust if I am missing something.",
    " I want us to move forward carefully.",
    " Does that work for you?",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_response(text: str) -> str:
    return " ".join(text.strip().split())


def make_variant_record(base: dict[str, Any], variant_idx: int, parts: tuple[str, str, str, str]) -> dict[str, Any]:
    a_prefix, b_prefix, a_suffix, b_suffix = parts
    rec = json.loads(json.dumps(base))
    rec["pair_id"] = f"{base['pair_id']}__bal_v{variant_idx:03d}"
    rec["candidate_a"] = normalize_response(f"{a_prefix}{base['candidate_a']}{a_suffix}")
    rec["candidate_b"] = normalize_response(f"{b_prefix}{base['candidate_b']}{b_suffix}")
    rec["notes"] = (
        f"{base.get('notes', '')} Balanced-pool variant {variant_idx}: "
        f"a_prefix={a_prefix!r}, b_prefix={b_prefix!r}, "
        f"a_suffix={a_suffix!r}, b_suffix={b_suffix!r}."
    )
    return rec


def generate_pool(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    source = read_jsonl(Path(args.source_jsonl))
    all_parts = list(product(PREFIXES, PREFIXES, SUFFIXES, SUFFIXES))
    pool: list[dict[str, Any]] = []

    for base in source:
        parts = all_parts[:]
        rng.shuffle(parts)

        # Always include the unmodified original first.
        ordered_parts = [("", "", "", "")]
        ordered_parts.extend(part for part in parts if part != ("", "", "", ""))
        for idx, variant_parts in enumerate(ordered_parts[: args.variants_per_base]):
            pool.append(make_variant_record(base, idx, variant_parts))

    write_jsonl(Path(args.pool_jsonl), pool)
    print(json.dumps({
        "source_pairs": len(source),
        "pool_pairs": len(pool),
        "variants_per_base": args.variants_per_base,
        "pool_jsonl": args.pool_jsonl,
    }, indent=2))


def read_scores(path: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    by_pair: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            pair_id = row["pair_id"]
            model = row["model"]
            state = row["state_label"]
            parsed: dict[str, Any] = dict(row)
            for key in [
                "score_candidate_a",
                "score_candidate_b",
                "signed_margin_a_minus_b",
                "correct_margin",
            ]:
                parsed[key] = float(parsed[key])
            parsed["is_correct"] = int(parsed["is_correct"])
            by_pair.setdefault(pair_id, {}).setdefault(model, {})[state] = parsed
    return by_pair


def root_pair_id(pair_id: str) -> str:
    return pair_id.split("__bal_v", maxsplit=1)[0]


def pair_metrics(pair_id: str, rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
    try:
        obs = rows["mental_observed_only"]["state_a"]["signed_margin_a_minus_b"]
        comp = rows["compression_observed"]["state_a"]["signed_margin_a_minus_b"]
        mz_a = rows["mental_correct_z"]["state_a"]["signed_margin_a_minus_b"]
        mz_b = rows["mental_correct_z"]["state_b"]["signed_margin_a_minus_b"]
        shuf_a = rows.get("mental_shuffled_z", {}).get("state_a", {}).get("signed_margin_a_minus_b")
        shuf_b = rows.get("mental_shuffled_z", {}).get("state_b", {}).get("signed_margin_a_minus_b")
        correct_a = rows["mental_correct_z"]["state_a"]["is_correct"]
        correct_b = rows["mental_correct_z"]["state_b"]["is_correct"]
        margin_a = rows["mental_correct_z"]["state_a"]["correct_margin"]
        margin_b = rows["mental_correct_z"]["state_b"]["correct_margin"]
    except KeyError:
        return None

    sensitivity = abs(mz_a - mz_b)
    shuffled_sensitivity = abs(shuf_a - shuf_b) if shuf_a is not None and shuf_b is not None else None
    return {
        "pair_id": pair_id,
        "root_pair_id": root_pair_id(pair_id),
        "mental_observed_margin": obs,
        "compression_observed_margin": comp,
        "max_observed_abs_margin": max(abs(obs), abs(comp)),
        "mental_state_a_margin": mz_a,
        "mental_state_b_margin": mz_b,
        "mental_state_sensitivity": sensitivity,
        "shuffled_state_sensitivity": shuffled_sensitivity,
        "mental_correct_both": int(correct_a == 1 and correct_b == 1),
        "mental_min_correct_margin": min(margin_a, margin_b),
        "mental_flip_direction": int(mz_a > 0 and mz_b < 0),
    }


def filter_pool(args: argparse.Namespace) -> None:
    records = {rec["pair_id"]: rec for rec in read_jsonl(Path(args.source_jsonl))}
    scored = read_scores(Path(args.scores_csv))
    metrics = []
    for pair_id, rows in scored.items():
        if pair_id not in records:
            continue
        item = pair_metrics(pair_id, rows)
        if item is not None:
            if args.root_mode == "source_turn":
                rec_source = records[pair_id].get("source", {})
                episode_id = rec_source.get("episode_id")
                turn_num = rec_source.get("turn_num")
                speaker = rec_source.get("speaker")
                if episode_id is not None and turn_num is not None:
                    item["root_pair_id"] = f"{episode_id}::t{turn_num}::{speaker}"
            metrics.append(item)

    candidates = [
        item for item in metrics
        if abs(item["mental_observed_margin"]) <= args.mental_observed_threshold
        and abs(item["compression_observed_margin"]) <= args.compression_observed_threshold
        and item["mental_state_sensitivity"] >= args.min_state_sensitivity
    ]
    if args.require_mental_flip:
        candidates = [
            item for item in candidates
            if item["mental_correct_both"] == 1 and item["mental_flip_direction"] == 1
        ]

    if args.selection_sort == "observed_only":
        candidates.sort(
            key=lambda item: (
                item["max_observed_abs_margin"],
                abs(item["mental_observed_margin"]),
                abs(item["compression_observed_margin"]),
                item["root_pair_id"],
                item["pair_id"],
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                item["max_observed_abs_margin"],
                -item["mental_correct_both"],
                -item["mental_min_correct_margin"],
                -item["mental_state_sensitivity"],
            )
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_roots: set[str] = set()
    for item in candidates:
        if len(selected) >= args.max_pairs:
            break
        root = item["root_pair_id"]
        if args.one_per_base and root in selected_roots:
            continue
        selected.append(item)
        selected_ids.add(item["pair_id"])
        selected_roots.add(root)

    selected_records = [records[item["pair_id"]] for item in selected]
    write_jsonl(Path(args.output_jsonl), selected_records)

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "root_pair_id",
        "selected",
        "mental_observed_margin",
        "compression_observed_margin",
        "max_observed_abs_margin",
        "mental_state_a_margin",
        "mental_state_b_margin",
        "mental_state_sensitivity",
        "shuffled_state_sensitivity",
        "mental_correct_both",
        "mental_min_correct_margin",
        "mental_flip_direction",
    ]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(metrics, key=lambda x: (x["max_observed_abs_margin"], -x["mental_state_sensitivity"])):
            row = dict(item)
            row["selected"] = int(item["pair_id"] in selected_ids)
            writer.writerow(row)

    summary = {
        "scored_pairs": len(metrics),
        "eligible_pairs": len(candidates),
        "selected_pairs": len(selected),
        "selected_roots": sorted(selected_roots),
        "thresholds": {
            "mental_observed_threshold": args.mental_observed_threshold,
            "compression_observed_threshold": args.compression_observed_threshold,
            "min_state_sensitivity": args.min_state_sensitivity,
            "require_mental_flip": args.require_mental_flip,
            "one_per_base": args.one_per_base,
            "selection_sort": args.selection_sort,
            "root_mode": args.root_mode,
        },
        "output_jsonl": args.output_jsonl,
        "metrics_csv": args.metrics_csv,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "filter"], required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pool-jsonl")
    parser.add_argument("--variants-per-base", type=int, default=96)

    parser.add_argument("--scores-csv")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--summary-json")
    parser.add_argument("--mental-observed-threshold", type=float, default=0.05)
    parser.add_argument("--compression-observed-threshold", type=float, default=0.08)
    parser.add_argument("--min-state-sensitivity", type=float, default=0.02)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--require-mental-flip", action="store_true")
    parser.add_argument("--one-per-base", action="store_true")
    parser.add_argument(
        "--selection-sort",
        choices=["balanced_then_mental", "observed_only"],
        default="balanced_then_mental",
        help=(
            "observed_only sorts selected examples only by observed/compression "
            "balance. balanced_then_mental preserves the older dev diagnostic "
            "behavior that prefers examples where mental_correct_z succeeds."
        ),
    )
    parser.add_argument(
        "--root-mode",
        choices=["pair_id", "source_turn"],
        default="pair_id",
        help="source_turn makes --one-per-base keep at most one pair per SOTOPIA source turn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "generate":
        if not args.pool_jsonl:
            raise ValueError("--pool-jsonl is required for --mode generate")
        generate_pool(args)
    else:
        required = ["scores_csv", "output_jsonl", "metrics_csv", "summary_json"]
        missing = [f"--{name.replace('_', '-')}" for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(f"Missing required arguments for --mode filter: {missing}")
        filter_pool(args)


if __name__ == "__main__":
    main()
