#!/usr/bin/env python3
"""Evaluate role-targeted counterfactual ranking items.

This compares whether BIT/Flat/Shuffled reward models correctly rank candidate
responses when the observable context is held fixed and exactly one hidden
mental role is flipped.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_counterfactual_flip_experiment import (  # noqa: E402
    SOTOPIA_DIMENSIONS,
    correct_margin,
    parse_scoring_dim_names,
    scoring_indices_for,
    score_two_candidates,
)
from stage2_grpo_agent_training_v3 import FrozenRewardModel  # noqa: E402


ROLE_ORDER = ["belief", "intent", "thought"]
MODEL_ORDER = ["observed_only", "shuffled", "flat", "bit"]
MODEL_LABELS = {
    "observed_only": "Observed only",
    "shuffled": "Shuffled",
    "flat": "Flat",
    "bit": "BIT",
}
MODEL_COLORS = {
    "observed_only": "#8A8A8A",
    "shuffled": "#5DA39A",
    "flat": "#CC8A4C",
    "bit": "#3A5F87",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def context_for(rec: dict[str, Any], state_key: str, context_mode: str) -> str:
    if context_mode == "observed":
        return rec["observable_context"]
    state = rec[state_key]
    if context_mode == "flat":
        return state.get("flat_context") or state["z_context"]
    if context_mode == "structured":
        return state["z_context"]
    raise ValueError(f"unknown context_mode={context_mode}")


def add_score_row(
    rows: list[dict[str, Any]],
    rec: dict[str, Any],
    model_name: str,
    state_label: str,
    context_mode: str,
    score_a: float,
    score_b: float,
) -> None:
    correct = rec[state_label]["correct"]
    margin = correct_margin(score_a, score_b, correct)
    rows.append({
        "pair_id": rec["pair_id"],
        "role": rec.get("role", ""),
        "model": model_name,
        "state_label": state_label,
        "context_mode": context_mode,
        "score_candidate_a": score_a,
        "score_candidate_b": score_b,
        "signed_margin_a_minus_b": score_a - score_b,
        "correct": correct,
        "correct_margin": margin,
        "is_correct": int(margin > 0),
    })


def score_records_with_model(
    reward_model: Any,
    records: list[dict[str, Any]],
    model_name: str,
    context_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        cand_a = rec["candidate_a"]
        cand_b = rec["candidate_b"]
        for state_label in ["state_a", "state_b"]:
            ctx = context_for(rec, state_label, context_mode)
            score_a, score_b = score_two_candidates(reward_model, ctx, cand_a, cand_b)
            add_score_row(rows, rec, model_name, state_label, context_mode, score_a, score_b)
    return rows


def load_reward(args: argparse.Namespace, checkpoint: str, scoring_indices: list[int] | None) -> FrozenRewardModel:
    return FrozenRewardModel(
        args.model_name,
        checkpoint,
        z_dim=args.z_dim,
        device=args.device,
        scoring_dim_indices=scoring_indices,
        ensemble_weight=args.ensemble_weight,
    )


def release_reward(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id", "role", "model", "state_label", "context_mode",
        "score_candidate_a", "score_candidate_b", "signed_margin_a_minus_b",
        "correct", "correct_margin", "is_correct",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], balance_threshold: float | None = None) -> dict[str, Any]:
    by_pair_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_pair_model.setdefault((row["pair_id"], row["model"]), []).append(row)

    observed_abs_margin_by_pair = {}
    for (pair_id, model), pair_rows in by_pair_model.items():
        if model != "observed_only":
            continue
        margins = [abs(float(row["signed_margin_a_minus_b"])) for row in pair_rows]
        if margins:
            observed_abs_margin_by_pair[pair_id] = float(np.mean(margins))

    kept_pair_ids = set(observed_abs_margin_by_pair)
    if balance_threshold is not None:
        kept_pair_ids = {pid for pid, margin in observed_abs_margin_by_pair.items() if margin <= balance_threshold}

    summary: dict[str, Any] = {
        "all_pairs": len({row["pair_id"] for row in rows}),
        "balanced_pairs": len(kept_pair_ids),
        "balance_threshold": balance_threshold,
        "models": {},
        "roles": {},
    }
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model and row["pair_id"] in kept_pair_ids]
        if not model_rows:
            continue
        pair_ids = sorted({row["pair_id"] for row in model_rows})
        flip_success = []
        flip_direction = []
        sensitivities = []
        for pair_id in pair_ids:
            pair_rows = [row for row in model_rows if row["pair_id"] == pair_id]
            state_a = next((row for row in pair_rows if row["state_label"] == "state_a"), None)
            state_b = next((row for row in pair_rows if row["state_label"] == "state_b"), None)
            if state_a is None or state_b is None:
                continue
            flip_success.append(int(state_a["is_correct"] == 1 and state_b["is_correct"] == 1))
            flip_direction.append(int(float(state_a["signed_margin_a_minus_b"]) > 0 and float(state_b["signed_margin_a_minus_b"]) < 0))
            sensitivities.append(abs(float(state_a["signed_margin_a_minus_b"]) - float(state_b["signed_margin_a_minus_b"])))

        summary["models"][model] = {
            "num_pairs": len(pair_ids),
            "state_accuracy": float(np.mean([int(row["is_correct"]) for row in model_rows])),
            "flip_accuracy": float(np.mean(flip_success)) if flip_success else None,
            "flip_direction": float(np.mean(flip_direction)) if flip_direction else None,
            "mean_correct_margin": float(np.mean([float(row["correct_margin"]) for row in model_rows])),
            "mean_state_sensitivity": float(np.mean(sensitivities)) if sensitivities else None,
        }

    for role in ROLE_ORDER:
        summary["roles"][role] = {}
        for model in MODEL_ORDER:
            role_rows = [
                row for row in rows
                if row["model"] == model and row["role"] == role and row["pair_id"] in kept_pair_ids
            ]
            if not role_rows:
                continue
            role_pair_ids = sorted({row["pair_id"] for row in role_rows})
            role_flip = []
            for pair_id in role_pair_ids:
                pair_rows = [row for row in role_rows if row["pair_id"] == pair_id]
                state_a = next((row for row in pair_rows if row["state_label"] == "state_a"), None)
                state_b = next((row for row in pair_rows if row["state_label"] == "state_b"), None)
                if state_a is not None and state_b is not None:
                    role_flip.append(int(state_a["is_correct"] == 1 and state_b["is_correct"] == 1))
            summary["roles"][role][model] = {
                "num_pairs": len(role_pair_ids),
                "state_accuracy": float(np.mean([int(row["is_correct"]) for row in role_rows])),
                "flip_accuracy": float(np.mean(role_flip)) if role_flip else None,
            }
    return summary


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Role-targeted counterfactual ranking summary",
        "",
        f"All pairs: {summary['all_pairs']}",
        f"Balanced pairs used: {summary['balanced_pairs']}",
        f"Observed-only balance threshold: {summary['balance_threshold']}",
        "",
        "| model | pairs | state accuracy | flip accuracy | flip direction | mean margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        if model not in summary["models"]:
            continue
        item = summary["models"][model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {item['num_pairs']} | "
            f"{item['state_accuracy']:.3f} | {item['flip_accuracy']:.3f} | "
            f"{item['flip_direction']:.3f} | {item['mean_correct_margin']:+.4f} |"
        )
    lines += ["", "## By flipped role", ""]
    for role in ROLE_ORDER:
        lines += [
            f"### {role}",
            "",
            "| model | pairs | state accuracy | flip accuracy |",
            "|---|---:|---:|---:|",
        ]
        for model in MODEL_ORDER:
            item = summary["roles"].get(role, {}).get(model)
            if not item:
                continue
            lines.append(
                f"| {MODEL_LABELS[model]} | {item['num_pairs']} | "
                f"{item['state_accuracy']:.3f} | {item['flip_accuracy']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def plot_role_accuracy(summary: dict[str, Any], out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })
    groups = ROLE_ORDER + ["avg"]
    x = np.arange(len(groups))
    width = 0.18
    fig, ax = plt.subplots(figsize=(6.2, 2.85), dpi=300)
    for mi, model in enumerate(MODEL_ORDER):
        vals = []
        for role in ROLE_ORDER:
            vals.append(summary["roles"].get(role, {}).get(model, {}).get("flip_accuracy", np.nan))
        vals.append(summary["models"].get(model, {}).get("flip_accuracy", np.nan))
        offset = (mi - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            color=MODEL_COLORS[model],
            edgecolor="#222222",
            linewidth=0.8,
            label=MODEL_LABELS[model],
            zorder=3,
        )
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x, ["belief", "intent", "thought", "avg"])
    ax.set_ylabel("flip accuracy")
    ax.set_title("Role-Targeted Counterfactual Ranking")
    ax.grid(axis="y", color="#D8DCE0", linewidth=0.85, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
                       edgecolor="#B8B8B8", framealpha=0.94, fancybox=True)
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.11, right=0.80, bottom=0.20, top=0.86)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-jsonl", default="projects/sotopia/experiments/runs/stage1/role_targeted_counterfactual/role_targeted_counterfactual_pairs_gpt4o_seed42.jsonl")
    parser.add_argument("--output-dir", default="projects/sotopia/experiments/runs/stage1/role_targeted_counterfactual/eval_bit_flat_shuffled")
    parser.add_argument("--model-name", default=os.environ.get("SOTOPIA_REWARD_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--bit-checkpoint", default="projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/epoch_2")
    parser.add_argument("--flat-checkpoint", default="projects/sotopia/experiments/runs/stage1/flat_mental_summary_qwen7b_seed42/best")
    parser.add_argument("--shuffled-checkpoint", default="projects/sotopia/experiments/runs/stage1/shuffled_mental_qwen7b_seed42/best")
    parser.add_argument("--device", default=os.environ.get("SOTOPIA_ROLE_CF_DEVICE", "cuda:2"))
    parser.add_argument("--scoring-dims", default=os.environ.get("SOTOPIA_COUNTERFACTUAL_DIMS", "goal,relationship,knowledge"))
    parser.add_argument("--ensemble-weight", type=float, default=0.7)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-threshold", type=float, default=None)
    parser.add_argument("--skip-shuffled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = read_jsonl(Path(args.counterfactual_jsonl))
    scoring_dim_names = parse_scoring_dim_names(args.scoring_dims)
    scoring_indices = scoring_indices_for(scoring_dim_names, SOTOPIA_DIMENSIONS, "role-targeted reward")
    rows: list[dict[str, Any]] = []

    model_specs = [
        ("bit", args.bit_checkpoint, "structured"),
        ("observed_only", args.bit_checkpoint, "observed"),
        ("flat", args.flat_checkpoint, "flat"),
    ]
    if not args.skip_shuffled:
        model_specs.append(("shuffled", args.shuffled_checkpoint, "structured"))

    # Score one checkpoint at a time to avoid keeping multiple 7B models resident.
    for model_name, checkpoint, context_mode in model_specs:
        print(f"\n=== Scoring {model_name}: {checkpoint} [{context_mode}] ===", flush=True)
        reward = load_reward(args, checkpoint, scoring_indices)
        rows.extend(score_records_with_model(reward, records, model_name, context_mode))
        release_reward(reward)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "role_targeted_scores.csv", rows)
    summary = summarize(rows, balance_threshold=args.balance_threshold)
    (out_dir / "role_targeted_summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(out_dir / "role_targeted_summary.md", summary)
    with (out_dir / "role_targeted_run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    plot_role_accuracy(summary, out_dir / "fig_role_targeted_flip_accuracy.png")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
