#!/usr/bin/env python3
"""Plot visuals for the intrinsic value of recursive mental decomposition.

Expected CSV columns for real results:
  case_id, case_type, condition, correct_margin

Optional alternative columns:
  score_candidate_a, score_candidate_b, correct

case_type values:
  first_order, second_order, mixed

condition values:
  observed_only, z1_intervention_only, z2_intervention_only,
  z1_then_z2_recursive, parallel_z1_z2, no_z2, z2_shuffled
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_GREY = "#7A7A7A"
PLOT_LIGHT_GREY = "#D6D8DC"
PLOT_DARK_GREY = "#4B4B4B"
PLOT_PURPLE = "#7B61B5"
PLOT_RED = "#C84C4C"
PLOT_TEAL = "#2A9D8F"

CASE_ORDER = ["first_order", "second_order", "mixed"]
CASE_LABELS = {
    "first_order": "First-order\npartner state",
    "second_order": "Second-order\nrecursive state",
    "mixed": "Mixed",
}

CONDITION_ORDER = [
    "observed_only",
    "z1_intervention_only",
    "z2_intervention_only",
    "z1_then_z2_recursive",
    "parallel_z1_z2",
    "no_z2",
    "z2_shuffled",
]
CONDITION_LABELS = {
    "observed_only": "Observed only",
    "z1_intervention_only": "z1 only",
    "z2_intervention_only": "z2 only",
    "z1_then_z2_recursive": "z1 -> z2",
    "parallel_z1_z2": "Parallel z1/z2",
    "no_z2": "No z2",
    "z2_shuffled": "Shuffled z2",
}
CONDITION_COLORS = {
    "observed_only": PLOT_GREY,
    "z1_intervention_only": PLOT_GREEN,
    "z2_intervention_only": PLOT_PURPLE,
    "z1_then_z2_recursive": PLOT_BLUE,
    "parallel_z1_z2": PLOT_TEAL,
    "no_z2": PLOT_LIGHT_GREY,
    "z2_shuffled": PLOT_RED,
}


def style_plot_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C7CBD1", linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color="#E8EAED", linewidth=0.5, alpha=0.85)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.1, length=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def style_plot_legend(legend: Any) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(PLOT_LIGHT_GREY)
    legend.get_frame().set_alpha(0.94)
    for text in legend.get_texts():
        text.set_fontsize(9)


def correct_margin_from_row(row: dict[str, str]) -> float:
    if row.get("correct_margin") not in {None, ""}:
        return float(row["correct_margin"])
    score_a = float(row["score_candidate_a"])
    score_b = float(row["score_candidate_b"])
    correct = row.get("correct", "a").strip().lower()
    if correct == "a":
        return score_a - score_b
    if correct == "b":
        return score_b - score_a
    raise ValueError(f"correct must be a or b, got {correct!r}")


def normalize_case_type(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "first": "first_order",
        "z1": "first_order",
        "second": "second_order",
        "recursive": "second_order",
        "z2": "second_order",
    }
    return aliases.get(value, value)


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_type = normalize_case_type(row.get("case_type", "mixed"))
            condition = row["condition"].strip()
            margin = correct_margin_from_row(row)
            rows.append({
                "case_id": row.get("case_id", row.get("pair_id", "")),
                "case_type": case_type,
                "condition": condition,
                "correct_margin": margin,
                "is_correct": int(margin > 0),
            })
    return rows


def demo_rows(seed: int = 42, n_per_case: int = 80) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    means = {
        "first_order": {
            "observed_only": 0.03,
            "z1_intervention_only": 0.34,
            "z2_intervention_only": 0.12,
            "z1_then_z2_recursive": 0.41,
            "parallel_z1_z2": 0.27,
            "no_z2": 0.23,
            "z2_shuffled": -0.04,
        },
        "second_order": {
            "observed_only": 0.02,
            "z1_intervention_only": 0.14,
            "z2_intervention_only": 0.35,
            "z1_then_z2_recursive": 0.48,
            "parallel_z1_z2": 0.27,
            "no_z2": 0.08,
            "z2_shuffled": -0.06,
        },
        "mixed": {
            "observed_only": 0.02,
            "z1_intervention_only": 0.24,
            "z2_intervention_only": 0.25,
            "z1_then_z2_recursive": 0.44,
            "parallel_z1_z2": 0.28,
            "no_z2": 0.15,
            "z2_shuffled": -0.05,
        },
    }
    rows = []
    for case_type in CASE_ORDER:
        for i in range(n_per_case):
            case_id = f"demo_{case_type}_{i:03d}"
            for condition in CONDITION_ORDER:
                margin = float(rng.normal(means[case_type][condition], 0.17))
                rows.append({
                    "case_id": case_id,
                    "case_type": case_type,
                    "condition": condition,
                    "correct_margin": margin,
                    "is_correct": int(margin > 0),
                })
    return rows


def condition_case_tables(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    by_case_condition = {}
    for row in rows:
        by_case_condition.setdefault((row["case_id"], row["condition"]), []).append(row["correct_margin"])

    baseline_by_case = {}
    for row in rows:
        if row["condition"] == "observed_only":
            baseline_by_case.setdefault(row["case_id"], []).append(row["correct_margin"])
    baseline_by_case = {k: float(np.mean(v)) for k, v in baseline_by_case.items()}

    gain_values = {}
    margin_values = {}
    for row in rows:
        key = (row["condition"], row["case_type"])
        margin_values.setdefault(key, []).append(row["correct_margin"])
        base = baseline_by_case.get(row["case_id"], 0.0)
        gain_values.setdefault(key, []).append(row["correct_margin"] - base)

    mean_margin = {key: float(np.mean(vals)) for key, vals in margin_values.items()}
    mean_gain = {key: float(np.mean(vals)) for key, vals in gain_values.items()}
    return mean_margin, mean_gain


def write_aggregate_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    mean_margin, mean_gain = condition_case_tables(rows)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "case_type", "mean_correct_margin", "mean_causal_gain"])
        writer.writeheader()
        for condition in CONDITION_ORDER:
            for case_type in CASE_ORDER:
                key = (condition, case_type)
                if key not in mean_margin:
                    continue
                writer.writerow({
                    "condition": condition,
                    "case_type": case_type,
                    "mean_correct_margin": mean_margin[key],
                    "mean_causal_gain": mean_gain[key],
                })


def plot_causal_graph(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "context": (0.12, 0.55, "Observed\nDialogue"),
        "z1": (0.36, 0.70, "z1\nPartner State"),
        "z2": (0.60, 0.70, "z2\nRecursive State"),
        "response": (0.36, 0.32, "Candidate\nResponse"),
        "reward": (0.82, 0.52, "Reward /\nPreference"),
    }
    colors = {
        "context": PLOT_LIGHT_GREY,
        "z1": PLOT_GREEN,
        "z2": PLOT_PURPLE,
        "response": "#F2D388",
        "reward": PLOT_BLUE,
    }

    for name, (x, y, label) in nodes.items():
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=colors[name], edgecolor=PLOT_DARK_GREY, linewidth=1.4),
        )

    def arrow(a: str, b: str, color: str = PLOT_DARK_GREY, lw: float = 2.0, style: str = "-") -> None:
        ax.annotate(
            "",
            xy=nodes[b][:2],
            xytext=nodes[a][:2],
            arrowprops=dict(arrowstyle="->", color=color, linewidth=lw, linestyle=style, shrinkA=28, shrinkB=32),
        )

    arrow("context", "z1", PLOT_GREEN)
    arrow("z1", "z2", PLOT_PURPLE, lw=2.6)
    arrow("context", "response", PLOT_DARK_GREY)
    arrow("response", "reward", "#A06B00")
    arrow("z1", "reward", PLOT_GREEN, lw=1.7)
    arrow("z2", "reward", PLOT_PURPLE, lw=2.2)
    ax.text(0.49, 0.83, "intrinsic causal path", color=PLOT_PURPLE, fontsize=11, fontweight="bold", ha="center")
    ax.text(0.52, 0.09, "Visual test: intervene on z1, z2, or z1 -> z2 and measure preference change.", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_gain_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    _, mean_gain = condition_case_tables(rows)
    conditions = [c for c in CONDITION_ORDER if any((c, case) in mean_gain for case in CASE_ORDER)]
    cases = [c for c in CASE_ORDER if any((cond, c) in mean_gain for cond in conditions)]
    mat = np.full((len(conditions), len(cases)), np.nan)
    for i, condition in enumerate(conditions):
        for j, case_type in enumerate(cases):
            mat[i, j] = mean_gain.get((condition, case_type), np.nan)

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(cases)), [CASE_LABELS[c] for c in cases])
    ax.set_yticks(np.arange(len(conditions)), [CONDITION_LABELS.get(c, c) for c in conditions])
    ax.set_title("Causal Gain Over Observed-Only Baseline", fontsize=13, fontweight="bold")
    for i in range(len(conditions)):
        for j in range(len(cases)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correct margin gain", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_intervention_ladder(rows: list[dict[str, Any]], out_path: Path) -> None:
    mean_margin, _ = condition_case_tables(rows)
    ladder = ["observed_only", "z1_intervention_only", "z2_intervention_only", "z1_then_z2_recursive"]
    xs = np.arange(len(ladder))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for case_type, color in zip(CASE_ORDER, [PLOT_GREEN, PLOT_PURPLE, PLOT_BLUE]):
        ys = [mean_margin.get((condition, case_type), np.nan) for condition in ladder]
        if not np.isfinite(ys).any():
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.4, color=color, label=CASE_LABELS[case_type].replace("\n", " "))
    ax.axhline(0, color=PLOT_DARK_GREY, linewidth=1.0, linestyle="--")
    ax.set_xticks(xs, [CONDITION_LABELS[c] for c in ladder])
    ax.set_title("Intervention Ladder", fontsize=13, fontweight="bold")
    style_plot_axes(ax, ylabel="mean correct margin")
    style_plot_legend(ax.legend(frameon=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_condition_accuracy(rows: list[dict[str, Any]], out_path: Path) -> None:
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row["is_correct"])
    conditions = [c for c in CONDITION_ORDER if c in by_condition]
    values = [float(np.mean(by_condition[c])) for c in conditions]
    colors = [CONDITION_COLORS.get(c, PLOT_DARK_GREY) for c in conditions]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(np.arange(len(conditions)), values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(conditions)), [CONDITION_LABELS.get(c, c) for c in conditions], rotation=25, ha="right")
    ax.set_title("Intervention Flip Accuracy", fontsize=13, fontweight="bold")
    style_plot_axes(ax, ylabel="accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--demo", action="store_true", help="Create stylized demo figures without real experiment results.")
    parser.add_argument("--demo-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_causal_graph(out_dir / "causal_decomposition_graph.png")
    rows = []
    if args.results_csv:
        rows = load_rows(args.results_csv)
    elif args.demo:
        rows = demo_rows(seed=args.demo_seed)

    if rows:
        write_aggregate_csv(rows, out_dir / "causal_decomposition_aggregates.csv")
        plot_gain_heatmap(rows, out_dir / "causal_gain_heatmap.png")
        plot_intervention_ladder(rows, out_dir / "intervention_ladder.png")
        plot_condition_accuracy(rows, out_dir / "intervention_accuracy.png")
        with (out_dir / "visualization_config.json").open("w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

    print(f"Wrote causal decomposition visuals to {out_dir}")


if __name__ == "__main__":
    main()
