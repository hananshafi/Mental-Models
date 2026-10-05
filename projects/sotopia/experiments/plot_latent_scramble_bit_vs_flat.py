#!/usr/bin/env python3
"""Plot BIT-vs-Flat latent scrambling intervention results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BIT = Path(
    "projects/sotopia/experiments/runs/stage1/"
    "latent_scramble_intervention_epoch2_fullval/summary.json"
)
DEFAULT_FLAT = Path(
    "projects/sotopia/experiments/runs/stage1/"
    "latent_scramble_intervention_flat_fullval/summary.json"
)
DEFAULT_OUT = Path(
    "projects/sotopia/experiments/runs/stage1/"
    "latent_scramble_bit_vs_flat"
)

MODEL_ORDER = ["BIT", "Flat"]
MODEL_COLORS = {"BIT": "#2F6FAE", "Flat": "#2CA25F"}
METRICS = [
    ("mental1_nll", "1st-order\nmental"),
    ("mental2_nll", "2nd-order\nmental"),
    ("z_combined_mse", "z-only\nreward"),
    ("joint_reward_mse", "joint\nreward"),
]


def load_ratios(path: Path, condition: str = "sample_shuffle_pair") -> dict[str, float]:
    data = json.loads(path.read_text())
    row = data[condition]
    return {metric: float(row[f"{metric}_ratio_to_baseline"]) for metric, _ in METRICS}


def set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.1,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )


def write_tables(out_dir: Path, ratios: dict[str, dict[str, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODEL_ORDER:
        for metric, label in METRICS:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "metric_label": label.replace("\n", " "),
                    "sample_shuffle_degradation": ratios[model][metric],
                }
            )
    with (out_dir / "bit_vs_flat_latent_scramble.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "bit_vs_flat_latent_scramble.json").open("w") as f:
        json.dump(ratios, f, indent=2, sort_keys=True)

    lines = [
        "# BIT vs Flat latent scramble",
        "",
        "Sample-shuffled z degradation factor relative to original z.",
        "",
        "| metric | BIT | Flat | BIT / Flat |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in METRICS:
        bit = ratios["BIT"][metric]
        flat = ratios["Flat"][metric]
        lines.append(f"| {label.replace(chr(10), ' ')} | {bit:.2f} | {flat:.2f} | {bit / flat:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines))


def plot_dot_compare(out_dir: Path, ratios: dict[str, dict[str, float]]) -> None:
    set_style()
    y = np.arange(len(METRICS))[::-1]

    fig, ax = plt.subplots(figsize=(5.75, 2.75), dpi=300)
    for yi, (metric, _) in zip(y, METRICS):
        flat = ratios["Flat"][metric]
        bit = ratios["BIT"][metric]
        ax.plot([flat, bit], [yi, yi], color="#C8CDD3", linewidth=5.5, solid_capstyle="round", zorder=1)
        ax.annotate(
            "",
            xy=(bit, yi),
            xytext=(flat, yi),
            arrowprops=dict(arrowstyle="-|>", color="#2F6FAE", linewidth=2.1, mutation_scale=14),
            zorder=2,
        )
        ax.scatter(flat, yi, s=135, color=MODEL_COLORS["Flat"], edgecolor="#2F2F2F", linewidth=1.0, zorder=3)
        ax.scatter(bit, yi, s=150, color=MODEL_COLORS["BIT"], edgecolor="#2F2F2F", linewidth=1.0, zorder=4)

    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 9.5)
    ax.set_xticks([1, 1.25, 1.5, 2, 4, 8])
    ax.set_xticklabels(["1", "1.25", "1.5", "2", "4", "8"])
    ax.set_yticks(y, [label for _, label in METRICS])
    ax.set_xlabel("error increase after sample-shuffling z", fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", color="#D8DDE3", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["BIT"],
                   markeredgecolor="#2F2F2F", markersize=8, label="BIT"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["Flat"],
                   markeredgecolor="#2F2F2F", markersize=8, label="Flat"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.22, top=0.96)
    fig.savefig(out_dir / "fig_latent_scramble_bit_vs_flat.png", pad_inches=0.02)
    fig.savefig(out_dir / "fig_latent_scramble_bit_vs_flat.pdf", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    out_dir = DEFAULT_OUT
    ratios = {
        "BIT": load_ratios(DEFAULT_BIT),
        "Flat": load_ratios(DEFAULT_FLAT),
    }
    write_tables(out_dir, ratios)
    plot_dot_compare(out_dir, ratios)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
