#!/usr/bin/env python3
"""Plot BIT role-swap donor-pull checkpoint sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["epoch_1", "epoch_2", "epoch_3", "epoch_4", "epoch_5", "best"]
LABELS = {
    "epoch_1": "epoch 1",
    "epoch_2": "epoch 2",
    "epoch_3": "epoch 3",
    "epoch_4": "epoch 4",
    "epoch_5": "epoch 5",
    "best": "best",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", default="projects/sotopia/experiments/runs/stage1/bit_role_swap_epoch_sweep/summary.json")
    parser.add_argument("--output", default="projects/sotopia/experiments/runs/stage1/bit_role_swap_epoch_sweep/fig_role_swap_epoch_sweep.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.summary_json).open() as f:
        summary = json.load(f)
    variants = [v for v in ORDER if v in summary]
    gaps = np.asarray([summary[v]["donor_pull_gap"] for v in variants], dtype=np.float32)
    diag_mass = np.asarray([summary[v]["diag_mass"] for v in variants], dtype=np.float32)
    winner = int(np.argmax(gaps))

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
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(5.4, 2.45), dpi=300)
    x = np.arange(len(variants))
    colors = ["#7EA6C9"] * len(variants)
    colors[winner] = "#1F4E8C"
    if "best" in variants and variants.index("best") != winner:
        colors[variants.index("best")] = "#9EA4AA"

    bars = ax.bar(x, gaps, color=colors, edgecolor="#222222", linewidth=0.9, zorder=3)
    ax.plot(x, gaps, color="#263238", linewidth=1.2, marker="o", markersize=4.2, zorder=4)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    for idx, (bar, value) in enumerate(zip(bars, gaps)):
        label = f"{value:+.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0016,
            label,
            ha="center",
            va="bottom",
            fontsize=8.7,
            fontweight="bold",
            color="#111111",
        )
    ax.text(
        winner,
        gaps[winner] + 0.007,
        "winner",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#1F4E8C",
    )

    ax.set_xticks(x, [LABELS.get(v, v) for v in variants])
    ax.set_ylabel("donor-pull gap")
    ax.set_xlabel("BIT checkpoint")
    ax.set_title("Role-Swap Specificity Across Checkpoints")
    ax.set_ylim(0.0, max(float(gaps.max()) + 0.013, 0.04))
    ax.grid(axis="y", color="#D8DCE0", linewidth=0.8, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, pad_inches=0.025)
    fig.savefig(out.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)

    metrics = {
        "best_checkpoint": variants[winner],
        "best_donor_pull_gap": float(gaps[winner]),
        "best_diag_mass": float(diag_mass[winner]),
    }
    (out.parent / "epoch_sweep_best.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
