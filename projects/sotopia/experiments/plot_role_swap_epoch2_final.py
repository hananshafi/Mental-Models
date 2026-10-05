#!/usr/bin/env python3
"""Final paper-style role-swap donor-pull figure using BIT epoch 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ORDER = ["structured_bit", "flat_mental_summary", "shuffled_mental"]
LABELS = {
    "structured_bit": "BIT",
    "flat_mental_summary": "Flat",
    "shuffled_mental": "Shuffled",
}
COLORS = {
    "structured_bit": "#3A5F87",
    "flat_mental_summary": "#CC8A4C",
    "shuffled_mental": "#5DA39A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", default="projects/sotopia/experiments/runs/stage1/bit_role_swap_epoch2_final/summary.json")
    parser.add_argument("--output", default="projects/sotopia/experiments/runs/stage1/bit_role_swap_epoch2_final/fig_role_swap_epoch2_final.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.summary_json).open() as f:
        summary = json.load(f)

    variants = [v for v in ORDER if v in summary]
    diag = np.asarray([summary[v]["diag_donor_pull"] for v in variants], dtype=np.float32)
    off = np.asarray([summary[v]["offdiag_donor_pull"] for v in variants], dtype=np.float32)
    gaps = diag - off
    y = np.arange(len(variants))[::-1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.15,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
    })

    fig, ax = plt.subplots(figsize=(5.35, 2.4), dpi=300)
    for yi, variant, off_val, diag_val, gap in zip(y, variants, off, diag, gaps):
        color = COLORS[variant]
        ax.plot([off_val, diag_val], [yi, yi], color="#B3BAC1", linewidth=7.5,
                alpha=0.75, solid_capstyle="round", zorder=1)
        ax.annotate(
            "",
            xy=(diag_val, yi),
            xytext=(off_val, yi),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.6,
                            mutation_scale=18),
            zorder=2,
        )
        ax.scatter([off_val], [yi], s=145, color="#D0D4D8",
                   edgecolor="#2A2A2A", linewidth=1.15, zorder=3)
        ax.scatter([diag_val], [yi], s=180, color=color,
                   edgecolor="#2A2A2A", linewidth=1.15, zorder=4)
    ax.set_yticks(y, [LABELS[v] for v in variants])
    ax.set_xlabel("semantic donor-pull score")
    ax.set_title("Role-Swap Donor-Pull")
    ax.grid(axis="x", color="#D8DCE0", linewidth=0.85, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(min(float(off.min()), float(diag.min())) - 0.025,
                max(float(off.max()), float(diag.max())) + 0.055)
    ax.set_ylim(-0.55, len(variants) - 0.45)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="matching role",
                   markerfacecolor="#3A5F87", markeredgecolor="#2A2A2A",
                   markeredgewidth=1.0, markersize=8.5),
            Line2D([0], [0], marker="o", color="w", label="non-matching roles",
                   markerfacecolor="#D0D4D8", markeredgecolor="#2A2A2A",
                   markeredgewidth=1.0, markersize=8.5),
        ],
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        edgecolor="#B8B8B8",
        fancybox=True,
    )
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.26, top=0.82)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, pad_inches=0.025)
    fig.savefig(out.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
