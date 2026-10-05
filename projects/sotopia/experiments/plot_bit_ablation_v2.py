#!/usr/bin/env python3
"""BIT vs flat summary ablation — paper-style dot-plot.

Mirrors the visual style of the reference "Probe Signal Accessibility" figure:
big circular markers, thick gray connector arrow pointing toward the winner,
bold lowercase axis labels, light gray major-x grid only, legend in upper-left
with rounded frame.

Two rows (preference + future); reward regression dropped per request.
Shuffled control shown as a small marker per row to indicate the noise floor
without overwhelming the BIT-vs-flat comparison.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

# Matched 5-epoch numbers (BIT_e5 vs flat_best_e2 vs shuffled_best_e2)
METRICS = [
    {"name": "Preference\nloss",
     "BIT": 0.0001, "flat": 0.0101, "shuffled": 0.0118},
    {"name": "Future\nNLL",
     "BIT": 0.62,   "flat": 2.45,   "shuffled": 2.45},
]

C_BIT      = "#3A5F87"
C_FLAT     = "#CC8A4C"
C_SHUFFLED = "#5DA39A"
C_ARROW    = "#A6ADB4"


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.4,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 240,
    })


def main():
    set_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    rows = list(reversed(METRICS))   # bottom row = first metric in list
    y_positions = np.arange(len(rows))

    for y, m in zip(y_positions, rows):
        # log-scale: arrow + dots at log positions
        x_bit = m["BIT"]
        x_flat = m["flat"]
        x_shuf = m["shuffled"]

        # connector arrow: from flat → BIT (toward the winner). Use a thick
        # rounded translucent band + arrowhead.
        ax.plot([x_flat, x_bit], [y, y], color=C_ARROW,
                linewidth=11, alpha=0.55, solid_capstyle="round", zorder=1)
        # arrow head at BIT
        ax.annotate(
            "", xy=(x_bit, y), xytext=(x_flat, y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.55,head_length=0.85",
                            color=C_ARROW, linewidth=2.2, alpha=0.95),
            zorder=2,
        )

        # markers (BIT primary, flat primary, shuffled small control)
        ax.plot(x_shuf, y, marker="D", markersize=11, markerfacecolor=C_SHUFFLED,
                markeredgecolor="#1F1F1F", markeredgewidth=1.4, zorder=3)
        ax.plot(x_flat, y, marker="o", markersize=22, markerfacecolor=C_FLAT,
                markeredgecolor="#1F1F1F", markeredgewidth=1.6, zorder=4)
        ax.plot(x_bit, y, marker="o", markersize=22, markerfacecolor=C_BIT,
                markeredgecolor="#1F1F1F", markeredgewidth=1.6, zorder=4)

        # value labels
        ax.annotate(f"{x_bit:.4f}".rstrip("0").rstrip("."),
                    (x_bit, y), xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=10.5, color="#222", fontweight="bold")
        ax.annotate(f"{x_flat:.4f}".rstrip("0").rstrip("."),
                    (x_flat, y), xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=10.5, color="#222", fontweight="bold")

        # multiplier label above arrow midline
        ratio = x_flat / x_bit
        mult_text = f"{ratio:.0f}× worse" if ratio >= 10 else f"{ratio:.1f}× worse"
        ax.annotate(mult_text,
                    (np.sqrt(x_bit * x_flat), y),
                    xytext=(0, 30), textcoords="offset points",
                    ha="center", fontsize=12, color="#5C6066",
                    fontstyle="italic", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(5e-5, 5e0)
    ax.set_ylim(-0.85, len(rows) + 0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m["name"] for m in rows], fontweight="bold")
    ax.set_ylabel("validation metric", fontweight="bold")
    ax.set_xlabel("heldout NLL  (lower = better)", fontweight="bold")
    ax.set_title("Mental supervision ablation: BIT vs flat summary", pad=14)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0,)))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5)))
    ax.grid(axis="x", which="major", color="#D6DADE", linewidth=0.9, alpha=0.85, zorder=0)
    ax.grid(axis="x", which="minor", color="#E5E8EB", linewidth=0.5, alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#222"); ax.spines[s].set_linewidth(1.2)
    ax.tick_params(axis="x", which="major", length=5)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="BIT (ours)",
               markerfacecolor=C_BIT, markeredgecolor="#1F1F1F",
               markeredgewidth=1.5, markersize=14),
        Line2D([0], [0], marker="o", color="w", label="flat summary",
               markerfacecolor=C_FLAT, markeredgecolor="#1F1F1F",
               markeredgewidth=1.5, markersize=14),
        Line2D([0], [0], marker="D", color="w", label="shuffled (control)",
               markerfacecolor=C_SHUFFLED, markeredgecolor="#1F1F1F",
               markeredgewidth=1.2, markersize=10),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper left",
                    bbox_to_anchor=(0.02, 0.99), ncol=1,
                    handletextpad=0.6, frameon=True, framealpha=0.95,
                    edgecolor="#888", fancybox=True)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    out_path = Path(
        "projects/sotopia/experiments/runs/stage1/variant_eval/figs_paper/"
        "fig_bit_ablation_dotplot.png"
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
