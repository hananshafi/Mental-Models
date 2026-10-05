#!/usr/bin/env python3
"""ToMi-2 zero-shot transfer dot-plot (BIT / flat / shuffled SOTOPIA variants).

Mirrors fig_bit_ablation_dotplot.png style.  X-axis = held-out probe F1
(higher = better) so the arrow points toward the higher-F1 variant on each
row, consistent with the reference convention "arrow points to winner."
This honestly displays that BIT does NOT transfer best — the figure is
suitable for a Limitations subsection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


METRICS = [
    {"name": "Recursive\nFB (order-2)",
     "BIT": 0.750, "flat": 0.794, "shuffled": 0.822},
    {"name": "ToM-required\nrecursive FB",
     "BIT": 0.580, "flat": 0.563, "shuffled": 0.637},
    {"name": "branch_3way\n(overall)",
     "BIT": 0.533, "flat": 0.630, "shuffled": 0.628},
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
        "axes.titlesize": 16,
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
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    rows = list(reversed(METRICS))
    y_positions = np.arange(len(rows))

    for y, m in zip(y_positions, rows):
        x_bit, x_flat, x_shuf = m["BIT"], m["flat"], m["shuffled"]
        winner_x = max(x_bit, x_flat, x_shuf)
        loser_x = min(x_bit, x_flat, x_shuf)

        # connector arrow points to highest-F1 variant
        ax.plot([loser_x, winner_x], [y, y], color=C_ARROW,
                linewidth=11, alpha=0.55, solid_capstyle="round", zorder=1)
        ax.annotate(
            "", xy=(winner_x, y), xytext=(loser_x, y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.55,head_length=0.85",
                            color=C_ARROW, linewidth=2.2, alpha=0.95),
            zorder=2,
        )

        ax.plot(x_shuf, y, marker="D", markersize=11, markerfacecolor=C_SHUFFLED,
                markeredgecolor="#1F1F1F", markeredgewidth=1.4, zorder=3)
        ax.plot(x_flat, y, marker="o", markersize=22, markerfacecolor=C_FLAT,
                markeredgecolor="#1F1F1F", markeredgewidth=1.6, zorder=4)
        ax.plot(x_bit, y, marker="o", markersize=22, markerfacecolor=C_BIT,
                markeredgecolor="#1F1F1F", markeredgewidth=1.6, zorder=4)

        for x_val, color, dy in [(x_bit, "#1F1F1F", -28),
                                  (x_flat, "#1F1F1F", -28)]:
            ax.annotate(f"{x_val:.3f}", (x_val, y),
                        xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=10.5, color=color, fontweight="bold")
        ax.annotate(f"{x_shuf:.3f}", (x_shuf, y),
                    xytext=(0, 18), textcoords="offset points",
                    ha="center", fontsize=10, color="#5C6066",
                    fontstyle="italic")

    ax.set_xlim(0.50, 0.86)
    ax.set_ylim(-0.85, len(rows) + 0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m["name"] for m in rows], fontweight="bold")
    ax.set_ylabel("ToMi-2 probe target", fontweight="bold")
    ax.set_xlabel("zero-shot probe macro-F1  (higher = better)", fontweight="bold")
    ax.set_title("ToMi-2 zero-shot transfer of SOTOPIA-trained variants", pad=12)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
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

    # Limitations note
    fig.text(0.5, 0.01,
             "Out-of-distribution: SOTOPIA-trained encoders applied zero-shot to ToMi-2 logical-state ToM. "
             "BIT specialization for in-domain mental supervision reduces zero-shot transfer.",
             ha="center", fontsize=9.5, color="#444", fontstyle="italic")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = Path(
        "projects/sotopia/experiments/runs/stage1/tomi_transfer/"
        "fig_tomi_transfer_dotplot.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
