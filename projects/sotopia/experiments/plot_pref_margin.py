#!/usr/bin/env python3
"""Render preference-margin distribution figure: BIT vs flat vs shuffled.

Single-panel KDE of the per-sample margin (pos - neg reward). The pref loss
= softplus(neg - pos), so the underlying random variable is the margin. Mean
accuracy is at ceiling (~99.9%) for all three variants, so the visual signal
lives in the *margin distribution*: how confidently pos beats neg per sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

VARIANT_EVAL = Path("projects/sotopia/experiments/runs/stage1/variant_eval/pref_margin")
VARIANTS = ["structured_bit", "flat_mental_summary", "shuffled_mental"]
PRETTY = {"structured_bit": "BIT (ours)",
          "flat_mental_summary": "flat summary",
          "shuffled_mental": "shuffled (control)"}

C_BIT = "#3A5F87"
C_FLAT = "#CC8A4C"
C_SHUFFLED = "#5DA39A"
COLORS = {"structured_bit": C_BIT,
          "flat_mental_summary": C_FLAT,
          "shuffled_mental": C_SHUFFLED}
LINESTYLES = {"structured_bit": "-",
              "flat_mental_summary": "-",
              "shuffled_mental": "--"}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "axes.titlesize": 21,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.6,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 240,
    })


def load_per_sample(variant: str) -> np.ndarray:
    arr = np.load(VARIANT_EVAL / f"margins_{variant}.npz")
    margins = arr["margins_per_dim"]   # (N, R)
    has_neg = arr["has_neg"]            # (N,)
    per_sample = margins.mean(axis=1)[has_neg > 0]
    return per_sample.astype(np.float64)


def main():
    set_style()
    data = {v: load_per_sample(v) for v in VARIANTS}

    fig, ax_kde = plt.subplots(figsize=(11.5, 6.2))

    # ── shared x range based on combined data ─────────────────────────────────
    all_vals = np.concatenate(list(data.values()))
    lo = float(np.percentile(all_vals, 0.5))
    hi = float(np.percentile(all_vals, 99.5))
    pad = 0.05 * (hi - lo)
    x_min, x_max = lo - pad, hi + pad
    grid = np.linspace(x_min, x_max, 500)

    # ── KDE density ───────────────────────────────────────────────────────────
    for v in VARIANTS:
        x = data[v]
        kde = gaussian_kde(x, bw_method=0.25)
        y = kde(grid)
        ax_kde.fill_between(grid, 0, y, color=COLORS[v], alpha=0.22, zorder=1)
        ax_kde.plot(grid, y, color=COLORS[v], linewidth=3.0,
                    linestyle=LINESTYLES[v], zorder=3, label=PRETTY[v])
        med = float(np.median(x))
        ax_kde.axvline(med, color=COLORS[v], linestyle=":", linewidth=1.8,
                       alpha=0.85, zorder=2)
    ax_kde.axvline(0, color="#444", linestyle="--", linewidth=1.2, alpha=0.7, zorder=0)
    ax_kde.set_xlim(x_min, x_max)
    ax_kde.set_xlabel("per-sample mean margin  (pos − neg reward)", fontweight="bold")
    ax_kde.set_ylabel("density", fontweight="bold")
    for s in ("top", "right"):
        ax_kde.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax_kde.spines[s].set_color("#222")
        ax_kde.spines[s].set_linewidth(1.4)
    ax_kde.grid(axis="x", color="#E5E8EB", linewidth=0.6, alpha=0.7, zorder=0)
    ax_kde.tick_params(axis="x", length=5)
    ax_kde.tick_params(axis="y", length=5)

    legend_lines = [
        Line2D([0], [0], color=COLORS[v], linewidth=3.0,
               linestyle=LINESTYLES[v],
               label=f"{PRETTY[v]}  (median = {np.median(data[v]):+.2f})")
        for v in VARIANTS
    ]
    leg = ax_kde.legend(
        handles=legend_lines, loc="upper right",
        bbox_to_anchor=(0.99, 0.99), frameon=True, framealpha=0.95,
        edgecolor="#888", fancybox=True,
    )
    for text in leg.get_texts():
        text.set_fontweight("bold")

    ax_kde.set_title(
        "Preference-margin distribution: BIT vs flat summary",
        pad=14,
    )

    fig.tight_layout()
    out_path = VARIANT_EVAL / "fig_pref_margin_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
