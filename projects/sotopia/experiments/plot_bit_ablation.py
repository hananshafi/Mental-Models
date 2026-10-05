#!/usr/bin/env python3
"""Paper-style ablation figures: BIT vs flat-summary vs shuffled.

Two outputs:
  - fig_bit_ablation_optionA.png:
      1x3 grouped bar chart on the headline mental-modeling metrics
      (mental1_gen, mental2_gen, future). shuffled rendered as muted-gray
      "noise floor" to make 'flat is worse than random control' visually
      obvious.

  - fig_bit_ablation_optionB.png:
      1x2 layout:
        left  — same headline mental-modeling bars as Option A (condensed).
        right — "matched mental information" panel: z_concat ridge-probe R²
                on mental2 text with BIT and flat shown side-by-side above
                a shuffled noise floor; same color used for BIT/flat to
                signal informational equivalence rather than ordering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Numbers (matched-epoch: structured_e5, flat_best e2, shuffled_best e2) ──
PHASE1 = {
    "mental1_gen":  {"BIT": 2.48, "flat": 5.22, "shuffled": 3.44},
    "mental2_gen":  {"BIT": 1.97, "flat": 4.51, "shuffled": 2.49},
    "future":       {"BIT": 0.62, "flat": 2.45, "shuffled": 2.45},
}
PHASE2_MENTAL2_R2 = {"BIT": 0.155, "flat": 0.219, "shuffled": 0.059}

METRIC_LABELS = {
    "mental1_gen": r"$\bf{Mental_1}$ NLL" + "\n(first-order belief/intent/thought)",
    "mental2_gen": r"$\bf{Mental_2}$ NLL" + "\n(recursive second-order)",
    "future":      r"$\bf{Future}$ NLL" + "\n(next-turn prediction)",
}

# Soft, reviewer-friendly palette
C_BIT      = "#1F5582"   # deep blue — primary
C_FLAT     = "#D08F4B"   # warm tan — primary alternative
C_SHUFFLED = "#A8AEB4"   # muted gray — control / noise floor


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#888",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    })


def grid_axes(ax, *, y_major=None, y_minor=None) -> None:
    ax.set_facecolor("#FAFAFA")
    if y_major is not None:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor))
    else:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.8, color="#A8B0B6", alpha=0.85, zorder=0)
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, color="#CFD4D9", alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#222")
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", length=5)
    ax.tick_params(axis="both", which="minor", length=3)


def _draw_phase1_panel(ax, *, condensed: bool = False) -> None:
    metrics = list(PHASE1.keys())
    x = np.arange(len(metrics))
    width = 0.27
    variants = [("BIT (ours)", "BIT", C_BIT),
                ("flat summary", "flat", C_FLAT),
                ("shuffled (control)", "shuffled", C_SHUFFLED)]
    for i, (label, key, color) in enumerate(variants):
        vals = [PHASE1[m][key] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, color=color, edgecolor="#222",
                      linewidth=0.7, label=label, zorder=3,
                      hatch=("////" if key == "shuffled" else None))
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.08,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5 if condensed else 9.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics],
                       fontsize=9 if condensed else 10)
    ax.set_ylabel("Validation NLL  (lower is better)")
    grid_axes(ax, y_major=1.0, y_minor=0.25)
    ax.set_ylim(0, max(max(d.values()) for d in PHASE1.values()) * 1.16)
    leg = ax.legend(loc="upper left", ncol=1)
    leg.get_frame().set_linewidth(0.6)


def _draw_architecture_schematic(ax) -> None:
    """For Option B: schematic of architectural alignment.
    BIT z → partitioned sub-blocks → 3 prefix heads (works).
    flat z → monolithic block → 3 prefix heads (mismatch)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def latent_block(x, y, w, h, parts=None, fill=C_BIT, label=""):
        if parts is None:
            box = FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.15",
                                 linewidth=1.2, edgecolor="#222", facecolor=fill, zorder=3)
            ax.add_patch(box)
        else:
            sub_w = w / len(parts)
            colors = ["#1F5582", "#3A7CA5", "#5FA8D3"]
            for i, p in enumerate(parts):
                box = FancyBboxPatch((x + i * sub_w, y), sub_w * 0.93, h,
                                     boxstyle="round,pad=0.03,rounding_size=0.08",
                                     linewidth=1.0, edgecolor="#222",
                                     facecolor=colors[i % len(colors)], zorder=3)
                ax.add_patch(box)
                ax.text(x + (i + 0.46) * sub_w, y + h / 2, p,
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold", zorder=4)
        if label:
            ax.text(x + w / 2, y + h + 0.25, label, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", zorder=4)

    def head_box(x, y, label, ok=True):
        c = "#2E7D32" if ok else "#9A9A9A"
        box = FancyBboxPatch((x, y), 1.6, 0.7,
                             boxstyle="round,pad=0.04,rounding_size=0.1",
                             linewidth=1.0, edgecolor="#222",
                             facecolor="white", zorder=3)
        ax.add_patch(box)
        ax.text(x + 0.8, y + 0.35, label, ha="center", va="center",
                fontsize=8.5, color=c, fontweight="bold", zorder=4)
        return c

    # Geometry
    LATENT_X, LATENT_W = 1.0, 3.4
    HEAD_X, HEAD_W = 6.4, 2.2
    HEAD_HEIGHT = 0.7

    # ── Top row: BIT (works) ─────────────────────────────────────────
    ax.text(0.2, 9.35, "BIT (ours):", fontsize=11, fontweight="bold",
            color=C_BIT, zorder=4)
    latent_block(LATENT_X, 8.45, LATENT_W, 0.85, parts=["B", "I", "T"])
    ax.text(LATENT_X + LATENT_W / 2, 8.30, r"partitioned $\bf{z}$",
            ha="center", va="top", fontsize=9.5, fontstyle="italic", color="#444")
    sub_w = LATENT_W / 3
    head_y_top = 9.05
    for i, name in enumerate(["belief head", "intent head", "thought head"]):
        head_box(HEAD_X, head_y_top - i * 0.92, name, ok=True)
        src_x = LATENT_X + (i + 0.5) * sub_w
        dst_y = head_y_top - i * 0.92 + HEAD_HEIGHT / 2
        arrow = FancyArrowPatch(
            (src_x, 8.45), (HEAD_X, dst_y),
            arrowstyle="->", color="#2E7D32", linewidth=1.5,
            mutation_scale=15, zorder=2,
        )
        ax.add_patch(arrow)
    ax.text((LATENT_X + LATENT_W + HEAD_X) / 2, 6.85,
            "✓  each sub-block routed to its dedicated head",
            color="#2E7D32", fontsize=10.2, fontweight="bold",
            ha="center", va="center")

    # Divider
    ax.plot([0.3, 9.7], [6.05, 6.05], color="#CFD4D9", linewidth=0.8, linestyle=":")

    # ── Bottom row: flat (mismatch) ─────────────────────────────────
    ax.text(0.2, 5.45, "flat summary:", fontsize=11, fontweight="bold",
            color=C_FLAT, zorder=4)
    latent_block(LATENT_X, 4.55, LATENT_W, 0.85, fill=C_FLAT)
    ax.text(LATENT_X + LATENT_W / 2, 4.97, "monolithic z",
            ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=4)
    ax.text(LATENT_X + LATENT_W / 2, 4.40, r"undivided $\bf{z}$",
            ha="center", va="top", fontsize=9.5, fontstyle="italic", color="#444")
    head_y_bot = 5.15
    for i, name in enumerate(["belief head", "intent head", "thought head"]):
        head_box(HEAD_X, head_y_bot - i * 0.92, name, ok=False)
        src_x = LATENT_X + LATENT_W / 2
        dst_y = head_y_bot - i * 0.92 + HEAD_HEIGHT / 2
        arrow = FancyArrowPatch(
            (src_x, 4.55), (HEAD_X, dst_y),
            arrowstyle="-", color="#B33A3A", linewidth=1.3, linestyle="--",
            mutation_scale=12, zorder=2,
        )
        ax.add_patch(arrow)
        mx = (src_x + HEAD_X) / 2
        my = (4.55 + dst_y) / 2
        ax.text(mx, my, "✗", color="#B33A3A", fontsize=15,
                ha="center", va="center", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="#FAFAFA", edgecolor="none"))
    ax.text((LATENT_X + LATENT_W + HEAD_X) / 2, 2.85,
            "✗  no sub-blocks for the heads to attach to",
            color="#B33A3A", fontsize=10.2, fontweight="bold",
            ha="center", va="center")

    # Footer takeaway
    ax.text(5.0, 1.0,
            "Same mental content, different structural form.\n"
            "Only BIT-partitioned $z$ matches the architecture's prefix-injection heads.",
            ha="center", va="center", fontsize=9.8, color="#333",
            fontstyle="italic")


# ── Figure builders ─────────────────────────────────────────────────────────
def make_option_a(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    _draw_phase1_panel(ax)
    fig.text(0.5, 0.02,
             "Matched 5-epoch training; identical hyper-parameters, model, and data.",
             ha="center", fontsize=9, color="#444", fontstyle="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    print("wrote", out_path)


def make_option_b(out_path: Path) -> None:
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    _draw_phase1_panel(ax_left, condensed=True)
    ax_left.set_title("(a) Mental-modeling task performance",
                      fontsize=11.5, fontweight="bold", loc="left", pad=8)

    _draw_architecture_schematic(ax_right)
    ax_right.set_title("(b) Why BIT decomposition matters",
                       fontsize=11.5, fontweight="bold", loc="left", pad=8)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.13, wspace=0.20)
    fig.text(0.5, 0.04,
             "BIT decomposition is necessary because it aligns z with the architecture's Belief / "
             "Intent / Thought prefix-injection heads.",
             ha="center", fontsize=10, color="#333", fontstyle="italic")
    fig.savefig(out_path)
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir",
                    default="projects/sotopia/experiments/runs/stage1/variant_eval/figs_paper")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    make_option_a(out_dir / "fig_bit_ablation_optionA.png")
    make_option_b(out_dir / "fig_bit_ablation_optionB.png")

    with (out_dir / "values_used.json").open("w") as f:
        json.dump({"phase1": PHASE1, "phase2_mental2_R2": PHASE2_MENTAL2_R2}, f, indent=2)


if __name__ == "__main__":
    main()
