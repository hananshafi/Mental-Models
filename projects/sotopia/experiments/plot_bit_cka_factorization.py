#!/usr/bin/env python3
"""Plot BIT sub-block CKA as factorization maps.

This figure replaces a grouped bar chart with a representation that matches
the quantity being measured: pairwise CKA between belief/intent/thought slices.
Off-diagonal cells are the inter-sub-block entanglement values; lower values
mean the slices are more factorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


VARIANT_ORDER = ["structured_bit", "flat_mental_summary", "shuffled_mental"]
VARIANT_LABELS = {
    "structured_bit": "BIT (ours)",
    "flat_mental_summary": "Flat summary",
    "shuffled_mental": "Shuffled control",
}
PAIR_KEYS = {
    (0, 1): "belief-intent",
    (0, 2): "belief-thought",
    (1, 2): "intent-thought",
}
SLICE_LABELS = ["Belief", "Intent", "Thought"]

PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_ORANGE = "#CC8A4C"
PLOT_RED = "#C84C4C"
PLOT_DARK = "#4B4B4B"
PLOT_GRID = "#C7CBD1"
PLOT_LIGHT = "#E8EAED"


def load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def cka_matrix(cka_row: dict[str, float]) -> np.ndarray:
    mat = np.full((3, 3), np.nan, dtype=np.float32)
    for (i, j), key in PAIR_KEYS.items():
        mat[i, j] = float(cka_row[key])
        mat[j, i] = float(cka_row[key])
    return mat


def style_ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=8.5, width=1.0, length=3)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK)
        spine.set_linewidth(0.9)


def plot_factorization(summary: dict, out_path: Path) -> None:
    cka = summary["strategy_8_cka"]
    variants = [v for v in VARIANT_ORDER if v in cka]
    means = {v: float(cka[v]["mean_off_diag"]) for v in variants}
    off_values = [
        float(cka[v][key])
        for v in variants
        for key in ["belief-intent", "belief-thought", "intent-thought"]
    ]
    vmin = max(0.0, min(off_values) - 0.03)
    vmax = min(1.0, max(off_values) + 0.03)
    cmap = LinearSegmentedColormap.from_list(
        "cka_factorization",
        ["#F7FBFF", "#9ECAE1", "#FEE08B", "#D73027"],
    )
    cmap.set_bad("#F2F2F2")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": PLOT_DARK,
        "axes.linewidth": 0.9,
    })

    fig = plt.figure(figsize=(7.25, 3.0), dpi=300)
    gs = fig.add_gridspec(1, len(variants) + 1, width_ratios=[1, 1, 1, 1.05], wspace=0.30)
    matrix_axes = [fig.add_subplot(gs[0, i]) for i in range(len(variants))]
    score_ax = fig.add_subplot(gs[0, len(variants)])

    im = None
    for ax, variant in zip(matrix_axes, variants):
        mat = np.ma.masked_invalid(cka_matrix(cka[variant]))
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(3), SLICE_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(3), SLICE_LABELS if ax is matrix_axes[0] else ["", "", ""])
        ax.set_title(f"{VARIANT_LABELS.get(variant, variant)}\nmean CKA={means[variant]:.2f}", fontsize=8.8)
        style_ticks(ax)

        for i in range(3):
            for j in range(3):
                if i == j:
                    ax.text(j, i, "self", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#777777")
                else:
                    value = float(mat[i, j])
                    color = "white" if value > (vmin + 0.68 * (vmax - vmin)) else "#1A1A1A"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.0, fontweight="bold", color=color)

        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)

    sorted_variants = sorted(variants, key=lambda v: means[v])
    y = np.arange(len(sorted_variants))[::-1]
    colors = {
        "structured_bit": PLOT_BLUE,
        "flat_mental_summary": PLOT_ORANGE,
        "shuffled_mental": PLOT_GREEN,
    }
    vals = [means[v] for v in sorted_variants]
    score_ax.hlines(y, 0.35, vals, color=PLOT_LIGHT, linewidth=7.5, zorder=1)
    score_ax.scatter(vals, y, s=135, color=[colors.get(v, "#888888") for v in sorted_variants],
                     edgecolor=PLOT_DARK, linewidth=0.8, zorder=3)
    for yi, value in zip(y, vals):
        score_ax.text(value + 0.012, yi, f"{value:.2f}", va="center", ha="left",
                      fontsize=8.8, fontweight="bold", color=PLOT_DARK)
    score_ax.set_yticks(y, [VARIANT_LABELS.get(v, v) for v in sorted_variants])
    score_ax.set_xlim(0.34, 0.72)
    score_ax.set_xlabel("mean off-diagonal CKA", fontsize=8.8, fontweight="bold")
    score_ax.set_title("Entanglement\n(lower better)", fontsize=8.8, fontweight="bold")
    score_ax.grid(True, axis="x", color=PLOT_GRID, linewidth=0.8, alpha=0.8)
    score_ax.set_axisbelow(True)
    score_ax.tick_params(axis="both", labelsize=8.5, width=1.0, length=3)
    for label in [*score_ax.get_xticklabels(), *score_ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in score_ax.spines.values():
        spine.set_color(PLOT_DARK)
        spine.set_linewidth(0.9)
    fig.suptitle("Belief-Intent-Thought sub-block factorization",
                 fontsize=11.8, fontweight="bold", y=0.985)
    fig.text(
        0.40,
        0.825,
        "Off-diagonal cells show pairwise CKA; lower values indicate less entanglement.",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=PLOT_DARK,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.24, top=0.70)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def interp_color(cmap: LinearSegmentedColormap, value: float, vmin: float, vmax: float):
    t = (value - vmin) / max(vmax - vmin, 1e-8)
    return cmap(float(np.clip(t, 0.0, 1.0)))


def plot_triangle_factorization(summary: dict, out_path: Path) -> None:
    cka = summary["strategy_8_cka"]
    variants = [v for v in VARIANT_ORDER if v in cka]
    off_values = [
        float(cka[v][key])
        for v in variants
        for key in ["belief-intent", "belief-thought", "intent-thought"]
    ]
    vmin = max(0.0, min(off_values) - 0.03)
    vmax = min(1.0, max(off_values) + 0.03)
    cmap = LinearSegmentedColormap.from_list(
        "cka_edges",
        ["#9ECAE1", "#FEE08B", "#D73027"],
    )
    positions = {
        "belief": np.array([0.50, 0.86]),
        "intent": np.array([0.18, 0.20]),
        "thought": np.array([0.82, 0.20]),
    }
    node_labels = {"belief": "B", "intent": "I", "thought": "T"}
    edge_defs = [
        ("belief", "intent", "belief-intent"),
        ("belief", "thought", "belief-thought"),
        ("intent", "thought", "intent-thought"),
    ]
    label_offsets = {
        "belief-intent": np.array([-0.05, 0.00]),
        "belief-thought": np.array([0.05, 0.00]),
        "intent-thought": np.array([0.00, -0.07]),
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, len(variants), figsize=(6.25, 2.35), dpi=300)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for a, b, key in edge_defs:
            value = float(cka[variant][key])
            p1 = positions[a]
            p2 = positions[b]
            t = (value - vmin) / max(vmax - vmin, 1e-8)
            width = 2.3 + 6.0 * float(np.clip(t, 0.0, 1.0))
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=interp_color(cmap, value, vmin, vmax),
                linewidth=width,
                solid_capstyle="round",
                alpha=0.92,
                zorder=1,
            )
            midpoint = 0.5 * (p1 + p2) + label_offsets[key]
            ax.text(
                midpoint[0],
                midpoint[1],
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold",
                color="#1A1A1A",
                bbox=dict(facecolor="white", edgecolor="#D6D8DC", alpha=0.88, pad=1.5),
                zorder=4,
            )
        for name, pos in positions.items():
            ax.scatter(
                [pos[0]],
                [pos[1]],
                s=360,
                color="white",
                edgecolor=PLOT_DARK,
                linewidth=1.4,
                zorder=5,
            )
            ax.text(
                pos[0],
                pos[1],
                node_labels[name],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=PLOT_DARK,
                zorder=6,
            )
        mean_value = float(cka[variant]["mean_off_diag"])
        ax.set_title(
            f"{VARIANT_LABELS.get(variant, variant)}\nmean CKA={mean_value:.2f}",
            fontsize=9.5,
            fontweight="bold",
            pad=6,
        )

    fig.suptitle(
        "Sub-block CKA factorization",
        fontsize=11.0,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.025,
        "Edge value/thickness = pairwise CKA; lower and thinner means more factorized.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=PLOT_DARK,
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.18, top=0.72, wspace=0.26)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def draw_curved_edge(
    ax: plt.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    rad: float,
    width: float,
    color,
    alpha: float,
) -> None:
    shadow = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=width + 2.0,
        color="#D7DCE2",
        alpha=0.55,
        capstyle="round",
        joinstyle="round",
        zorder=1,
    )
    edge = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=width,
        color=color,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_patch(shadow)
    ax.add_patch(edge)


def plot_entanglement_map(summary: dict, out_path: Path) -> None:
    """Clean paper-facing map: no numbers on edges, visual edge strength only."""
    cka = summary["strategy_8_cka"]
    variants = [v for v in VARIANT_ORDER if v in cka]
    off_values = [
        float(cka[v][key])
        for v in variants
        for key in ["belief-intent", "belief-thought", "intent-thought"]
    ]
    vmin = max(0.0, min(off_values) - 0.03)
    vmax = min(1.0, max(off_values) + 0.03)
    cmap = LinearSegmentedColormap.from_list("cka_polished", ["#8EC9DF", "#F3D26A", "#E4573F"])
    positions = {
        "belief": np.array([0.50, 0.78]),
        "intent": np.array([0.21, 0.26]),
        "thought": np.array([0.79, 0.26]),
    }
    edge_defs = [
        ("belief", "intent", "belief-intent", 0.08),
        ("belief", "thought", "belief-thought", -0.08),
        ("intent", "thought", "intent-thought", 0.00),
    ]
    node_names = {"belief": "B", "intent": "I", "thought": "T"}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, len(variants), figsize=(5.45, 1.85), dpi=300)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for a, b, key, rad in edge_defs:
            value = float(cka[variant][key])
            t = (value - vmin) / max(vmax - vmin, 1e-8)
            t = float(np.clip(t, 0.0, 1.0))
            draw_curved_edge(
                ax,
                positions[a],
                positions[b],
                rad=rad,
                width=3.0 + 7.8 * t,
                color=cmap(t),
                alpha=0.58 + 0.34 * t,
            )

        for name, pos in positions.items():
            ax.scatter(
                [pos[0]],
                [pos[1]],
                s=390,
                color="white",
                edgecolor=PLOT_DARK,
                linewidth=1.5,
                zorder=5,
            )
            ax.text(
                pos[0],
                pos[1],
                node_names[name],
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=PLOT_DARK,
                zorder=6,
            )

        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=9.0, fontweight="bold", pad=2)

    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.03, top=0.82, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def plot_seaborn_style_heatmap(summary: dict, out_path: Path) -> None:
    """Compact seaborn-style heatmap of pairwise CKA values."""
    cka = summary["strategy_8_cka"]
    variants = [v for v in VARIANT_ORDER if v in cka]
    columns = [
        ("B-I", "belief-intent"),
        ("B-T", "belief-thought"),
        ("I-T", "intent-thought"),
        ("Mean", "mean_off_diag"),
    ]
    data = np.asarray(
        [[float(cka[v][key]) for _, key in columns] for v in variants],
        dtype=np.float32,
    )
    short_row_labels = {
        "structured_bit": "BIT",
        "flat_mental_summary": "Flat",
        "shuffled_mental": "Shuffled",
    }
    row_labels = [short_row_labels.get(v, VARIANT_LABELS.get(v, v)) for v in variants]
    col_labels = [label for label, _ in columns]
    vmin = float(np.min(data[:, :3]) - 0.03)
    vmax = float(np.max(data[:, :3]) + 0.03)

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#B8B8B8",
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
    })
    cmap = LinearSegmentedColormap.from_list(
        "seaborn_like_cka",
        ["#1F4E8C", "#D6E6F2", "#F7F3EA", "#E6A07C", "#B2182B"],
    )

    fig = plt.figure(figsize=(3.75, 2.05), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.10], hspace=0.36)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        resample=False,
    )

    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="both", width=0.7, length=0, colors="#2B2B2B")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")

    # Subtle divider before the mean column.
    ax.axvline(2.5, color="#696969", linewidth=0.9, alpha=0.45)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Inter-sub-block CKA", fontweight="bold", pad=6)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7.4, length=0, pad=1.5, colors="#2B2B2B")
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight("bold")
    cbar.set_label("entanglement level", fontsize=7.8, fontweight="bold", labelpad=2, color="#2B2B2B")

    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.23, top=0.86)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        default="projects/sotopia/experiments/runs/stage1/bit_structure/summary.json",
    )
    parser.add_argument(
        "--output",
        default="projects/sotopia/experiments/runs/stage1/bit_structure/fig_strategy8_cka_factorization_map.png",
    )
    parser.add_argument(
        "--triangle-output",
        default="projects/sotopia/experiments/runs/stage1/bit_structure/fig_strategy8_cka_triangle_map.png",
    )
    parser.add_argument(
        "--entanglement-output",
        default="projects/sotopia/experiments/runs/stage1/bit_structure/fig_strategy8_cka_entanglement_map.png",
    )
    parser.add_argument(
        "--heatmap-output",
        default="projects/sotopia/experiments/runs/stage1/bit_structure/fig_strategy8_cka_seaborn_heatmap.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(Path(args.summary_json))
    plot_factorization(summary, Path(args.output))
    plot_triangle_factorization(summary, Path(args.triangle_output))
    plot_entanglement_map(summary, Path(args.entanglement_output))
    plot_seaborn_style_heatmap(summary, Path(args.heatmap_output))
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.triangle_output}", flush=True)
    print(f"Wrote {args.entanglement_output}", flush=True)
    print(f"Wrote {args.heatmap_output}", flush=True)


if __name__ == "__main__":
    main()
