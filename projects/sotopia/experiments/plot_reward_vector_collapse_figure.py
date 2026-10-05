#!/usr/bin/env python3
"""Paper-style reward-vector collapse figure.

Uses the same visual language as the probe accessibility plot, but the y-axis
rows are latent models and the x-axis is the intervention effect:

    reward-head error increase = MSE(sample-scrambled z) - MSE(original z)

This is the anti-bypass diagnostic: does each model's own reward head actually
depend on its latent z?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_LIGHT_GREY = "#E8EAED"
PLOT_DARK_GREY = "#4B4B4B"


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C7CBD1", linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color=PLOT_LIGHT_GREY, linewidth=0.5, alpha=0.85)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.0, length=4)
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.5)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def make_figure(args: argparse.Namespace) -> None:
    mental_summary = read_json(Path(args.mental_summary_json))
    compression_summary = read_json(Path(args.compression_summary_json))
    mental_value = float(mental_summary["sample_shuffle_pair"]["joint_reward_mse_delta"])
    compression_value = float(compression_summary["sample_shuffle"]["reward_probe_mse_delta"])
    models = [
        ("Compression VAE", compression_value, PLOT_GREEN, 1.0),
        ("Mental-model z", mental_value, PLOT_BLUE, 0.0),
    ]

    fig, ax = plt.subplots(figsize=(4.15, 2.25), dpi=300)
    start_x = 0.0
    for label, value, color, y in models:
        ax.plot(
            [start_x, value],
            [y, y],
            color="#BFC5CC",
            linewidth=7.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.annotate(
            "",
            xy=(max(value - 0.002, 0.0), y),
            xytext=(start_x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.65, shrinkA=0, shrinkB=0),
            zorder=2,
        )
        ax.scatter(
            value,
            y,
            s=135,
            color=color,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.8,
            zorder=3,
        )

    ax.set_yticks([1.0, 0.0], ["Compression\nVAE", "Mental-model\nz"])
    ax.tick_params(axis="y", which="major", pad=8)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(-0.005, 0.090)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08], ["0.00", "0.02", "0.04", "0.06", "0.08"])
    ax.set_xlabel("reward-head MSE increase", fontsize=10, fontweight="bold")
    ax.set_title("Reward Vector Collapse Under z-Scrambling", fontsize=11, fontweight="bold", pad=6)
    style_axes(ax)

    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.25, top=0.84)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"paper_reward_vector_collapse_{args.checkpoint}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="epoch2")
    parser.add_argument(
        "--mental-summary-json",
        default="projects/sotopia/experiments/runs/stage1/latent_scramble_intervention_epoch2_fullval/summary.json",
    )
    parser.add_argument(
        "--compression-summary-json",
        default="projects/sotopia/experiments/runs/stage1/latent_scramble_intervention_compression_fullval/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="projects/sotopia/experiments/runs/visuals/task_signal_efficiency_checkpoint_sweep_episodeval1516/epoch2/reward_vector_collapse",
    )
    return parser.parse_args()


def main() -> None:
    make_figure(parse_args())


if __name__ == "__main__":
    main()
