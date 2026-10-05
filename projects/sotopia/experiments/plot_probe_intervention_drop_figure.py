#!/usr/bin/env python3
"""Paper figure: held-out probe collapse after scrambling z.

This figure uses intervention drops rather than raw probe scores:
  drop = score(original z) - score(sample-scrambled z)

The two panels keep metrics separate, so macro-F1 and R2 are never put on the
same numeric axis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_LIGHT_GREY = "#E8EAED"
PLOT_DARK_GREY = "#4B4B4B"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], *, task: str, model: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["model"] == model:
            return row
    raise ValueError(f"Missing task={task!r}, model={model!r}")


def style_axes(ax: plt.Axes, xlabel: str) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="x", color="#C7CBD1", linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", axis="x", color=PLOT_LIGHT_GREY, linewidth=0.5, alpha=0.85)
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.0, length=4)
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.5)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def write_combined_summary(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(args: argparse.Namespace) -> None:
    social_rows = read_rows(Path(args.macro_f1_summary))
    reward_rows = read_rows(Path(args.reward_r2_summary))

    models = [
        ("Compression VAE", "compression_vae_z", PLOT_GREEN),
        ("Mental recursive z", "mental_recursive_z", PLOT_BLUE),
    ]

    intent_values = []
    reward_values = []
    combined = []
    for label, model_key, color in models:
        intent = find_row(social_rows, task="intent_label", model=model_key)
        reward = find_row(reward_rows, task="reward_vec", model=model_key)
        intent_drop = float(intent["macro_f1_drop"])
        reward_drop = float(reward["r2_drop"])
        intent_values.append((label, intent_drop, color))
        reward_values.append((label, reward_drop, color))
        combined.append(
            {
                "probe": "intent_labels",
                "metric": "macro_f1_drop",
                "model": label,
                "original": float(intent["macro_f1_original"]),
                "scrambled": float(intent["macro_f1_z_scrambled"]),
                "drop": intent_drop,
            }
        )
        combined.append(
            {
                "probe": "reward_vector",
                "metric": "r2_drop",
                "model": label,
                "original": float(reward["r2_original"]),
                "scrambled": float(reward["r2_z_scrambled"]),
                "drop": reward_drop,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_combined_summary(out_dir / "probe_intervention_drop_summary.csv", combined)

    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.15), dpi=300, sharey=True)
    panels = [
        (axes[0], intent_values, "Intent labels", "macro-F1 drop", 0.12),
        (axes[1], reward_values, "Reward vector", "R2 drop", 0.70),
    ]
    y = np.arange(len(models))[::-1]
    for ax, values, title, xlabel, xmax in panels:
        vals = [value for _, value, _ in values]
        colors = [color for _, _, color in values]
        ax.barh(y, vals, height=0.46, color=colors, edgecolor=PLOT_DARK_GREY, linewidth=0.75)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)
        style_axes(ax, xlabel)

    axes[0].set_yticks(y, [label for label, _, _ in intent_values])
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)
    fig.suptitle("Probe Collapse Under z-Scrambling", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.26, top=0.78, wspace=0.32)

    out_path = out_dir / f"paper_probe_intervention_drop_{args.checkpoint}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="epoch2")
    parser.add_argument(
        "--macro-f1-summary",
        default="projects/sotopia/experiments/runs/visuals/task_signal_efficiency_checkpoint_sweep_episodeval1516/epoch2/macro_f1_intervention/macro_f1_intervention_summary.csv",
    )
    parser.add_argument(
        "--reward-r2-summary",
        default="projects/sotopia/experiments/runs/visuals/task_signal_efficiency_checkpoint_sweep_episodeval1516/epoch2/reward_r2_intervention/reward_r2_intervention_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="projects/sotopia/experiments/runs/visuals/task_signal_efficiency_checkpoint_sweep_episodeval1516/epoch2/intervention_drop_figure",
    )
    return parser.parse_args()


def main() -> None:
    make_figure(parse_args())


if __name__ == "__main__":
    main()
