#!/usr/bin/env python3
"""Plot latent scrambling: BIT mental z vs compression-only z."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BIT_PATH = Path(
    "projects/sotopia/experiments/runs/stage1/"
    "latent_scramble_intervention_epoch2_fullval/summary.json"
)
COMPRESSION_PATH = Path(
    "projects/sotopia/experiments/runs/stage1/"
    "latent_scramble_intervention_compression_fullval/summary.json"
)
OUT_DIR = Path(
    "projects/sotopia/experiments/runs/stage1/"
    "latent_scramble_bit_vs_compression"
)

COLORS = {"BIT": "#2F6FAE", "Compression VAE": "#2CA25F"}


def load_values() -> dict[str, dict[str, float]]:
    bit = json.loads(BIT_PATH.read_text())["sample_shuffle_pair"]
    comp = json.loads(COMPRESSION_PATH.read_text())["sample_shuffle"]
    bit_decoder = 0.5 * (
        bit["mental1_nll_ratio_to_baseline"] + bit["mental2_nll_ratio_to_baseline"]
    )
    return {
        "BIT": {
            "decoder": bit_decoder,
            "reward": bit["joint_reward_mse_ratio_to_baseline"],
            "mental1": bit["mental1_nll_ratio_to_baseline"],
            "mental2": bit["mental2_nll_ratio_to_baseline"],
            "z_only_reward": bit["z_combined_mse_ratio_to_baseline"],
        },
        "Compression VAE": {
            "decoder": comp["recon_nll_ratio_to_baseline"],
            "reward": comp["reward_probe_mse_ratio_to_baseline"],
        },
    }


def set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update({
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
    })


def write_summary(values: dict[str, dict[str, float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "BIT",
            "metric": "own_decoder",
            "description": "average mental1/mental2 NLL increase after sample-shuffling z",
            "degradation_factor": values["BIT"]["decoder"],
        },
        {
            "model": "BIT",
            "metric": "joint_reward",
            "description": "joint reward MSE increase after sample-shuffling z",
            "degradation_factor": values["BIT"]["reward"],
        },
        {
            "model": "Compression VAE",
            "metric": "own_decoder",
            "description": "summary reconstruction NLL increase after sample-shuffling z",
            "degradation_factor": values["Compression VAE"]["decoder"],
        },
        {
            "model": "Compression VAE",
            "metric": "reward_probe",
            "description": "reward probe MSE increase after sample-shuffling z",
            "degradation_factor": values["Compression VAE"]["reward"],
        },
    ]
    with (OUT_DIR / "bit_vs_compression_latent_scramble.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "bit_vs_compression_latent_scramble.json").open("w") as f:
        json.dump(values, f, indent=2, sort_keys=True)

    ratio = values["BIT"]["reward"] / max(values["Compression VAE"]["reward"], 1e-12)
    lines = [
        "# BIT vs Compression VAE latent scramble",
        "",
        "Sample-shuffled z degradation factor relative to original z.",
        "",
        "| model | own decoder | reward head |",
        "|---|---:|---:|",
        f"| BIT mental z | {values['BIT']['decoder']:.2f} | {values['BIT']['reward']:.2f} |",
        f"| Compression z | {values['Compression VAE']['decoder']:.2f} | {values['Compression VAE']['reward']:.2f} |",
        "",
        f"Reward-head degradation is {ratio:.2f}x larger for BIT than compression z.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def plot(values: dict[str, dict[str, float]]) -> None:
    set_style()
    metric_labels = ["own decoder", "reward head"]
    models = ["BIT", "Compression VAE"]
    data = np.array([
        [values["BIT"]["decoder"], values["BIT"]["reward"]],
        [values["Compression VAE"]["decoder"], values["Compression VAE"]["reward"]],
    ])
    x = np.arange(len(metric_labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(5.3, 2.65), dpi=300)
    for i, model in enumerate(models):
        ax.bar(
            x + (i - 0.5) * width,
            data[i],
            width=width,
            color=COLORS[model],
            edgecolor="#2F2F2F",
            linewidth=0.8,
            label=model,
        )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.85, 10.5)
    ax.set_yticks([1, 1.25, 2, 4, 8])
    ax.set_yticklabels(["1", "1.25", "2", "4", "8"])
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("error increase after z shuffle", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.9)
    ax.grid(axis="x", visible=False)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.legend(loc="upper left", frameon=True, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.22, top=0.96)
    fig.savefig(OUT_DIR / "fig_latent_scramble_bit_vs_compression.png", pad_inches=0.02)
    fig.savefig(OUT_DIR / "fig_latent_scramble_bit_vs_compression.pdf", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    values = load_values()
    write_summary(values)
    plot(values)
    print(f"Wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
