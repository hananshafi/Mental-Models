#!/usr/bin/env python3
"""Paper figure for task-signal accessibility vs compression VAE.

The figure is designed for the "not just compression" argument. It does not
claim that the mental latent stores more total information; it shows that task
signals are more accessible from the mental z space under matched probes.
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
PLOT_LIGHT_GREEN = "#A8DDB5"
PLOT_GREY = "#7A7A7A"
PLOT_LIGHT_GREY = "#E8EAED"
PLOT_DARK_GREY = "#4B4B4B"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def find_checkpoint(rows: list[dict[str, str]], checkpoint: str) -> dict[str, float]:
    for row in rows:
        if row["checkpoint"] == checkpoint:
            return {key: float(value) for key, value in row.items() if key != "checkpoint"}
    raise ValueError(f"Checkpoint {checkpoint!r} not found in CSV")


def style_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C7CBD1", linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color=PLOT_LIGHT_GREY, linewidth=0.5, alpha=0.85)
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.0, length=4)
    ax.tick_params(axis="both", which="minor", width=0.8, length=2.5)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def make_signal_figure(row: dict[str, float], checkpoint: str, out_path: Path) -> None:
    tasks = [
        {
            "label": "Social\nfull z",
            "mental": row["social_macro_f1_full_mental_recursive"],
            "compression": row["social_macro_f1_full_compression_vae"],
            "metric": "macro F1",
        },
        {
            "label": "Social\n32D",
            "mental": row["social_macro_f1_32_mental_recursive"],
            "compression": row["social_macro_f1_32_compression_vae"],
            "metric": "macro F1",
        },
        {
            "label": "Reward\n32D",
            "mental": row["reward_r2_32_mental_recursive"],
            "compression": row["reward_r2_32_compression_vae"],
            "metric": "R2",
        },
    ]
    labels = [task["label"] for task in tasks]
    mental = np.asarray([task["mental"] for task in tasks], dtype=np.float32)
    compression = np.asarray([task["compression"] for task in tasks], dtype=np.float32)
    gaps = mental - compression
    lift = 100.0 * gaps / np.maximum(np.abs(compression), 1e-8)

    fig = plt.figure(figsize=(7.2, 3.15), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.95], wspace=0.28)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_lift = fig.add_subplot(gs[0, 1])

    y = np.arange(len(tasks))[::-1]
    for idx, yi in enumerate(y):
        ax_score.plot(
            [compression[idx], mental[idx]],
            [yi, yi],
            color="#BFC5CC",
            linewidth=8.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax_score.scatter(
            compression[idx],
            yi,
            s=150,
            color=PLOT_GREEN,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.8,
            zorder=3,
            label="Compression VAE" if idx == 0 else None,
        )
        ax_score.scatter(
            mental[idx],
            yi,
            s=150,
            color=PLOT_BLUE,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.8,
            zorder=4,
            label="Mental recursive z" if idx == 0 else None,
        )
        ax_score.annotate(
            "",
            xy=(mental[idx], yi),
            xytext=(compression[idx], yi),
            arrowprops=dict(arrowstyle="-|>", color=PLOT_BLUE, lw=1.7, shrinkA=12, shrinkB=12),
            zorder=2,
        )

    ax_score.set_yticks(y, labels)
    ax_score.set_xlim(0.09, 0.235)
    style_axes(ax_score, xlabel="heldout probe score", ylabel="task signal")
    ax_score.set_ylim(-0.25, len(tasks) - 0.45)
    legend = ax_score.legend(
        loc="upper left",
        frameon=True,
        fontsize=8,
        handlelength=1.2,
        borderpad=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D8DC")
    ax_score.set_title("Task Signal Accessibility", fontsize=11, fontweight="bold", pad=6)

    colors = [PLOT_BLUE, PLOT_BLUE, PLOT_BLUE]
    ax_lift.barh(y, lift, color=colors, edgecolor=PLOT_DARK_GREY, linewidth=0.55)
    ax_lift.axvline(0, color=PLOT_DARK_GREY, linewidth=1.0)
    ax_lift.set_yticks(y, ["", "", ""])
    ax_lift.set_ylim(-0.25, len(tasks) - 0.45)
    max_lift = max(110.0, float(np.max(lift)) * 1.18)
    ax_lift.set_xlim(0, max_lift)
    style_axes(ax_lift, xlabel="relative lift (%)", ylabel=None)
    ax_lift.set_title("Mental z Lift", fontsize=11, fontweight="bold", pad=6)

    for yi, value in zip(y, lift):
        ax_lift.text(
            value + max_lift * 0.025,
            yi,
            f"+{value:.0f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=PLOT_DARK_GREY,
        )

    fig.suptitle(
        f"Matched-Latent Task Probes ({checkpoint})",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.17, top=0.76, wspace=0.34)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_accessibility_only_figure(row: dict[str, float], checkpoint: str, out_path: Path) -> None:
    tasks = [
        {
            "label": "Social\nlabels",
            "mental": row["social_macro_f1_full_mental_recursive"],
            "compression": row["social_macro_f1_full_compression_vae"],
        },
        {
            "label": "Reward\nvector",
            "mental": row["reward_r2_32_mental_recursive"],
            "compression": row["reward_r2_32_compression_vae"],
        },
    ]
    labels = [task["label"] for task in tasks]
    mental = np.asarray([task["mental"] for task in tasks], dtype=np.float32)
    compression = np.asarray([task["compression"] for task in tasks], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.15, 2.55), dpi=300)
    y = np.arange(len(tasks))[::-1]
    for idx, yi in enumerate(y):
        ax.plot(
            [compression[idx], mental[idx]],
            [yi, yi],
            color="#BFC5CC",
            linewidth=7.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            compression[idx],
            yi,
            s=130,
            color=PLOT_GREEN,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.75,
            zorder=3,
            label="Compression VAE" if idx == 0 else None,
        )
        ax.scatter(
            mental[idx],
            yi,
            s=130,
            color=PLOT_BLUE,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.75,
            zorder=4,
            label="Mental recursive z" if idx == 0 else None,
        )
        ax.annotate(
            "",
            xy=(mental[idx], yi),
            xytext=(compression[idx], yi),
            arrowprops=dict(arrowstyle="-|>", color=PLOT_BLUE, lw=1.55, shrinkA=11, shrinkB=11),
            zorder=2,
        )

    ax.set_yticks(y, labels, rotation=90, ha="center", va="center", rotation_mode="anchor")
    ax.tick_params(axis="y", which="major", pad=12)
    ax.set_ylim(-0.35, len(tasks) - 0.65)
    ax.set_xlim(0.09, 0.235)
    style_axes(ax, xlabel="heldout probe score (macro-F1 / R2)", ylabel="probe signal")
    ax.set_ylabel("probe signal", fontsize=10, fontweight="bold", labelpad=18)
    ax.set_title("Probe Signal Accessibility", fontsize=11, fontweight="bold", pad=6)
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fontsize=8.2,
        handlelength=1.0,
        borderpad=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D8DC")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.70)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def make_reward_head_only_figure(row: dict[str, float], checkpoint: str, out_path: Path) -> None:
    mental = float(row["reward_r2_32_mental_recursive"])
    compression = float(row["reward_r2_32_compression_vae"])

    fig, ax = plt.subplots(figsize=(4.15, 2.05), dpi=300)
    start_x = 0.098
    lanes = [
        ("Compression VAE", compression, PLOT_GREEN, 0.11),
        ("Mental recursive z", mental, PLOT_BLUE, -0.11),
    ]
    for label, score, color, y in lanes:
        ax.plot(
            [start_x, score],
            [y, y],
            color="#BFC5CC",
            linewidth=6.2,
            solid_capstyle="round",
            zorder=1,
        )
        ax.annotate(
            "",
            xy=(score - 0.004, y),
            xytext=(start_x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.55, shrinkA=0, shrinkB=0),
            zorder=2,
        )
        ax.scatter(
            score,
            y,
            s=130,
            color=color,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.75,
            zorder=3,
            label=label,
        )

    ax.set_yticks([0.0], ["Reward\nhead"], rotation=90, ha="center", va="center", rotation_mode="anchor")
    ax.tick_params(axis="y", which="major", pad=12)
    ax.set_ylim(-0.38, 0.72)
    ax.set_xlim(0.09, 0.235)
    style_axes(ax, xlabel="heldout reward-probe score (R2)", ylabel="probe signal")
    ax.set_ylabel("probe signal", fontsize=10, fontweight="bold", labelpad=18)
    ax.set_title("Probe Signal Accessibility", fontsize=11, fontweight="bold", pad=6)
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fontsize=8.2,
        handlelength=1.0,
        borderpad=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D8DC")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.27, top=0.83)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def aggregate_task_probe(
    aggregate_rows: list[dict[str, str]],
    *,
    task: str,
    dim: str,
    model: str,
) -> float | None:
    values = [
        float(row["macro_f1_mean"])
        for row in aggregate_rows
        if row["task"] == task and str(row["dim"]) == dim and row["model"] == model
    ]
    if not values:
        return None
    return float(np.mean(values))


def make_reward_plus_intent_figure(row: dict[str, float], checkpoint: str, output_dir: Path) -> None:
    aggregate_path = output_dir / checkpoint / "task_signal_classification_aggregate.csv"
    if aggregate_path.exists():
        aggregate_rows = read_rows(aggregate_path)
        intent_compression = aggregate_task_probe(
            aggregate_rows,
            task="intent_label",
            dim="full",
            model="compression_vae",
        )
        intent_mental = aggregate_task_probe(
            aggregate_rows,
            task="intent_label",
            dim="full",
            model="mental_recursive",
        )
    else:
        intent_compression = None
        intent_mental = None

    tasks = [
        {
            "label": "Intent\nlabels",
            "compression": intent_compression
            if intent_compression is not None
            else float(row["social_macro_f1_full_compression_vae"]),
            "mental": intent_mental if intent_mental is not None else float(row["social_macro_f1_full_mental_recursive"]),
        },
        {
            "label": "Reward\nhead",
            "compression": float(row["reward_r2_32_compression_vae"]),
            "mental": float(row["reward_r2_32_mental_recursive"]),
        },
    ]

    fig, ax = plt.subplots(figsize=(4.15, 2.55), dpi=300)
    start_x = 0.098
    y_centers = np.arange(len(tasks))[::-1].astype(float)
    lane_offsets = {"Compression VAE": 0.11, "Mental recursive z": -0.11}
    colors = {"Compression VAE": PLOT_GREEN, "Mental recursive z": PLOT_BLUE}

    for idx, (task, y_center) in enumerate(zip(tasks, y_centers)):
        for model_label, score_key in [("Compression VAE", "compression"), ("Mental recursive z", "mental")]:
            y = y_center + lane_offsets[model_label]
            score = float(task[score_key])
            color = colors[model_label]
            ax.plot(
                [start_x, score],
                [y, y],
                color="#BFC5CC",
                linewidth=5.7,
                solid_capstyle="round",
                zorder=1,
            )
            ax.annotate(
                "",
                xy=(score - 0.004, y),
                xytext=(start_x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.45, shrinkA=0, shrinkB=0),
                zorder=2,
            )
            ax.scatter(
                score,
                y,
                s=115,
                color=color,
                edgecolor=PLOT_DARK_GREY,
                linewidth=0.7,
                zorder=3,
                label=model_label if idx == 0 else None,
            )

    ax.set_yticks(
        y_centers,
        [str(task["label"]) for task in tasks],
        rotation=90,
        ha="center",
        va="center",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="y", which="major", pad=12)
    ax.set_ylim(-0.45, len(tasks) + 0.05)
    ax.set_xlim(0.09, 0.235)
    style_axes(ax, xlabel="heldout probe score (macro-F1 / R2)", ylabel="probe signal")
    ax.set_ylabel("probe signal", fontsize=10, fontweight="bold", labelpad=18)
    ax.set_title("Probe Signal Accessibility", fontsize=11, fontweight="bold", pad=6)
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=8.2,
        handlelength=1.0,
        borderpad=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D8DC")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.85)
    out_path = output_dir / f"paper_reward_plus_intent_accessibility_{checkpoint}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def make_minimal_lift_strip(row: dict[str, float], checkpoint: str, out_path: Path) -> None:
    tasks = [
        ("Social full", row["social_macro_f1_full_mental_recursive"], row["social_macro_f1_full_compression_vae"]),
        ("Social 32D", row["social_macro_f1_32_mental_recursive"], row["social_macro_f1_32_compression_vae"]),
        ("Reward 32D", row["reward_r2_32_mental_recursive"], row["reward_r2_32_compression_vae"]),
    ]
    labels = [task[0] for task in tasks]
    mental = np.asarray([task[1] for task in tasks], dtype=np.float32)
    compression = np.asarray([task[2] for task in tasks], dtype=np.float32)
    lift = 100.0 * (mental - compression) / np.maximum(np.abs(compression), 1e-8)

    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=300)
    x = np.arange(len(tasks))
    ax.bar(x, lift, color=[PLOT_BLUE, PLOT_BLUE, PLOT_BLUE], edgecolor=PLOT_DARK_GREY, linewidth=0.5)
    ax.set_xticks(x, labels)
    style_axes(ax, xlabel="probe target", ylabel="mental z lift (%)")
    ax.set_title(f"Task-Signal Lift ({checkpoint})", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def task_display_name(task: str) -> str:
    return {
        "intent_label": "Intent",
        "interaction_label": "Interaction",
        "strategy_label": "Strategy",
        "knowledge_label": "Knowledge",
    }.get(task, task.replace("_label", "").title())


def aggregate_social_tasks(class_rows: list[dict[str, str]], dim: str) -> list[dict[str, float | str]]:
    tasks = sorted({row["task"] for row in class_rows})
    out: list[dict[str, float | str]] = []
    for task in tasks:
        mental = [
            float(row["macro_f1_mean"])
            for row in class_rows
            if row["task"] == task and str(row["dim"]) == dim and row["model"] == "mental_recursive"
        ]
        compression = [
            float(row["macro_f1_mean"])
            for row in class_rows
            if row["task"] == task and str(row["dim"]) == dim and row["model"] == "compression_vae"
        ]
        if not mental or not compression:
            continue
        mental_mean = float(np.mean(mental))
        compression_mean = float(np.mean(compression))
        gap = mental_mean - compression_mean
        lift = 100.0 * gap / max(abs(compression_mean), 1e-8)
        out.append(
            {
                "task": task,
                "label": task_display_name(task),
                "mental": mental_mean,
                "compression": compression_mean,
                "gap": gap,
                "lift": lift,
            }
        )
    return sorted(out, key=lambda item: float(item["lift"]), reverse=True)


def compute_social_win_rates(score_rows: list[dict[str, str]], dims: list[str]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for dim in dims:
        wins = 0
        total = 0
        tasks = sorted({row["task"] for row in score_rows if str(row["dim"]) == dim})
        for task in tasks:
            budgets = sorted(
                {row["train_per_class"] for row in score_rows if row["task"] == task and str(row["dim"]) == dim},
                key=int,
            )
            for budget in budgets:
                seeds = sorted(
                    {
                        row["seed"]
                        for row in score_rows
                        if row["task"] == task and str(row["dim"]) == dim and row["train_per_class"] == budget
                    },
                    key=int,
                )
                for seed in seeds:
                    mental = [
                        float(row["macro_f1"])
                        for row in score_rows
                        if row["task"] == task
                        and str(row["dim"]) == dim
                        and row["train_per_class"] == budget
                        and row["seed"] == seed
                        and row["model"] == "mental_recursive"
                    ]
                    compression = [
                        float(row["macro_f1"])
                        for row in score_rows
                        if row["task"] == task
                        and str(row["dim"]) == dim
                        and row["train_per_class"] == budget
                        and row["seed"] == seed
                        and row["model"] == "compression_vae"
                    ]
                    if mental and compression:
                        total += 1
                        wins += int(mental[0] > compression[0])
        if total:
            out.append({"label": f"{dim} z" if dim == "full" else f"{dim}D", "wins": wins, "total": total, "rate": wins / total})
    return out


def make_social_breakdown_figure(output_dir: Path, checkpoint: str) -> None:
    checkpoint_dir = output_dir / checkpoint
    aggregate_path = checkpoint_dir / "task_signal_classification_aggregate.csv"
    scores_path = checkpoint_dir / "task_signal_classification_scores.csv"
    if not aggregate_path.exists() or not scores_path.exists():
        return
    aggregate_rows = read_rows(aggregate_path)
    score_rows = read_rows(scores_path)
    full_rows = aggregate_social_tasks(aggregate_rows, "full")
    win_rates = compute_social_win_rates(score_rows, ["full", "32"])
    if not full_rows:
        return

    fig = plt.figure(figsize=(7.2, 3.25), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.85], wspace=0.36)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_consistency = fig.add_subplot(gs[0, 1])

    labels = [str(row["label"]) for row in full_rows]
    y = np.arange(len(labels))[::-1]
    compression = np.asarray([float(row["compression"]) for row in full_rows], dtype=np.float32)
    mental = np.asarray([float(row["mental"]) for row in full_rows], dtype=np.float32)
    lift = np.asarray([float(row["lift"]) for row in full_rows], dtype=np.float32)

    for idx, yi in enumerate(y):
        ax_score.plot(
            [compression[idx], mental[idx]],
            [yi, yi],
            color="#BFC5CC",
            linewidth=6.5,
            solid_capstyle="round",
            zorder=1,
        )
        ax_score.scatter(
            compression[idx],
            yi,
            s=105,
            color=PLOT_GREEN,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.7,
            zorder=3,
            label="Compression VAE" if idx == 0 else None,
        )
        ax_score.scatter(
            mental[idx],
            yi,
            s=105,
            color=PLOT_BLUE,
            edgecolor=PLOT_DARK_GREY,
            linewidth=0.7,
            zorder=4,
            label="Mental recursive z" if idx == 0 else None,
        )
        ax_score.annotate(
            "",
            xy=(mental[idx], yi),
            xytext=(compression[idx], yi),
            arrowprops=dict(arrowstyle="-|>", color=PLOT_BLUE, lw=1.35, shrinkA=10, shrinkB=10),
            zorder=2,
        )
        ax_score.text(
            max(mental[idx], compression[idx]) + 0.006,
            yi,
            f"+{lift[idx]:.0f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color=PLOT_DARK_GREY,
        )

    ax_score.set_yticks(y, labels)
    ax_score.set_xlim(0.13, 0.265)
    ax_score.set_ylim(-0.65, len(labels) - 0.65)
    style_axes(ax_score, xlabel="heldout macro F1", ylabel=None)
    legend = ax_score.legend(
        loc="lower center",
        bbox_to_anchor=(0.54, 0.02),
        ncol=2,
        frameon=True,
        fontsize=7.5,
        handlelength=1.0,
        borderpad=0.30,
        columnspacing=0.9,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D8DC")
    ax_score.set_title("Social Signal Breakdown", fontsize=11, fontweight="bold", pad=6)

    if win_rates:
        wr_labels = [str(row["label"]) for row in win_rates]
        wr_y = np.arange(len(wr_labels))[::-1]
        rates = np.asarray([float(row["rate"]) for row in win_rates], dtype=np.float32)
        ax_consistency.barh(wr_y, rates, color=[PLOT_BLUE, PLOT_LIGHT_GREEN], edgecolor=PLOT_DARK_GREY, linewidth=0.55)
        ax_consistency.set_yticks(wr_y, wr_labels)
        ax_consistency.set_xlim(0, 1.0)
        style_axes(ax_consistency, xlabel="mental wins", ylabel=None)
        ax_consistency.set_title("Seed/Budget Consistency", fontsize=11, fontweight="bold", pad=6)
        ax_consistency.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"])
        for idx, (yi, row) in enumerate(zip(wr_y, win_rates)):
            rate = float(row["rate"])
            ax_consistency.text(
                max(rate - 0.025, 0.05),
                yi,
                f"{int(row['wins'])}/{int(row['total'])}",
                va="center",
                ha="right",
                fontsize=8.5,
                fontweight="bold",
                color="white" if idx == 0 else PLOT_DARK_GREY,
            )

    fig.suptitle(f"Social Probe Breakdown ({checkpoint})", fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.18, top=0.78, wspace=0.36)
    png_path = output_dir / f"paper_social_task_breakdown_{checkpoint}.png"
    fig.savefig(png_path)
    fig.savefig(png_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default="projects/sotopia/experiments/runs/visuals/task_signal_efficiency_checkpoint_sweep_episodeval1516/checkpoint_task_signal_gap_summary.csv",
    )
    parser.add_argument("--checkpoint", default="epoch2")
    parser.add_argument(
        "--output-dir",
        default="projects/sotopia/experiments/runs/visuals/task_signal_efficiency_checkpoint_sweep_episodeval1516",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.summary_csv))
    row = find_checkpoint(rows, args.checkpoint)
    out_dir = Path(args.output_dir)
    make_signal_figure(row, args.checkpoint, out_dir / f"paper_task_signal_accessibility_{args.checkpoint}.png")
    make_accessibility_only_figure(
        row,
        args.checkpoint,
        out_dir / f"paper_task_signal_accessibility_only_{args.checkpoint}.png",
    )
    make_reward_head_only_figure(
        row,
        args.checkpoint,
        out_dir / f"paper_reward_head_accessibility_only_{args.checkpoint}.png",
    )
    make_reward_plus_intent_figure(row, args.checkpoint, out_dir)
    make_minimal_lift_strip(row, args.checkpoint, out_dir / f"paper_task_signal_lift_strip_{args.checkpoint}.png")
    make_social_breakdown_figure(out_dir, args.checkpoint)
    print(f"Wrote task-signal accessibility figures to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
