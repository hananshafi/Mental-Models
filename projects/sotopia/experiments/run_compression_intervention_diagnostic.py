#!/usr/bin/env python3
"""Intervention diagnostic: mental VAE vs compression VAE.

This experiment avoids the overly permissive "raw latent probe" comparison.
For each latent family, it trains a probe on the correct held-out latent and
then evaluates the same trained probe after latent interventions:

  mental reward:      [z1, z2] -> [z1, shuffled z2] / [z1, zero z2]
  compression VAE:    [c1, c2] -> [c1, shuffled c2] / [c1, zero c2]

The compression split is deliberately arbitrary: it is a capacity-matched
single-latent baseline, not a recursive model. If the mental model is doing
something beyond generic compression, the recursive second half should show a
more targeted relationship to mental-state targets than the compression split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_recursive_decomposition_empirical import fit_text_targets, mean_cosine


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_GREY = "#7A7A7A"
PLOT_DARK_GREY = "#4B4B4B"
PLOT_RED = "#C84C4C"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key].astype(np.float32) for key in data.files}


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n > 1:
        tries = 0
        while np.any(perm == np.arange(n)) and tries < 100:
            stuck = perm == np.arange(n)
            perm[stuck] = np.roll(perm[stuck], 1)
            tries += 1
    return perm


def intervention_probe_cv(
    train_features: np.ndarray,
    eval_features: dict[str, np.ndarray],
    target: np.ndarray,
    folds: int,
    seed: int,
    alpha: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    accum: dict[str, dict[str, list[float]]] = {
        name: {"r2": [], "cosine": []} for name in eval_features
    }
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(train_features):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_features[train_idx], target[train_idx])
        for name, X_eval in eval_features.items():
            pred = model.predict(X_eval[test_idx])
            accum[name]["r2"].append(r2_score(target[test_idx], pred, multioutput="variance_weighted"))
            accum[name]["cosine"].append(mean_cosine(target[test_idx], pred))

    for name, metrics in accum.items():
        rows.append({
            "condition": name,
            "r2": float(np.mean(metrics["r2"])),
            "cosine": float(np.mean(metrics["cosine"])),
        })
    return rows


def style_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C7CBD1", linewidth=0.9, alpha=0.85)
    ax.grid(True, which="minor", color="#E8EAED", linewidth=0.5, alpha=0.85)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.1, length=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_intervention_r2(rows: list[dict[str, Any]], out_path: Path) -> None:
    targets = ["context_text", "mental2_text", "reward_vec"]
    target_labels = ["Observed\ncontext", "Second-order\nmental", "Reward\ncontrol"]
    models = ["mental_recursive", "compression_vae"]
    model_labels = ["Mental recursive", "Compression VAE"]
    conditions = ["correct", "shuffle_second"]
    colors = {
        ("mental_recursive", "correct"): PLOT_BLUE,
        ("mental_recursive", "shuffle_second"): "#9EC3E6",
        ("compression_vae", "correct"): PLOT_GREEN,
        ("compression_vae", "shuffle_second"): "#A8DDB5",
    }
    lookup = {
        (row["model"], row["target"], row["condition"]): float(row["r2"])
        for row in rows
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, target, target_label in zip(axes, targets, target_labels):
        xs = np.arange(len(models))
        width = 0.34
        for j, condition in enumerate(conditions):
            vals = [lookup[(model, target, condition)] for model in models]
            ax.bar(
                xs + (j - 0.5) * width,
                vals,
                width=width,
                color=[colors[(model, condition)] for model in models],
                edgecolor="#555555",
                linewidth=0.45,
                label="correct" if condition == "correct" else "shuffle second half",
            )
        ax.set_xticks(xs, model_labels, rotation=18, ha="right")
        ax.set_title(target_label, fontsize=13, fontweight="bold")
        style_axes(ax, ylabel="probe R2" if ax is axes[0] else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_drop(rows: list[dict[str, Any]], out_path: Path) -> None:
    targets = ["context_text", "mental1_text", "mental2_text", "reward_vec"]
    target_labels = ["Context", "Mental-1", "Mental-2", "Reward"]
    models = ["mental_recursive", "compression_vae"]
    model_labels = ["Mental recursive", "Compression VAE"]
    colors = [PLOT_BLUE, PLOT_GREEN]
    lookup = {
        (row["model"], row["target"], row["condition"]): float(row["r2"])
        for row in rows
    }
    drops = {
        model: [
            lookup[(model, target, "correct")] - lookup[(model, target, "shuffle_second")]
            for target in targets
        ]
        for model in models
    }

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(targets))
    width = 0.36
    for i, model in enumerate(models):
        ax.bar(
            x + (i - 0.5) * width,
            drops[model],
            width,
            label=model_labels[i],
            color=colors[i],
            edgecolor="#555555",
            linewidth=0.45,
        )
    ax.axhline(0, color=PLOT_DARK_GREY, linewidth=1.0)
    ax.set_xticks(x, target_labels)
    style_axes(ax, xlabel="probe target", ylabel="R2 drop after second-half shuffle")
    leg = ax.legend(frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D6D8DC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_specificity(rows: list[dict[str, Any]], out_path: Path) -> None:
    lookup = {
        (row["model"], row["target"], row["condition"]): float(row["r2"])
        for row in rows
    }
    models = ["mental_recursive", "compression_vae"]
    labels = ["Mental recursive", "Compression VAE"]
    values = []
    for model in models:
        drop_m2 = lookup[(model, "mental2_text", "correct")] - lookup[(model, "mental2_text", "shuffle_second")]
        drop_reward = lookup[(model, "reward_vec", "correct")] - lookup[(model, "reward_vec", "shuffle_second")]
        drop_context = lookup[(model, "context_text", "correct")] - lookup[(model, "context_text", "shuffle_second")]
        values.append({
            "mental_minus_reward": drop_m2 - drop_reward,
            "mental_minus_context": drop_m2 - drop_context,
        })

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, [v["mental_minus_reward"] for v in values], width, color=PLOT_BLUE, label="Mental-2 drop - reward drop")
    ax.bar(x + width / 2, [v["mental_minus_context"] for v in values], width, color=PLOT_GREEN, label="Mental-2 drop - context drop")
    ax.axhline(0, color=PLOT_DARK_GREY, linewidth=1.0)
    ax.set_xticks(x, labels)
    style_axes(ax, xlabel="latent model", ylabel="specificity of second-half shuffle")
    leg = ax.legend(frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D6D8DC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mental-arrays", required=True)
    parser.add_argument("--compression-arrays", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--text-components", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.records_jsonl)
    mental = load_npz(args.mental_arrays)
    compression = load_npz(args.compression_arrays)

    n = min(len(records), len(mental["z1"]), len(compression["z_compress"]))
    records = records[:n]
    z1 = mental["z1"][:n]
    z2 = mental["z2"][:n]
    zc = compression["z_compress"][:n]
    c1, c2 = zc[:, : zc.shape[1] // 2], zc[:, zc.shape[1] // 2 :]
    perm = derangement(n, args.seed)

    mental_features = {
        "correct": np.concatenate([z1, z2], axis=1),
        "shuffle_second": np.concatenate([z1, z2[perm]], axis=1),
        "zero_second": np.concatenate([z1, np.zeros_like(z2)], axis=1),
        "shuffle_first": np.concatenate([z1[perm], z2], axis=1),
        "zero_first": np.concatenate([np.zeros_like(z1), z2], axis=1),
    }
    compression_features = {
        "correct": np.concatenate([c1, c2], axis=1),
        "shuffle_second": np.concatenate([c1, c2[perm]], axis=1),
        "zero_second": np.concatenate([c1, np.zeros_like(c2)], axis=1),
        "shuffle_first": np.concatenate([c1[perm], c2], axis=1),
        "zero_first": np.concatenate([np.zeros_like(c1), c2], axis=1),
    }

    reward = StandardScaler().fit_transform(
        np.asarray([rec["reward_vec"] for rec in records], dtype=np.float32)
    ).astype(np.float32)
    targets = {
        "context_text": fit_text_targets([rec.get("context_text", "") for rec in records], args.text_components, args.seed),
        "mental1_text": fit_text_targets([rec.get("mental1_text", "") for rec in records], args.text_components, args.seed),
        "mental2_text": fit_text_targets([rec.get("mental2_text", "") for rec in records], args.text_components, args.seed),
        "reward_vec": reward,
    }

    rows: list[dict[str, Any]] = []
    for model_name, features in [
        ("mental_recursive", mental_features),
        ("compression_vae", compression_features),
    ]:
        train_X = features["correct"]
        for target_name, Y in targets.items():
            result_rows = intervention_probe_cv(
                train_X,
                features,
                Y,
                folds=args.folds,
                seed=args.seed,
                alpha=args.ridge_alpha,
            )
            for row in result_rows:
                rows.append({
                    "model": model_name,
                    "target": target_name,
                    **row,
                })

    write_csv(out_dir / "compression_intervention_scores.csv", rows)

    drop_rows: list[dict[str, Any]] = []
    lookup = {
        (row["model"], row["target"], row["condition"]): row
        for row in rows
    }
    for model_name in ["mental_recursive", "compression_vae"]:
        for target_name in targets:
            correct = float(lookup[(model_name, target_name, "correct")]["r2"])
            for intervention in ["shuffle_second", "zero_second", "shuffle_first", "zero_first"]:
                value = float(lookup[(model_name, target_name, intervention)]["r2"])
                drop_rows.append({
                    "model": model_name,
                    "target": target_name,
                    "intervention": intervention,
                    "correct_r2": correct,
                    "intervention_r2": value,
                    "r2_drop": correct - value,
                })
    write_csv(out_dir / "compression_intervention_drops.csv", drop_rows)

    plot_intervention_r2(rows, out_dir / "paper_compression_intervention_r2.png")
    plot_drop(rows, out_dir / "paper_compression_second_half_shuffle_drop.png")
    plot_specificity(rows, out_dir / "paper_compression_specificity.png")

    summary = {
        "n": n,
        "config": vars(args),
        "main_findings": {
            "mental_recursive_flip_is_not_tested_here": True,
            "diagnostic": "probe trained on correct latent; evaluated under latent shuffle/zero interventions",
        },
    }
    with (out_dir / "compression_intervention_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote compression intervention diagnostic to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
