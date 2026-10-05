#!/usr/bin/env python3
"""Per-reward-dimension R² analysis: BIT vs flat vs shuffled.

Recreates the same val split used by eval_mental_variants.py, extracts the
7-D reward_vec target, and fits one ridge regressor per reward dimension
from the cached frozen latents (z1, z2, z_concat). Reports per-dim R² with
5-fold CV.

Goal: show that flat summary loses the dimensional structure of partner
mental state, leading to systematically worse modeling on the
mental-laden reward axes (relationship, secret, social_rules, knowledge).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
from transformers import AutoTokenizer
import torch

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    SOTOPIA_DIMENSIONS, RecursiveToMDataset,
)

VARIANT_EVAL = SOTOPIA_ROOT / "experiments/runs/stage1/variant_eval"
OUT_DIR = VARIANT_EVAL / "per_reward_dim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["structured_bit", "flat_mental_summary", "shuffled_mental"]
FEATURES = ["z1", "z_concat"]   # focus on encoder latent only; z2 is reward-conditioned
SEED = 42
FOLDS = 5
RIDGE_ALPHA = 10.0
DATA_PATH = "projects/sotopia/data/sotopia_turn_rewards_v3.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

C_BIT = "#3A5F87"
C_FLAT = "#CC8A4C"
C_SHUFFLED = "#5DA39A"
C_ARROW = "#A6ADB4"
COLORS = {"structured_bit": C_BIT,
          "flat_mental_summary": C_FLAT,
          "shuffled_mental": C_SHUFFLED}


def get_val_reward_vecs() -> np.ndarray:
    """Recreate the same val split used by eval_mental_variants.py
    (deterministic via torch random_split with seed=42, val_ratio=0.1)
    and pull reward_vec per sample."""
    print("Loading dataset & recreating val split...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = RecursiveToMDataset(
        DATA_PATH, tokenizer,
        max_ctx_len=1024, max_resp_len=256, max_mental_len=256,
    )
    val_size = min(max(1, int(len(dataset) * 0.1)), len(dataset) - 1)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(SEED)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)

    base = val_dataset.dataset
    indices = val_dataset.indices
    reward_vecs = np.asarray(
        [base.samples[i]["reward_vec"] for i in indices],
        dtype=np.float32,
    )
    print(f"  val size: {reward_vecs.shape[0]}, reward dim: {reward_vecs.shape[1]}", flush=True)
    return reward_vecs


def per_dim_ridge_cv(
    X: np.ndarray, Y: np.ndarray, folds: int, seed: int, alpha: float,
) -> np.ndarray:
    """Return per-dim mean R² across folds, shape (Y.shape[1],)."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_r2 = []
    for train_idx, test_idx in kf.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        fold_r2.append(r2_score(Y[test_idx], pred, multioutput="raw_values"))
    return np.mean(np.stack(fold_r2, axis=0), axis=0)


def main():
    reward_vecs = get_val_reward_vecs()  # (N, 7)
    Y = StandardScaler().fit_transform(reward_vecs).astype(np.float32)

    rng = np.random.default_rng(SEED)
    results: dict[str, dict[str, np.ndarray]] = {}
    shuffled_floor: dict[str, np.ndarray] = {}

    for variant in VARIANTS:
        npz_path = VARIANT_EVAL / f"latents_{variant}.npz"
        arrays = np.load(npz_path)
        results[variant] = {}
        for feat in FEATURES:
            X = arrays[feat].astype(np.float32)
            r2_per_dim = per_dim_ridge_cv(X, Y, FOLDS, SEED, RIDGE_ALPHA)
            results[variant][feat] = r2_per_dim
            print(f"  {variant:22s} {feat:9s}  "
                  + " ".join(f"{r2:+.3f}" for r2 in r2_per_dim), flush=True)

        # shuffled-label noise floor (per variant just to confirm probe is honest)
        if variant == VARIANTS[0]:
            X = arrays[FEATURES[0]].astype(np.float32)
            perm = rng.permutation(Y.shape[0])
            shuffled_floor[FEATURES[0]] = per_dim_ridge_cv(X, Y[perm], FOLDS, SEED, RIDGE_ALPHA)
            print(f"  shuffled-label floor    {FEATURES[0]:9s}  "
                  + " ".join(f"{r2:+.3f}" for r2 in shuffled_floor[FEATURES[0]]), flush=True)

    # ── save raw numbers ──────────────────────────────────────────────────────
    summary = {
        "reward_dim_names": SOTOPIA_DIMENSIONS,
        "feature_sets": FEATURES,
        "variants": VARIANTS,
        "r2_per_dim": {
            v: {f: results[v][f].tolist() for f in FEATURES} for v in VARIANTS
        },
        "shuffled_label_floor_z1": shuffled_floor.get("z1", np.zeros(7)).tolist(),
        "n_val": int(reward_vecs.shape[0]),
        "folds": FOLDS,
        "ridge_alpha": RIDGE_ALPHA,
    }
    (OUT_DIR / "per_reward_dim_r2.json").write_text(json.dumps(summary, indent=2))

    # markdown table
    lines = ["# Per-reward-dimension R² (frozen-z ridge probe, 5-fold CV)", ""]
    for feat in FEATURES:
        lines += [f"## feature = {feat}", "",
                  "| reward dim | " + " | ".join(VARIANTS) + " | shuffled-label floor (BIT) |",
                  "|---|" + "|".join(["---:"] * (len(VARIANTS) + 1)) + "|"]
        for di, dim in enumerate(SOTOPIA_DIMENSIONS):
            cells = [f"{results[v][feat][di]:+.3f}" for v in VARIANTS]
            floor_val = shuffled_floor.get(feat, np.zeros(7))[di] if feat == FEATURES[0] else float("nan")
            cells.append(f"{floor_val:+.3f}" if not np.isnan(floor_val) else "—")
            lines.append(f"| {dim} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT_DIR / "per_reward_dim_r2.md").write_text("\n".join(lines))

    # ── figure (z1 only, paper style) ──────────────────────────────────────────
    plot_per_dim_dotplot(results, "z1", shuffled_floor.get("z1"))
    plot_per_dim_dotplot(results, "z_concat", None)

    print(f"\nWrote {OUT_DIR}/", flush=True)


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.3,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 240,
    })


PRETTY_DIM = {
    "believability": "believability",
    "relationship": "relationship",
    "knowledge": "knowledge",
    "secret": "secret",
    "social_rules": "social rules",
    "financial_and_material_benefits": "financial /\nmaterial",
    "goal": "goal",
}

# Mark which dims are mental-laden (have substantial partner-mental-state component)
MENTAL_DIMS = {"relationship", "secret", "social_rules", "knowledge"}


def plot_per_dim_dotplot(results, feature, floor):
    set_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    n_dims = len(SOTOPIA_DIMENSIONS)

    # row order: keep SOTOPIA_DIMENSIONS order, mark mental-laden via background shade
    rows = list(reversed(SOTOPIA_DIMENSIONS))
    y_positions = np.arange(len(rows))

    for y, dim in zip(y_positions, rows):
        di = SOTOPIA_DIMENSIONS.index(dim)
        x_bit = results["structured_bit"][feature][di]
        x_flat = results["flat_mental_summary"][feature][di]
        x_shuf = results["shuffled_mental"][feature][di]

        # background shade for mental-laden dims
        if dim in MENTAL_DIMS:
            ax.axhspan(y - 0.45, y + 0.45, color="#FFF4D6", alpha=0.55, zorder=0)

        # connector arrow loser → winner
        winner = max(x_bit, x_flat)
        loser = min(x_bit, x_flat)
        ax.plot([loser, winner], [y, y], color=C_ARROW,
                linewidth=10, alpha=0.5, solid_capstyle="round", zorder=1)
        ax.annotate(
            "", xy=(winner, y), xytext=(loser, y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.8",
                            color=C_ARROW, linewidth=2.0, alpha=0.95),
            zorder=2,
        )

        ax.plot(x_shuf, y, marker="D", markersize=9, markerfacecolor=C_SHUFFLED,
                markeredgecolor="#1F1F1F", markeredgewidth=1.2, zorder=3)
        ax.plot(x_flat, y, marker="o", markersize=18, markerfacecolor=C_FLAT,
                markeredgecolor="#1F1F1F", markeredgewidth=1.4, zorder=4)
        ax.plot(x_bit, y, marker="o", markersize=18, markerfacecolor=C_BIT,
                markeredgecolor="#1F1F1F", markeredgewidth=1.4, zorder=4)

        # value labels for BIT and flat
        ax.annotate(f"{x_bit:+.3f}", (x_bit, y), xytext=(0, -22),
                    textcoords="offset points", ha="center",
                    fontsize=9.5, color="#222", fontweight="bold")
        ax.annotate(f"{x_flat:+.3f}", (x_flat, y), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=9.5, color="#222", fontweight="bold")

    if floor is not None:
        # overlay shuffled-label floor markers (semi-transparent gray X)
        for y, dim in zip(y_positions, rows):
            di = SOTOPIA_DIMENSIONS.index(dim)
            ax.plot(floor[di], y, marker="x", markersize=8,
                    color="#888", markeredgewidth=1.5, zorder=2.5)

    # zero line
    ax.axvline(0, color="#444", linewidth=0.9, linestyle="--", alpha=0.7, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([PRETTY_DIM[d] for d in rows], fontweight="bold")
    ax.set_ylabel("SOTOPIA reward dimension", fontweight="bold")
    ax.set_xlabel(f"held-out R² (frozen {feature} → reward)  (higher = better)",
                  fontweight="bold")
    title_feat = "z₁ encoder latent" if feature == "z1" else "z₁ ⊕ z₂ concatenated latent"
    ax.set_title(f"Per-reward-dimension probe ({title_feat})", pad=14)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.grid(axis="x", which="major", color="#D6DADE", linewidth=0.8, alpha=0.85, zorder=0)
    ax.grid(axis="x", which="minor", color="#E5E8EB", linewidth=0.4, alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#222")
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(axis="x", which="major", length=4)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="BIT (ours)",
               markerfacecolor=C_BIT, markeredgecolor="#1F1F1F",
               markeredgewidth=1.3, markersize=12),
        Line2D([0], [0], marker="o", color="w", label="flat summary",
               markerfacecolor=C_FLAT, markeredgecolor="#1F1F1F",
               markeredgewidth=1.3, markersize=12),
        Line2D([0], [0], marker="D", color="w", label="shuffled (control)",
               markerfacecolor=C_SHUFFLED, markeredgecolor="#1F1F1F",
               markeredgewidth=1.0, markersize=9),
    ]
    if floor is not None:
        legend_handles.append(
            Line2D([0], [0], marker="x", color="#888", label="shuffled-label floor",
                   markersize=9, linestyle="None", markeredgewidth=1.5)
        )
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(0.99, 0.02), frameon=True, framealpha=0.95,
              edgecolor="#888", fancybox=True)

    # annotate the mental-band region
    fig.text(0.015, 0.5,
             "yellow band = mental-laden reward axes",
             rotation=90, va="center", fontsize=9.5, color="#8B6F1B",
             fontstyle="italic")

    fig.tight_layout(rect=(0.03, 0, 1, 1))
    out_path = OUT_DIR / f"fig_per_reward_dim_{feature}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
