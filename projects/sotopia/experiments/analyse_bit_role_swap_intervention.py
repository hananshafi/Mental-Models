#!/usr/bin/env python3
"""Held-out role-swap intervention for BIT vs flat mental supervision.

For each grouped CV fold:
  1. Train full-z role probes on train scenarios.
  2. On held-out scenarios, predict belief / intent / thought targets.
  3. Swap exactly one latent sub-block between held-out examples.
  4. Measure which role prediction changes.

If BIT is the right interface, swapping z_belief should mostly move the belief
prediction, swapping z_intent should mostly move intent, and swapping z_thought
should mostly move thought.  Flat summaries can still encode mental content,
but the same intervention should be more diffuse across roles.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_bit_role_routing import (  # noqa: E402
    ROLE_LABELS,
    ROLE_ORDER,
    SUB_SLICES,
    FIELD_PATTERNS,
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_ORDER,
    extract_first_match,
    fit_role_targets,
    load_latents,
    load_val_records,
)


def mean_normalized_l2_shift(before: np.ndarray, after: np.ndarray) -> float:
    """Average prediction movement in standardized embedding units."""
    return float(np.mean(np.linalg.norm(after - before, axis=1) / np.sqrt(before.shape[1])))


def mean_cosine_shift(before: np.ndarray, after: np.ndarray) -> float:
    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    denom = np.maximum(before_norm * after_norm, 1e-8)
    cosine = np.sum(before * after, axis=1) / denom
    return float(np.mean(1.0 - cosine))


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    return np.sum(a * b, axis=1) / denom


def donor_pull_score(
    before: np.ndarray,
    after: np.ndarray,
    receiver_target: np.ndarray,
    donor_target: np.ndarray,
) -> float:
    """Positive means the swap moved predictions toward the donor target and
    away from the receiver's original target."""
    donor_gain = row_cosine(after, donor_target) - row_cosine(before, donor_target)
    receiver_gain = row_cosine(after, receiver_target) - row_cosine(before, receiver_target)
    return float(np.mean(donor_gain - receiver_gain))


def fit_ordered_role_targets(
    records: list[dict],
    target_order: str,
    n_components: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if target_order == "combined":
        return fit_role_targets(records, n_components, seed)

    pattern_idx = 0 if target_order == "first" else 1
    text_key = "mental1_text" if target_order == "first" else "mental2_text"
    targets = {}
    for role in ROLE_ORDER:
        texts = []
        for record in records:
            text = extract_first_match(record.get(text_key, ""), FIELD_PATTERNS[role][pattern_idx])
            texts.append(text if text.strip() else "N/A")
        vectorizer = TfidfVectorizer(
            max_features=12000,
            min_df=2,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(texts)
        max_components = max(2, min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        emb = TruncatedSVD(n_components=max_components, random_state=seed).fit_transform(tfidf)
        targets[role] = StandardScaler().fit_transform(emb).astype(np.float32)
    return targets


def make_fold_plans(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two scenario groups for grouped CV.")
    rng = np.random.default_rng(seed)
    plans = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups):
        n_test = len(test_idx)
        if n_test < 2:
            continue
        perm = rng.permutation(n_test)
        donors = np.empty(n_test, dtype=np.int64)
        donors[perm] = np.roll(perm, 1)
        plans.append((train_idx, test_idx, donors))
    return plans


def run_swap_intervention(
    latents: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, dict]]:
    fold_plans = make_fold_plans(groups, args.folds, args.seed)
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    variant_order = [v for v in getattr(args, "variants", VARIANT_ORDER) if v in latents]
    for variant in variant_order:
        X = latents[variant][args.feature]
        l2_matrix = np.zeros((len(ROLE_ORDER), len(ROLE_ORDER)), dtype=np.float64)
        cosine_matrix = np.zeros_like(l2_matrix)
        donor_pull_matrix = np.zeros_like(l2_matrix)
        count = 0

        for train_idx, test_idx, donors in fold_plans:
            X_train = X[train_idx]
            X_test = X[test_idx]
            probes = {}
            original_preds = {}
            for target_role in ROLE_ORDER:
                model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
                model.fit(X_train, targets[target_role][train_idx])
                probes[target_role] = model
                original_preds[target_role] = model.predict(X_test)

            for source_i, source_role in enumerate(ROLE_ORDER):
                X_swapped = X_test.copy()
                sl = SUB_SLICES[source_role]
                X_swapped[:, sl] = X_test[donors][:, sl]
                for target_i, target_role in enumerate(ROLE_ORDER):
                    swapped_preds = probes[target_role].predict(X_swapped)
                    l2_matrix[source_i, target_i] += mean_normalized_l2_shift(
                        original_preds[target_role], swapped_preds
                    )
                    cosine_matrix[source_i, target_i] += mean_cosine_shift(
                        original_preds[target_role], swapped_preds
                    )
                    y_test = targets[target_role][test_idx]
                    donor_pull_matrix[source_i, target_i] += donor_pull_score(
                        original_preds[target_role],
                        swapped_preds,
                        receiver_target=y_test,
                        donor_target=y_test[donors],
                    )
            count += 1

        l2_matrix = (l2_matrix / max(count, 1)).astype(np.float32)
        cosine_matrix = (cosine_matrix / max(count, 1)).astype(np.float32)
        donor_pull_matrix = (donor_pull_matrix / max(count, 1)).astype(np.float32)
        row_mass = l2_matrix / np.maximum(l2_matrix.sum(axis=1, keepdims=True), 1e-8)
        diag_mask = np.eye(len(ROLE_ORDER), dtype=bool)
        off_mask = ~diag_mask
        diag_l2 = float(l2_matrix[diag_mask].mean())
        off_l2 = float(l2_matrix[off_mask].mean())
        diag_pull = float(donor_pull_matrix[diag_mask].mean())
        off_pull = float(donor_pull_matrix[off_mask].mean())
        diag_mass = float(row_mass[diag_mask].mean())
        uniform = 1.0 / len(ROLE_ORDER)
        specificity = float((diag_mass - uniform) / (1.0 - uniform))

        summary[variant] = {
            "l2_shift_matrix": l2_matrix,
            "cosine_shift_matrix": cosine_matrix,
            "donor_pull_matrix": donor_pull_matrix,
            "row_normalized_mass": row_mass.astype(np.float32),
            "diag_l2_shift": diag_l2,
            "offdiag_l2_shift": off_l2,
            "diag_minus_off_l2": float(diag_l2 - off_l2),
            "diag_donor_pull": diag_pull,
            "offdiag_donor_pull": off_pull,
            "donor_pull_gap": float(diag_pull - off_pull),
            "diag_mass": diag_mass,
            "specificity_over_uniform": specificity,
            "offtarget_leakage": float(off_l2 / max(diag_l2, 1e-8)),
        }

        print(
            f"{variant}: diag mass={diag_mass:.3f}, "
            f"specificity={specificity:+.3f}, "
            f"donor-pull gap={summary[variant]['donor_pull_gap']:+.3f}, "
            f"leakage={summary[variant]['offtarget_leakage']:.3f}",
            flush=True,
        )

        for source_i, source_role in enumerate(ROLE_ORDER):
            for target_i, target_role in enumerate(ROLE_ORDER):
                rows.append({
                    "variant": variant,
                    "swapped_subblock": source_role,
                    "affected_prediction": target_role,
                    "l2_shift": float(l2_matrix[source_i, target_i]),
                    "cosine_shift": float(cosine_matrix[source_i, target_i]),
                    "donor_pull": float(donor_pull_matrix[source_i, target_i]),
                    "row_normalized_mass": float(row_mass[source_i, target_i]),
                    "is_diagonal": source_role == target_role,
                })
    return rows, summary


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
    })


def plot_swap_heatmaps(summary: dict[str, dict], out_path: Path, variant_order: list[str] | None = None) -> None:
    set_style()
    variants = [v for v in (variant_order or VARIANT_ORDER) if v in summary]
    cmap = LinearSegmentedColormap.from_list(
        "swap_mass",
        ["#F7F3EA", "#D6E6F2", "#7EA6C9", "#1F4E8C"],
    )

    fig, axes = plt.subplots(1, len(variants), figsize=(5.9, 1.95), dpi=300)
    if len(variants) == 1:
        axes = [axes]
    im = None
    for ax, variant in zip(axes, variants):
        mat = summary[variant]["row_normalized_mass"]
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.set_yticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.tick_params(length=0)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
        for idx in range(3):
            ax.add_patch(plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False,
                                       edgecolor="#111111", linewidth=1.6))
        ax.set_title(
            f"{VARIANT_LABELS.get(variant, variant)}\n"
            f"diag mass={summary[variant]['diag_mass']:.2f}",
            pad=5,
        )
        if ax is axes[0]:
            ax.set_ylabel("swapped block")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.82, pad=0.18)
    cbar.outline.set_visible(False)
    cbar.set_ticks([1 / 3, 1.0])
    cbar.set_ticklabels(["uniform", "targeted"])
    cbar.ax.tick_params(labelsize=7.5, length=0, pad=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight("bold")
    cbar.set_label("within-swap share of prediction change", fontsize=8, fontweight="bold", labelpad=2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.37, top=0.80, wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def plot_specificity(summary: dict[str, dict], out_path: Path, variant_order: list[str] | None = None) -> None:
    set_style()
    variants = [v for v in (variant_order or VARIANT_ORDER) if v in summary]
    y = np.arange(len(variants))[::-1]
    uniform = 1.0 / len(ROLE_ORDER)
    diag_masses = [float(summary[v]["diag_mass"]) for v in variants]
    fig, ax = plt.subplots(figsize=(4.6, 1.9), dpi=300)
    ax.axvline(uniform, color="#8D939A", linewidth=1.4, linestyle="--", zorder=0)
    ax.text(uniform, len(variants) - 0.55, "uniform", ha="center", va="bottom",
            fontsize=8.2, fontweight="bold", color="#626870")
    for yi, variant, value in zip(y, variants, diag_masses):
        ax.plot([uniform, value], [yi, yi], color="#B9BEC5", linewidth=6,
                solid_capstyle="round", zorder=1)
        ax.annotate(
            "",
            xy=(value, yi),
            xytext=(uniform, yi),
            arrowprops=dict(arrowstyle="-|>", color=VARIANT_COLORS.get(variant, "#555555"), linewidth=2.0),
            zorder=2,
        )
        ax.scatter([value], [yi], s=140, color=VARIANT_COLORS.get(variant, "#555555"),
                   edgecolor="#333333", linewidth=0.9, zorder=3)
        ax.text(value + 0.012, yi, f"{value:.3f}", ha="left", va="center",
                fontsize=8.8, fontweight="bold")
    ax.set_yticks(y, [VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("diagonal share of swap-induced change")
    ax.set_title("Role-swap specificity")
    ax.set_xlim(0.25, max(diag_masses) + 0.08)
    ax.grid(axis="x", color="#D8DCE0", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.30, top=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def plot_donor_pull_heatmaps(summary: dict[str, dict], out_path: Path, variant_order: list[str] | None = None) -> None:
    set_style()
    variants = [v for v in (variant_order or VARIANT_ORDER) if v in summary]
    mats = [summary[v]["donor_pull_matrix"] for v in variants]
    vmax = max(abs(float(np.min(m))) for m in mats)
    vmax = max(vmax, max(abs(float(np.max(m))) for m in mats), 1e-6)
    cmap = LinearSegmentedColormap.from_list(
        "donor_pull",
        ["#1F4E8C", "#D6E6F2", "#F7F3EA", "#E6A07C", "#B2182B"],
    )

    fig, axes = plt.subplots(1, len(variants), figsize=(5.9, 1.95), dpi=300)
    if len(variants) == 1:
        axes = [axes]
    im = None
    for ax, variant in zip(axes, variants):
        mat = summary[variant]["donor_pull_matrix"]
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.set_yticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.tick_params(length=0)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
        for idx in range(3):
            ax.add_patch(plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False,
                                       edgecolor="#111111", linewidth=1.6))
        ax.set_title(
            f"{VARIANT_LABELS.get(variant, variant)}\n"
            f"gap={summary[variant]['donor_pull_gap']:+.3f}",
            pad=5,
        )
        if ax is axes[0]:
            ax.set_ylabel("swapped block")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.82, pad=0.18)
    cbar.outline.set_visible(False)
    cbar.set_ticks([-vmax, 0.0, vmax])
    cbar.set_ticklabels(["receiver", "none", "donor"])
    cbar.ax.tick_params(labelsize=7.5, length=0, pad=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight("bold")
    cbar.set_label("semantic pull after swap", fontsize=8, fontweight="bold", labelpad=2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.37, top=0.80, wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def plot_donor_pull_gap(summary: dict[str, dict], out_path: Path, variant_order: list[str] | None = None) -> None:
    set_style()
    variants = [v for v in (variant_order or VARIANT_ORDER) if v in summary]
    y = np.arange(len(variants))[::-1]
    off = [float(summary[v]["offdiag_donor_pull"]) for v in variants]
    diag = [float(summary[v]["diag_donor_pull"]) for v in variants]
    fig, ax = plt.subplots(figsize=(4.8, 1.9), dpi=300)
    for yi, variant, off_val, diag_val in zip(y, variants, off, diag):
        ax.plot([off_val, diag_val], [yi, yi], color="#B9BEC5", linewidth=6,
                solid_capstyle="round", zorder=1)
        ax.annotate(
            "",
            xy=(diag_val, yi),
            xytext=(off_val, yi),
            arrowprops=dict(arrowstyle="-|>", color=VARIANT_COLORS.get(variant, "#555555"), linewidth=2.0),
            zorder=2,
        )
        ax.scatter([off_val], [yi], s=95, color="#C8CDD3", edgecolor="#333333", linewidth=0.8, zorder=3)
        ax.scatter([diag_val], [yi], s=135, color=VARIANT_COLORS.get(variant, "#555555"),
                   edgecolor="#333333", linewidth=0.9, zorder=4)
        ax.text(max(off_val, diag_val) + 0.006, yi, f"{diag_val - off_val:+.3f}",
                ha="left", va="center", fontsize=8.8, fontweight="bold")
    ax.set_yticks(y, [VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("semantic donor-pull score")
    ax.set_title("Role-swap donor-pull gap")
    ax.grid(axis="x", color="#D8DCE0", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    x_min = min(min(off), min(diag)) - 0.025
    x_max = max(max(off), max(diag)) + 0.055
    ax.set_xlim(x_min, x_max)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.30, top=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def write_outputs(out_dir: Path, rows: list[dict], summary: dict[str, dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "role_swap_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_ready = {}
    for variant, values in summary.items():
        json_ready[variant] = {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in values.items()
        }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(json_ready, f, indent=2)

    md = [
        "# BIT role-swap intervention",
        "",
        "| variant | diagonal change share ↑ | specificity over uniform ↑ | "
        "donor-pull gap ↑ | diag L2 shift | offdiag L2 shift | off-target leakage ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_variants = [v for v in (list(summary.keys())) if v in summary]
    for variant in ordered_variants:
        values = summary[variant]
        md.append(
            f"| {variant} | {values['diag_mass']:.4f} | "
            f"{values['specificity_over_uniform']:+.4f} | "
            f"{values['donor_pull_gap']:+.4f} | "
            f"{values['diag_l2_shift']:.4f} | {values['offdiag_l2_shift']:.4f} | "
            f"{values['offtarget_leakage']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents_dir", default="projects/sotopia/experiments/runs/stage1/variant_eval")
    parser.add_argument("--output_dir", default="projects/sotopia/experiments/runs/stage1/bit_role_swap")
    parser.add_argument("--variants", nargs="+", default=VARIANT_ORDER)
    parser.add_argument("--feature", default="z1", choices=["z1", "z2", "z_concat"])
    parser.add_argument("--target_order", default="first", choices=["first", "second", "combined"])
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--max_mental_len", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--text_components", type=int, default=48)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    print("Loading held-out split records...", flush=True)
    records, groups = load_val_records(args)
    print(f"  records={len(records)}, scenarios={len(np.unique(groups))}", flush=True)

    print("Loading cached variant latents...", flush=True)
    latents = load_latents(Path(args.latents_dir), args.variants)
    n_latents = next(iter(latents.values()))[args.feature].shape[0]
    if n_latents != len(records):
        raise RuntimeError(f"latent count {n_latents} != record count {len(records)}")

    print(f"Fitting {args.target_order}-order role text targets...", flush=True)
    targets = fit_ordered_role_targets(records, args.target_order, args.text_components, args.seed)

    print("Running held-out sub-block swap interventions...", flush=True)
    rows, summary = run_swap_intervention(latents, targets, groups, args)
    write_outputs(out_dir, rows, summary)
    plot_swap_heatmaps(summary, out_dir / "fig_role_swap_heatmaps.png", args.variants)
    plot_specificity(summary, out_dir / "fig_role_swap_specificity.png", args.variants)
    plot_donor_pull_heatmaps(summary, out_dir / "fig_role_swap_donor_pull_heatmaps.png", args.variants)
    plot_donor_pull_gap(summary, out_dir / "fig_role_swap_donor_pull_gap.png", args.variants)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
