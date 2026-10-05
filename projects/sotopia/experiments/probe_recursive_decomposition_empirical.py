#!/usr/bin/env python3
"""Empirically probe the intrinsic value of z1 -> z2 recursive decomposition.

This script consumes cached latent arrays from analyse_tom_latents.py:
  latent_arrays.npz with z1, z2, z_concat, context_hidden, reward_vec
  analysis_records_subset.jsonl or latent_records.jsonl with mental1/mental2 text

It produces paper-facing latent probing figures:
  1. z1/z2 probe heatmaps for first-order text, second-order text, reward.
  2. order-selectivity bars: z1 advantage on first-order, z2 advantage on second-order.
  3. recursive unique contribution bars: z1 vs z1+z2 vs z1+shuffled-z2 vs residual z2.
  4. z1->z2 mediation summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_GREY = "#7A7A7A"
PLOT_LIGHT_GREY = "#D6D8DC"
PLOT_DARK_GREY = "#4B4B4B"
PLOT_PURPLE = "#7B61B5"
PLOT_RED = "#C84C4C"


def style_plot_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
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


def load_records(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_arrays(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key].astype(np.float32) for key in data.files}


def fit_text_targets(texts: list[str], n_components: int, seed: int) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        max_features=12000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform([text if text.strip() else "N/A" for text in texts])
    max_components = max(2, min(n_components, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=seed)
    Z = svd.fit_transform(X).astype(np.float32)
    return StandardScaler().fit_transform(Z).astype(np.float32)


def mean_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    denom = np.maximum(true_norm * pred_norm, 1e-8)
    return float(np.mean(np.sum(y_true * y_pred, axis=1) / denom))


def regression_probe_cv(
    X: np.ndarray,
    Y: np.ndarray,
    folds: int,
    seed: int,
    alpha: float,
) -> dict[str, float]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    r2s = []
    cosines = []
    for train_idx, test_idx in kf.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        r2s.append(r2_score(Y[test_idx], pred, multioutput="variance_weighted"))
        cosines.append(mean_cosine(Y[test_idx], pred))
    return {"r2": float(np.mean(r2s)), "cosine": float(np.mean(cosines))}


def residualize_cv(X_base: np.ndarray, X_target: np.ndarray, folds: int, seed: int, alpha: float) -> np.ndarray:
    residual = np.zeros_like(X_target, dtype=np.float32)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X_base):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_base[train_idx], X_target[train_idx])
        residual[test_idx] = X_target[test_idx] - model.predict(X_base[test_idx])
    return residual


def text_cluster_probe_cv(
    X: np.ndarray,
    text_embedding: np.ndarray,
    n_clusters: int,
    folds: int,
    seed: int,
) -> dict[str, float]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(text_embedding)
    counts = np.bincount(labels)
    valid = counts[labels] >= folds
    X_valid = X[valid]
    y_valid = labels[valid]
    if len(set(y_valid.tolist())) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
    )
    acc = cross_val_score(clf, X_valid, y_valid, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X_valid, y_valid, cv=cv, scoring="f1_macro")
    return {"accuracy": float(np.mean(acc)), "macro_f1": float(np.mean(f1))}


def build_feature_sets(arrays: dict[str, np.ndarray], seed: int, folds: int, alpha: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1 = arrays["z1"]
    z2 = arrays["z2"]
    perm = rng.permutation(len(z2))
    z2_shuffled = z2[perm]
    z2_resid = residualize_cv(z1, z2, folds=folds, seed=seed, alpha=alpha)
    return {
        "context_hidden": arrays["context_hidden"],
        "z1": z1,
        "z2": z2,
        "z1_plus_z2": np.concatenate([z1, z2], axis=1),
        "z1_plus_shuffled_z2": np.concatenate([z1, z2_shuffled], axis=1),
        "z2_residual_given_z1": z2_resid,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(rows: list[dict[str, Any]], metric: str, out_path: Path) -> None:
    feature_order = ["context_hidden", "z1", "z2", "z1_plus_z2", "z1_plus_shuffled_z2", "z2_residual_given_z1"]
    target_order = ["mental1_text", "mental2_text", "reward_vec"]
    labels_feature = {
        "context_hidden": "Context hidden",
        "z1": "z1",
        "z2": "z2",
        "z1_plus_z2": "z1 + z2",
        "z1_plus_shuffled_z2": "z1 + shuffled z2",
        "z2_residual_given_z1": "z2 residual",
    }
    labels_target = {
        "mental1_text": "First-order\nmental text",
        "mental2_text": "Second-order\nmental text",
        "reward_vec": "Reward\nvector",
    }

    mat = np.full((len(feature_order), len(target_order)), np.nan)
    lookup = {(row["feature"], row["target"]): float(row[metric]) for row in rows if row["probe_type"] == "regression"}
    for i, feature in enumerate(feature_order):
        for j, target in enumerate(target_order):
            mat[i, j] = lookup.get((feature, target), np.nan)

    finite = mat[np.isfinite(mat)]
    if metric == "r2":
        vmin = min(0.0, float(finite.min())) if finite.size else 0.0
        vmax = max(0.05, float(finite.max())) if finite.size else 1.0
        cmap = "Blues"
    else:
        vmin = min(0.0, float(finite.min())) if finite.size else 0.0
        vmax = max(0.05, float(finite.max())) if finite.size else 1.0
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(target_order)), [labels_target[t] for t in target_order])
    ax.set_yticks(np.arange(len(feature_order)), [labels_feature[f] for f in feature_order])
    ax.set_title(f"Latent Probe Alignment ({metric})", fontsize=13, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_order_selectivity(rows: list[dict[str, Any]], metric: str, out_path: Path) -> None:
    lookup = {(row["feature"], row["target"]): float(row[metric]) for row in rows if row["probe_type"] == "regression"}
    bars = [
        ("z1 - z2\non first-order", lookup[("z1", "mental1_text")] - lookup[("z2", "mental1_text")], PLOT_GREEN),
        ("z2 - z1\non second-order", lookup[("z2", "mental2_text")] - lookup[("z1", "mental2_text")], PLOT_PURPLE),
        ("(z1+z2) - z1\non second-order", lookup[("z1_plus_z2", "mental2_text")] - lookup[("z1", "mental2_text")], PLOT_BLUE),
        ("(z1+z2) - shuffled\non reward", lookup[("z1_plus_z2", "reward_vec")] - lookup[("z1_plus_shuffled_z2", "reward_vec")], PLOT_RED),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(np.arange(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars])
    ax.axhline(0, color=PLOT_DARK_GREY, linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(bars)), [b[0] for b in bars], rotation=20, ha="right")
    ax.set_title("Order Selectivity And Recursive Value", fontsize=13, fontweight="bold")
    style_plot_axes(ax, ylabel=f"probe score difference ({metric})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_recursive_unique(rows: list[dict[str, Any]], metric: str, out_path: Path) -> None:
    lookup = {(row["feature"], row["target"]): float(row[metric]) for row in rows if row["probe_type"] == "regression"}
    features = ["z1", "z2", "z1_plus_z2", "z1_plus_shuffled_z2", "z2_residual_given_z1"]
    labels = ["z1", "z2", "z1+z2", "z1+shuffled z2", "z2 residual"]
    colors = [PLOT_GREEN, PLOT_PURPLE, PLOT_BLUE, PLOT_RED, PLOT_DARK_GREY]
    targets = ["mental2_text", "reward_vec"]
    titles = ["Second-Order Target", "Reward Target"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    for ax, target, title in zip(axes, targets, titles):
        values = [lookup.get((feature, target), np.nan) for feature in features]
        ax.bar(np.arange(len(features)), values, color=colors)
        ax.set_xticks(np.arange(len(features)), labels, rotation=25, ha="right")
        ax.set_title(title, fontsize=12, fontweight="bold")
        style_plot_axes(ax, ylabel=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cluster_f1_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    feature_order = ["context_hidden", "z1", "z2", "z1_plus_z2", "z1_plus_shuffled_z2", "z2_residual_given_z1"]
    target_order = ["mental1_clusters", "mental2_clusters"]
    lookup = {(row["feature"], row["target"]): float(row["macro_f1"]) for row in rows if row["probe_type"] == "cluster_classification"}
    mat = np.full((len(feature_order), len(target_order)), np.nan)
    for i, feature in enumerate(feature_order):
        for j, target in enumerate(target_order):
            mat[i, j] = lookup.get((feature, target), np.nan)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=max(0.05, float(np.nanmax(mat))))
    ax.set_xticks(np.arange(len(target_order)), ["First-order\nclusters", "Second-order\nclusters"])
    ax.set_yticks(np.arange(len(feature_order)), ["Context hidden", "z1", "z2", "z1 + z2", "z1 + shuffled z2", "z2 residual"])
    ax.set_title("Discrete Mental-Cluster Probe (macro-F1)", fontsize=13, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_probes(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    arrays = load_arrays(args.latent_arrays)
    records = load_records(args.records_jsonl)
    n = min(len(records), len(arrays["z1"]))
    records = records[:n]
    arrays = {key: value[:n] for key, value in arrays.items()}
    if args.max_examples and n > args.max_examples:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n, size=args.max_examples, replace=False))
        records = [records[i] for i in idx]
        arrays = {key: value[idx] for key, value in arrays.items()}

    mental1 = [r.get("mental1_text", "") for r in records]
    mental2 = [r.get("mental2_text", "") for r in records]
    mental1_target = fit_text_targets(mental1, args.text_components, args.seed)
    mental2_target = fit_text_targets(mental2, args.text_components, args.seed)
    reward_target = StandardScaler().fit_transform(arrays["reward_vec"]).astype(np.float32)
    features = build_feature_sets(arrays, seed=args.seed, folds=args.folds, alpha=args.ridge_alpha)

    targets = {
        "mental1_text": mental1_target,
        "mental2_text": mental2_target,
        "reward_vec": reward_target,
    }

    regression_rows = []
    for feature_name, X in features.items():
        for target_name, Y in targets.items():
            metrics = regression_probe_cv(X, Y, folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
            regression_rows.append({
                "probe_type": "regression",
                "feature": feature_name,
                "target": target_name,
                **metrics,
            })

    cluster_rows = []
    for feature_name, X in features.items():
        for target_name, Y in [("mental1_clusters", mental1_target), ("mental2_clusters", mental2_target)]:
            metrics = text_cluster_probe_cv(X, Y, n_clusters=args.num_clusters, folds=args.folds, seed=args.seed)
            cluster_rows.append({
                "probe_type": "cluster_classification",
                "feature": feature_name,
                "target": target_name,
                **metrics,
            })

    z2_from_z1 = regression_probe_cv(arrays["z1"], arrays["z2"], folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
    z1_from_z2 = regression_probe_cv(arrays["z2"], arrays["z1"], folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
    mediation = {
        "z2_from_z1_r2": z2_from_z1["r2"],
        "z2_from_z1_cosine": z2_from_z1["cosine"],
        "z1_from_z2_r2": z1_from_z2["r2"],
        "z1_from_z2_cosine": z1_from_z2["cosine"],
        "num_examples": len(records),
    }
    return regression_rows, cluster_rows, mediation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-arrays", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--text-components", type=int, default=64)
    parser.add_argument("--num-clusters", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regression_rows, cluster_rows, mediation = run_probes(args)
    all_rows = regression_rows + cluster_rows
    write_rows(out_dir / "recursive_decomposition_probe_scores.csv", all_rows)
    with (out_dir / "recursive_decomposition_mediation.json").open("w") as f:
        json.dump(mediation, f, indent=2, sort_keys=True)
    with (out_dir / "probe_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    plot_heatmap(regression_rows, "r2", out_dir / "latent_probe_r2_heatmap.png")
    plot_heatmap(regression_rows, "cosine", out_dir / "latent_probe_cosine_heatmap.png")
    plot_order_selectivity(regression_rows, "cosine", out_dir / "order_selectivity_cosine.png")
    plot_recursive_unique(regression_rows, "cosine", out_dir / "recursive_unique_contribution.png")
    plot_cluster_f1_heatmap(cluster_rows, out_dir / "mental_cluster_probe_f1_heatmap.png")
    print(f"Wrote recursive decomposition probes to {out_dir}")


if __name__ == "__main__":
    main()
