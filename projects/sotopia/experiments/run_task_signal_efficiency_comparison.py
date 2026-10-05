#!/usr/bin/env python3
"""Compare task-signal efficiency of mental latents vs compression latents.

This is intentionally different from "which latent contains more information?"
Compression latents can be very informative. The question here is whether a
simple, low-data probe can extract task-relevant social/reward signals more
easily from the mental-model latent space.

Tasks:
  - low-data classification of SOTOPIA heuristic social-action labels
  - low-data reward-vector regression
  - optional PCA bottlenecks to compare signal accessibility at small dims
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from probe_recursive_decomposition_empirical import mean_cosine


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_GREY = "#7A7A7A"
PLOT_DARK_GREY = "#4B4B4B"


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def make_probe(dim: int | None, task: str, seed: int, alpha: float):
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if dim is not None:
        steps.append(("pca", PCA(n_components=dim, random_state=seed)))
    if task == "classification":
        steps.append((
            "clf",
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
        ))
    else:
        steps.append(("ridge", Ridge(alpha=alpha)))
    from sklearn.pipeline import Pipeline

    return Pipeline(steps)


def filter_label(records: list[dict[str, Any]], key: str, min_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels = np.array([str(rec.get(key, "other")) for rec in records])
    values, counts = np.unique(labels, return_counts=True)
    keep_values = {v for v, c in zip(values, counts) if c >= min_count}
    mask = np.array([label in keep_values for label in labels])
    enc = LabelEncoder()
    y = enc.fit_transform(labels[mask])
    class_counts = {label: int(np.sum(labels[mask] == label)) for label in enc.classes_}
    return mask, y, class_counts


def sample_balanced_train(y_train_full: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray | None:
    idxs = []
    for cls in np.unique(y_train_full):
        cls_idx = np.where(y_train_full == cls)[0]
        if len(cls_idx) < per_class:
            return None
        idxs.extend(rng.choice(cls_idx, size=per_class, replace=False).tolist())
    return np.array(sorted(idxs), dtype=np.int64)


def classification_efficiency(
    X_by_model: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    label_key: str,
    per_class_values: list[int],
    dims: list[int | None],
    seeds: list[int],
    min_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mask, y, class_counts = filter_label(records, label_key, min_count)
    if len(np.unique(y)) < 2:
        return [], class_counts

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
        train_full_idx, test_idx = next(splitter.split(np.zeros_like(y), y))
        y_train_full = y[train_full_idx]
        y_test = y[test_idx]
        rng = np.random.default_rng(seed)

        for per_class in per_class_values:
            local_train = sample_balanced_train(y_train_full, per_class, rng)
            if local_train is None:
                continue
            train_idx = train_full_idx[local_train]
            for dim in dims:
                for model_name, X_all in X_by_model.items():
                    X = X_all[mask]
                    if dim is not None and dim >= len(train_idx):
                        continue
                    probe = make_probe(dim, "classification", seed, alpha=10.0)
                    probe.fit(X[train_idx], y[train_idx])
                    pred = probe.predict(X[test_idx])
                    rows.append({
                        "task_type": "classification",
                        "task": label_key,
                        "model": model_name,
                        "seed": seed,
                        "dim": "full" if dim is None else dim,
                        "train_per_class": per_class,
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(test_idx)),
                        "num_classes": int(len(np.unique(y))),
                        "macro_f1": float(f1_score(y_test, pred, average="macro")),
                        "accuracy": float(accuracy_score(y_test, pred)),
                    })
    return rows, class_counts


def regression_efficiency(
    X_by_model: dict[str, np.ndarray],
    Y: np.ndarray,
    train_sizes: list[int],
    dims: list[int | None],
    seeds: list[int],
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(Y)
    for seed in seeds:
        all_train, test_idx = train_test_split(np.arange(n), test_size=0.35, random_state=seed)
        rng = np.random.default_rng(seed)
        for train_size in train_sizes:
            if train_size > len(all_train):
                continue
            train_idx = np.array(sorted(rng.choice(all_train, size=train_size, replace=False)), dtype=np.int64)
            for dim in dims:
                if dim is not None and dim >= len(train_idx):
                    continue
                for model_name, X in X_by_model.items():
                    probe = make_probe(dim, "regression", seed, alpha=alpha)
                    probe.fit(X[train_idx], Y[train_idx])
                    pred = probe.predict(X[test_idx])
                    rows.append({
                        "task_type": "regression",
                        "task": "reward_vec",
                        "model": model_name,
                        "seed": seed,
                        "dim": "full" if dim is None else dim,
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(test_idx)),
                        "r2": float(r2_score(Y[test_idx], pred, multioutput="variance_weighted")),
                        "cosine": float(mean_cosine(Y[test_idx], pred)),
                    })
    return rows


def aggregate(rows: list[dict[str, Any]], metric: str, group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if metric not in row:
            continue
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(float(row[metric]))
    out = []
    for key, values in groups.items():
        rec = {k: v for k, v in zip(group_keys, key)}
        rec[f"{metric}_mean"] = float(np.mean(values))
        rec[f"{metric}_std"] = float(np.std(values))
        rec["n"] = len(values)
        out.append(rec)
    return out


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


def plot_classification_curves(rows: list[dict[str, Any]], out_path: Path, dim_value: str = "full") -> None:
    tasks = sorted({row["task"] for row in rows if row["dim"] == dim_value})
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    colors = {"mental_recursive": PLOT_BLUE, "compression_vae": PLOT_GREEN}
    labels = {"mental_recursive": "Mental recursive", "compression_vae": "Compression VAE"}
    for ax, task in zip(axes, tasks):
        for model in ["mental_recursive", "compression_vae"]:
            sub = [r for r in rows if r["task"] == task and r["model"] == model and r["dim"] == dim_value]
            by_budget: dict[int, list[float]] = {}
            for row in sub:
                by_budget.setdefault(int(row["train_per_class"]), []).append(float(row["macro_f1"]))
            xs = sorted(by_budget)
            ys = [np.mean(by_budget[x]) for x in xs]
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=colors[model], label=labels[model])
        ax.set_title(task.replace("_label", ""), fontsize=13, fontweight="bold")
        style_axes(ax, xlabel="train examples per class", ylabel="macro F1" if ax is axes[0] else None)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_reward_curve(rows: list[dict[str, Any]], out_path: Path, dim_value: str = "full") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"mental_recursive": PLOT_BLUE, "compression_vae": PLOT_GREEN}
    labels = {"mental_recursive": "Mental recursive", "compression_vae": "Compression VAE"}
    for model in ["mental_recursive", "compression_vae"]:
        sub = [r for r in rows if r["model"] == model and r["dim"] == dim_value]
        by_budget: dict[int, list[float]] = {}
        for row in sub:
            by_budget.setdefault(int(row["train_n"]), []).append(float(row["r2"]))
        xs = sorted(by_budget)
        ys = [np.mean(by_budget[x]) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.2, color=colors[model], label=labels[model])
    style_axes(ax, xlabel="train examples", ylabel="reward-vector R2")
    leg = ax.legend(frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D6D8DC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_auc_summary(class_rows: list[dict[str, Any]], reward_rows: list[dict[str, Any]], out_path: Path) -> None:
    values = []
    for model in ["mental_recursive", "compression_vae"]:
        cls_vals = [float(r["macro_f1"]) for r in class_rows if r["model"] == model and r["dim"] == "full"]
        rew_vals = [float(r["r2"]) for r in reward_rows if r["model"] == model and r["dim"] == "full"]
        values.append((model, "social-label macro F1", float(np.mean(cls_vals)) if cls_vals else float("nan")))
        values.append((model, "reward R2", float(np.mean(rew_vals)) if rew_vals else float("nan")))
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    labels = ["Social labels", "Reward"]
    x = np.arange(len(labels))
    width = 0.35
    lookup = {(m, t): v for m, t, v in values}
    ax.bar(x - width / 2, [lookup[("mental_recursive", "social-label macro F1")], lookup[("mental_recursive", "reward R2")]], width, color=PLOT_BLUE, label="Mental recursive")
    ax.bar(x + width / 2, [lookup[("compression_vae", "social-label macro F1")], lookup[("compression_vae", "reward R2")]], width, color=PLOT_GREEN, label="Compression VAE")
    ax.set_xticks(x, labels)
    style_axes(ax, xlabel="task family", ylabel="mean low-data probe score")
    leg = ax.legend(frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D6D8DC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_int_or_full(value: str) -> int | None:
    if value.lower() == "full":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mental-arrays", required=True)
    parser.add_argument("--compression-arrays", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-tasks", default="intent_label,strategy_label,interaction_label,knowledge_label")
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--per-class-values", default="4,8,16,32")
    parser.add_argument("--reward-train-sizes", default="64,128,256,512")
    parser.add_argument("--dims", default="16,32,64,full")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.records_jsonl)
    mental = load_npz(args.mental_arrays)
    compression = load_npz(args.compression_arrays)
    n = min(len(records), len(mental["z_concat"]), len(compression["z_compress"]))
    records = records[:n]
    X_by_model = {
        "mental_recursive": mental["z_concat"][:n],
        "compression_vae": compression["z_compress"][:n],
    }
    reward = StandardScaler().fit_transform(
        np.asarray([rec["reward_vec"] for rec in records], dtype=np.float32)
    ).astype(np.float32)
    label_tasks = [task.strip() for task in args.label_tasks.split(",") if task.strip()]
    per_class_values = [int(x) for x in args.per_class_values.split(",") if x.strip()]
    reward_train_sizes = [int(x) for x in args.reward_train_sizes.split(",") if x.strip()]
    dims = [parse_int_or_full(x.strip()) for x in args.dims.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    class_rows: list[dict[str, Any]] = []
    class_meta: dict[str, Any] = {}
    for task in label_tasks:
        rows, counts = classification_efficiency(
            X_by_model,
            records,
            label_key=task,
            per_class_values=per_class_values,
            dims=dims,
            seeds=seeds,
            min_count=args.min_class_count,
        )
        class_rows.extend(rows)
        class_meta[task] = counts

    reward_rows = regression_efficiency(
        X_by_model,
        reward,
        train_sizes=reward_train_sizes,
        dims=dims,
        seeds=seeds,
        alpha=args.ridge_alpha,
    )

    write_csv(out_dir / "task_signal_classification_scores.csv", class_rows)
    write_csv(out_dir / "task_signal_reward_scores.csv", reward_rows)
    write_csv(
        out_dir / "task_signal_classification_aggregate.csv",
        aggregate(class_rows, "macro_f1", ["task", "model", "dim", "train_per_class"]),
    )
    write_csv(
        out_dir / "task_signal_reward_aggregate.csv",
        aggregate(reward_rows, "r2", ["task", "model", "dim", "train_n"]),
    )

    plot_classification_curves(class_rows, out_dir / "paper_task_signal_classification_lowdata_full.png", dim_value="full")
    plot_classification_curves(class_rows, out_dir / "paper_task_signal_classification_lowdata_dim32.png", dim_value=32)
    plot_reward_curve(reward_rows, out_dir / "paper_task_signal_reward_lowdata_full.png", dim_value="full")
    plot_reward_curve(reward_rows, out_dir / "paper_task_signal_reward_lowdata_dim32.png", dim_value=32)
    plot_auc_summary(class_rows, reward_rows, out_dir / "paper_task_signal_summary.png")

    summary = {
        "n": n,
        "class_meta": class_meta,
        "config": vars(args),
        "interpretation": (
            "Compares task-signal accessibility under simple low-data probes; "
            "not a test of total information content."
        ),
    }
    with (out_dir / "task_signal_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote task signal efficiency comparison to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
