#!/usr/bin/env python3
"""Sweep reward-model checkpoints for recursive z2 pairing probe strength.

This is a focused companion to analyse_tom_latents.py. It extracts only the
latent arrays needed for the recursive-pairing test, then evaluates whether
the true (z1, z2) pairing improves mental-state probing over shuffled z2.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

SOTOPIA_DIR = Path(__file__).resolve().parents[1]
if str(SOTOPIA_DIR) not in sys.path:
    sys.path.insert(0, str(SOTOPIA_DIR))

from analyse_tom_latents import (  # noqa: E402
    RESPONSE_HEURISTIC_VERSION,
    load_or_extract_latents,
    load_records_jsonl,
    make_records_signature,
    save_records_jsonl,
)
from probe_recursive_decomposition_empirical import (  # noqa: E402
    fit_text_targets,
    regression_probe_cv,
)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def latent_meta(args: argparse.Namespace, checkpoint_dir: Path, records_path: Path, num_records: int, signature: str) -> dict[str, Any]:
    return {
        "balanced_subset_enabled": False,
        "balanced_subset_field": "interaction",
        "balanced_subset_keep_other": False,
        "balanced_subset_min_label_count": 40,
        "balanced_subset_per_label": 200,
        "base_model_name": args.base_model_name,
        "cache_type": "latents",
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "data_path": os.path.abspath(args.data_path),
        "label_jsonl": None,
        "label_source": "response_heuristic",
        "max_ctx_len": args.max_ctx_len,
        "num_records": num_records,
        "records_path": os.path.abspath(records_path),
        "records_signature": signature,
        "response_heuristic_version": RESPONSE_HEURISTIC_VERSION,
        "z_dim": args.z_dim,
    }


def ensure_epoch_latents(args: argparse.Namespace, epoch: int, records: list[Any], signature: str) -> Path:
    checkpoint_dir = Path(args.checkpoint_root) / f"epoch_{epoch}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    out_dir = Path(args.latent_output_root) / args.latent_name_template.format(epoch=epoch)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "analysis_records_subset.jsonl"
    save_records_jsonl(records, records_path)

    cache_path = out_dir / "latent_arrays.npz"
    meta_path = out_dir / "latent_arrays_meta.json"
    extract_args = SimpleNamespace(
        base_model_name=args.base_model_name,
        checkpoint_dir=str(checkpoint_dir),
        data_path=args.data_path,
        label_source="response_heuristic",
        label_jsonl=None,
        z_dim=args.z_dim,
        max_ctx_len=args.max_ctx_len,
        use_full_records=True,
        balanced_subset_field="interaction",
        balanced_subset_per_label=200,
        balanced_subset_min_label_count=40,
        balanced_subset_keep_other=False,
        recompute_latents=args.recompute_latents,
        batch_size=args.batch_size,
        device=args.device,
        ensemble_weight=args.ensemble_weight,
    )
    meta = latent_meta(args, checkpoint_dir, records_path, len(records), signature)
    load_or_extract_latents(extract_args, records, cache_path, meta_path, meta)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out_dir


def pairing_probe(
    arrays: dict[str, np.ndarray],
    mental2_target: np.ndarray,
    reward_target: np.ndarray,
    *,
    folds: int,
    seed: int,
    ridge_alpha: float,
    permutations: int,
) -> dict[str, Any]:
    z1 = arrays["z1"].astype(np.float32)
    z2 = arrays["z2"].astype(np.float32)
    n = len(z1)
    paired_x = np.concatenate([z1, z2], axis=1)
    paired = {
        "mental2_text": regression_probe_cv(paired_x, mental2_target, folds=folds, seed=seed, alpha=ridge_alpha),
        "reward_vec": regression_probe_cv(paired_x, reward_target, folds=folds, seed=seed, alpha=ridge_alpha),
    }

    rng = np.random.default_rng(seed + 1000)
    null_rows: list[dict[str, Any]] = []
    for perm_idx in range(permutations):
        perm = rng.permutation(n)
        shuffled_x = np.concatenate([z1, z2[perm]], axis=1)
        for target_name, target in [("mental2_text", mental2_target), ("reward_vec", reward_target)]:
            metrics = regression_probe_cv(shuffled_x, target, folds=folds, seed=seed, alpha=ridge_alpha)
            null_rows.append({"perm": perm_idx, "target": target_name, **metrics})

    summary: dict[str, Any] = {"paired": paired, "null": {}, "num_permutations": permutations}
    for target_name in ["mental2_text", "reward_vec"]:
        for metric in ["r2", "cosine"]:
            vals = np.array([row[metric] for row in null_rows if row["target"] == target_name], dtype=np.float64)
            paired_score = float(paired[target_name][metric])
            null_mean = float(vals.mean())
            null_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            abs_gain = paired_score - null_mean
            rel_gain = abs_gain / max(abs(null_mean), 1e-12)
            empirical_p_ge = float((1 + np.sum(vals >= paired_score)) / (len(vals) + 1))
            summary["null"].setdefault(target_name, {})[metric] = {
                "mean": null_mean,
                "std": null_std,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "abs_gain": float(abs_gain),
                "rel_gain": float(rel_gain),
                "paired_score": paired_score,
                "z": float(abs_gain / max(null_std, 1e-12)),
                "empirical_p_ge": empirical_p_ge,
            }
    return {"summary": summary, "null_rows": null_rows}


def write_epoch_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "epoch",
            "target",
            "metric",
            "paired_score",
            "shuffle_mean",
            "shuffle_std",
            "abs_gain",
            "rel_gain_percent",
            "z",
            "empirical_p_ge",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict[str, Any]], out_path: Path) -> None:
    r2_rows = [row for row in rows if row["metric"] == "r2"]
    epochs = sorted({int(row["epoch"]) for row in r2_rows})
    by_key = {(int(row["epoch"]), row["target"]): row for row in r2_rows}
    mental_abs = np.array([float(by_key[(epoch, "mental2_text")]["abs_gain"]) for epoch in epochs])
    reward_abs = np.array([float(by_key[(epoch, "reward_vec")]["abs_gain"]) for epoch in epochs])
    mental_rel = np.array([float(by_key[(epoch, "mental2_text")]["rel_gain_percent"]) for epoch in epochs])
    reward_rel = np.array([float(by_key[(epoch, "reward_vec")]["rel_gain_percent"]) for epoch in epochs])

    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=240)
    x = np.arange(len(epochs))
    width = 0.36

    axes[0].bar(x - width / 2, mental_abs, width, color="#2f6fae", label="second-order mental")
    axes[0].bar(x + width / 2, reward_abs, width, color="#7d7d7d", label="reward control")
    axes[0].set_title("Absolute R2 Pairing Gain")
    axes[0].set_ylabel("true pairing - shuffle null")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(epoch) for epoch in epochs])
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].bar(x - width / 2, mental_rel, width, color="#2f6fae", label="second-order mental")
    axes[1].bar(x + width / 2, reward_rel, width, color="#7d7d7d", label="reward control")
    axes[1].set_title("Relative R2 Pairing Gain")
    axes[1].set_ylabel("gain over shuffle null (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(epoch) for epoch in epochs])
    axes[1].set_xlabel("checkpoint epoch")

    for ax in axes:
        ax.grid(axis="y", color="#d4d8dd", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    best_idx = int(np.argmax(mental_rel - reward_rel))
    axes[1].annotate(
        f"best: epoch {epochs[best_idx]}",
        xy=(best_idx - width / 2, mental_rel[best_idx]),
        xytext=(best_idx, max(mental_rel) * 1.12),
        ha="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
    )
    fig.suptitle("Recursive Pairing Strength Across Checkpoints", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint_root", default="projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3")
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument("--records_jsonl", default="projects/sotopia/runs/analysis/mental_latent_qwen_v3_epoch5/analysis_records_subset.jsonl")
    parser.add_argument("--latent_output_root", default="projects/sotopia")
    parser.add_argument("--latent_name_template", default="mental_latent_analysis_qwen_v3_epoch{epoch}")
    parser.add_argument("--output_dir", default="projects/sotopia/experiments/runs/visuals/recursive_pairing_epoch_sweep")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--ensemble_weight", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--text_components", type=int, default=64)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute_latents", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records_jsonl(Path(args.records_jsonl))
    signature = make_records_signature(records)
    mental2_target = fit_text_targets([record.mental2_text for record in records], args.text_components, args.seed)
    reward_target = StandardScaler().fit_transform(np.asarray([record.reward_vec for record in records], dtype=np.float32)).astype(np.float32)

    epoch_rows: list[dict[str, Any]] = []
    for epoch in args.epochs:
        print(f"[epoch {epoch}] extracting/loading latents", flush=True)
        latent_dir = ensure_epoch_latents(args, epoch, records, signature)
        arrays_npz = np.load(latent_dir / "latent_arrays.npz")
        arrays = {key: arrays_npz[key].astype(np.float32) for key in arrays_npz.files}
        print(f"[epoch {epoch}] running {args.permutations} shuffle probes", flush=True)
        result = pairing_probe(
            arrays,
            mental2_target,
            reward_target,
            folds=args.folds,
            seed=args.seed,
            ridge_alpha=args.ridge_alpha,
            permutations=args.permutations,
        )
        epoch_out = out_dir / f"epoch_{epoch}"
        epoch_out.mkdir(parents=True, exist_ok=True)
        save_json(epoch_out / "pairing_probe_summary.json", result["summary"])
        with (epoch_out / "pairing_probe_shuffle_scores.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["perm", "target", "cosine", "r2"])
            writer.writeheader()
            writer.writerows(result["null_rows"])

        for target in ["mental2_text", "reward_vec"]:
            for metric in ["r2", "cosine"]:
                item = result["summary"]["null"][target][metric]
                epoch_rows.append(
                    {
                        "epoch": epoch,
                        "target": target,
                        "metric": metric,
                        "paired_score": item["paired_score"],
                        "shuffle_mean": item["mean"],
                        "shuffle_std": item["std"],
                        "abs_gain": item["abs_gain"],
                        "rel_gain_percent": item["rel_gain"] * 100.0,
                        "z": item["z"],
                        "empirical_p_ge": item["empirical_p_ge"],
                    }
                )
        del arrays
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_epoch_rows(out_dir / "recursive_pairing_epoch_sweep_scores.csv", epoch_rows)
    plot_sweep(epoch_rows, out_dir / "recursive_pairing_epoch_sweep_r2.png")
    save_json(out_dir / "sweep_config.json", vars(args))
    print(f"Wrote epoch sweep to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
