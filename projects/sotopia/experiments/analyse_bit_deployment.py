#!/usr/bin/env python3
"""Deployment-relevance tests for BIT factorization.

These three tests answer the question: "Does the geometric factorization
(Strategy 8) translate into something a deployment team can actually use?"

  Test A — Sub-block sufficiency:
        Can a single 48-dim sub-block alone predict reward as well as the
        full 128-dim z1?  If BIT factorizes useful information per sub-block,
        z_belief alone should ≈ full z1 (compression).  flat should require
        all dimensions and degrade much more when restricted to a subset.

  Test B — Selective control via perturbation:
        Perturb each sub-block along its first principal direction; measure
        which reward dimensions move the most.  BIT should show *concentrated*
        responses (perturbing z_belief moves a small subset of reward dims a
        lot).  flat should show *diffuse* responses (uniform spread).

  Test C — Coefficient-localization interpretability:
        Fit ridge from z1 → each individual reward dimension; look at how
        the coefficient mass distributes across sub-blocks.  BIT should have
        lower per-dim entropy (each reward dim relies on one sub-block); flat
        should be near-uniform (each dim relies on the whole z).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    RecursiveToMDataset, RecursiveToMModel, SOTOPIA_DIMENSIONS,
)
from transformers import AutoTokenizer  # noqa: E402

Z_BELIEF_DIM = RecursiveToMModel.Z_BELIEF_DIM     # 48
Z_INTENT_DIM = RecursiveToMModel.Z_INTENT_DIM     # 40
Z_THOUGHT_DIM = RecursiveToMModel.Z_THOUGHT_DIM   # 40

SUB_SLICES = {
    "belief":  slice(0, Z_BELIEF_DIM),
    "intent":  slice(Z_BELIEF_DIM, Z_BELIEF_DIM + Z_INTENT_DIM),
    "thought": slice(Z_BELIEF_DIM + Z_INTENT_DIM,
                     Z_BELIEF_DIM + Z_INTENT_DIM + Z_THOUGHT_DIM),
}
SUB_NAMES = list(SUB_SLICES.keys())
SUB_DIMS = [Z_BELIEF_DIM, Z_INTENT_DIM, Z_THOUGHT_DIM]

C_BIT      = "#3A5F87"
C_FLAT     = "#CC8A4C"
C_SHUFFLED = "#5DA39A"
VARIANT_COLORS = {
    "structured_bit":      C_BIT,
    "flat_mental_summary": C_FLAT,
    "shuffled_mental":     C_SHUFFLED,
}
VARIANT_LABELS = {
    "structured_bit":      "BIT (ours)",
    "flat_mental_summary": "flat summary",
    "shuffled_mental":     "shuffled (control)",
}


# ── Data ─────────────────────────────────────────────────────────────────────
def load_val_records(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = RecursiveToMDataset(
        args.data_path, tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
    )
    val_size = min(max(1, int(len(dataset) * args.val_ratio)), len(dataset) - 1)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
    records = []
    groups = []
    for i in val_dataset.indices:
        s = dataset.samples[i]
        records.append({"reward_vec": s["reward_vec"]})
        groups.append(s.get("scenario", str(i)))
    return records, np.asarray(groups)


def load_latents(latents_dir: Path, variants: list[str]):
    out = {}
    for v in variants:
        path = latents_dir / f"latents_{v}.npz"
        d = np.load(path)
        out[v] = {k: d[k].astype(np.float32) for k in d.files}
    return out


def cv_ridge_per_dim(X, Y, groups, alpha=10.0, n_splits=5):
    """Group-CV ridge fitting one regressor per output dim.  Returns per-dim
    mean R² and the *full-data* coefficient matrix (Y.shape[1] × X.shape[1])."""
    n_unique = len(np.unique(groups))
    n_splits = min(n_splits, n_unique)
    splits = list(GroupKFold(n_splits=n_splits).split(X, Y, groups=groups))
    n_dim = Y.shape[1]
    r2 = np.zeros(n_dim)
    for tr, te in splits:
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        for d in range(n_dim):
            clf = Ridge(alpha=alpha).fit(Xtr, Y[tr, d])
            r2[d] += r2_score(Y[te, d], clf.predict(Xte))
    r2 /= n_splits
    # full-data coefficients (after scaling) for coefficient-spectrum analyses
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    coefs = np.stack([Ridge(alpha=alpha).fit(Xs, Y[:, d]).coef_ for d in range(n_dim)], axis=0)
    return r2.astype(np.float32), coefs.astype(np.float32)


# ── Test A — Sub-block sufficiency ───────────────────────────────────────────
def test_a(latents, reward_vec, groups):
    out = {}
    for variant, arrays in latents.items():
        z1 = arrays["z1"]
        result = {}
        full_r2, _ = cv_ridge_per_dim(z1, reward_vec, groups)
        result["full_z1"] = {"r2_per_dim": full_r2, "r2_mean": float(full_r2.mean())}
        for sname, sl in SUB_SLICES.items():
            sub_r2, _ = cv_ridge_per_dim(z1[:, sl], reward_vec, groups)
            result[sname] = {
                "r2_per_dim": sub_r2,
                "r2_mean": float(sub_r2.mean()),
                "ratio_to_full": float(sub_r2.mean() / max(full_r2.mean(), 1e-8)),
                "n_dim": int(sl.stop - sl.start),
            }
        out[variant] = result
        print(f"  [test A] {variant}: full R² = {result['full_z1']['r2_mean']:+.3f}, "
              f"belief alone = {result['belief']['r2_mean']:+.3f} "
              f"({result['belief']['ratio_to_full']:.2f}× full)", flush=True)
    return out


# ── Test B — Selective control via PCA-direction perturbation ────────────────
def test_b(latents, reward_vec, groups, alpha_perturb=2.0):
    """For each variant: train per-dim ridge on full-z1.  Find each sub-block's
    first principal direction (within the sub-block).  Perturb z by α·u within
    that sub-block, predict reward shift per dim.  Result: 3×7 matrix of |Δ
    pred| per (perturbed sub-block, reward dim).  Selectivity index = entropy
    of the row (lower = more concentrated = more selective)."""
    out = {}
    for variant, arrays in latents.items():
        z1 = arrays["z1"]
        scaler = StandardScaler().fit(z1)
        Xs = scaler.transform(z1)
        # Per-dim ridge models on standardized z
        models = [Ridge(alpha=10.0).fit(Xs, reward_vec[:, d]) for d in range(reward_vec.shape[1])]
        baseline = np.stack([m.predict(Xs) for m in models], axis=1)  # n × 7

        delta_matrix = np.zeros((len(SUB_NAMES), reward_vec.shape[1]), dtype=np.float32)
        for si, (sname, sl) in enumerate(SUB_SLICES.items()):
            # principal direction within this sub-block (in standardized space)
            sub = Xs[:, sl]
            sub_centered = sub - sub.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(sub_centered, full_matrices=False)
            principal = Vt[0]                          # (sub_dim,)
            perturb = np.zeros(z1.shape[1], dtype=np.float32)
            perturb[sl] = principal * alpha_perturb    # in standardized space
            X_pert = Xs + perturb[None, :]
            y_pert = np.stack([m.predict(X_pert) for m in models], axis=1)
            delta = np.abs(y_pert - baseline).mean(0)  # mean |Δpred| per dim
            delta_matrix[si] = delta

        # selectivity per row: lower entropy of normalized deltas = more concentrated
        eps = 1e-9
        row_probs = delta_matrix / (delta_matrix.sum(1, keepdims=True) + eps)
        row_entropy = -(row_probs * np.log(row_probs + eps)).sum(1)
        max_entropy = float(np.log(reward_vec.shape[1]))
        normalized_entropy = row_entropy / max_entropy   # 1 = uniform spread
        out[variant] = {
            "delta_matrix": delta_matrix,
            "row_entropy_normalized": normalized_entropy.astype(np.float32),
            "selectivity_index": float(1.0 - normalized_entropy.mean()),
        }
        print(f"  [test B] {variant}: selectivity index = "
              f"{out[variant]['selectivity_index']:.4f}", flush=True)
    return out


# ── Test C — Coefficient-localization interpretability ───────────────────────
def test_c(latents, reward_vec, groups):
    """For each variant: ridge fit z1 → reward_dim_i.  Per dim, compute
    L2-norm of coefficients restricted to each sub-block, normalize → 3-vector
    distribution.  Entropy of this distribution measures "diffuseness" of
    reward-dim's reliance on sub-blocks.  Lower entropy = better localized
    (one sub-block carries the dim) = more interpretable."""
    out = {}
    for variant, arrays in latents.items():
        z1 = arrays["z1"]
        _, coefs = cv_ridge_per_dim(z1, reward_vec, groups)
        # coefs: (n_dim × 128)
        per_dim_dist = np.zeros((coefs.shape[0], len(SUB_NAMES)), dtype=np.float32)
        for d in range(coefs.shape[0]):
            for si, sl in enumerate(SUB_SLICES.values()):
                per_dim_dist[d, si] = np.linalg.norm(coefs[d, sl])
        per_dim_dist = per_dim_dist / (per_dim_dist.sum(1, keepdims=True) + 1e-9)
        eps = 1e-9
        per_dim_entropy = -(per_dim_dist * np.log(per_dim_dist + eps)).sum(1)
        max_entropy = float(np.log(len(SUB_NAMES)))
        normalized_entropy = per_dim_entropy / max_entropy
        out[variant] = {
            "per_dim_distribution": per_dim_dist,         # n_dim × 3
            "per_dim_entropy_normalized": normalized_entropy.astype(np.float32),
            "localization_index": float(1.0 - normalized_entropy.mean()),
        }
        print(f"  [test C] {variant}: localization index = "
              f"{out[variant]['localization_index']:.4f}", flush=True)
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 240,
    })


def plot_test_a(res_a, out_path):
    set_style()
    variants = list(res_a.keys())
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.21
    groups = ["full z1\n(128 dim)", "belief\n(48 dim)", "intent\n(40 dim)", "thought\n(40 dim)"]
    x = np.arange(len(groups))
    for i, v in enumerate(variants):
        r = res_a[v]
        vals = [r["full_z1"]["r2_mean"], r["belief"]["r2_mean"],
                r["intent"]["r2_mean"], r["thought"]["r2_mean"]]
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=VARIANT_COLORS.get(v, "#888"),
                      edgecolor="#1F1F1F", linewidth=1.0,
                      label=VARIANT_LABELS.get(v, v))
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    val + 0.005 if val >= 0 else val - 0.018,
                    f"{val:.2f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(r"mean held-out $R^2$  (across 7 reward dims)", fontweight="bold")
    ax.set_title("Test A — Sub-block sufficiency: can a single sub-block alone predict reward?",
                 fontsize=11.5)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", which="major", color="#D6DADE", linewidth=0.9, alpha=0.85, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#888")
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_test_b(res_b, out_path):
    set_style()
    variants = list(res_b.keys())
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]
    dim_short = [d.replace("financial_and_material_benefits", "financial")
                 .replace("believability", "believ.")
                 .replace("relationship", "relation.")
                 .replace("social_rules", "social")
                 .replace("knowledge", "knowl.")
                 for d in SOTOPIA_DIMENSIONS]
    vmax = float(max(np.max(res_b[v]["delta_matrix"]) for v in variants))
    for ax, v in zip(axes, variants):
        mat = res_b[v]["delta_matrix"]
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        for si in range(len(SUB_NAMES)):
            for di in range(len(SOTOPIA_DIMENSIONS)):
                col = "white" if mat[si, di] < vmax * 0.55 else "#1A1A1A"
                ax.text(di, si, f"{mat[si, di]:.2f}", ha="center", va="center",
                        fontsize=8.5, color=col)
        ax.set_xticks(range(len(SOTOPIA_DIMENSIONS)))
        ax.set_xticklabels(dim_short, rotation=40, ha="right")
        ax.set_yticks(range(len(SUB_NAMES)))
        ax.set_yticklabels([f"perturb z1_{s}" for s in SUB_NAMES], fontweight="bold")
        ax.set_title(f"{VARIANT_LABELS.get(v, v)}\nselectivity: {res_b[v]['selectivity_index']:.3f}",
                     fontsize=11.5)
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"|Δ predicted reward|")
    fig.suptitle("Test B — Selective control via principal-direction perturbation",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_test_c(res_c, out_path):
    set_style()
    variants = list(res_c.keys())
    fig, axes = plt.subplots(1, len(variants) + 1,
                             figsize=(4.0 * (len(variants) + 1), 4.4),
                             gridspec_kw={"width_ratios": [1] * len(variants) + [0.7]},
                             constrained_layout=True)
    dim_short = [d.replace("financial_and_material_benefits", "financial")
                 .replace("believability", "believ.")
                 .replace("relationship", "relation.")
                 .replace("social_rules", "social")
                 .replace("knowledge", "knowl.")
                 for d in SOTOPIA_DIMENSIONS]
    for ax, v in zip(axes[:-1], variants):
        dist = res_c[v]["per_dim_distribution"]   # n_dim × 3
        bottoms = np.zeros(dist.shape[0])
        cluster_palette = ["#1F5582", "#3A7CA5", "#5FA8D3"]
        for si, sname in enumerate(SUB_NAMES):
            ax.bar(range(dist.shape[0]), dist[:, si], bottom=bottoms,
                   color=cluster_palette[si], edgecolor="white", linewidth=0.5,
                   label=f"z1_{sname}")
            bottoms += dist[:, si]
        ax.set_xticks(range(dist.shape[0]))
        ax.set_xticklabels(dim_short, rotation=40, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("coef-norm share", fontweight="bold")
        ax.set_title(f"{VARIANT_LABELS.get(v, v)}\nlocalization: {res_c[v]['localization_index']:.3f}",
                     fontsize=11.5)
        if v == variants[0]:
            leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                            ncol=3, frameon=False, fontsize=9)

    # Summary bar of localization indices
    ax = axes[-1]
    indices = [res_c[v]["localization_index"] for v in variants]
    bars = ax.bar(range(len(variants)), indices,
                  color=[VARIANT_COLORS.get(v, "#888") for v in variants],
                  edgecolor="#1F1F1F", linewidth=1.0)
    for rect, val in zip(bars, indices):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=15)
    ax.set_ylabel("localization index", fontweight="bold")
    ax.set_title("Higher = more interpretable", fontsize=11)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", which="major", color="#D6DADE", linewidth=0.9, alpha=0.85, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Test C — Coefficient-localization (per reward-dim ridge weights)",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(out_dir, res_a, res_b, res_c):
    rows = []
    variants = list(res_a.keys())
    for v in variants:
        rows.append({
            "variant": v,
            "test_A_full_z1_R2": res_a[v]["full_z1"]["r2_mean"],
            "test_A_belief_alone_R2": res_a[v]["belief"]["r2_mean"],
            "test_A_belief_ratio_to_full": res_a[v]["belief"]["ratio_to_full"],
            "test_A_intent_alone_R2": res_a[v]["intent"]["r2_mean"],
            "test_A_thought_alone_R2": res_a[v]["thought"]["r2_mean"],
            "test_B_selectivity_index": res_b[v]["selectivity_index"],
            "test_C_localization_index": res_c[v]["localization_index"],
        })
    md = ["# BIT deployment-relevance summary", "",
          "| variant | A: full R² | A: belief alone | A: belief / full | "
          "B: selectivity ↑ | C: localization ↑ |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['variant']} | "
            f"{r['test_A_full_z1_R2']:+.3f} | "
            f"{r['test_A_belief_alone_R2']:+.3f} | "
            f"{r['test_A_belief_ratio_to_full']:.2f}× | "
            f"{r['test_B_selectivity_index']:.4f} | "
            f"{r['test_C_localization_index']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md))
    with (out_dir / "summary.json").open("w") as f:
        json.dump({
            "rows": rows,
            "test_a": {v: {k: (val.tolist() if hasattr(val, 'tolist') else val)
                            for k, val in res_a[v].items()
                            if not isinstance(val, dict)} | {
                                kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                                for sub_k in res_a[v]
                                if isinstance(res_a[v][sub_k], dict)
                                for kk, vv in res_a[v][sub_k].items()
                            }
                       for v in variants},
            "test_b": {v: {"delta_matrix": res_b[v]["delta_matrix"].tolist(),
                            "row_entropy_normalized": res_b[v]["row_entropy_normalized"].tolist(),
                            "selectivity_index": res_b[v]["selectivity_index"]}
                       for v in variants},
            "test_c": {v: {"per_dim_distribution": res_c[v]["per_dim_distribution"].tolist(),
                            "per_dim_entropy_normalized": res_c[v]["per_dim_entropy_normalized"].tolist(),
                            "localization_index": res_c[v]["localization_index"]}
                       for v in variants},
        }, f, indent=2)
    print(f"wrote {out_dir / 'summary.md'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents_dir",
                   default="projects/sotopia/experiments/runs/stage1/variant_eval")
    p.add_argument("--output_dir",
                   default="projects/sotopia/experiments/runs/stage1/bit_deployment")
    p.add_argument("--variants", nargs="+",
                   default=["structured_bit", "flat_mental_summary", "shuffled_mental"])
    p.add_argument("--data_path",
                   default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_ctx_len", type=int, default=1024)
    p.add_argument("--max_resp_len", type=int, default=256)
    p.add_argument("--max_mental_len", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha_perturb", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Re-deriving val records …", flush=True)
    records, groups = load_val_records(args)
    reward_vec = np.asarray([r["reward_vec"] for r in records], dtype=np.float32)
    print(f"  val records: {len(records)}, scenarios: {len(np.unique(groups))}")

    print("Loading cached latents …", flush=True)
    latents = load_latents(Path(args.latents_dir), args.variants)
    n_lat = next(iter(latents.values()))["z1"].shape[0]
    if n_lat != len(records):
        raise RuntimeError(f"latent count {n_lat} != val record count {len(records)}")

    print("\n=== Test A: sub-block sufficiency ===", flush=True)
    res_a = test_a(latents, reward_vec, groups)
    plot_test_a(res_a, out_dir / "fig_test_a_sufficiency.png")

    print("\n=== Test B: selective control via perturbation ===", flush=True)
    res_b = test_b(latents, reward_vec, groups, alpha_perturb=args.alpha_perturb)
    plot_test_b(res_b, out_dir / "fig_test_b_selectivity.png")

    print("\n=== Test C: coefficient-localization interpretability ===", flush=True)
    res_c = test_c(latents, reward_vec, groups)
    plot_test_c(res_c, out_dir / "fig_test_c_localization.png")

    write_summary(out_dir, res_a, res_b, res_c)


if __name__ == "__main__":
    main()
