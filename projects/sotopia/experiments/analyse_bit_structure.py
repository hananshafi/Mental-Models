#!/usr/bin/env python3
"""Decoder-free structural analysis of BIT sub-blocks.

Three strategies, all using cached val-set latents from variant_eval/:

  Strategy 1b — Sub-block × reward-dim ridge probe (3×7 R² matrix per variant).
                Tests whether each BIT sub-block selectively predicts its
                semantic-cluster reward dimensions. flat z should show no
                preference; BIT z should diagonalize across (B/I/T sub-block,
                aligned reward cluster).

  Strategy 6  — Sub-block ablation effect on a fresh z→reward ridge probe.
                Train one ridge on full z1, then evaluate with each sub-block
                zeroed.  ΔR² per ablated sub-block tells us which dimensions
                the head is causally relying on.  BIT: differential ablation
                effect by sub-block; flat: uniform.

  Strategy 8  — Linear CKA between sub-block pairs (B-I, B-T, I-T).  Lower
                CKA = more factorized representation.  BIT should be lower
                (sub-blocks encode independent things); flat should be
                higher (no factorial pressure).

All probes use scenario-grouped 5-fold CV to prevent leakage.
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

# BIT sub-block dims (from RecursiveToMModel constants)
Z_BELIEF_DIM = RecursiveToMModel.Z_BELIEF_DIM   # 48
Z_INTENT_DIM = RecursiveToMModel.Z_INTENT_DIM   # 40
Z_THOUGHT_DIM = RecursiveToMModel.Z_THOUGHT_DIM # 40

SUB_SLICES = {
    "belief":  slice(0, Z_BELIEF_DIM),
    "intent":  slice(Z_BELIEF_DIM, Z_BELIEF_DIM + Z_INTENT_DIM),
    "thought": slice(Z_BELIEF_DIM + Z_INTENT_DIM,
                     Z_BELIEF_DIM + Z_INTENT_DIM + Z_THOUGHT_DIM),
}

# Semantic cluster mapping of SOTOPIA reward dimensions
REWARD_CLUSTERS = {
    "believability":                   "belief",
    "knowledge":                       "belief",
    "secret":                          "belief",
    "goal":                            "intent",
    "financial_and_material_benefits": "intent",
    "social_rules":                    "thought",
    "relationship":                    "thought",
}


# ── Data loading ─────────────────────────────────────────────────────────────
def load_val_records(args) -> tuple[list[dict], np.ndarray]:
    """Re-derive the val split records (reward_vec + scenario_id) using the
    same dataset + random_split + seed used in eval_mental_variants.py."""
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
        records.append({"reward_vec": s["reward_vec"], "scenario": s.get("scenario", "")})
        groups.append(s.get("scenario", str(i)))
    return records, np.asarray(groups)


def load_latents(latents_dir: Path, variants: list[str]) -> dict[str, dict]:
    """Returns {variant_name: {z1, z2, z_concat, context_hidden}} arrays."""
    out = {}
    for v in variants:
        path = latents_dir / f"latents_{v}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        d = np.load(path)
        out[v] = {k: d[k].astype(np.float32) for k in d.files}
    return out


# ── Strategy 1b: sub-block × reward-dim ridge probe ──────────────────────────
def ridge_cv_r2(X, y, groups, n_splits=5, alpha=10.0, seed=42) -> float:
    """Group-aware 5-fold ridge CV. Returns mean R² across folds."""
    n_unique = len(np.unique(groups))
    n_splits = min(n_splits, n_unique)
    if n_splits < 2:
        return float("nan")
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    r2s = []
    for tr, te in splits:
        clf = Ridge(alpha=alpha, random_state=seed)
        scaler = StandardScaler().fit(X[tr])
        clf.fit(scaler.transform(X[tr]), y[tr])
        y_hat = clf.predict(scaler.transform(X[te]))
        r2s.append(r2_score(y[te], y_hat))
    return float(np.mean(r2s))


def strategy_1b(latents: dict[str, dict], reward_vec: np.ndarray,
                groups: np.ndarray) -> dict[str, np.ndarray]:
    """For each variant: returns a (3, 7) R² matrix indexed by
       (sub-block, reward dimension)."""
    sub_names = list(SUB_SLICES.keys())
    out = {}
    for variant, arrays in latents.items():
        z1 = arrays["z1"]
        mat = np.zeros((len(sub_names), len(SOTOPIA_DIMENSIONS)), dtype=np.float32)
        for si, sname in enumerate(sub_names):
            X = z1[:, SUB_SLICES[sname]]
            for di, dname in enumerate(SOTOPIA_DIMENSIONS):
                y = reward_vec[:, di]
                mat[si, di] = ridge_cv_r2(X, y, groups)
        out[variant] = mat
        print(f"  [strategy 1b] {variant} done", flush=True)
    return out


def alignment_score(mat: np.ndarray) -> float:
    """Mean R² on semantically-aligned (sub-block, reward-dim) cells minus
    mean R² on off-cluster cells."""
    sub_names = list(SUB_SLICES.keys())
    aligned, off = [], []
    for si, sname in enumerate(sub_names):
        for di, dname in enumerate(SOTOPIA_DIMENSIONS):
            cluster = REWARD_CLUSTERS[dname]
            (aligned if cluster == sname else off).append(float(mat[si, di]))
    return float(np.mean(aligned) - np.mean(off))


# ── Strategy 6: sub-block ablation effect on full-z reward probe ─────────────
def strategy_6(latents: dict[str, dict], reward_vec: np.ndarray,
               groups: np.ndarray) -> dict[str, dict]:
    """For each variant: train ridge on full z1 → reward_vec.  Evaluate
    baseline R² (per dim, mean across folds), then ablate each sub-block
    (zero out at test time) and recompute ΔR²."""
    out = {}
    for variant, arrays in latents.items():
        z1 = arrays["z1"]
        n_unique = len(np.unique(groups))
        n_splits = min(5, n_unique)
        splits = list(GroupKFold(n_splits=n_splits).split(z1, reward_vec, groups=groups))

        baseline_per_dim = np.zeros(len(SOTOPIA_DIMENSIONS), dtype=np.float32)
        ablated = {sname: np.zeros(len(SOTOPIA_DIMENSIONS), dtype=np.float32)
                   for sname in SUB_SLICES}
        for tr, te in splits:
            scaler = StandardScaler().fit(z1[tr])
            X_tr = scaler.transform(z1[tr])
            X_te = scaler.transform(z1[te])
            for di in range(len(SOTOPIA_DIMENSIONS)):
                clf = Ridge(alpha=10.0).fit(X_tr, reward_vec[tr, di])
                baseline_per_dim[di] += r2_score(reward_vec[te, di], clf.predict(X_te))
                for sname, sl in SUB_SLICES.items():
                    X_te_abl = X_te.copy()
                    X_te_abl[:, sl] = 0.0
                    ablated[sname][di] += r2_score(reward_vec[te, di], clf.predict(X_te_abl))
        baseline_per_dim /= n_splits
        for sname in SUB_SLICES:
            ablated[sname] /= n_splits

        # ΔR² = baseline - ablated (positive = sub-block was contributing)
        delta = {s: (baseline_per_dim - ablated[s]).astype(np.float32) for s in SUB_SLICES}
        out[variant] = {"baseline": baseline_per_dim, "ablated": ablated, "delta": delta}
        print(f"  [strategy 6] {variant} done", flush=True)
    return out


def causal_alignment_score(delta: dict) -> float:
    """Mean ΔR² on semantically-aligned cells minus mean ΔR² on off-cluster
    cells.  Positive = sub-block ablation hurts its own cluster more than
    other clusters (causal specialization)."""
    aligned, off = [], []
    for sname, delta_vec in delta.items():
        for di, dname in enumerate(SOTOPIA_DIMENSIONS):
            cluster = REWARD_CLUSTERS[dname]
            (aligned if cluster == sname else off).append(float(delta_vec[di]))
    return float(np.mean(aligned) - np.mean(off))


# ── Strategy 8: linear CKA between sub-block pairs ───────────────────────────
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered kernel alignment.  Range [0, 1]; 0 = orthogonal."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / (den + 1e-12))


def strategy_8(latents: dict[str, dict]) -> dict[str, dict]:
    """For each variant: pairwise CKA between z1 sub-blocks."""
    out = {}
    sub_names = list(SUB_SLICES.keys())
    for variant, arrays in latents.items():
        z1 = arrays["z1"]
        cka = {}
        for i, a in enumerate(sub_names):
            for j, b in enumerate(sub_names):
                if i < j:
                    cka[f"{a}-{b}"] = linear_cka(z1[:, SUB_SLICES[a]],
                                                 z1[:, SUB_SLICES[b]])
        cka["mean_off_diag"] = float(np.mean(list(cka.values())))
        out[variant] = cka
        print(f"  [strategy 8] {variant} done", flush=True)
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────
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
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 240,
    })


def plot_strategy_1b(s1b: dict, out_path: Path):
    set_style()
    variants = list(s1b.keys())
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]

    sub_names = list(SUB_SLICES.keys())
    dim_short = [d.replace("financial_and_material_benefits", "financial")
                 .replace("believability", "believ.")
                 .replace("relationship", "relation.")
                 .replace("social_rules", "social")
                 .replace("knowledge", "knowl.")
                 for d in SOTOPIA_DIMENSIONS]
    cluster_for_dim = [REWARD_CLUSTERS[d] for d in SOTOPIA_DIMENSIONS]

    vmin = min(np.min(m) for m in s1b.values())
    vmax = max(np.max(m) for m in s1b.values())
    for ax, v in zip(axes, variants):
        mat = s1b[v]
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for si in range(len(sub_names)):
            for di in range(len(SOTOPIA_DIMENSIONS)):
                txt_color = "white" if mat[si, di] < (vmin + vmax) / 2 else "#1A1A1A"
                ax.text(di, si, f"{mat[si, di]:.2f}", ha="center", va="center",
                        fontsize=8.5, color=txt_color)
                # mark aligned cells
                if cluster_for_dim[di] == sub_names[si]:
                    ax.add_patch(plt.Rectangle((di - 0.5, si - 0.5), 1, 1, fill=False,
                                                edgecolor="#FFD93D", linewidth=2.0))
        ax.set_xticks(range(len(SOTOPIA_DIMENSIONS)))
        ax.set_xticklabels(dim_short, rotation=40, ha="right")
        ax.set_yticks(range(len(sub_names)))
        ax.set_yticklabels([f"z1_{s}" for s in sub_names], fontweight="bold")
        score = alignment_score(mat)
        ax.set_title(f"{VARIANT_LABELS.get(v, v)}\nalignment score: {score:+.3f}",
                     fontsize=11.5)
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"held-out probe $R^2$")
    fig.suptitle("Strategy 1b — sub-block × reward-dim probe (yellow box = semantic alignment)",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_strategy_6(s6: dict, out_path: Path):
    set_style()
    variants = list(s6.keys())
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), constrained_layout=True)
    if n == 1:
        axes = [axes]

    sub_names = list(SUB_SLICES.keys())
    dim_short = [d.replace("financial_and_material_benefits", "financial")
                 .replace("believability", "believ.")
                 .replace("relationship", "relation.")
                 .replace("social_rules", "social")
                 .replace("knowledge", "knowl.")
                 for d in SOTOPIA_DIMENSIONS]
    cluster_for_dim = [REWARD_CLUSTERS[d] for d in SOTOPIA_DIMENSIONS]

    all_deltas = np.array([np.stack([s6[v]["delta"][s] for s in sub_names])
                           for v in variants])
    vmax = float(np.max(np.abs(all_deltas)))
    for ax, v in zip(axes, variants):
        mat = np.stack([s6[v]["delta"][s] for s in sub_names])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for si in range(len(sub_names)):
            for di in range(len(SOTOPIA_DIMENSIONS)):
                txt_color = "white" if abs(mat[si, di]) > vmax * 0.55 else "#1A1A1A"
                ax.text(di, si, f"{mat[si, di]:+.2f}", ha="center", va="center",
                        fontsize=8.5, color=txt_color)
                if cluster_for_dim[di] == sub_names[si]:
                    ax.add_patch(plt.Rectangle((di - 0.5, si - 0.5), 1, 1, fill=False,
                                                edgecolor="#1A1A1A", linewidth=1.6))
        ax.set_xticks(range(len(SOTOPIA_DIMENSIONS)))
        ax.set_xticklabels(dim_short, rotation=40, ha="right")
        ax.set_yticks(range(len(sub_names)))
        ax.set_yticklabels([f"ablate z1_{s}" for s in sub_names], fontweight="bold")
        score = causal_alignment_score(s6[v]["delta"])
        ax.set_title(f"{VARIANT_LABELS.get(v, v)}\ncausal alignment: {score:+.3f}",
                     fontsize=11.5)
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$\Delta R^2$  (ablation - baseline)")
    fig.suptitle("Strategy 6 — sub-block ablation effect on reward prediction",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_strategy_8(s8: dict, out_path: Path):
    set_style()
    variants = list(s8.keys())
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    pairs = ["belief-intent", "belief-thought", "intent-thought"]
    x = np.arange(len(pairs) + 1)
    width = 0.27
    for i, v in enumerate(variants):
        vals = [s8[v][p] for p in pairs] + [s8[v]["mean_off_diag"]]
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=VARIANT_COLORS.get(v, "#888"),
                      edgecolor="#1F1F1F", linewidth=1.0,
                      label=VARIANT_LABELS.get(v, v))
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs + ["mean off-diag\n(entanglement score)"],
                       rotation=15, ha="right")
    ax.set_ylabel("linear CKA  (0 = orthogonal, 1 = identical)", fontweight="bold")
    ax.set_title("Strategy 8 — inter-sub-block entanglement (lower = more factorized)",
                 fontsize=12.5, fontweight="bold")
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", which="major", color="#D6DADE", linewidth=0.9, alpha=0.85, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#888")
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(out_dir: Path, s1b, s6, s8):
    rows = []
    for v in s1b:
        rows.append({
            "variant": v,
            "strategy_1b_alignment": alignment_score(s1b[v]),
            "strategy_6_causal_alignment": causal_alignment_score(s6[v]["delta"]),
            "strategy_8_mean_cka": s8[v]["mean_off_diag"],
        })
    md = ["# BIT structural probe summary", "",
          "| variant | 1b alignment ↑ | 6 causal align ↑ | 8 mean CKA ↓ |",
          "|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['strategy_1b_alignment']:+.4f} | "
                  f"{r['strategy_6_causal_alignment']:+.4f} | "
                  f"{r['strategy_8_mean_cka']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(md))
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"rows": rows,
                   "strategy_1b_R2_matrices": {k: v.tolist() for k, v in s1b.items()},
                   "strategy_6_delta": {v: {s: arr.tolist() for s, arr in d["delta"].items()}
                                         for v, d in s6.items()},
                   "strategy_8_cka": s8}, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents_dir",
                   default="projects/sotopia/experiments/runs/stage1/variant_eval")
    p.add_argument("--output_dir",
                   default="projects/sotopia/experiments/runs/stage1/bit_structure")
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
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Re-deriving val records …", flush=True)
    records, groups = load_val_records(args)
    reward_vec = np.asarray([r["reward_vec"] for r in records], dtype=np.float32)
    print(f"  val records: {len(records)}, scenarios: {len(np.unique(groups))}")

    print("Loading cached latents …", flush=True)
    latents = load_latents(Path(args.latents_dir), args.variants)
    n_lat = next(iter(latents.values()))["z1"].shape[0]
    if n_lat != len(records):
        raise RuntimeError(
            f"latent count {n_lat} != val record count {len(records)}; "
            f"check seed/val_ratio match between this script and "
            f"eval_mental_variants.py")

    print("\n=== Strategy 1b: sub-block × reward-dim probe ===", flush=True)
    s1b = strategy_1b(latents, reward_vec, groups)
    plot_strategy_1b(s1b, out_dir / "fig_strategy1b_subblock_reward.png")

    print("\n=== Strategy 6: sub-block ablation effect ===", flush=True)
    s6 = strategy_6(latents, reward_vec, groups)
    plot_strategy_6(s6, out_dir / "fig_strategy6_ablation.png")

    print("\n=== Strategy 8: inter-sub-block CKA ===", flush=True)
    s8 = strategy_8(latents)
    plot_strategy_8(s8, out_dir / "fig_strategy8_cka.png")

    write_summary(out_dir, s1b, s6, s8)
    print(f"\nWrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
