#!/usr/bin/env python3
"""Role-routing probes for BIT vs flat mental supervision.

This analysis asks a more direct question than CKA:

  If the latent is partitioned into belief / intent / thought sub-blocks, does
  each sub-block linearly expose its matching role target on held-out samples?

Flat summaries may still preserve mental content, but without role-structured
supervision the information should be less addressable through the intended
sub-block interface.  The main score is diagonal routing gap:

  mean(probe score for matching sub-block -> role target)
  minus mean(probe score for non-matching sub-block -> role target).

All probes use the cached validation latents from eval_mental_variants.py and
the same held-out split.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split
from transformers import AutoTokenizer

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import RecursiveToMDataset, RecursiveToMModel  # noqa: E402


Z_BELIEF_DIM = RecursiveToMModel.Z_BELIEF_DIM
Z_INTENT_DIM = RecursiveToMModel.Z_INTENT_DIM
Z_THOUGHT_DIM = RecursiveToMModel.Z_THOUGHT_DIM

SUB_SLICES = {
    "belief": slice(0, Z_BELIEF_DIM),
    "intent": slice(Z_BELIEF_DIM, Z_BELIEF_DIM + Z_INTENT_DIM),
    "thought": slice(Z_BELIEF_DIM + Z_INTENT_DIM, Z_BELIEF_DIM + Z_INTENT_DIM + Z_THOUGHT_DIM),
}
ROLE_ORDER = ["belief", "intent", "thought"]
ROLE_LABELS = {"belief": "Belief", "intent": "Intent", "thought": "Thought"}

VARIANT_ORDER = ["structured_bit", "flat_mental_summary", "shuffled_mental"]
VARIANT_LABELS = {
    "structured_bit": "BIT",
    "flat_mental_summary": "Flat",
    "shuffled_mental": "Shuffled",
}
VARIANT_COLORS = {
    "structured_bit": "#3A5F87",
    "flat_mental_summary": "#CC8A4C",
    "shuffled_mental": "#5DA39A",
}

FIELD_PATTERNS = {
    "belief": [
        r"Partner Belief:\s*(.*?)(?=\s*\|\s*Strategic Intent:|\s*\|\s*Thought Process:|$)",
        r"Second-Order Belief:\s*(.*?)(?=\s*\|\s*Second-Order Intent:|\s*\|\s*Second-Order Thought:|$)",
    ],
    "intent": [
        r"Strategic Intent:\s*(.*?)(?=\s*\|\s*Thought Process:|$)",
        r"Second-Order Intent:\s*(.*?)(?=\s*\|\s*Second-Order Thought:|$)",
    ],
    "thought": [
        r"Thought Process:\s*(.*)$",
        r"Second-Order Thought:\s*(.*)$",
    ],
}


def extract_first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def role_texts(mental1: str, mental2: str) -> dict[str, str]:
    joined = {"belief": [], "intent": [], "thought": []}
    for role, patterns in FIELD_PATTERNS.items():
        first = extract_first_match(mental1, patterns[0])
        second = extract_first_match(mental2, patterns[1])
        if first:
            joined[role].append(first)
        if second:
            joined[role].append(second)
    return {role: " ".join(parts) if parts else "N/A" for role, parts in joined.items()}


def load_val_records(args: argparse.Namespace) -> tuple[list[dict], np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = RecursiveToMDataset(
        args.data_path,
        tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
    )
    val_size = min(max(1, int(len(dataset) * args.val_ratio)), len(dataset) - 1)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)

    records: list[dict] = []
    groups = []
    for idx in val_dataset.indices:
        sample = dataset.samples[idx]
        roles = role_texts(sample.get("mental1_text", ""), sample.get("mental2_text", ""))
        records.append({
            "roles": roles,
            "mental1_text": sample.get("mental1_text", ""),
            "mental2_text": sample.get("mental2_text", ""),
            "scenario": sample.get("scenario", str(idx)),
        })
        groups.append(sample.get("scenario", str(idx)))
    return records, np.asarray(groups)


def load_latents(latents_dir: Path, variants: list[str]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for variant in variants:
        path = latents_dir / f"latents_{variant}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path)
        out[variant] = {k: arr[k].astype(np.float32) for k in arr.files}
    return out


def fit_role_targets(records: list[dict], n_components: int, seed: int) -> dict[str, np.ndarray]:
    targets = {}
    for role in ROLE_ORDER:
        texts = [record["roles"][role] for record in records]
        vectorizer = TfidfVectorizer(
            max_features=12000,
            min_df=2,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform([text if text.strip() else "N/A" for text in texts])
        max_components = max(2, min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        emb = TruncatedSVD(n_components=max_components, random_state=seed).fit_transform(tfidf)
        targets[role] = StandardScaler().fit_transform(emb).astype(np.float32)
    return targets


def mean_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    denom = np.maximum(true_norm * pred_norm, 1e-8)
    return float(np.mean(np.sum(y_true * y_pred, axis=1) / denom))


def group_ridge_probe(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    alpha: float,
) -> dict[str, float]:
    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two groups for grouped CV.")
    r2s: list[float] = []
    cosines: list[float] = []
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(X, Y, groups=groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        r2s.append(r2_score(Y[test_idx], pred, multioutput="variance_weighted"))
        cosines.append(mean_cosine(Y[test_idx], pred))
    return {"r2": float(np.mean(r2s)), "cosine": float(np.mean(cosines))}


def compute_role_routing(
    latents: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray | float]]]:
    rows: list[dict] = []
    summary: dict[str, dict[str, np.ndarray | float]] = {}
    for variant in [v for v in VARIANT_ORDER if v in latents]:
        matrix = np.zeros((len(ROLE_ORDER), len(ROLE_ORDER)), dtype=np.float32)
        r2_matrix = np.zeros_like(matrix)
        for source_i, source_role in enumerate(ROLE_ORDER):
            X = latents[variant]["z1"][:, SUB_SLICES[source_role]]
            for target_i, target_role in enumerate(ROLE_ORDER):
                metrics = group_ridge_probe(
                    X,
                    targets[target_role],
                    groups,
                    folds=args.folds,
                    alpha=args.ridge_alpha,
                )
                matrix[source_i, target_i] = metrics["cosine"]
                r2_matrix[source_i, target_i] = metrics["r2"]
                rows.append({
                    "variant": variant,
                    "source_subblock": source_role,
                    "target_role": target_role,
                    "cosine": metrics["cosine"],
                    "r2": metrics["r2"],
                    "is_diagonal": source_role == target_role,
                })
        diag_mask = np.eye(len(ROLE_ORDER), dtype=bool)
        off_mask = ~diag_mask
        summary[variant] = {
            "cosine_matrix": matrix,
            "r2_matrix": r2_matrix,
            "diag_cosine": float(matrix[diag_mask].mean()),
            "offdiag_cosine": float(matrix[off_mask].mean()),
            "diag_gap_cosine": float(matrix[diag_mask].mean() - matrix[off_mask].mean()),
            "diag_r2": float(r2_matrix[diag_mask].mean()),
            "offdiag_r2": float(r2_matrix[off_mask].mean()),
            "diag_gap_r2": float(r2_matrix[diag_mask].mean() - r2_matrix[off_mask].mean()),
        }
        print(
            f"{variant}: cosine diag={summary[variant]['diag_cosine']:.3f}, "
            f"off={summary[variant]['offdiag_cosine']:.3f}, "
            f"gap={summary[variant]['diag_gap_cosine']:+.3f}",
            flush=True,
        )
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
    })


def plot_routing_heatmaps(summary: dict[str, dict], out_path: Path) -> None:
    set_style()
    variants = [v for v in VARIANT_ORDER if v in summary]
    mats = [summary[v]["cosine_matrix"] for v in variants]
    vmin = min(float(np.min(m)) for m in mats)
    vmax = max(float(np.max(m)) for m in mats)

    fig, axes = plt.subplots(1, len(variants), figsize=(5.9, 1.95), dpi=300)
    if len(variants) == 1:
        axes = [axes]
    cmap = "Blues"
    im = None
    for ax, variant in zip(axes, variants):
        mat = summary[variant]["cosine_matrix"]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.set_yticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.set_title(VARIANT_LABELS.get(variant, variant), pad=5)
        ax.tick_params(length=0)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
        for idx in range(3):
            ax.add_patch(plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False,
                                       edgecolor="#111111", linewidth=1.6))
        if ax is axes[0]:
            ax.set_ylabel("latent block")
        ax.set_xlabel("target role")
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.82, pad=0.19)
    cbar.outline.set_visible(False)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight("bold")
    cbar.set_label("held-out text-probe cosine", fontsize=8, fontweight="bold", labelpad=2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.83, wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def plot_gap_dotplot(summary: dict[str, dict], out_path: Path) -> None:
    set_style()
    variants = [v for v in VARIANT_ORDER if v in summary]
    y = np.arange(len(variants))[::-1]
    gaps = [float(summary[v]["diag_gap_cosine"]) for v in variants]
    off = [float(summary[v]["offdiag_cosine"]) for v in variants]
    diag = [float(summary[v]["diag_cosine"]) for v in variants]
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
                ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y, [VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("role-probe cosine")
    ax.set_title("Role routing gap")
    ax.grid(axis="x", color="#D8DCE0", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    x_min = min(min(off), min(diag)) - 0.01
    x_max = max(max(off), max(diag)) + 0.045
    ax.set_xlim(x_min, x_max)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.30, top=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.025)
    plt.close(fig)


def write_outputs(out_dir: Path, rows: list[dict], summary: dict[str, dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "role_routing_scores.csv").open("w", newline="") as f:
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
        "# BIT role-routing probe",
        "",
        "| variant | diagonal cosine | off-diagonal cosine | routing gap | diagonal R2 | off-diagonal R2 | R2 gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in [v for v in VARIANT_ORDER if v in summary]:
        values = summary[variant]
        md.append(
            f"| {variant} | {values['diag_cosine']:.4f} | {values['offdiag_cosine']:.4f} | "
            f"{values['diag_gap_cosine']:+.4f} | {values['diag_r2']:.4f} | "
            f"{values['offdiag_r2']:.4f} | {values['diag_gap_r2']:+.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents_dir", default="projects/sotopia/experiments/runs/stage1/variant_eval")
    parser.add_argument("--output_dir", default="projects/sotopia/experiments/runs/stage1/bit_role_routing")
    parser.add_argument("--variants", nargs="+", default=VARIANT_ORDER)
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
    n_latents = next(iter(latents.values()))["z1"].shape[0]
    if n_latents != len(records):
        raise RuntimeError(f"latent count {n_latents} != record count {len(records)}")

    print("Fitting role text targets...", flush=True)
    targets = fit_role_targets(records, args.text_components, args.seed)

    print("Running grouped role-routing probes...", flush=True)
    rows, summary = compute_role_routing(latents, targets, groups, args)
    write_outputs(out_dir, rows, summary)
    plot_routing_heatmaps(summary, out_dir / "fig_role_routing_heatmaps.png")
    plot_gap_dotplot(summary, out_dir / "fig_role_routing_gap.png")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
