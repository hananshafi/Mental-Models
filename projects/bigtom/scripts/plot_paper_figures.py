"""
Paper-ready figures from the step_300 held-out wide analysis.

Constraints:
  - no plot titles
  - x/y labels are bold and visible
  - legend font not too small (>= 10pt)
  - axes backgrounds gridded with both major and minor squared gridlines

Output: <analysis_dir>/figs_paper/
  - fig1_umap.png
  - fig2_classifier.png
  - fig2_reward.png
  - fig3_probe.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DEFAULT_DIR = "projects/bigtom/runs/analysis/mental_latent_qwen_step300_heldout_wide"


# ── Style helpers ────────────────────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#222",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#888",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    })


def grid_axes(ax, *, x_major=None, x_minor=None, y_major=None, y_minor=None):
    ax.set_facecolor("#FAFAFA")
    if x_major is not None:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_major))
    if y_major is not None:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major))
    if x_minor is not None:
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(x_minor))
    else:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    if y_minor is not None:
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor))
    else:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.grid(which="major", linestyle="-", linewidth=0.8, color="#A8B0B6", alpha=0.85, zorder=0)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="#CFD4D9", alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#222")
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", length=5)
    ax.tick_params(axis="both", which="minor", length=3)


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_traversal(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["alpha"] = float(r["alpha"])
        r["anchor_rank"] = int(r["anchor_rank"])
        for k in ("p_branch_aware", "p_belief_class0", "z1_only_reward", "z_combined_reward"):
            r[k] = float(r[k])
        if "direction" not in r:
            r["direction"] = "belief"
    return rows


def aggregate_random(rows, anchor_rank, key):
    rs = [r for r in rows if r["anchor_rank"] == anchor_rank and r["direction"] == "random"]
    out: dict[float, tuple[float, float]] = {}
    for a in sorted({r["alpha"] for r in rs}):
        sub = [r[key] for r in rs if r["alpha"] == a]
        out[a] = (float(np.mean(sub)), float(np.std(sub)))
    xs = sorted(out)
    means = np.array([out[a][0] for a in xs])
    stds = np.array([out[a][1] for a in xs])
    return np.array(xs), means, stds


def parse_table1(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    out: dict = {}
    for r in rows:
        out.setdefault(r["target"], {}).setdefault(r["feature"], {})[r["label_condition"]] = (
            float(r["f1_mean"]), float(r["f1_std"])
        )
    return out


# ── Figures ──────────────────────────────────────────────────────────────────
def fig1_umap(analysis_dir: Path, out_dir: Path):
    import json as _json
    rows = list(csv.DictReader((analysis_dir / "figure1_z_concat_umap_points.csv").open()))
    xs = np.array([float(r["umap_x"]) for r in rows])
    ys = np.array([float(r["umap_y"]) for r in rows])
    branch = np.array([r["branch"] for r in rows])

    tasks: list[str] = []
    with (analysis_dir / "analysis_records.jsonl").open() as f:
        for line in f:
            tasks.append(_json.loads(line)["task"])
    tasks = np.array(tasks[: len(xs)])

    branch_colors = {"aware": "#2E86AB", "not_aware": "#E76F51"}
    branch_labels = {"aware": "aware (true belief)", "not_aware": "not_aware (false belief)"}

    unique_tasks = sorted(set(tasks.tolist()))
    cmap = plt.get_cmap("tab10" if len(unique_tasks) <= 10 else "tab20")
    task_colors = {t: cmap(i % cmap.N) for i, t in enumerate(unique_tasks)}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

    ax = axes[0]
    for t in unique_tasks:
        m = tasks == t
        ax.scatter(xs[m], ys[m], s=14, alpha=0.65, c=[task_colors[t]],
                   edgecolors="white", linewidths=0.25, label=t.replace("_", " "), zorder=3)
    grid_axes(ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    leg = ax.legend(loc="best", markerscale=1.6, scatterpoints=1, fontsize=10, ncol=1,
                    title="task", title_fontsize=11)
    leg.get_frame().set_linewidth(0.6)

    ax = axes[1]
    for cond in ("aware", "not_aware"):
        m = branch == cond
        ax.scatter(xs[m], ys[m], s=14, alpha=0.65, c=branch_colors[cond],
                   edgecolors="white", linewidths=0.25, label=branch_labels[cond], zorder=3)
    grid_axes(ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    leg = ax.legend(loc="best", markerscale=1.6, scatterpoints=1,
                    title="branch", title_fontsize=11)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_umap.png")
    plt.close(fig)
    print("wrote", out_dir / "fig1_umap.png")


def aggregate_belief(rows, key):
    """Average over anchors at each alpha for the belief-direction curves."""
    rs = [r for r in rows if r["direction"] == "belief"]
    out: dict[float, list[float]] = {}
    for r in rs:
        out.setdefault(r["alpha"], []).append(r[key])
    xs = np.array(sorted(out))
    means = np.array([float(np.mean(out[a])) for a in xs])
    stds = np.array([float(np.std(out[a])) for a in xs])
    return xs, means, stds


def aggregate_random_all(rows, key):
    """Average over anchors x random seeds at each alpha."""
    rs = [r for r in rows if r["direction"] == "random"]
    out: dict[float, list[float]] = {}
    for r in rs:
        out.setdefault(r["alpha"], []).append(r[key])
    xs = np.array(sorted(out))
    means = np.array([float(np.mean(out[a])) for a in xs])
    stds = np.array([float(np.std(out[a])) for a in xs])
    return xs, means, stds


def fig2_traversal(analysis_dir: Path, out_dir: Path):
    rows = load_traversal(analysis_dir / "figure2_traversal_branch_aware.csv")

    c_belief_cls = "#0B3C5D"
    c_belief_rew = "#1E8449"
    c_random = "#888888"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # Left: classifier
    ax = axes[0]
    bx, bm, bs = aggregate_belief(rows, "p_belief_class0")
    rx, rm, rs = aggregate_random_all(rows, "p_belief_class0")
    ax.fill_between(bx, bm - bs, bm + bs, color=c_belief_cls, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(bx, bm, "-o", color=c_belief_cls, linewidth=2.6, markersize=8.0,
            label="belief direction (mean ± std over anchors)", zorder=4)
    ax.fill_between(rx, rm - rs, rm + rs, color=c_random, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(rx, rm, "--s", color=c_random, linewidth=2.0, markersize=6.5,
            label="random direction (mean ± std, n=12)", zorder=3)
    grid_axes(ax, x_major=5, x_minor=1, y_major=0.05, y_minor=0.01)
    lo = float(min(bm.min() - bs.max(), rm.min() - rs.max()) - 0.03)
    hi = float(max(bm.max() + bs.max(), rm.max() + rs.max()) + 0.03)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Traversal coefficient $\mathbf{\alpha}$  (units of $\sigma_{\mathrm{proj}}$)")
    ax.set_ylabel(r"Belief-classifier probability  $\mathbf{p(\mathrm{class}\,0\,|\,z)}$")
    leg = ax.legend(loc="upper left", ncol=1)
    leg.get_frame().set_linewidth(0.6)

    # Right: z1-only reward head
    ax = axes[1]
    bx, bm, bs = aggregate_belief(rows, "z1_only_reward")
    rx, rm, rs = aggregate_random_all(rows, "z1_only_reward")
    ax.fill_between(bx, bm - bs, bm + bs, color=c_belief_rew, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(bx, bm, "-o", color=c_belief_rew, linewidth=2.6, markersize=8.0,
            label="belief direction (mean ± std over anchors)", zorder=4)
    ax.fill_between(rx, rm - rs, rm + rs, color=c_random, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(rx, rm, "--s", color=c_random, linewidth=2.0, markersize=6.5,
            label="random direction (mean ± std, n=12)", zorder=3)
    grid_axes(ax, x_major=5, x_minor=1, y_minor=0.05)
    ax.set_xlabel(r"Traversal coefficient $\mathbf{\alpha}$  (units of $\sigma_{\mathrm{proj}}$)")
    ax.set_ylabel(r"$\mathbf{z_1}$-only reward head")
    leg = ax.legend(loc="upper left", ncol=1)
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_traversal.png")
    plt.close(fig)
    print("wrote", out_dir / "fig2_traversal.png")


def fig3_probe(analysis_dir: Path, out_dir: Path):
    table = parse_table1(analysis_dir / "table1_linear_probe.csv")
    targets = ["branch", "init_belief", "task"]
    features = ["z1", "z2", "z_concat", "context_hidden", "reward_vec", "tfidf_text"]
    feat_colors = {
        "z1": "#2E86AB", "z2": "#1B5E7A", "z_concat": "#0B3C5D",
        "context_hidden": "#6C757D", "reward_vec": "#E76F51", "tfidf_text": "#888888",
    }
    feat_labels = {
        "z1": "z1", "z2": "z2", "z_concat": "z_concat",
        "context_hidden": "context_hidden",
        "reward_vec": "reward_vec",
        "tfidf_text": "tfidf_text",
    }

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    bar_w = 0.135
    centers = np.arange(len(targets))
    null_means: list[float] = []

    for ti, tgt in enumerate(targets):
        for fi, feat in enumerate(features):
            real_m, real_s = table[tgt][feat]["real"]
            null_m, _ = table[tgt][feat]["shuffled_summary"]
            null_means.append(null_m)
            x = centers[ti] + (fi - (len(features) - 1) / 2) * bar_w
            ax.bar(
                x, real_m, width=bar_w * 0.95,
                yerr=real_s, capsize=3,
                color=feat_colors[feat], edgecolor="black", linewidth=0.5,
                label=feat_labels[feat] if ti == 0 else None,
                zorder=4,
            )
            ax.text(
                x, real_m + real_s + 0.012,
                f"{real_m:.2f}", ha="center", va="bottom", fontsize=8, zorder=5,
            )

    avg_null = float(np.mean(null_means))
    ax.axhline(avg_null, color="crimson", linestyle="--", linewidth=1.6, zorder=3,
               label=f"shuffled-label null (≈{avg_null:.2f})")
    ax.axhspan(0, avg_null, color="crimson", alpha=0.06, zorder=1)

    grid_axes(ax, y_major=0.2, y_minor=0.05)
    ax.set_xticks(centers)
    ax.set_xticklabels([t.replace("_", " ") for t in targets], fontweight="bold")
    ax.set_xlabel("Target")
    ax.set_ylabel("Macro-F1  (5-fold episode-grouped CV)")
    ax.set_ylim(0, 1.10)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_probe.png")
    plt.close(fig)
    print("wrote", out_dir / "fig3_probe.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", type=str, default=DEFAULT_DIR)
    ap.add_argument("--out_subdir", type=str, default="figs_paper")
    args = ap.parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_dir = analysis_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    fig1_umap(analysis_dir, out_dir)
    fig2_traversal(analysis_dir, out_dir)
    fig3_probe(analysis_dir, out_dir)


if __name__ == "__main__":
    main()
