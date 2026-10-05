"""
Publication-style figures from the step_500 analysis run.

Reads from mental_latent_analysis_qwen_step500/:
  - figure1_z_concat_umap_points.csv  (umap_x, umap_y, branch)
  - figure2_traversal_branch_aware.csv

Writes figures to mental_latent_analysis_qwen_step500/figs_styled/:
  - fig1_umap.png
  - fig2_probe_classifier.png
  - fig2_reward_heads.png
  - fig3_probes_grouped.png  (linear-probe macro-F1 with shuffled null)

Style: white background, major + minor gridlines, sans-serif.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import argparse

DEFAULT_DIR = "projects/bigtom/runs/analysis/mental_latent_qwen_step500"


def styled_axes(ax, *, x_minor_step=None, y_minor_step=None):
    ax.set_facecolor("#FAFAFA")
    ax.grid(which="major", linestyle="-", linewidth=0.7, color="#BFC6CC", alpha=0.9)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="#D6DBE0", alpha=0.7)
    if x_minor_step is not None:
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(x_minor_step))
    else:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    if y_minor_step is not None:
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor_step))
    else:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=9, length=4)
    ax.tick_params(axis="both", which="minor", length=2.5)


def set_global_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#444",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#BBB",
    })


# ── Figure 1: UMAP ────────────────────────────────────────────────────────────
def plot_umap():
    rows = list(csv.DictReader((analysis_dir / "figure1_z_concat_umap_points.csv").open()))
    xs = np.array([float(r["umap_x"]) for r in rows])
    ys = np.array([float(r["umap_y"]) for r in rows])
    branch = np.array([r["branch"] for r in rows])

    colors = {"aware": "#2E86AB", "not_aware": "#E76F51"}
    labels = {"aware": "aware (true belief)", "not_aware": "not_aware (false belief)"}

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for cond in ("aware", "not_aware"):
        m = branch == cond
        ax.scatter(
            xs[m], ys[m], s=10, alpha=0.6, c=colors[cond],
            edgecolors="white", linewidths=0.2, label=labels[cond],
        )
    styled_axes(ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP of $z_{\\mathrm{concat}}$ — colored by perceptual-access branch\n"
                 "(BigToM stage1, step_500, N=1500)")
    leg = ax.legend(loc="upper right", fontsize=9, markerscale=1.6, scatterpoints=1)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_umap.png")
    plt.close(fig)
    print("wrote", out_dir / "fig1_umap.png")


# ── Figure 2: Traversal panels ────────────────────────────────────────────────
def load_traversal():
    rows = list(csv.DictReader((analysis_dir / "figure2_traversal_branch_aware.csv").open()))
    for r in rows:
        r["alpha"] = float(r["alpha"])
        r["anchor_rank"] = int(r["anchor_rank"])
        for k in ("p_branch_aware", "p_belief_class0", "z1_only_reward", "z_combined_reward"):
            r[k] = float(r[k])
        if "direction" not in r:
            r["direction"] = "belief"
    return rows


def _aggregate_random(rows, anchor_rank, key):
    """Mean ± std over random_id at each alpha for this anchor."""
    rs = [r for r in rows if r["anchor_rank"] == anchor_rank and r["direction"] == "random"]
    out: dict[float, tuple[float, float]] = {}
    for a in sorted({r["alpha"] for r in rs}):
        sub = [r[key] for r in rs if r["alpha"] == a]
        out[a] = (float(np.mean(sub)), float(np.std(sub)))
    xs = sorted(out)
    means = [out[a][0] for a in xs]
    stds = [out[a][1] for a in xs]
    return np.array(xs), np.array(means), np.array(stds)


def plot_probe_classifier():
    rows = load_traversal()
    anchors = sorted({r["anchor_rank"] for r in rows})
    palette = ["#2E86AB", "#1E8449", "#7B3F99", "#E07B00"]
    has_random = any(r["direction"] == "random" for r in rows)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, rk in enumerate(anchors):
        belief_rows = sorted(
            [r for r in rows if r["anchor_rank"] == rk and r["direction"] == "belief"],
            key=lambda r: r["alpha"],
        )
        rec_cond = belief_rows[0]["condition"]
        xs = [r["alpha"] for r in belief_rows]
        cls = [r["p_belief_class0"] for r in belief_rows]
        c = palette[i % len(palette)]
        ax.plot(xs, cls, "-o", color=c, linewidth=2.0, markersize=6.5,
                label=f"anchor {rk} belief-dir ({rec_cond})")
        if has_random:
            rxs, rmean, rstd = _aggregate_random(rows, rk, "p_belief_class0")
            ax.fill_between(rxs, rmean - rstd, rmean + rstd, color=c, alpha=0.10, linewidth=0)
            ax.plot(rxs, rmean, ":", color=c, linewidth=1.1, alpha=0.7,
                    label=f"anchor {rk} random ctrl")

    ax.axhline(0.5, color="#555", linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.7)
    ax.text(ax.get_xlim()[0] + 0.05, 0.52, "decision boundary",
            color="#555", fontsize=8.5, alpha=0.85)
    styled_axes(ax, x_minor_step=0.25, y_minor_step=0.05)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"traversal coefficient $\alpha$  (in units of $\sigma_{\mathrm{proj}}$)")
    ax.set_ylabel(r"$p(\mathrm{aware}\,|\,z)$")
    ax.set_title("Held-out belief-direction traversal — trained belief classifier\n"
                 "(belief direction vs random unit-direction control, matched ‖α·σ‖)")
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, ncol=1)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_probe_classifier.png")
    plt.close(fig)
    print("wrote", out_dir / "fig2_probe_classifier.png")


def plot_reward_heads():
    rows = load_traversal()
    anchors = sorted({r["anchor_rank"] for r in rows})
    palette = ["#2E86AB", "#1E8449", "#7B3F99", "#E07B00"]
    has_random = any(r["direction"] == "random" for r in rows)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, rk in enumerate(anchors):
        belief_rows = sorted(
            [r for r in rows if r["anchor_rank"] == rk and r["direction"] == "belief"],
            key=lambda r: r["alpha"],
        )
        rec_cond = belief_rows[0]["condition"]
        xs = [r["alpha"] for r in belief_rows]
        zr1 = [r["z1_only_reward"] for r in belief_rows]
        zrc = [r["z_combined_reward"] for r in belief_rows]
        c = palette[i % len(palette)]
        ax.plot(xs, zr1, "-o", color=c, linewidth=1.8, markersize=6,
                label=f"anchor {rk} z1_only belief ({rec_cond})")
        ax.plot(xs, zrc, "--s", color=c, linewidth=1.4, markersize=5, alpha=0.85,
                label=f"anchor {rk} z_combined belief")
        if has_random:
            rxs, r_zr1_m, r_zr1_s = _aggregate_random(rows, rk, "z1_only_reward")
            rxs2, r_zrc_m, r_zrc_s = _aggregate_random(rows, rk, "z_combined_reward")
            ax.fill_between(rxs, r_zr1_m - r_zr1_s, r_zr1_m + r_zr1_s, color=c, alpha=0.08, linewidth=0)
            ax.plot(rxs, r_zr1_m, ":", color=c, linewidth=0.9, alpha=0.55)
            ax.fill_between(rxs2, r_zrc_m - r_zrc_s, r_zrc_m + r_zrc_s, color=c, alpha=0.08, linewidth=0)
            ax.plot(rxs2, r_zrc_m, "-.", color=c, linewidth=0.9, alpha=0.5)

    styled_axes(ax, x_minor_step=0.25, y_minor_step=0.1)
    ax.set_xlabel(r"traversal coefficient $\alpha$  (in units of $\sigma_{\mathrm{proj}}$)")
    ax.set_ylabel("reward-head output")
    ax.set_title("Reward heads under held-out belief-direction traversal\n"
                 "(reward outputs are α-invariant for both belief and random directions)")
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.0, ncol=1)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_reward_heads.png")
    plt.close(fig)
    print("wrote", out_dir / "fig2_reward_heads.png")


# ── Figure 3: linear-probe summary ────────────────────────────────────────────
def parse_table1() -> dict:
    csv_path = analysis_dir / "table1_linear_probe.csv"
    rows = list(csv.DictReader(csv_path.open()))
    out: dict = {}
    for r in rows:
        out.setdefault(r["target"], {}).setdefault(r["feature"], {})[r["label_condition"]] = (
            float(r["f1_mean"]), float(r["f1_std"])
        )
    return out


def plot_probe_summary():
    table = parse_table1()
    targets = ["branch", "init_belief", "task"]
    features = ["z1", "z2", "z_concat", "context_hidden", "reward_vec", "tfidf_text"]
    feat_colors = {
        "z1": "#2E86AB", "z2": "#1B5E7A", "z_concat": "#0B3C5D",
        "context_hidden": "#6C757D", "reward_vec": "#E76F51", "tfidf_text": "#888888",
    }
    feat_labels = {
        "z1": "z1", "z2": "z2", "z_concat": "z_concat",
        "context_hidden": "context_hidden\n(upper bound)",
        "reward_vec": "reward_vec",
        "tfidf_text": "tfidf_text",
    }

    fig, ax = plt.subplots(figsize=(11, 5.4))
    bar_w = 0.13
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
                zorder=3,
            )
            ax.text(
                x, real_m + real_s + 0.01,
                f"{real_m:.2f}", ha="center", va="bottom", fontsize=7.5, zorder=4,
            )

    avg_null = float(np.mean(null_means))
    ax.axhline(avg_null, color="crimson", linestyle="--", linewidth=1.4, zorder=2,
               label=f"shuffled-label null (≈{avg_null:.2f})")
    ax.axhspan(0, avg_null, color="crimson", alpha=0.06, zorder=1)

    styled_axes(ax, y_minor_step=0.05)
    ax.set_xticks(centers)
    ax.set_xticklabels([t.replace("_", " ") for t in targets], fontsize=10.5)
    ax.set_ylabel("Macro-F1 (episode-grouped 5-fold CV)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Linear-probe macro-F1 by target × feature\n"
                 "(BigToM stage1, step_500, N=1500, 20 shuffled permutations)")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.8, ncol=1)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_probes_grouped.png")
    plt.close(fig)
    print("wrote", out_dir / "fig3_probes_grouped.png")


def main() -> None:
    global analysis_dir, out_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", type=str, default=DEFAULT_DIR)
    ap.add_argument("--out_subdir", type=str, default="figs_styled")
    args = ap.parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_dir = analysis_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_style()
    plot_umap()
    plot_probe_classifier()
    plot_reward_heads()
    plot_probe_summary()


if __name__ == "__main__":
    main()
