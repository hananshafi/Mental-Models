"""
Side-by-side comparison of two analyse_tom_latents.py output dirs.

Reads each run's table1_linear_probe.csv and figure2_traversal_branch_aware.csv
and produces:
  - compare_table1.md  (linear-probe F1 side by side)
  - compare_traversal_probe.png  (p(aware) probe + classifier vs alpha)
  - compare_traversal_reward.png (z1_only / z_combined reward vs alpha)

Usage:
  python compare_analyses.py \
    --runs step_500=/path/to/mental_latent_analysis_qwen_step500 \
           step_1000=/path/to/mental_latent_analysis_qwen_step1000 \
    --out /path/to/analysis_compare_step500_vs_step1000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_probe_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        for k in ("acc_mean", "acc_std", "f1_mean", "f1_std"):
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = float("nan")
    return rows


def load_traversal_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["alpha"] = float(r["alpha"])
        r["anchor_rank"] = int(r["anchor_rank"])
        for k in ("p_branch_aware", "p_belief_class0", "z1_only_reward", "z_combined_reward"):
            r[k] = float(r[k])
    return rows


def make_table_md(runs: dict[str, list[dict]], out: Path) -> None:
    targets = sorted({r["target"] for rs in runs.values() for r in rs})
    features = ["z1", "z2", "z_concat", "context_hidden", "reward_vec", "tfidf_text"]

    lines = ["# Linear-probe comparison", ""]
    for tgt in targets:
        lines += [f"## target = {tgt}", "",
                  "| feature | labels | " + " | ".join(f"{k} F1" for k in runs) + " |",
                  "|---|---|" + "|".join(["---:"] * len(runs)) + "|"]
        for feat in features:
            for cond in ("real", "shuffled_summary"):
                cells = []
                for tag, rs in runs.items():
                    hit = next((r for r in rs if r["target"] == tgt and r["feature"] == feat
                                and r["label_condition"] == cond), None)
                    cells.append(f"{hit['f1_mean']:.3f} ± {hit['f1_std']:.3f}" if hit else "—")
                lines.append(f"| {feat} | {cond} | " + " | ".join(cells) + " |")
        lines.append("")
    out.write_text("\n".join(lines))


def make_traversal_plots(runs: dict[str, list[dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, (tag, rs) in zip(axes, runs.items()):
        anchors = sorted({r["anchor_rank"] for r in rs})
        for rank in anchors:
            sub = [r for r in rs if r["anchor_rank"] == rank]
            sub.sort(key=lambda r: r["alpha"])
            xs = [r["alpha"] for r in sub]
            ax.plot(xs, [r["p_branch_aware"] for r in sub], "-o", alpha=0.75, label=f"anchor {rank} probe")
            ax.plot(xs, [r["p_belief_class0"] for r in sub], "--", alpha=0.55, label=f"anchor {rank} cls")
        ax.axhline(0.5, color="k", linestyle=":", alpha=0.3)
        ax.set_title(f"{tag}: probe + classifier vs α")
        ax.set_xlabel("α"); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, ncol=2, loc="best")
    axes[0].set_ylabel("p(aware) / classifier prob")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_traversal_probe.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, (tag, rs) in zip(axes, runs.items()):
        anchors = sorted({r["anchor_rank"] for r in rs})
        for rank in anchors:
            sub = [r for r in rs if r["anchor_rank"] == rank]
            sub.sort(key=lambda r: r["alpha"])
            xs = [r["alpha"] for r in sub]
            ax.plot(xs, [r["z1_only_reward"] for r in sub], "-o", alpha=0.75, label=f"anchor {rank} z1_only")
            ax.plot(xs, [r["z_combined_reward"] for r in sub], "--", alpha=0.55, label=f"anchor {rank} z_comb")
        ax.set_title(f"{tag}: reward heads vs α")
        ax.set_xlabel("α")
        ax.legend(fontsize=7, ncol=2, loc="best")
    axes[0].set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_traversal_reward.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Each entry is tag=path/to/analysis_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes: dict[str, list[dict]] = {}
    travs: dict[str, list[dict]] = {}
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        path = Path(path)
        probes[tag] = load_probe_csv(path / "table1_linear_probe.csv")
        travs[tag] = load_traversal_csv(path / "figure2_traversal_branch_aware.csv")

    make_table_md(probes, out_dir / "compare_table1.md")
    make_traversal_plots(travs, out_dir)
    print(f"Wrote {out_dir}/compare_table1.md")
    print(f"Wrote {out_dir}/compare_traversal_probe.png")
    print(f"Wrote {out_dir}/compare_traversal_reward.png")


if __name__ == "__main__":
    main()
