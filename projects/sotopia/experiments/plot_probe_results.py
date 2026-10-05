import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ANALYSIS_DIR = Path("projects/sotopia/runs/analysis/mental_latent")

TARGETS = ["intent", "knowledge", "strategy"]
FEATURES = ["z1", "z2", "z_concat", "reward_vec", "context_hidden"]

FEATURE_COLORS = {
    "z1": "#2E86AB",
    "z2": "#1B5E7A",
    "z_concat": "#0B3C5D",
    "reward_vec": "#E76F51",
    "context_hidden": "#6C757D",
}
FEATURE_LABELS = {
    "z1": "z1",
    "z2": "z2",
    "z_concat": "z_concat",
    "reward_vec": "reward_vec (baseline)",
    "context_hidden": "context_hidden (upper bound)",
}


def load_table(table_csv: Path) -> dict:
    rows = list(csv.DictReader(table_csv.open()))
    out: dict = {}
    for r in rows:
        target = r["target"].strip()
        feature = r["feature"].strip()
        labels = r["label_condition"].strip()
        f1_mean = float(r["macro_f1_mean"])
        f1_std = float(r["macro_f1_std"])
        out.setdefault(target, {}).setdefault(feature, {})[labels] = (f1_mean, f1_std)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table_csv",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "table1_linear_probe.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "figure_probe_signal_vs_baselines.png",
    )
    args = parser.parse_args()
    table = load_table(args.table_csv)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_features = len(FEATURES)
    n_targets = len(TARGETS)
    bar_w = 0.14
    group_centers = np.arange(n_targets)

    null_means: list[float] = []
    for ti, target in enumerate(TARGETS):
        for fi, feat in enumerate(FEATURES):
            real_mean, real_std = table[target][feat]["real"]
            shuf_mean, _ = table[target][feat]["shuffled_summary"]
            null_means.append(shuf_mean)

            x = group_centers[ti] + (fi - (n_features - 1) / 2.0) * bar_w
            ax.bar(
                x, real_mean, width=bar_w * 0.95,
                yerr=real_std, capsize=3,
                color=FEATURE_COLORS[feat],
                edgecolor="black", linewidth=0.6,
                label=FEATURE_LABELS[feat] if ti == 0 else None,
            )
            ax.text(
                x, real_mean + real_std + 0.012,
                f"{real_mean:.2f}", ha="center", va="bottom", fontsize=8,
            )

    avg_null = float(np.mean(null_means))
    ax.axhline(
        avg_null, color="crimson", linestyle="--", linewidth=1.4,
        label=f"shuffled-label null (~{avg_null:.2f})",
    )
    ax.axhspan(0, avg_null, color="crimson", alpha=0.06)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([t.capitalize() for t in TARGETS], fontsize=11)
    ax.set_ylabel("Macro-F1 (episode-grouped CV)", fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.set_title(
        "Linear probe: ToM latents carry real signal beyond reward,\n"
        "but trail the raw context hidden state",
        fontsize=12, pad=10,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.95)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
