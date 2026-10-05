#!/usr/bin/env python3
"""Run the downstream BIT-vs-flat-vs-shuffled policy ablation.

This is the main-paper experiment for the question:

  Does the structured BIT mental model matter beyond being another compressed
  reward representation?

The harness keeps the policy side fixed and changes only the frozen Stage-1
reward checkpoint used by GRPO:

  BIT reward checkpoint       -> GRPO policy -> official SOTOPIA evaluation
  Flat-summary checkpoint     -> GRPO policy -> official SOTOPIA evaluation
  Shuffled-mental checkpoint  -> GRPO policy -> official SOTOPIA evaluation

To avoid SFT noise, all variants start from the same SFT LoRA adapter. The
script can write commands, launch training/evaluation, and plot aggregated
official-judge results once evaluation summaries exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
STAGE2 = SOTOPIA_ROOT / "stage2_grpo_agent_training_v3.py"
STAGE3 = SOTOPIA_ROOT / "stage3_evaluate_sotopia.py"

DEFAULT_VARIANTS = {
    "BIT": SOTOPIA_ROOT / "coupled_mental_reward_checkpoint_qwen_v3" / "epoch_2",
    "Flat": SOTOPIA_ROOT
    / "experiments"
    / "runs"
    / "stage1"
    / "flat_mental_summary_qwen7b_seed42"
    / "best",
    "Shuffled": SOTOPIA_ROOT
    / "experiments"
    / "runs"
    / "stage1"
    / "shuffled_mental_qwen7b_seed42"
    / "best",
}

PLOT_COLORS = {
    "BIT": "#2F6FAE",
    "Flat": "#D28B45",
    "Shuffled": "#5BA79D",
}

METRIC_LABELS = {
    "overall_score": "Overall",
    "goal": "Goal",
    "knowledge": "Knowledge",
    "believability": "Believability",
    "relationship": "Relationship",
}


def parse_variant_specs(specs: Iterable[str] | None) -> dict[str, Path]:
    if not specs:
        return DEFAULT_VARIANTS.copy()
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Variant spec must be Label=/path, got {spec!r}")
        label, path = spec.split("=", 1)
        out[label.strip()] = Path(path.strip())
    return out


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def read_api_key(path: str | None) -> str | None:
    if not path:
        return None
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(key_path)
    key = key_path.read_text().strip()
    return key or None


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + shell_join(cmd) + "\n\n")
    print(f"\n[cmd] {shell_join(cmd)}", flush=True)
    print(f"[log] {log_path}", flush=True)
    if dry_run:
        return 0
    with log_path.open("a") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0:
        print(f"[failed] returncode={proc.returncode}; see {log_path}", flush=True)
    return proc.returncode


def train_command(label: str, reward_ckpt: Path, args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(STAGE2),
        "--policy_model_name",
        args.policy_model_name,
        "--reward_model_name",
        args.reward_model_name,
        "--reward_checkpoint_dir",
        str(reward_ckpt),
        "--reward_version",
        "v3",
        "--data_path",
        args.data_path,
        "--output_dir",
        str(run_dir),
        "--preset",
        args.preset,
        "--group_size",
        str(args.group_size),
        "--grpo_epochs",
        str(args.grpo_epochs),
        "--prompts_per_step",
        str(args.prompts_per_step),
        "--num_ppo_epochs",
        str(args.num_ppo_epochs),
        "--clip_eps",
        str(args.clip_eps),
        "--kl_coeff",
        str(args.kl_coeff),
        "--lr",
        str(args.lr),
        "--temperature",
        str(args.train_temperature),
        "--top_p",
        str(args.train_top_p),
        "--max_gen_len",
        str(args.max_gen_len),
        "--max_ctx_len",
        str(args.max_ctx_len),
        "--sft_checkpoint",
        args.sft_checkpoint,
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--num_lora_layers",
        str(args.num_lora_layers),
        "--ensemble_weight",
        str(args.ensemble_weight),
        "--trajectory_weight",
        str(args.trajectory_weight),
        "--grpo_grad_accum",
        str(args.grpo_grad_accum),
        "--reward_clip",
        str(args.reward_clip),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--save_every",
        str(args.save_every),
        "--reward_scoring_dims",
        args.reward_scoring_dims,
        "--seed",
        str(args.seed),
        "--gpu",
        args.train_gpus,
    ]
    if args.patience > 0:
        cmd += ["--patience", str(args.patience)]
    return cmd


def adapter_path_for_eval(run_dir: Path, checkpoint_name: str) -> Path:
    preferred = run_dir / checkpoint_name
    if preferred.exists():
        return preferred
    for fallback in ["best", "epoch_0", "step_50", "sft_warmup"]:
        candidate = run_dir / fallback
        if candidate.exists():
            return candidate
    return preferred


def eval_command(
    label: str,
    adapter_path: Path,
    agent_index: int,
    args: argparse.Namespace,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(STAGE3),
        "--policy_model_name",
        args.policy_model_name,
        "--policy_adapter_path",
        str(adapter_path),
        "--merge_adapter",
        "--use_hf",
        "--deduplicate_envs",
        "--task",
        args.eval_task,
        "--output_path",
        str(output_path),
        "--max_turns",
        str(args.max_turns),
        "--max_episodes",
        str(args.max_episodes),
        "--policy_agent_index",
        str(agent_index),
        "--partner_model",
        args.partner_model,
        "--judge_model",
        args.judge_model,
        "--temperature",
        str(args.eval_temperature),
        "--top_p",
        str(args.eval_top_p),
        "--max_gen_len",
        str(args.max_gen_len),
        "--seed",
        str(args.seed),
        "--gpu",
        args.eval_gpu,
        "--tag",
        f"downstream_decomp_{label}_agent{agent_index}",
    ]


def collect_eval_values(eval_dir: Path, label: str, agent_indices: list[int], metrics: list[str]) -> dict[str, list[float]]:
    values = {metric: [] for metric in metrics}
    for agent_index in agent_indices:
        summary_path = eval_dir / f"{label}_agent{agent_index}_{eval_dir.name}_summary.json"
        if not summary_path.exists():
            # stage3 creates summary by replacing .jsonl suffix, so keep this
            # fallback for manually renamed files.
            candidates = sorted(eval_dir.glob(f"{label}_agent{agent_index}*_summary.json"))
            if not candidates:
                continue
            summary_path = candidates[0]
        with summary_path.open() as f:
            summary = json.load(f)
        policy_scores = summary.get("policy_agent", {})
        for metric in metrics:
            metric_values = policy_scores.get(metric, {}).get("values", [])
            values[metric].extend(float(x) for x in metric_values)
    return values


def write_result_table(
    out_dir: Path,
    variants: dict[str, Path],
    agent_indices: list[int],
    metrics: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    eval_dir = out_dir / "eval"
    table = {}
    lines = [
        "# Downstream decomposition ablation",
        "",
        "| variant | metric | mean | stderr | n |",
        "|---|---|---:|---:|---:|",
    ]
    for label in variants:
        metric_values = collect_eval_values(eval_dir, label, agent_indices, metrics)
        table[label] = {}
        for metric, vals in metric_values.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            stderr = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            mean = float(arr.mean())
            table[label][metric] = {"mean": mean, "stderr": stderr, "n": int(arr.size)}
            lines.append(
                f"| {label} | {METRIC_LABELS.get(metric, metric)} | {mean:.4f} | {stderr:.4f} | {arr.size} |"
            )
    with (out_dir / "downstream_summary.json").open("w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    (out_dir / "downstream_summary.md").write_text("\n".join(lines))
    return table


def plot_results(table: dict[str, dict[str, dict[str, float]]], metrics: list[str], out_path: Path) -> None:
    if not table:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    labels = list(table.keys())
    y = np.arange(len(metrics))
    height = 0.22
    offsets = np.linspace(-height, height, max(len(labels), 1))

    fig, ax = plt.subplots(figsize=(5.7, 2.55), dpi=300)
    for offset, label in zip(offsets, labels):
        means = [table.get(label, {}).get(metric, {}).get("mean", np.nan) for metric in metrics]
        errs = [table.get(label, {}).get(metric, {}).get("stderr", 0.0) for metric in metrics]
        ax.barh(
            y + offset,
            means,
            height=height * 0.9,
            xerr=errs,
            color=PLOT_COLORS.get(label, "#777777"),
            edgecolor="#2F2F2F",
            linewidth=0.7,
            label=label,
            error_kw={"elinewidth": 0.9, "capsize": 2.0, "capthick": 0.9},
        )
    ax.set_yticks(y, [METRIC_LABELS.get(metric, metric) for metric in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("official SOTOPIA judge score", fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.9)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.legend(loc="lower right", frameon=True, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.23, top=0.96)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.02)
    plt.close(fig)


def write_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    lines = ["# Downstream decomposition ablation commands", ""]
    for section in ["train", "eval"]:
        if not manifest.get(section):
            continue
        lines.append(f"## {section}")
        lines.append("")
        for item in manifest[section]:
            lines.append(f"### {item['name']}")
            lines.append("")
            lines.append("```bash")
            lines.append(item["command"])
            lines.append("```")
            lines.append("")
    (out_dir / "commands.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["write", "train", "eval", "plot", "all"], default="write")
    parser.add_argument("--dry_run", action="store_true", help="Print/write commands without executing them.")
    parser.add_argument(
        "--output_dir",
        default=str(SOTOPIA_ROOT / "experiments" / "runs" / "stage2" / "downstream_decomposition_ablation"),
    )
    parser.add_argument("--variants", nargs="+", default=None, help="Override variants as Label=/checkpoint/path.")

    parser.add_argument("--policy_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reward_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default=str(SOTOPIA_ROOT / "sotopia_turn_rewards_v3.jsonl"))
    parser.add_argument(
        "--sft_checkpoint",
        default=str(SOTOPIA_ROOT / "grpo_agent_checkpoint_qwen_v3" / "sft_warmup"),
        help="Common SFT LoRA adapter used to initialize every GRPO variant.",
    )

    parser.add_argument("--preset", default="qwen", choices=["qwen", "llama", "mistral"])
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--grpo_epochs", type=int, default=1)
    parser.add_argument("--prompts_per_step", type=int, default=4)
    parser.add_argument("--num_ppo_epochs", type=int, default=1)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--train_temperature", type=float, default=0.8)
    parser.add_argument("--train_top_p", type=float, default=0.95)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)
    parser.add_argument("--ensemble_weight", type=float, default=0.7)
    parser.add_argument("--trajectory_weight", type=float, default=0.2)
    parser.add_argument("--grpo_grad_accum", type=int, default=1)
    parser.add_argument("--reward_clip", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--reward_scoring_dims", default="goal,knowledge,believability")
    parser.add_argument("--train_gpus", default="0,6", help="Visible GPUs for stage2; first=policy, last=reward/ref.")

    parser.add_argument("--eval_task", default="hard", choices=["all", "hard", "cooperative", "competitive"])
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--eval_agent_indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--partner_model", default="gpt-4o-mini")
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--eval_temperature", type=float, default=0.7)
    parser.add_argument("--eval_top_p", type=float, default=0.9)
    parser.add_argument("--eval_gpu", default="6")
    parser.add_argument("--eval_checkpoint", default="best")
    parser.add_argument("--openai_api_key_file", default=None)
    parser.add_argument(
        "--plot_metrics",
        nargs="+",
        default=["overall_score", "goal", "knowledge", "believability"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    log_dir = out_dir / "logs"
    variants = parse_variant_specs(args.variants)

    manifest = {"train": [], "eval": [], "config": vars(args)}
    for label, ckpt in variants.items():
        run_dir = train_dir / label
        cmd = train_command(label, ckpt, args, run_dir)
        manifest["train"].append(
            {
                "name": label,
                "reward_checkpoint": str(ckpt),
                "output_dir": str(run_dir),
                "command": shell_join(cmd),
            }
        )
        adapter_path = adapter_path_for_eval(run_dir, args.eval_checkpoint)
        for agent_index in args.eval_agent_indices:
            output_path = eval_dir / f"{label}_agent{agent_index}_{args.eval_task}.jsonl"
            ecmd = eval_command(label, adapter_path, agent_index, args, output_path)
            manifest["eval"].append(
                {
                    "name": f"{label}_agent{agent_index}",
                    "adapter_path": str(adapter_path),
                    "output_path": str(output_path),
                    "command": shell_join(ecmd),
                }
            )
    write_manifest(out_dir, manifest)

    env = os.environ.copy()
    api_key = read_api_key(args.openai_api_key_file)
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    env.pop("OPENAI_BASE_URL", None)

    if args.mode in {"train", "all"}:
        for item in manifest["train"]:
            rc = run_logged(
                shlex.split(item["command"]),
                log_dir / f"train_{item['name']}.log",
                env=env,
                dry_run=args.dry_run,
            )
            if rc != 0:
                raise SystemExit(rc)

    if args.mode in {"eval", "all"}:
        if not api_key and not args.dry_run:
            raise RuntimeError("OPENAI_API_KEY is required for stage3 evaluation.")
        for item in manifest["eval"]:
            rc = run_logged(
                shlex.split(item["command"]),
                log_dir / f"eval_{item['name']}.log",
                env=env,
                dry_run=args.dry_run,
            )
            if rc != 0:
                raise SystemExit(rc)

    if args.mode in {"plot", "all"}:
        table = write_result_table(out_dir, variants, args.eval_agent_indices, args.plot_metrics)
        plot_results(table, args.plot_metrics, out_dir / "fig_downstream_decomposition_ablation.png")
        print(f"Wrote downstream summary/figure under {out_dir}", flush=True)
    else:
        print(f"Wrote command manifest under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
