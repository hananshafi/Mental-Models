#!/usr/bin/env python3
"""Latent scrambling intervention for the recursive mental model.

The test is deliberately causal:

  - keep the held-out examples, targets, response text, and model weights fixed;
  - feed the original cached z1/z2 into the fixed heads;
  - feed scrambled z1/z2 into the same fixed heads;
  - measure how much mental decoding and reward prediction degrade.

If z were collapsed or ignored, scrambling would have little effect. If z is a
usable sample-specific mental state, distribution-preserving scrambles should
increase held-out mental NLL and reward error.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    CUSTOM_HEAD_NAMES,
    REWARD_DIM,
    RecursiveToMDataset,
    RecursiveToMModel,
    collate_fn,
)
from experiments.stage1_train_mental_supervision_variants import (  # noqa: E402
    VariantRecursiveToMDataset,
    load_flat_summary_cache,
)


CONDITION_ORDER = [
    "baseline",
    "sample_shuffle_pair",
    "sample_shuffle_independent",
    "block_shuffle",
    "coordinate_permute",
]
CONDITION_LABELS = {
    "baseline": "original z",
    "sample_shuffle_pair": "sample-shuffled z",
    "sample_shuffle_independent": "independent z1/z2 shuffle",
    "block_shuffle": "B/I/T block shuffle",
    "coordinate_permute": "coordinate scramble",
}
CONDITION_COLORS = {
    "baseline": "#2F6FAE",
    "sample_shuffle_pair": "#5BA79D",
    "sample_shuffle_independent": "#D28B45",
    "block_shuffle": "#8B6BB8",
    "coordinate_permute": "#C84C4C",
}
PLOT_METRICS = [
    ("mental1_nll", "1st-order mental NLL"),
    ("mental2_nll", "2nd-order mental NLL"),
    ("z_combined_mse", "z-only reward MSE"),
    ("joint_reward_mse", "joint reward MSE"),
]


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_model(model_name: str, checkpoint_dir: Path, device: torch.device, z_dim: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    base = PeftModel.from_pretrained(base, checkpoint_dir / "lora_adapter")
    model = RecursiveToMModel(base, reward_dim=REWARD_DIM, z_dim=z_dim)
    for head_name in CUSTOM_HEAD_NAMES:
        path = checkpoint_dir / f"{head_name}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        getattr(model, head_name).load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def decoder_token_losses(
    model: RecursiveToMModel,
    z: torch.Tensor,
    decoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    z_prefix = model._expand_z_to_prefix(z, decoder)
    embeds = model.base_model.get_input_embeddings()(input_ids)
    attended, _ = decoder["cross_attn"](query=embeds, key=z_prefix, value=z_prefix)
    h = decoder["ln"](embeds + attended)
    h = h + decoder["ffn"](h)
    h = decoder["ln2"](h)

    output_embedding = model.base_model.get_output_embeddings()
    logits = F.linear(
        h,
        output_embedding.weight,
        output_embedding.bias
        if hasattr(output_embedding, "bias") and output_embedding.bias is not None
        else None,
    )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].clone().contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels[shift_mask == 0] = -100
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    return losses.view(input_ids.size(0), -1)


def nll_sum_count(token_losses: torch.Tensor, attention_mask: torch.Tensor) -> tuple[float, int]:
    valid = attention_mask[:, 1:].to(token_losses.device, dtype=token_losses.dtype)
    loss_sum = (token_losses * valid).sum().float().item()
    count = int(valid.sum().item())
    return loss_sum, count


def cosine_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    denom = pred.norm(dim=1).clamp_min(1e-8) * target.norm(dim=1).clamp_min(1e-8)
    return ((pred * target).sum(dim=1) / denom).mean().float().item()


def build_val_dataset(args, tokenizer):
    if args.target_variant == "structured_bit":
        dataset = RecursiveToMDataset(
            args.data_path,
            tokenizer,
            max_ctx_len=args.max_ctx_len,
            max_resp_len=args.max_resp_len,
            max_mental_len=args.max_mental_len,
        )
    else:
        dataset = VariantRecursiveToMDataset(
            args.data_path,
            tokenizer,
            variant=args.target_variant,
            seed=args.seed,
            max_ctx_len=args.max_ctx_len,
            max_resp_len=args.max_resp_len,
            max_mental_len=args.max_mental_len,
            flat_summary_cache=load_flat_summary_cache(args.flat_summary_jsonl),
        )
    val_size = min(max(1, int(len(dataset) * args.val_ratio)), len(dataset) - 1)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
    return val_dataset


def load_latents(path: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(path)
    z1 = arrays["z1"].astype(np.float32)
    z2 = arrays["z2"].astype(np.float32)
    if max_samples and max_samples > 0:
        z1 = z1[:max_samples]
        z2 = z2[:max_samples]
    return z1, z2


def make_condition_latents(
    condition: str,
    z1_all: torch.Tensor,
    z2_all: torch.Tensor,
    batch_idx: np.ndarray,
    perms: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = z1_all[batch_idx]
    z2 = z2_all[batch_idx]
    if condition == "baseline":
        return z1, z2
    if condition == "sample_shuffle_pair":
        perm_idx = perms["pair"][batch_idx]
        return z1_all[perm_idx], z2_all[perm_idx]
    if condition == "sample_shuffle_independent":
        return z1_all[perms["z1"][batch_idx]], z2_all[perms["z2"][batch_idx]]
    if condition == "coordinate_permute":
        return z1[:, perms["dim"]], z2[:, perms["dim"]]
    if condition == "block_shuffle":
        z1_new = z1.clone()
        z2_new = z2.clone()
        slices = [
            slice(0, RecursiveToMModel.Z_BELIEF_DIM),
            slice(RecursiveToMModel.Z_BELIEF_DIM, RecursiveToMModel.Z_BELIEF_DIM + RecursiveToMModel.Z_INTENT_DIM),
            slice(RecursiveToMModel.Z_BELIEF_DIM + RecursiveToMModel.Z_INTENT_DIM, None),
        ]
        for block_i, sl in enumerate(slices):
            z1_new[:, sl] = z1_all[perms[f"z1_block_{block_i}"][batch_idx]][:, sl]
            z2_new[:, sl] = z2_all[perms[f"z2_block_{block_i}"][batch_idx]][:, sl]
        return z1_new, z2_new
    raise ValueError(f"Unknown condition: {condition}")


def init_metric_accumulators() -> dict[str, dict[str, float]]:
    return {
        condition: {
            "mental1_loss_sum": 0.0,
            "mental1_token_count": 0,
            "mental2_loss_sum": 0.0,
            "mental2_token_count": 0,
            "joint_reward_sse": 0.0,
            "z_combined_sse": 0.0,
            "z1_only_sse": 0.0,
            "reward_count": 0,
            "joint_reward_cosine_sum": 0.0,
            "z_combined_cosine_sum": 0.0,
            "preference_correct": 0,
            "preference_count": 0,
        }
        for condition in CONDITION_ORDER
    }


def finalize_metrics(acc: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for condition, m in acc.items():
        reward_count = max(int(m["reward_count"]), 1)
        pref_count = max(int(m["preference_count"]), 1)
        out[condition] = {
            "mental1_nll": m["mental1_loss_sum"] / max(int(m["mental1_token_count"]), 1),
            "mental2_nll": m["mental2_loss_sum"] / max(int(m["mental2_token_count"]), 1),
            "joint_reward_mse": m["joint_reward_sse"] / reward_count,
            "z_combined_mse": m["z_combined_sse"] / reward_count,
            "z1_only_mse": m["z1_only_sse"] / reward_count,
            "joint_reward_cosine": m["joint_reward_cosine_sum"] / pref_count,
            "z_combined_cosine": m["z_combined_cosine_sum"] / pref_count,
            "preference_acc": m["preference_correct"] / pref_count,
            "n_reward_examples": reward_count,
            "n_preference_examples": pref_count,
            "mental1_tokens": int(m["mental1_token_count"]),
            "mental2_tokens": int(m["mental2_token_count"]),
        }
    baseline = out["baseline"]
    for condition, row in out.items():
        for metric, _ in PLOT_METRICS:
            base = max(float(baseline[metric]), 1e-12)
            row[f"{metric}_ratio_to_baseline"] = float(row[metric] / base)
            row[f"{metric}_delta"] = float(row[metric] - baseline[metric])
        row["preference_acc_delta"] = float(row["preference_acc"] - baseline["preference_acc"])
    return out


@torch.no_grad()
def run_intervention(args) -> dict[str, dict[str, float]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    tokenizer, model = load_model(args.model_name, Path(args.checkpoint_dir), device, args.z_dim)
    val_dataset = build_val_dataset(args, tokenizer)
    z1_np, z2_np = load_latents(Path(args.latents_path), args.max_samples)
    n = min(len(val_dataset), z1_np.shape[0])
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)
    z1_np = z1_np[:n]
    z2_np = z2_np[:n]
    print(f"Evaluating {n} held-out samples with checkpoint {args.checkpoint_dir}", flush=True)

    perms = {
        "pair": rng.permutation(n),
        "z1": rng.permutation(n),
        "z2": rng.permutation(n),
        "dim": rng.permutation(z1_np.shape[1]),
    }
    for block_i in range(3):
        perms[f"z1_block_{block_i}"] = rng.permutation(n)
        perms[f"z2_block_{block_i}"] = rng.permutation(n)

    z1_all = torch.from_numpy(z1_np).to(device)
    z2_all = torch.from_numpy(z2_np).to(device)
    acc = init_metric_accumulators()

    for start in range(0, n, args.batch_size):
        end = min(n, start + args.batch_size)
        batch_positions = np.arange(start, end)
        items = [val_dataset[int(i)] for i in batch_positions]
        batch = collate_fn(items, tokenizer)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        mental1_ids = batch["mental1_input_ids"].to(device)
        mental1_mask = batch["mental1_attention_mask"].to(device)
        mental2_ids = batch["mental2_input_ids"].to(device)
        mental2_mask = batch["mental2_attention_mask"].to(device)
        rewards = batch["reward_vec"].to(device)
        has_neg = batch["has_negative"].to(device) > 0

        for condition in CONDITION_ORDER:
            z1, z2 = make_condition_latents(condition, z1_all, z2_all, batch_positions, perms)
            with amp_context(device):
                mental1_losses = decoder_token_losses(model, z1, model.mental1_decoder, mental1_ids, mental1_mask)
                mental2_losses = decoder_token_losses(model, z2, model.mental2_decoder, mental2_ids, mental2_mask)
                pos_reward = model.forward_reward_with_z(z1, z2, pos_ids, pos_mask)
                neg_reward = model.forward_reward_with_z(z1, z2, neg_ids, neg_mask)
                z1_only_reward = model.z1_only_reward_head(z1)
                z_combined_reward = model.z_combined_reward_head(torch.cat([z1, z2], dim=1))

            m = acc[condition]
            loss_sum, count = nll_sum_count(mental1_losses, mental1_mask)
            m["mental1_loss_sum"] += loss_sum
            m["mental1_token_count"] += count
            loss_sum, count = nll_sum_count(mental2_losses, mental2_mask)
            m["mental2_loss_sum"] += loss_sum
            m["mental2_token_count"] += count

            reward_dim_count = rewards.numel()
            m["joint_reward_sse"] += F.mse_loss(pos_reward.float(), rewards.float(), reduction="sum").item()
            m["z_combined_sse"] += F.mse_loss(z_combined_reward.float(), rewards.float(), reduction="sum").item()
            m["z1_only_sse"] += F.mse_loss(z1_only_reward.float(), rewards.float(), reduction="sum").item()
            m["reward_count"] += reward_dim_count
            m["joint_reward_cosine_sum"] += cosine_mean(pos_reward.float(), rewards.float()) * rewards.size(0)
            m["z_combined_cosine_sum"] += cosine_mean(z_combined_reward.float(), rewards.float()) * rewards.size(0)
            pos_scalar = pos_reward.float().mean(dim=1)
            neg_scalar = neg_reward.float().mean(dim=1)
            valid = has_neg
            if valid.any():
                m["preference_correct"] += int((pos_scalar[valid] > neg_scalar[valid]).sum().item())
                m["preference_count"] += int(valid.sum().item())

        if (start // args.batch_size + 1) % args.log_every == 0:
            print(f"  processed {end}/{n}", flush=True)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return finalize_metrics(acc)


def write_outputs(metrics: dict[str, dict[str, float]], args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    fieldnames = ["condition"] + list(next(iter(metrics.values())).keys())
    with (out_dir / "latent_scramble_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITION_ORDER:
            writer.writerow({"condition": condition, **metrics[condition]})

    lines = [
        "# Latent scramble intervention",
        "",
        "| condition | mental1 NLL | mental2 NLL | z-only MSE | joint MSE | pref acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        row = metrics[condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {row['mental1_nll']:.4f} | "
            f"{row['mental2_nll']:.4f} | {row['z_combined_mse']:.5f} | "
            f"{row['joint_reward_mse']:.5f} | {row['preference_acc']:.4f} |"
        )
    lines += [
        "",
        "## Degradation relative to original z",
        "",
        "| condition | mental1 x | mental2 x | z-only MSE x | joint MSE x |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER[1:]:
        row = metrics[condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | "
            f"{row['mental1_nll_ratio_to_baseline']:.2f} | "
            f"{row['mental2_nll_ratio_to_baseline']:.2f} | "
            f"{row['z_combined_mse_ratio_to_baseline']:.2f} | "
            f"{row['joint_reward_mse_ratio_to_baseline']:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))


def set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.1,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_degradation(metrics: dict[str, dict[str, float]], out_dir: Path) -> None:
    set_plot_style()
    conditions = CONDITION_ORDER[1:]
    metric_keys = [m[0] for m in PLOT_METRICS]
    metric_labels = [m[1] for m in PLOT_METRICS]
    data = np.asarray(
        [[metrics[c][f"{metric}_ratio_to_baseline"] for c in conditions] for metric in metric_keys],
        dtype=float,
    )
    y = np.arange(len(metric_keys))
    bar_h = 0.16
    offsets = np.linspace(-0.24, 0.24, len(conditions))

    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=300)
    for offset, condition_i, condition in zip(offsets, range(len(conditions)), conditions):
        ax.barh(
            y + offset,
            data[:, condition_i],
            height=bar_h,
            color=CONDITION_COLORS[condition],
            edgecolor="#2F2F2F",
            linewidth=0.6,
            label=CONDITION_LABELS[condition],
        )
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yticks(y, metric_labels)
    ax.invert_yaxis()
    ax.set_xlabel("degradation factor vs original z", fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", color="#D8DDE3", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.legend(loc="lower right", frameon=True, fontsize=7.7)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.28, right=0.985, bottom=0.21, top=0.96)
    fig.savefig(out_dir / "fig_latent_scramble_degradation.png", pad_inches=0.02)
    fig.savefig(out_dir / "fig_latent_scramble_degradation.pdf", pad_inches=0.02)
    plt.close(fig)


def plot_baseline_vs_shuffle(metrics: dict[str, dict[str, float]], out_dir: Path) -> None:
    set_plot_style()
    chosen = ["baseline", "sample_shuffle_pair", "coordinate_permute"]
    metric_keys = ["mental1_nll", "mental2_nll", "z_combined_mse", "joint_reward_mse"]
    labels = ["1st-order\nmental", "2nd-order\nmental", "z-only\nreward", "joint\nreward"]
    x = np.arange(len(metric_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(5.2, 2.55), dpi=300)
    for i, condition in enumerate(chosen):
        vals = [metrics[condition][key] for key in metric_keys]
        # Use baseline-normalized values so NLL and MSE can share one axis.
        vals = [val / max(metrics["baseline"][key], 1e-12) for val, key in zip(vals, metric_keys)]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=CONDITION_COLORS[condition],
            edgecolor="#2F2F2F",
            linewidth=0.7,
            label=CONDITION_LABELS[condition],
        )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("normalized error", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.9)
    ax.grid(axis="x", visible=False)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.legend(loc="upper left", frameon=True, fontsize=8)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.96)
    fig.savefig(out_dir / "fig_latent_scramble_original_vs_scrambled.png", pad_inches=0.02)
    fig.savefig(out_dir / "fig_latent_scramble_original_vs_scrambled.pdf", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--checkpoint_dir",
        default="projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/epoch_2",
    )
    parser.add_argument(
        "--latents_path",
        default="projects/sotopia/experiments/runs/stage1/bit_epoch_sweep_latents_stage1val1508/latents_epoch_2.npz",
    )
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument(
        "--target_variant",
        default="structured_bit",
        choices=["structured_bit", "flat_mental_summary", "shuffled_mental", "no_mental"],
        help="Mental target format used for held-out decoder NLL.",
    )
    parser.add_argument(
        "--flat_summary_jsonl",
        default="projects/sotopia/experiments/runs/flat_mental_summary_gpt4omini_cache.jsonl",
        help="LLM-fused flat summaries used when --target_variant flat_mental_summary.",
    )
    parser.add_argument(
        "--output_dir",
        default="projects/sotopia/experiments/runs/stage1/latent_scramble_intervention_epoch2",
    )
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--max_mental_len", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_intervention(args)
    out_dir = Path(args.output_dir)
    write_outputs(metrics, args)
    plot_degradation(metrics, out_dir)
    plot_baseline_vs_shuffle(metrics, out_dir)
    print(f"Wrote latent scramble intervention to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
