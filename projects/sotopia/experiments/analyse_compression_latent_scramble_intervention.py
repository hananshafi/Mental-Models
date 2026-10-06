#!/usr/bin/env python3
"""Latent scrambling intervention for the compression-only VAE baseline.

This mirrors analyse_latent_scramble_intervention.py, but for the pure
compression VAE. It keeps examples/model weights fixed and only scrambles
z_compress, then measures:

  - reconstruction NLL of the compression target;
  - reward-probe MSE for [z_compress, response_hidden] -> reward.

If the compressed z is meaningful, scrambling should hurt reconstruction and
reward probing. The comparison with BIT should focus especially on reward-head
degradation, since both heads use z plus response hidden.
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
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_train_compression_vae import (  # noqa: E402
    COMPRESSION_CUSTOM_HEAD_NAMES,
    CompressionVAEModel,
    SotopiaCompressionDataset,
    collate_fn,
)


CONDITION_ORDER = ["baseline", "sample_shuffle", "coordinate_permute", "mean_z"]
CONDITION_LABELS = {
    "baseline": "original z",
    "sample_shuffle": "sample-shuffled z",
    "coordinate_permute": "coordinate scramble",
    "mean_z": "mean z",
}
CONDITION_COLORS = {
    "baseline": "#2F6FAE",
    "sample_shuffle": "#5BA79D",
    "coordinate_permute": "#C84C4C",
    "mean_z": "#8B6BB8",
}


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_model(args, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    base = PeftModel.from_pretrained(base, Path(args.checkpoint_dir) / "lora_adapter")
    model = CompressionVAEModel(
        base_model=base,
        z_dim=args.z_dim,
        num_memory_tokens=args.num_memory_tokens,
        max_target_len=args.max_target_len,
        decoder_layers=args.decoder_layers,
    )
    for head_name in COMPRESSION_CUSTOM_HEAD_NAMES:
        path = Path(args.checkpoint_dir) / f"{head_name}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        getattr(model, head_name).load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def build_val_loader(args, tokenizer) -> DataLoader:
    dataset = SotopiaCompressionDataset(
        args.data_path,
        tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_target_len=args.max_target_len,
        max_resp_len=args.max_resp_len,
        target_mode=args.target_mode,
        summary_history_turns=args.summary_history_turns,
    )
    val_size = min(max(1, int(len(dataset) * args.val_ratio)), len(dataset) - 1)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
    if args.max_samples and args.max_samples > 0:
        val_dataset.indices = list(val_dataset.indices[: args.max_samples])
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def precompute_z(model: CompressionVAEModel, loader: DataLoader, device: torch.device) -> torch.Tensor:
    chunks = []
    for batch in loader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        with amp_context(device):
            z = model.encode_context_to_z(ctx_ids, ctx_mask)
        chunks.append(z.float().cpu())
    return torch.cat(chunks, dim=0).to(device)


def token_nll_from_logits(logits: torch.Tensor, target_ids: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, int]:
    labels = target_ids.clone()
    labels[target_mask == 0] = -100
    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(target_ids.size(0), -1)
    valid = target_mask.to(logits.device, dtype=losses.dtype)
    return float((losses * valid).sum().float().item()), int(valid.sum().item())


def cosine_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    denom = pred.norm(dim=1).clamp_min(1e-8) * target.norm(dim=1).clamp_min(1e-8)
    return ((pred * target).sum(dim=1) / denom).mean().float().item()


def make_condition_z(
    condition: str,
    z_all: torch.Tensor,
    batch_idx: np.ndarray,
    perm: np.ndarray,
    dim_perm: np.ndarray,
    z_mean: torch.Tensor,
) -> torch.Tensor:
    z = z_all[batch_idx]
    if condition == "baseline":
        return z
    if condition == "sample_shuffle":
        return z_all[perm[batch_idx]]
    if condition == "coordinate_permute":
        return z[:, dim_perm]
    if condition == "mean_z":
        return z_mean.unsqueeze(0).expand_as(z)
    raise ValueError(condition)


def init_acc() -> dict[str, dict[str, float]]:
    return {
        condition: {
            "recon_loss_sum": 0.0,
            "recon_token_count": 0,
            "reward_sse": 0.0,
            "reward_count": 0,
            "reward_cosine_sum": 0.0,
            "pref_correct": 0,
            "pref_count": 0,
        }
        for condition in CONDITION_ORDER
    }


def finalize(acc: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for condition, row in acc.items():
        reward_count = max(int(row["reward_count"]), 1)
        pref_count = max(int(row["pref_count"]), 1)
        out[condition] = {
            "recon_nll": row["recon_loss_sum"] / max(int(row["recon_token_count"]), 1),
            "reward_probe_mse": row["reward_sse"] / reward_count,
            "reward_probe_cosine": row["reward_cosine_sum"] / pref_count,
            "preference_acc": row["pref_correct"] / pref_count,
            "n_reward_examples": reward_count,
            "n_preference_examples": pref_count,
            "recon_tokens": int(row["recon_token_count"]),
        }
    baseline = out["baseline"]
    for condition, row in out.items():
        for metric in ["recon_nll", "reward_probe_mse"]:
            row[f"{metric}_ratio_to_baseline"] = row[metric] / max(baseline[metric], 1e-12)
            row[f"{metric}_delta"] = row[metric] - baseline[metric]
        row["preference_acc_delta"] = row["preference_acc"] - baseline["preference_acc"]
    return out


@torch.no_grad()
def run(args) -> dict[str, dict[str, float]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    tokenizer, model = load_model(args, device)
    loader = build_val_loader(args, tokenizer)
    n = len(loader.dataset)
    print(f"Evaluating compression VAE z scramble on {n} held-out samples", flush=True)
    z_all = precompute_z(model, loader, device)
    perm = rng.permutation(n)
    dim_perm = rng.permutation(z_all.size(1))
    z_mean = z_all.mean(dim=0)
    acc = init_acc()

    cursor = 0
    for batch_i, batch in enumerate(loader, start=1):
        batch_size = batch["ctx_input_ids"].size(0)
        batch_idx = np.arange(cursor, cursor + batch_size)
        cursor += batch_size
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        rewards = batch["reward_vec"].to(device)
        has_neg = batch["has_negative"].to(device) > 0

        with amp_context(device):
            pos_hidden = model.encode_response(pos_ids, pos_mask)
            neg_hidden = model.encode_response(neg_ids, neg_mask)

        for condition in CONDITION_ORDER:
            z = make_condition_z(condition, z_all, batch_idx, perm, dim_perm, z_mean)
            with amp_context(device):
                logits = model._decode_from_z(z, target_ids.size(1))
                pos_reward = model.reward_probe_head(torch.cat([z, pos_hidden], dim=-1))
                neg_reward = model.reward_probe_head(torch.cat([z, neg_hidden], dim=-1))

            loss_sum, token_count = token_nll_from_logits(logits, target_ids, target_mask)
            row = acc[condition]
            row["recon_loss_sum"] += loss_sum
            row["recon_token_count"] += token_count
            row["reward_sse"] += F.mse_loss(pos_reward.float(), rewards.float(), reduction="sum").item()
            row["reward_count"] += rewards.numel()
            row["reward_cosine_sum"] += cosine_mean(pos_reward.float(), rewards.float()) * rewards.size(0)
            pos_scalar = pos_reward.float().mean(dim=1)
            neg_scalar = neg_reward.float().mean(dim=1)
            if has_neg.any():
                row["pref_correct"] += int((pos_scalar[has_neg] > neg_scalar[has_neg]).sum().item())
                row["pref_count"] += int(has_neg.sum().item())

        if batch_i % args.log_every == 0:
            print(f"  processed {min(cursor, n)}/{n}", flush=True)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return finalize(acc)


def write_outputs(metrics: dict[str, dict[str, float]], args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    fieldnames = ["condition"] + list(next(iter(metrics.values())).keys())
    with (out_dir / "compression_latent_scramble_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITION_ORDER:
            writer.writerow({"condition": condition, **metrics[condition]})

    lines = [
        "# Compression VAE latent scramble intervention",
        "",
        "| condition | recon NLL | reward probe MSE | pref acc |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        row = metrics[condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {row['recon_nll']:.4f} | "
            f"{row['reward_probe_mse']:.5f} | {row['preference_acc']:.4f} |"
        )
    lines += [
        "",
        "## Degradation relative to original z",
        "",
        "| condition | recon x | reward probe MSE x |",
        "|---|---:|---:|",
    ]
    for condition in CONDITION_ORDER[1:]:
        row = metrics[condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | "
            f"{row['recon_nll_ratio_to_baseline']:.2f} | "
            f"{row['reward_probe_mse_ratio_to_baseline']:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))


def set_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update({
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
    })


def plot(metrics: dict[str, dict[str, float]], out_dir: Path) -> None:
    set_style()
    conditions = CONDITION_ORDER[1:]
    metric_keys = ["recon_nll", "reward_probe_mse"]
    metric_labels = ["summary\nreconstruction", "reward\nprobe"]
    x = np.arange(len(metric_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(4.5, 2.35), dpi=300)
    for i, condition in enumerate(conditions):
        vals = [metrics[condition][f"{key}_ratio_to_baseline"] for key in metric_keys]
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
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("error increase", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.9)
    ax.grid(axis="x", visible=False)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.legend(frameon=True, fontsize=7.5)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.25, top=0.96)
    fig.savefig(out_dir / "fig_compression_latent_scramble.png", pad_inches=0.02)
    fig.savefig(out_dir / "fig_compression_latent_scramble.pdf", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--checkpoint_dir",
        default="projects/sotopia/experiments/runs/stage1/compression_vae_qwen7b_summary_seed42/best",
    )
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument(
        "--output_dir",
        default="projects/sotopia/experiments/runs/stage1/latent_scramble_intervention_compression_fullval",
    )
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--target_mode", choices=["summary", "context"], default="summary")
    parser.add_argument("--summary_history_turns", type=int, default=3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--num_memory_tokens", type=int, default=24)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(args)
    out_dir = Path(args.output_dir)
    write_outputs(metrics, args)
    plot(metrics, out_dir)
    print(f"Wrote compression latent scramble intervention to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
