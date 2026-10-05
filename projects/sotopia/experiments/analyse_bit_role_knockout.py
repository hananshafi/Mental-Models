#!/usr/bin/env python3
"""Causal role-knockout analysis for BIT vs flat/shuffled supervision.

This experiment is meant to answer the decomposition concern more directly
than a generic downstream score:

  If z is partitioned as belief / intent / thought, does removing the belief
  block selectively damage belief-token decoding, and likewise for intent and
  thought?

For each held-out SOTOPIA validation example, the script:
  1. encodes the observed context into deterministic z1 and z2 means;
  2. decodes the original structured mental text;
  3. replaces one latent sub-block with the held-out mean sub-block;
  4. measures the increase in token NLL on each role span.

The resulting 3x3 matrix is a causal role-addressability score. Rows are the
knocked-out latent block, columns are the target role span. A good BIT model
should have larger diagonal values than off-diagonal values.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    CUSTOM_HEAD_NAMES,
    REWARD_DIM,
    RecursiveToMDataset,
    RecursiveToMModel,
)


ROLE_ORDER = ["belief", "intent", "thought"]
ROLE_LABELS = {"belief": "Belief", "intent": "Intent", "thought": "Thought"}
ORDER_LABELS = {"mental1": "first_order", "mental2": "second_order"}

VARIANT_ORDER = ["structured_bit", "flat_mental_summary", "shuffled_mental"]
VARIANT_LABELS = {
    "structured_bit": "BIT",
    "flat_mental_summary": "Flat",
    "shuffled_mental": "Shuffled",
}
VARIANT_COLORS = {
    "structured_bit": "#2F6FAE",
    "flat_mental_summary": "#D28B45",
    "shuffled_mental": "#5BA79D",
}

SUB_SLICES = {
    "belief": slice(0, RecursiveToMModel.Z_BELIEF_DIM),
    "intent": slice(
        RecursiveToMModel.Z_BELIEF_DIM,
        RecursiveToMModel.Z_BELIEF_DIM + RecursiveToMModel.Z_INTENT_DIM,
    ),
    "thought": slice(
        RecursiveToMModel.Z_BELIEF_DIM + RecursiveToMModel.Z_INTENT_DIM,
        RecursiveToMModel.Z_BELIEF_DIM
        + RecursiveToMModel.Z_INTENT_DIM
        + RecursiveToMModel.Z_THOUGHT_DIM,
    ),
}

ROLE_VALUE_PATTERNS = {
    "mental1": {
        "belief": r"Partner Belief:\s*(.*?)(?=\s*\|\s*Strategic Intent:|\s*\|\s*Thought Process:|$)",
        "intent": r"Strategic Intent:\s*(.*?)(?=\s*\|\s*Thought Process:|$)",
        "thought": r"Thought Process:\s*(.*)$",
    },
    "mental2": {
        "belief": r"Second-Order Belief:\s*(.*?)(?=\s*\|\s*Second-Order Intent:|\s*\|\s*Second-Order Thought:|$)",
        "intent": r"Second-Order Intent:\s*(.*?)(?=\s*\|\s*Second-Order Thought:|$)",
        "thought": r"Second-Order Thought:\s*(.*)$",
    },
}


def parse_variant_specs(specs: Iterable[str]) -> dict[str, Path]:
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Variant spec must be name=/path, got {spec!r}")
        name, path = spec.split("=", 1)
        parsed[name.strip()] = Path(path.strip())
    return parsed


def role_value_spans(text: str, order: str) -> dict[str, tuple[int, int]]:
    spans = {}
    for role, pattern in ROLE_VALUE_PATTERNS[order].items():
        match = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        start, end = match.span(1)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans[role] = (start, end)
    return spans


def token_role_mask(text: str, offsets, order: str) -> torch.Tensor:
    spans = role_value_spans(text, order)
    mask = torch.zeros((len(ROLE_ORDER), len(offsets)), dtype=torch.float32)
    for role_i, role in enumerate(ROLE_ORDER):
        if role not in spans:
            continue
        start, end = spans[role]
        for tok_i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:
                continue
            if max(tok_start, start) < min(tok_end, end):
                mask[role_i, tok_i] = 1.0
    return mask


class RoleKnockoutDataset(Dataset):
    def __init__(
        self,
        base_dataset: RecursiveToMDataset,
        indices: list[int],
        tokenizer,
        max_ctx_len: int,
        max_mental_len: int,
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Role span masking requires a fast tokenizer with offset mappings.")
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_mental_len = max_mental_len

    def __len__(self) -> int:
        return len(self.indices)

    def _encode_mental(self, text: str, order: str) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text if text and text.strip() else "N/A",
            truncation=True,
            max_length=self.max_mental_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").squeeze(0).tolist()
        role_mask = token_role_mask(text, offsets, order)
        return {
            "input_ids": enc.input_ids.squeeze(0).long(),
            "attention_mask": enc.attention_mask.squeeze(0).long(),
            "role_mask": role_mask,
        }

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.base_dataset.samples[self.indices[item]]
        ctx_text = self.base_dataset._format_context(sample)
        ctx_enc = self.tokenizer(
            ctx_text,
            truncation=True,
            max_length=self.max_ctx_len,
            padding="max_length",
            return_tensors="pt",
        )
        mental1 = self._encode_mental(sample.get("mental1_text", ""), "mental1")
        mental2 = self._encode_mental(sample.get("mental2_text", ""), "mental2")
        return {
            "ctx_input_ids": ctx_enc.input_ids.squeeze(0).long(),
            "ctx_attention_mask": ctx_enc.attention_mask.squeeze(0).long(),
            "mental1_input_ids": mental1["input_ids"],
            "mental1_attention_mask": mental1["attention_mask"],
            "mental1_role_mask": mental1["role_mask"],
            "mental2_input_ids": mental2["input_ids"],
            "mental2_attention_mask": mental2["attention_mask"],
            "mental2_role_mask": mental2["role_mask"],
        }


def load_checkpoint(model_name: str, ckpt_dir: Path, device: torch.device, z_dim: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    base.config.use_cache = False
    base = PeftModel.from_pretrained(base, ckpt_dir / "lora_adapter")
    model = RecursiveToMModel(base, reward_dim=REWARD_DIM, z_dim=z_dim)
    for head_name in CUSTOM_HEAD_NAMES:
        path = ckpt_dir / f"{head_name}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        getattr(model, head_name).load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode_deterministic(model: RecursiveToMModel, ctx_ids, ctx_mask):
    ctx_last = model._encode_context(ctx_ids, ctx_mask)
    ctx_for_heads = ctx_last.to(model.z1_mu.weight.dtype)
    z1 = model.z1_mu(ctx_for_heads)
    z2_input = torch.cat([ctx_for_heads, z1], dim=1)
    z2 = model.z2_mu(z2_input)
    return z1, z2


def decoder_token_nll(
    model: RecursiveToMModel,
    z: torch.Tensor,
    decoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    z_prefix = model._expand_z_to_prefix(z, decoder)
    embedding_layer = model.base_model.get_input_embeddings()
    mental_embeds = embedding_layer(input_ids)

    attended, _ = decoder["cross_attn"](query=mental_embeds, key=z_prefix, value=z_prefix)
    h = decoder["ln"](mental_embeds + attended)
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
    shift_valid = attention_mask[:, 1:].contiguous()
    shift_labels[shift_valid == 0] = -100
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    return losses.view(input_ids.size(0), -1)


def accumulate_role_nll(
    token_losses: torch.Tensor,
    role_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    shifted_role_mask = role_mask[:, :, 1:].to(token_losses.device, dtype=token_losses.dtype)
    shifted_valid = attention_mask[:, 1:].unsqueeze(1).to(token_losses.device, dtype=token_losses.dtype)
    shifted_role_mask = shifted_role_mask * shifted_valid
    role_sums = (token_losses.unsqueeze(1) * shifted_role_mask).sum(dim=(0, 2))
    role_counts = shifted_role_mask.sum(dim=(0, 2))
    sums += role_sums.float().cpu().numpy()
    counts += role_counts.float().cpu().numpy()


@torch.no_grad()
def compute_latent_means(model, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    z1_sum = torch.zeros(RecursiveToMModel.Z_BELIEF_DIM + RecursiveToMModel.Z_INTENT_DIM + RecursiveToMModel.Z_THOUGHT_DIM)
    z2_sum = torch.zeros_like(z1_sum)
    n = 0
    for batch in loader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            z1, z2 = encode_deterministic(model, ctx_ids, ctx_mask)
        z1_sum += z1.float().sum(dim=0).cpu()
        z2_sum += z2.float().sum(dim=0).cpu()
        n += z1.size(0)
    return (z1_sum / max(n, 1)).to(device), (z2_sum / max(n, 1)).to(device)


def mean_ablate(z: torch.Tensor, role: str, mean_z: torch.Tensor) -> torch.Tensor:
    z_ablated = z.clone()
    sl = SUB_SLICES[role]
    z_ablated[:, sl] = mean_z[sl].to(device=z.device, dtype=z.dtype).unsqueeze(0)
    return z_ablated


@torch.no_grad()
def run_variant_knockout(
    variant: str,
    ckpt_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    print(f"\n=== {variant}: loading checkpoint ===", flush=True)
    tokenizer, model = load_checkpoint(args.model_name, ckpt_dir, device, args.z_dim)
    base_dataset = RecursiveToMDataset(
        args.data_path,
        tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
    )
    val_size = min(max(1, int(len(base_dataset) * args.val_ratio)), len(base_dataset) - 1)
    train_size = len(base_dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(base_dataset, [train_size, val_size], generator=gen)
    indices = list(val_dataset.indices)
    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]

    eval_dataset = RoleKnockoutDataset(
        base_dataset,
        indices,
        tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_mental_len=args.max_mental_len,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"  heldout samples={len(eval_dataset)}", flush=True)
    z1_mean, z2_mean = compute_latent_means(model, loader, device)

    base_sums = {order: np.zeros(3, dtype=np.float64) for order in ORDER_LABELS}
    base_counts = {order: np.zeros(3, dtype=np.float64) for order in ORDER_LABELS}
    ablate_sums = {
        order: {source: np.zeros(3, dtype=np.float64) for source in ROLE_ORDER}
        for order in ORDER_LABELS
    }
    ablate_counts = {
        order: {source: np.zeros(3, dtype=np.float64) for source in ROLE_ORDER}
        for order in ORDER_LABELS
    }

    for batch_i, batch in enumerate(loader, start=1):
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        mental1_ids = batch["mental1_input_ids"].to(device)
        mental1_mask = batch["mental1_attention_mask"].to(device)
        mental1_role = batch["mental1_role_mask"].to(device)
        mental2_ids = batch["mental2_input_ids"].to(device)
        mental2_mask = batch["mental2_attention_mask"].to(device)
        mental2_role = batch["mental2_role_mask"].to(device)

        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            z1, z2 = encode_deterministic(model, ctx_ids, ctx_mask)
            base1 = decoder_token_nll(model, z1, model.mental1_decoder, mental1_ids, mental1_mask)
            base2 = decoder_token_nll(model, z2, model.mental2_decoder, mental2_ids, mental2_mask)
        accumulate_role_nll(base1, mental1_role, mental1_mask, base_sums["mental1"], base_counts["mental1"])
        accumulate_role_nll(base2, mental2_role, mental2_mask, base_sums["mental2"], base_counts["mental2"])

        for source_role in ROLE_ORDER:
            with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
                z1_ablated = mean_ablate(z1, source_role, z1_mean)
                z2_ablated = mean_ablate(z2, source_role, z2_mean)
                abl1 = decoder_token_nll(
                    model, z1_ablated, model.mental1_decoder, mental1_ids, mental1_mask
                )
                abl2 = decoder_token_nll(
                    model, z2_ablated, model.mental2_decoder, mental2_ids, mental2_mask
                )
            accumulate_role_nll(
                abl1,
                mental1_role,
                mental1_mask,
                ablate_sums["mental1"][source_role],
                ablate_counts["mental1"][source_role],
            )
            accumulate_role_nll(
                abl2,
                mental2_role,
                mental2_mask,
                ablate_sums["mental2"][source_role],
                ablate_counts["mental2"][source_role],
            )
        if batch_i % args.log_every == 0:
            print(f"  processed {min(batch_i * args.batch_size, len(eval_dataset))}/{len(eval_dataset)}", flush=True)

    rows = []
    matrices_by_order = {}
    for order in ORDER_LABELS:
        base_nll = base_sums[order] / np.maximum(base_counts[order], 1.0)
        matrix = np.zeros((3, 3), dtype=np.float64)
        for source_i, source_role in enumerate(ROLE_ORDER):
            ablated_nll = (
                ablate_sums[order][source_role]
                / np.maximum(ablate_counts[order][source_role], 1.0)
            )
            delta = ablated_nll - base_nll
            matrix[source_i, :] = delta
            for target_i, target_role in enumerate(ROLE_ORDER):
                rows.append(
                    {
                        "variant": variant,
                        "order": ORDER_LABELS[order],
                        "ablated_block": source_role,
                        "target_role": target_role,
                        "base_nll": float(base_nll[target_i]),
                        "ablated_nll": float(ablated_nll[target_i]),
                        "delta_nll": float(delta[target_i]),
                        "token_count": int(base_counts[order][target_i]),
                        "is_diagonal": source_role == target_role,
                    }
                )
        matrices_by_order[order] = matrix

    matrix_avg = np.mean(list(matrices_by_order.values()), axis=0)
    diag_mask = np.eye(3, dtype=bool)
    off_mask = ~diag_mask
    summary = {
        "variant": variant,
        "n_samples": len(eval_dataset),
        "matrix_average": matrix_avg.tolist(),
        "matrix_mental1": matrices_by_order["mental1"].tolist(),
        "matrix_mental2": matrices_by_order["mental2"].tolist(),
        "diag_delta_nll": float(matrix_avg[diag_mask].mean()),
        "offdiag_delta_nll": float(matrix_avg[off_mask].mean()),
        "selectivity_gap": float(matrix_avg[diag_mask].mean() - matrix_avg[off_mask].mean()),
    }
    print(
        f"  diag delta={summary['diag_delta_nll']:.4f}, "
        f"off={summary['offdiag_delta_nll']:.4f}, "
        f"gap={summary['selectivity_gap']:+.4f}",
        flush=True,
    )

    del model, tokenizer, base_dataset, eval_dataset, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


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
            "axes.linewidth": 1.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def plot_heatmaps(summary: dict[str, dict], out_path: Path) -> None:
    set_plot_style()
    variants = [v for v in VARIANT_ORDER if v in summary]
    matrices = [np.asarray(summary[v]["matrix_average"], dtype=float) for v in variants]
    vmin = min(0.0, min(float(np.min(m)) for m in matrices))
    vmax = max(float(np.max(m)) for m in matrices)
    cmap = "YlGnBu" if vmin >= 0 else "RdBu_r"

    fig, axes = plt.subplots(1, len(variants), figsize=(6.3, 2.05), dpi=300)
    if len(variants) == 1:
        axes = [axes]
    im = None
    for ax, variant, matrix in zip(axes, variants, matrices):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=11, fontweight="bold", pad=5)
        ax.set_xticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.set_yticks(range(3), [ROLE_LABELS[r][0] for r in ROLE_ORDER])
        ax.tick_params(length=0)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
        for idx in range(3):
            ax.add_patch(
                plt.Rectangle(
                    (idx - 0.5, idx - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.5,
                )
            )
        if ax is axes[0]:
            ax.set_ylabel("knocked-out block", fontsize=9, fontweight="bold")
        ax.set_xlabel("target role", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.86, pad=0.20)
    cbar.outline.set_visible(False)
    cbar.set_label("increase in held-out role-token NLL", fontsize=8.5, fontweight="bold", labelpad=2)
    cbar.ax.tick_params(labelsize=7.5, length=0)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.82, wspace=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.02)
    plt.close(fig)


def plot_selectivity(summary: dict[str, dict], out_path: Path) -> None:
    set_plot_style()
    variants = [v for v in VARIANT_ORDER if v in summary]
    y = np.arange(len(variants))[::-1]
    diag = np.asarray([summary[v]["diag_delta_nll"] for v in variants], dtype=float)
    off = np.asarray([summary[v]["offdiag_delta_nll"] for v in variants], dtype=float)

    fig, ax = plt.subplots(figsize=(4.85, 1.85), dpi=300)
    for yi, variant, off_val, diag_val in zip(y, variants, off, diag):
        ax.plot([off_val, diag_val], [yi, yi], color="#C4CAD1", linewidth=6, solid_capstyle="round")
        ax.annotate(
            "",
            xy=(diag_val, yi),
            xytext=(off_val, yi),
            arrowprops=dict(
                arrowstyle="-|>",
                color=VARIANT_COLORS.get(variant, "#555555"),
                linewidth=2.2,
                mutation_scale=14,
            ),
        )
        ax.scatter(off_val, yi, s=100, color="#D2D7DD", edgecolor="#303030", linewidth=0.9, zorder=3)
        ax.scatter(
            diag_val,
            yi,
            s=145,
            color=VARIANT_COLORS.get(variant, "#555555"),
            edgecolor="#303030",
            linewidth=1.0,
            zorder=4,
        )
    ax.set_yticks(y, [VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("role-token NLL increase after knockout", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#D7DDE3", linewidth=0.9)
    ax.set_axisbelow(True)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    xmin = min(float(np.min(off)), float(np.min(diag)), 0.0) - 0.002
    xmax = max(float(np.max(off)), float(np.max(diag)), 0.001) + 0.010
    ax.set_xlim(xmin, xmax)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.31, top=0.93)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), pad_inches=0.02)
    plt.close(fig)


def write_outputs(out_dir: Path, rows: list[dict], summary: dict[str, dict], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "role_knockout_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    lines = [
        "# Role-knockout selectivity",
        "",
        "| variant | diagonal ΔNLL | off-diagonal ΔNLL | selectivity gap | samples |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in [v for v in VARIANT_ORDER if v in summary]:
        s = summary[variant]
        lines.append(
            f"| {VARIANT_LABELS.get(variant, variant)} | {s['diag_delta_nll']:.4f} | "
            f"{s['offdiag_delta_nll']:.4f} | {s['selectivity_gap']:+.4f} | {s['n_samples']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "structured_bit=projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/epoch_2",
            "flat_mental_summary=projects/sotopia/experiments/runs/stage1/flat_mental_summary_qwen7b_seed42/best",
            "shuffled_mental=projects/sotopia/experiments/runs/stage1/shuffled_mental_qwen7b_seed42/best",
        ],
    )
    parser.add_argument(
        "--output_dir",
        default="projects/sotopia/experiments/runs/stage1/bit_role_knockout",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--max_mental_len", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Debug/sample cap; 0 means all val samples.")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    variant_paths = parse_variant_specs(args.variants)
    all_rows = []
    all_summary = {}
    for variant in [v for v in VARIANT_ORDER if v in variant_paths]:
        rows, summary = run_variant_knockout(variant, variant_paths[variant], args, device)
        all_rows.extend(rows)
        all_summary[variant] = summary

    out_dir = Path(args.output_dir)
    write_outputs(out_dir, all_rows, all_summary, args)
    plot_heatmaps(all_summary, out_dir / "fig_role_knockout_heatmaps.png")
    plot_selectivity(all_summary, out_dir / "fig_role_knockout_selectivity.png")
    print(f"\nWrote role-knockout analysis to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
