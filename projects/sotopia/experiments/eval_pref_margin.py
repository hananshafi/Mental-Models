#!/usr/bin/env python3
"""Compute per-sample preference margins (pos_reward - neg_reward) for each
SOTOPIA Stage-1 variant on the same val split, save raw arrays, and render a
distribution comparison figure.

The preference loss is softplus(neg - pos), so the underlying random variable
is the margin (pos - neg). Mean preference loss compresses this into one number;
this script visualizes the full distribution to expose calibration / confidence
gaps that mean loss conceals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    CUSTOM_HEAD_NAMES, REWARD_DIM,
    RecursiveToMDataset, RecursiveToMModel, collate_fn,
)


def load_variant(model_name: str, ckpt_dir: Path, device: torch.device, z_dim: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    base = PeftModel.from_pretrained(base, ckpt_dir / "lora_adapter")
    model = RecursiveToMModel(base, reward_dim=REWARD_DIM, z_dim=z_dim)
    for head_name in CUSTOM_HEAD_NAMES:
        path = ckpt_dir / f"{head_name}.pth"
        if path.exists():
            getattr(model, head_name).load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
    model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def collect_margins(model, dataloader, device) -> dict[str, np.ndarray]:
    """Return per-(sample,dim) margins (pos - neg)."""
    margin_chunks = []
    has_neg_chunks = []
    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        mental1_ids = batch["mental1_input_ids"].to(device)
        mental1_mask = batch["mental1_attention_mask"].to(device)
        mental2_ids = batch["mental2_input_ids"].to(device)
        mental2_mask = batch["mental2_attention_mask"].to(device)
        expl_ids = batch["expl_input_ids"].to(device)
        expl_mask = batch["expl_attention_mask"].to(device)
        first_pos_token = batch["first_pos_token"].to(device)
        has_neg = batch["has_negative"].to(device)

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            out = model.forward_all(
                ctx_ids, ctx_mask, pos_ids, pos_mask, neg_ids, neg_mask,
                mental1_ids, mental1_mask, mental2_ids, mental2_mask,
                expl_ids, expl_mask, first_pos_token, stop_grad_z1=False,
            )
        pos = out["pos_joint_reward"].float().cpu().numpy()    # (B, R)
        neg = out["neg_joint_reward"].float().cpu().numpy()    # (B, R)
        margin = pos - neg
        margin_chunks.append(margin)
        has_neg_chunks.append(has_neg.cpu().numpy())

    margins = np.concatenate(margin_chunks, axis=0)             # (N, R)
    has_neg = np.concatenate(has_neg_chunks, axis=0)            # (N,)
    return {"margins_per_dim": margins, "has_neg": has_neg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", required=True, help="tag=ckpt_path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path",
                   default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_ctx_len", type=int, default=1024)
    p.add_argument("--max_resp_len", type=int, default=256)
    p.add_argument("--max_mental_len", type=int, default=256)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for spec in args.variants:
        tag, path = spec.split("=", 1)
        ckpt_dir = Path(path.strip())
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: missing {ckpt_dir}", flush=True)
            continue

        print(f"\n=== {tag}: collecting per-sample margins ===", flush=True)
        tokenizer, model = load_variant(args.model_name, ckpt_dir, device, args.z_dim)

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
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            num_workers=args.num_workers,
        )

        out = collect_margins(model, val_loader, device)
        np.savez_compressed(out_dir / f"margins_{tag}.npz", **out)

        margins_per_dim = out["margins_per_dim"]
        has_neg = out["has_neg"]
        # per-sample mean margin (averaged across reward dims), filter to has_neg
        per_sample_mean = margins_per_dim.mean(axis=1)[has_neg > 0]
        # all (sample, dim) judgments
        per_dim_flat = margins_per_dim[has_neg > 0].ravel()

        s = {
            "n_samples_with_neg": int((has_neg > 0).sum()),
            "per_sample_mean_margin": {
                "mean": float(per_sample_mean.mean()),
                "median": float(np.median(per_sample_mean)),
                "p10": float(np.percentile(per_sample_mean, 10)),
                "p25": float(np.percentile(per_sample_mean, 25)),
                "p75": float(np.percentile(per_sample_mean, 75)),
                "p90": float(np.percentile(per_sample_mean, 90)),
                "min": float(per_sample_mean.min()),
                "max": float(per_sample_mean.max()),
                "frac_correct": float((per_sample_mean > 0).mean()),
            },
            "per_dim_judgment": {
                "mean": float(per_dim_flat.mean()),
                "median": float(np.median(per_dim_flat)),
                "p10": float(np.percentile(per_dim_flat, 10)),
                "frac_correct": float((per_dim_flat > 0).mean()),
            },
        }
        summary[tag] = s
        print(f"  n_with_neg={s['n_samples_with_neg']}  "
              f"mean={s['per_sample_mean_margin']['mean']:+.4f}  "
              f"median={s['per_sample_mean_margin']['median']:+.4f}  "
              f"p10={s['per_sample_mean_margin']['p10']:+.4f}  "
              f"frac>0={s['per_sample_mean_margin']['frac_correct']:.4f}", flush=True)

        del model, tokenizer
        torch.cuda.empty_cache()

    with (out_dir / "margin_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nWrote {out_dir}/margin_summary.json", flush=True)


if __name__ == "__main__":
    main()
