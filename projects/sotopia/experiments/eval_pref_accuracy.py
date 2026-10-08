#!/usr/bin/env python3
"""Compute preference accuracy (pos_reward > neg_reward fraction) for each
SOTOPIA Stage-1 variant checkpoint, on the same val split used for the
existing phase1 metrics.

Output: runs/stage1/variant_eval/phase1_pref_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
def compute_pref_accuracy(model, dataloader, device) -> dict[str, float]:
    """Return {per_dim_acc, per_sample_acc, n_pairs}."""
    n_correct_dim = 0
    n_total_dim = 0
    sum_per_sample = 0.0
    n_samples = 0
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
        pos = out["pos_joint_reward"].float()      # (B, R)
        neg = out["neg_joint_reward"].float()      # (B, R)
        correct = (pos > neg).float()              # (B, R)

        # per-dim accuracy (each (sample, reward_dim) is one judgment)
        mask = has_neg.unsqueeze(1).expand_as(correct)
        n_correct_dim += (correct * mask).sum().item()
        n_total_dim += mask.sum().item()

        # per-sample accuracy (fraction of dims correct, then mean over samples)
        per_sample = correct.mean(dim=1)
        sum_per_sample += (per_sample * has_neg).sum().item()
        n_samples += has_neg.sum().item()

    return {
        "per_dim_acc": n_correct_dim / max(n_total_dim, 1),
        "per_sample_mean_dim_frac": sum_per_sample / max(n_samples, 1),
        "n_pairs_evaluated": int(n_samples),
        "n_dim_judgments": int(n_total_dim),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", required=True, help="tag=ckpt_path")
    p.add_argument("--output_path", required=True)
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
    p.add_argument("--gpu", default="")
    args = p.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for spec in args.variants:
        tag, path = spec.split("=", 1)
        ckpt_dir = Path(path.strip())
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: missing {ckpt_dir}", flush=True)
            continue

        print(f"\n=== {tag}: pref accuracy ===", flush=True)
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

        acc = compute_pref_accuracy(model, val_loader, device)
        print(f"  per_dim_acc          = {acc['per_dim_acc']:.4f}")
        print(f"  per_sample_mean_frac = {acc['per_sample_mean_dim_frac']:.4f}")
        print(f"  n_pairs              = {acc['n_pairs_evaluated']}")
        results[tag] = acc

        del model, tokenizer
        torch.cuda.empty_cache()

    with out_path.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
