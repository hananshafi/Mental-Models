#!/usr/bin/env python3
"""Extract held-out validation latents for BIT checkpoint epoch sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mental_variants import extract_val_latents, load_variant_checkpoint  # noqa: E402
from stage1_train_coupled_mental_reward_v3 import RecursiveToMDataset, collate_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", default="projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3")
    parser.add_argument("--tags", nargs="+", default=["epoch_1", "epoch_2", "epoch_3", "epoch_4", "epoch_5", "best"])
    parser.add_argument("--output_dir", default="projects/sotopia/experiments/runs/stage1/bit_epoch_sweep_latents_stage1val1508")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--max_mental_len", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing fixed Stage-1 held-out split...", flush=True)
    # Use the first tokenizer/model load only after dataset construction needs tokenizer.
    first_ckpt = Path(args.checkpoint_root) / args.tags[0]
    tokenizer, model = load_variant_checkpoint(args.model_name, first_ckpt, device, args.z_dim)
    dataset = RecursiveToMDataset(
        args.data_path,
        tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
    )
    val_size = min(max(1, int(len(dataset) * args.val_ratio)), len(dataset) - 1)
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=args.num_workers,
    )
    print(f"  held-out records={len(val_dataset)}", flush=True)

    metadata = {
        "checkpoint_root": args.checkpoint_root,
        "tags": args.tags,
        "model_name": args.model_name,
        "data_path": args.data_path,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_records": len(val_dataset),
        "val_indices": list(map(int, val_dataset.indices)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for tag_i, tag in enumerate(args.tags):
        out_path = out_dir / f"latents_{tag}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {tag}: {out_path} exists", flush=True)
            continue
        ckpt = Path(args.checkpoint_root) / tag
        if not ckpt.exists():
            print(f"[skip] missing {ckpt}", flush=True)
            continue
        if tag_i > 0:
            del model
            torch.cuda.empty_cache()
            _, model = load_variant_checkpoint(args.model_name, ckpt, device, args.z_dim)
        print(f"[extract] {tag} from {ckpt}", flush=True)
        arrays, _ = extract_val_latents(model, val_loader, device)
        np.savez_compressed(out_path, **arrays)
        print(f"  wrote {out_path}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
