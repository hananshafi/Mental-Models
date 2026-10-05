#!/usr/bin/env python3
"""Train Stage-1 mental-supervision variants for SOTOPIA.

Variants:
  structured_bit       Original Partner Belief / Strategic Intent / Thought Process targets.
  flat_mental_summary  Same target content fused into an unsegmented narrative.
  shuffled_mental      Mental targets randomly reassigned across examples.
  no_mental            Mental generation losses disabled; reward/future/expl losses remain.

The checkpoint format is the same as stage1_train_coupled_mental_reward_v3.py,
so stage2_grpo_agent_training_v3.FrozenRewardModel can load these checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    CUSTOM_HEAD_NAMES,
    REWARD_DIM,
    MentalPrewarmDataset,
    RecursiveToMDataset,
    RecursiveToMModel,
    _save_checkpoint,
    collate_fn,
    evaluate_epoch,
    run_mental_prewarm,
    train_epoch,
)

ALIASES = {
    "structured": "structured_bit",
    "bit": "structured_bit",
    "flat": "flat_mental_summary",
    "flat_summary": "flat_mental_summary",
    "shuffled": "shuffled_mental",
    "shuffled_bit": "shuffled_mental",
    "no-mental": "no_mental",
    "none": "no_mental",
}
DESCRIPTIONS = {
    "structured_bit": "Original structured BIT mental targets.",
    "flat_mental_summary": "Same mental target content fused into an unsegmented narrative.",
    "shuffled_mental": "Mental targets shuffled across examples.",
    "no_mental": "No mental generation supervision.",
}


def normalize_variant(name: str) -> str:
    variant = ALIASES.get(name.strip().lower(), name.strip().lower())
    if variant not in DESCRIPTIONS:
        raise ValueError(f"Unknown variant {name!r}; valid variants are {sorted(DESCRIPTIONS)}")
    return variant


def parse_tagged_text(text: str) -> dict[str, str]:
    fields = {}
    if not text or not text.strip() or text.strip().upper() == "N/A":
        return fields
    for part in text.split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        value = value.strip()
        if value:
            fields[key.strip().lower()] = value
    return fields


def clean_clause(text: str) -> str:
    return " ".join(str(text).strip().split()).rstrip(".!?")


def fuse_tagged_text(text: str, second_order: bool = False) -> str:
    """Fuse tagged mental content into one narrative with no segment labels.

    This deterministic fallback removes the explicit B/I/T boundaries. For the
    paper-strength baseline, pass --flat_summary_jsonl with LLM-paraphrased
    fused summaries.
    """
    fields = parse_tagged_text(text)
    if not fields:
        return "N/A"

    if second_order:
        clauses = [
            clean_clause(fields.get("second-order belief", "")),
            clean_clause(fields.get("second-order intent", "")),
            clean_clause(fields.get("second-order thought", "")),
        ]
        clauses = [c for c in clauses if c]
        if len(clauses) >= 3:
            return (
                f"{clauses[0]}, which leads the speaker to anticipate {clauses[1].lower()} "
                f"while accounting for {clauses[2].lower()}."
            )
        return ". ".join(f"{c}." for c in clauses)

    clauses = [
        clean_clause(fields.get("partner belief", "")),
        clean_clause(fields.get("strategic intent", "")),
        clean_clause(fields.get("thought process", "")),
    ]
    clauses = [c for c in clauses if c]
    if len(clauses) >= 3:
        return (
            f"{clauses[0]}, so the speaker moves toward {clauses[1].lower()} "
            f"while weighing {clauses[2].lower()}."
        )
    return ". ".join(f"{c}." for c in clauses)


def mental_pair_key(mental1: str, mental2: str) -> str:
    payload = json.dumps([mental1 or "", mental2 or ""], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_flat_summary_cache(path: str | None) -> dict[str, tuple[str, str]]:
    if not path:
        return {}
    cache = {}
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("key") or mental_pair_key(
                rec.get("source_mental1_text", ""),
                rec.get("source_mental2_text", ""),
            )
            flat1 = rec.get("flat_mental1_text", rec.get("mental1_flat", ""))
            flat2 = rec.get("flat_mental2_text", rec.get("mental2_flat", ""))
            if not flat1 or not flat2:
                raise ValueError(f"{path}:{line_no} missing flat_mental1_text/flat_mental2_text")
            cache[key] = (flat1, flat2)
    return cache


def deranged_indices(n: int, seed: int) -> list[int]:
    indices = list(range(n))
    if n <= 1:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    if any(i == j for i, j in enumerate(indices)):
        indices = indices[1:] + indices[:1]
    return indices


class VariantRecursiveToMDataset(RecursiveToMDataset):
    def __init__(self, data_path: str, tokenizer, variant: str, seed: int,
                 max_ctx_len: int = 1024, max_resp_len: int = 256, max_mental_len: int = 256,
                 flat_summary_cache: dict[str, tuple[str, str]] | None = None):
        self.variant = normalize_variant(variant)
        self.variant_seed = seed
        self.flat_summary_cache = flat_summary_cache or {}
        self.flat_summary_cache_hits = 0
        super().__init__(
            data_path, tokenizer,
            max_ctx_len=max_ctx_len, max_resp_len=max_resp_len, max_mental_len=max_mental_len,
        )
        self._apply_variant()
        self.variant_stats = self._stats()
        print(f"Applied mental supervision variant: {self.variant}", flush=True)
        print(json.dumps(self.variant_stats, indent=2), flush=True)

    def _apply_variant(self) -> None:
        if self.variant == "structured_bit":
            return
        if self.variant == "flat_mental_summary":
            for sample in self.samples:
                mental1 = sample.get("mental1_text", "")
                mental2 = sample.get("mental2_text", "")
                cached = self.flat_summary_cache.get(mental_pair_key(mental1, mental2))
                if cached:
                    sample["mental1_text"], sample["mental2_text"] = cached
                    self.flat_summary_cache_hits += 1
                else:
                    sample["mental1_text"] = fuse_tagged_text(mental1, second_order=False)
                    sample["mental2_text"] = fuse_tagged_text(mental2, second_order=True)
            return
        if self.variant == "shuffled_mental":
            pairs = [(s.get("mental1_text", ""), s.get("mental2_text", "")) for s in self.samples]
            for sample, source_idx in zip(self.samples, deranged_indices(len(pairs), self.variant_seed)):
                sample["mental1_text"], sample["mental2_text"] = pairs[source_idx]
            return
        if self.variant == "no_mental":
            for sample in self.samples:
                sample["mental1_text"] = "N/A"
                sample["mental2_text"] = "N/A"
            return
        raise ValueError(f"Unhandled variant {self.variant}")

    def _stats(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "description": DESCRIPTIONS[self.variant],
            "num_samples": len(self.samples),
            "mental1_nonempty": sum(1 for s in self.samples if s.get("mental1_text", "").strip() not in {"", "N/A"}),
            "mental2_nonempty": sum(1 for s in self.samples if s.get("mental2_text", "").strip() not in {"", "N/A"}),
            "flat_summary_cache_size": len(self.flat_summary_cache),
            "flat_summary_cache_hits": self.flat_summary_cache_hits,
            "variant_seed": self.variant_seed,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    p.add_argument("--output_dir", default="projects/sotopia/experiments/runs/stage1/mental_variant")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_ctx_len", type=int, default=1024)
    p.add_argument("--max_resp_len", type=int, default=256)
    p.add_argument("--max_mental_len", type=int, default=256)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_lora_layers", type=int, default=16)
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument("--future_weight", type=float, default=0.5)
    p.add_argument("--mental1_weight", type=float, default=0.3)
    p.add_argument("--mental2_weight", type=float, default=0.3)
    p.add_argument("--expl_weight", type=float, default=0.3)
    p.add_argument("--z_only_weight", type=float, default=0.5)
    p.add_argument("--kl_anneal_steps", type=int, default=200)
    p.add_argument("--z2_kl_delay_steps", type=int, default=100)
    p.add_argument("--z2_warmup_steps", type=int, default=100)
    p.add_argument("--mental_prewarm_data", default=None)
    p.add_argument("--mental_prewarm_epochs", type=int, default=1)
    p.add_argument("--mental_prewarm_lr", type=float, default=2e-4)
    p.add_argument("--allow_variant_prewarm", action="store_true")
    p.add_argument("--flat_summary_jsonl", default=None,
                   help="Optional LLM-fused flat summaries keyed by source mental1/mental2 text.")
    p.add_argument("--head_lr_mult", type=float, default=10.0)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def apply_variant_overrides(args: argparse.Namespace) -> None:
    args.variant = normalize_variant(args.variant)
    if args.variant == "no_mental":
        args.mental1_weight = 0.0
        args.mental2_weight = 0.0
        args.mental_prewarm_data = None
    if args.mental_prewarm_data and args.variant != "structured_bit" and not args.allow_variant_prewarm:
        raise ValueError(
            "mental_prewarm_data uses structured mental targets. Disable it for this variant, "
            "or pass --allow_variant_prewarm intentionally."
        )


def freeze_no_mental_heads(model: RecursiveToMModel, variant: str) -> None:
    if variant != "no_mental":
        return
    for name, param in model.named_parameters():
        if "mental1_decoder" in name or "mental2_decoder" in name:
            param.requires_grad = False


def main() -> None:
    args = parse_args()
    apply_variant_overrides(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args).copy()
    config["variant_description"] = DESCRIPTIONS[args.variant]
    config["checkpoint_compatible_with"] = "stage2_grpo_agent_training_v3.FrozenRewardModel"
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.model_name}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="auto")
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    base_model.config.use_cache = False

    top_layers = list(range(base_model.config.num_hidden_layers - args.num_lora_layers, base_model.config.num_hidden_layers))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        layers_to_transform=top_layers,
        layers_pattern="layers",
    )
    base_model = get_peft_model(base_model, lora_config)
    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_" in name
    base_model.print_trainable_parameters()

    model = RecursiveToMModel(base_model, reward_dim=REWARD_DIM, z_dim=args.z_dim).to(device)
    for name, param in model.named_parameters():
        if any(k in name for k in CUSTOM_HEAD_NAMES):
            param.requires_grad = True
    freeze_no_mental_heads(model, args.variant)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    if args.mental_prewarm_data:
        prewarm = MentalPrewarmDataset(args.mental_prewarm_data, tokenizer, max_ctx_len=args.max_ctx_len)
        model = run_mental_prewarm(
            model, prewarm, device,
            num_epochs=args.mental_prewarm_epochs,
            lr=args.mental_prewarm_lr,
            batch_size=args.batch_size,
        )
        del prewarm
        gc.collect()
        torch.cuda.empty_cache()

    dataset = VariantRecursiveToMDataset(
        args.data_path,
        tokenizer,
        variant=args.variant,
        seed=args.seed,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
        flat_summary_cache=load_flat_summary_cache(args.flat_summary_jsonl),
    )
    with open(os.path.join(args.output_dir, "variant_stats.json"), "w") as f:
        json.dump(dataset.variant_stats, f, indent=2, sort_keys=True)

    train_dataset = dataset
    val_dataset = None
    if args.val_ratio > 0 and len(dataset) > 1:
        val_size = min(max(1, int(len(dataset) * args.val_ratio)), len(dataset) - 1)
        train_size = len(dataset) - val_size
        gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
        print(f"Dataset split: train={train_size}, val={val_size}", flush=True)
    else:
        print(f"Dataset split: train={len(dataset)}, val=0", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    head_params, lora_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if any(k in name for k in CUSTOM_HEAD_NAMES) else lora_params).append(param)

    head_lr = args.lr * args.head_lr_mult
    print(f"Param groups: LoRA lr={args.lr}, Head lr={head_lr}", flush=True)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
        ]
    )
    total_steps = max(1, len(train_loader) * args.num_epochs // args.grad_accum_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best_metric = float("inf")
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss, metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            kl_weight=args.kl_weight,
            future_weight=args.future_weight,
            mental1_weight=args.mental1_weight,
            mental2_weight=args.mental2_weight,
            expl_weight=args.expl_weight,
            z_only_weight=args.z_only_weight,
            grad_accum_steps=args.grad_accum_steps,
            kl_anneal_steps=args.kl_anneal_steps,
            z2_kl_delay_steps=args.z2_kl_delay_steps,
            z2_warmup_steps=args.z2_warmup_steps,
            global_step_offset=global_step,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}: avg_loss={avg_loss:.4f}", flush=True)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}", flush=True)

        monitor_metric = avg_loss
        monitor_name = "train_loss"
        if val_loader is not None:
            val_loss, val_metrics = evaluate_epoch(
                model, val_loader, device, current_opt_step=max(global_step, 1),
                kl_weight=args.kl_weight,
                future_weight=args.future_weight,
                mental1_weight=args.mental1_weight,
                mental2_weight=args.mental2_weight,
                expl_weight=args.expl_weight,
                z_only_weight=args.z_only_weight,
                kl_anneal_steps=args.kl_anneal_steps,
                z2_kl_delay_steps=args.z2_kl_delay_steps,
            )
            print(f"  val_loss: {val_loss:.4f}", flush=True)
            for key, value in val_metrics.items():
                print(f"  val_{key}: {value:.4f}", flush=True)
            monitor_metric = val_loss
            monitor_name = "val_loss"

        _save_checkpoint(model, os.path.join(args.output_dir, f"epoch_{epoch}"))
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            _save_checkpoint(model, os.path.join(args.output_dir, "best"))
            print(f"  -> Best model saved ({monitor_name}={best_metric:.4f})", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nVariant training complete: {args.variant}. Best metric: {best_metric:.4f}", flush=True)


if __name__ == "__main__":
    main()
