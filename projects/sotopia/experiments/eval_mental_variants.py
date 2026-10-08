#!/usr/bin/env python3
"""Evaluate mental-supervision Stage-1 variant checkpoints.

Two analyses are run for each variant checkpoint:
  (1) Val-set Stage-1 metrics (preference accuracy, reward regression,
      mental gen NLL, KL terms) by re-running evaluate_epoch on a fixed
      held-out split.
  (2) Frozen-z linear (ridge) probe for SOTOPIA-aligned semantic targets:
        - reward_vec        (7-D normalized SOTOPIA scores)
        - mental2_text      (TF-IDF-SVD compressed second-order mental text)
        - context_text      (TF-IDF-SVD compressed observed context)
      Features probed: z1, z2, z_concat, context_hidden.
      A shuffled-label control is included for every (feature, target) pair.

Usage:
  python experiments/eval_mental_variants.py \
      --variants \
        structured_bit=projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/best \
        flat_mental_summary=projects/sotopia/experiments/runs/stage1/flat_mental_summary_qwen7b_seed42/best \
        shuffled_mental=projects/sotopia/experiments/runs/stage1/shuffled_mental_qwen7b_seed42/best \
      --output_dir projects/sotopia/experiments/runs/stage1/variant_eval \
      --gpu 0
"""
from __future__ import annotations

import argparse
import csv
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
    RecursiveToMDataset, RecursiveToMModel,
    collate_fn, evaluate_epoch,
)
from experiments.probe_recursive_decomposition_empirical import (  # noqa: E402
    fit_text_targets, regression_probe_cv,
)


# ── Checkpoint loader ────────────────────────────────────────────────────────
def load_variant_checkpoint(
    model_name: str, ckpt_dir: Path, device: torch.device, z_dim: int,
) -> tuple[AutoTokenizer, RecursiveToMModel]:
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
        else:
            print(f"  [warn] missing head {head_name} in {ckpt_dir}", flush=True)
    model.to(device).eval()
    return tokenizer, model


# ── Latent extraction (val set) ──────────────────────────────────────────────
@torch.no_grad()
def extract_val_latents(
    model: RecursiveToMModel, dataloader: DataLoader, device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    z1_chunks, z2_chunks, ctx_chunks = [], [], []
    sample_records: list[dict] = []
    dataset = dataloader.dataset
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(base_dataset))

    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            ctx_last = model._encode_context(ctx_ids, ctx_mask)
            z1_mu = model.z1_mu(ctx_last.to(model.z1_mu.weight.dtype))
            z2_inp = torch.cat([ctx_last.to(z1_mu.dtype), z1_mu], dim=1)
            z2_mu = model.z2_mu(z2_inp)
        z1_chunks.append(z1_mu.float().cpu().numpy())
        z2_chunks.append(z2_mu.float().cpu().numpy())
        ctx_chunks.append(ctx_last.float().cpu().numpy())

    for i in indices:
        s = base_dataset.samples[i]
        ctx_text = base_dataset._format_context(s) if hasattr(base_dataset, "_format_context") else s.get("context_text", "")
        sample_records.append({
            "context_text": ctx_text,
            "mental1_text": s.get("mental1_text", ""),
            "mental2_text": s.get("mental2_text", ""),
            "reward_vec": s.get("reward_vec", []),
        })

    z1 = np.concatenate(z1_chunks, axis=0)
    z2 = np.concatenate(z2_chunks, axis=0)
    ctx = np.concatenate(ctx_chunks, axis=0)
    arrays = {
        "z1": z1.astype(np.float32),
        "z2": z2.astype(np.float32),
        "z_concat": np.concatenate([z1, z2], axis=1).astype(np.float32),
        "context_hidden": ctx.astype(np.float32),
    }
    return arrays, sample_records


# ── Phase 1: per-variant val metrics ────────────────────────────────────────
def run_phase1(
    variant_name: str, ckpt_dir: Path, args, device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict]]:
    print(f"\n=== {variant_name}: phase 1 (val metrics) ===", flush=True)
    tokenizer, model = load_variant_checkpoint(args.model_name, ckpt_dir, device, args.z_dim)

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

    val_loss, val_metrics = evaluate_epoch(
        model, val_loader, device,
        current_opt_step=args.dummy_opt_step,
        kl_weight=args.kl_weight, future_weight=args.future_weight,
        mental1_weight=args.mental1_weight, mental2_weight=args.mental2_weight,
        expl_weight=args.expl_weight, z_only_weight=args.z_only_weight,
        kl_anneal_steps=args.kl_anneal_steps, z2_kl_delay_steps=args.z2_kl_delay_steps,
    )
    metrics_out = {"val_loss": float(val_loss), **{k: float(v) for k, v in val_metrics.items()}}

    arrays, sample_records = extract_val_latents(model, val_loader, device)

    del model, tokenizer
    torch.cuda.empty_cache()
    return metrics_out, arrays, sample_records


# ── Phase 2: frozen-z probe ──────────────────────────────────────────────────
def run_phase2(
    variant_name: str,
    arrays: dict[str, np.ndarray],
    sample_records: list[dict],
    args,
    rng: np.random.Generator,
) -> list[dict]:
    print(f"\n=== {variant_name}: phase 2 (frozen-z probes) ===", flush=True)

    reward_vecs = np.asarray([r["reward_vec"] for r in sample_records], dtype=np.float32)
    if reward_vecs.size == 0 or reward_vecs.ndim != 2:
        raise RuntimeError(f"{variant_name}: reward_vec missing in records")
    from sklearn.preprocessing import StandardScaler
    reward_target = StandardScaler().fit_transform(reward_vecs).astype(np.float32)

    mental2_target = fit_text_targets(
        [r.get("mental2_text", "") for r in sample_records],
        args.text_components, args.seed,
    )
    context_target = fit_text_targets(
        [r.get("context_text", "") for r in sample_records],
        args.text_components, args.seed,
    )

    targets = {
        "reward_vec": reward_target,
        "mental2_text": mental2_target,
        "context_text": context_target,
    }
    feature_names = ["z1", "z2", "z_concat", "context_hidden"]
    rows: list[dict] = []
    for fname in feature_names:
        X = arrays[fname]
        for tname, Y in targets.items():
            real = regression_probe_cv(X, Y, folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
            rows.append({
                "variant": variant_name, "feature": fname, "target": tname,
                "label_condition": "real",
                "r2_mean": real["r2"], "cosine_mean": real["cosine"],
                "n_samples": int(X.shape[0]), "feature_dim": int(X.shape[1]),
            })
            # shuffled control: permute Y rows
            perm = rng.permutation(Y.shape[0])
            shuf = regression_probe_cv(X, Y[perm], folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
            rows.append({
                "variant": variant_name, "feature": fname, "target": tname,
                "label_condition": "shuffled",
                "r2_mean": shuf["r2"], "cosine_mean": shuf["cosine"],
                "n_samples": int(X.shape[0]), "feature_dim": int(X.shape[1]),
            })
    return rows


# ── Output writers ───────────────────────────────────────────────────────────
def write_phase1_md(out_dir: Path, all_metrics: dict[str, dict]) -> None:
    keys = ["val_loss", "preference", "reward_reg", "z1_only_reg", "z_combined_reg",
            "kl1", "kl2", "future", "mental1_gen", "mental2_gen", "expl_reward"]
    variants = list(all_metrics.keys())
    md = ["# Phase 1: val-set Stage-1 metrics", "",
          "| metric | " + " | ".join(variants) + " |",
          "|---|" + "|".join(["---:"] * len(variants)) + "|"]
    for k in keys:
        cells = [f"{all_metrics[v].get(k, float('nan')):.4f}" for v in variants]
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    (out_dir / "phase1_val_metrics.md").write_text("\n".join(md))
    with (out_dir / "phase1_val_metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)


def write_phase2_md(out_dir: Path, rows: list[dict]) -> None:
    csv_path = out_dir / "phase2_probe_scores.csv"
    fieldnames = ["variant", "feature", "target", "label_condition",
                  "r2_mean", "cosine_mean", "n_samples", "feature_dim"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    targets = ["reward_vec", "mental2_text", "context_text"]
    features = ["z1", "z2", "z_concat", "context_hidden"]
    variants = sorted({r["variant"] for r in rows})

    md = ["# Phase 2: frozen-z ridge probe (R²)", ""]
    for tgt in targets:
        md += [f"## target = {tgt}", "",
               "| feature | labels | " + " | ".join(variants) + " |",
               "|---|---|" + "|".join(["---:"] * len(variants)) + "|"]
        for feat in features:
            for cond in ("real", "shuffled"):
                cells = []
                for v in variants:
                    hit = next((r for r in rows
                                if r["variant"] == v and r["feature"] == feat
                                and r["target"] == tgt and r["label_condition"] == cond), None)
                    cells.append(f"{hit['r2_mean']:.3f}" if hit else "—")
                md.append(f"| {feat} | {cond} | " + " | ".join(cells) + " |")
        md.append("")
    (out_dir / "phase2_probe_scores.md").write_text("\n".join(md))


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", required=True,
                   help="tag=ckpt_path entries")
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
    # phase-1 metric weights (cosmetic for evaluate_epoch — matches training defaults)
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument("--future_weight", type=float, default=0.5)
    p.add_argument("--mental1_weight", type=float, default=0.3)
    p.add_argument("--mental2_weight", type=float, default=0.3)
    p.add_argument("--expl_weight", type=float, default=0.3)
    p.add_argument("--z_only_weight", type=float, default=0.5)
    p.add_argument("--kl_anneal_steps", type=int, default=200)
    p.add_argument("--z2_kl_delay_steps", type=int, default=100)
    p.add_argument("--dummy_opt_step", type=int, default=10_000,
                   help="Pinned high so KL anneal is fully on during eval (=training-end behavior).")
    # phase-2 probe knobs
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--text_components", type=int, default=64)
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--gpu", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    variant_specs: list[tuple[str, Path]] = []
    for spec in args.variants:
        if "=" not in spec:
            raise ValueError(f"--variants expects tag=path, got {spec!r}")
        tag, path = spec.split("=", 1)
        variant_specs.append((tag.strip(), Path(path.strip())))

    all_metrics: dict[str, dict] = {}
    all_probe_rows: list[dict] = []
    for tag, ckpt_dir in variant_specs:
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: missing {ckpt_dir}", flush=True)
            continue
        metrics, arrays, samples = run_phase1(tag, ckpt_dir, args, device)
        all_metrics[tag] = metrics
        np.savez_compressed(out_dir / f"latents_{tag}.npz", **arrays)
        all_probe_rows.extend(run_phase2(tag, arrays, samples, args, rng))

    write_phase1_md(out_dir, all_metrics)
    write_phase2_md(out_dir, all_probe_rows)
    print(f"\nWrote {out_dir}/phase1_val_metrics.md", flush=True)
    print(f"Wrote {out_dir}/phase2_probe_scores.md", flush=True)


if __name__ == "__main__":
    main()
