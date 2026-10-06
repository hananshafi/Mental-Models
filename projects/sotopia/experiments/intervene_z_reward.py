#!/usr/bin/env python3
"""z-scramble intervention test on REWARD predictions (all three reward heads).

The earlier intervene_z_scramble.py showed scrambling z barely affects the
joint_outcome_head's preference *margin* — because that head also sees the
response hidden state and can route around z. This script tests the heads
that receive *only* z (z1_only_reward_head, z_combined_reward_head), where
any scrambling of z must propagate through if z is causally informative.

For each variant and each intervention we report MAE between predicted and
ground-truth reward (7-D normalized SOTOPIA scores).

Reward heads:
  joint_outcome_head        : [z1, z2, response_hidden]   — gets text bypass
  z1_only_reward_head       : [z1]                         — pure z1 → reward
  z_combined_reward_head    : [z1, z2]                     — pure (z1, z2) → reward

Interventions:
  baseline, permute_z1, permute_z2, permute_both,
  zero_z1, zero_z2, noise_z1, noise_z2, mean_z1, mean_z2

Output:
  runs/stage1/variant_eval/z_intervention/reward_intervention_summary.json
  runs/stage1/variant_eval/z_intervention/reward_intervention_table.md
  runs/stage1/variant_eval/z_intervention/fig_z_intervention_reward.png
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
def cache_features(model, dataloader, device):
    """Cache per-sample (z1, z2, pos_h, reward_target, has_neg)."""
    z1_chunks, z2_chunks, pos_h_chunks = [], [], []
    rew_chunks, has_neg_chunks = [], []
    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        reward_targets = batch["reward_vec"]                  # CPU tensor
        has_neg = batch["has_negative"]                       # CPU tensor

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            ctx_last = model._encode_context(ctx_ids, ctx_mask)
            mu1 = model.z1_mu(ctx_last.to(model.z1_mu.weight.dtype))
            z2_inp = torch.cat([ctx_last.to(mu1.dtype), mu1], dim=1)
            mu2 = model.z2_mu(z2_inp)

            pos_out = model.transformer(
                input_ids=pos_ids, attention_mask=pos_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state
            last_idx = pos_mask.sum(dim=1) - 1
            last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, pos_out.size(-1))
            pos_h = pos_out.gather(1, last_exp).squeeze(1)

        z1_chunks.append(mu1.float().cpu().numpy())
        z2_chunks.append(mu2.float().cpu().numpy())
        pos_h_chunks.append(pos_h.float().cpu().numpy())
        rew_chunks.append(reward_targets.numpy())
        has_neg_chunks.append(has_neg.numpy())

    return {
        "z1": np.concatenate(z1_chunks, axis=0),
        "z2": np.concatenate(z2_chunks, axis=0),
        "pos_h": np.concatenate(pos_h_chunks, axis=0),
        "reward_target": np.concatenate(rew_chunks, axis=0),
        "has_neg": np.concatenate(has_neg_chunks, axis=0),
    }


@torch.no_grad()
def predict_rewards(
    model, z1: torch.Tensor, z2: torch.Tensor, pos_h: torch.Tensor, chunk: int = 256,
) -> dict[str, np.ndarray]:
    """Run all three reward heads and return numpy arrays of predictions."""
    head_dtype = next(model.joint_outcome_head.parameters()).dtype
    z1c = z1.to(head_dtype)
    z2c = z2.to(head_dtype)
    pc = pos_h.to(head_dtype)
    N = z1c.size(0)

    joint_chunks, z1_chunks, zc_chunks = [], [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        joint_in = torch.cat([z1c[s:e], z2c[s:e], pc[s:e]], dim=1)
        joint = model.joint_outcome_head(joint_in).float().cpu().numpy()

        z1_only = model.z1_only_reward_head(z1c[s:e]).float().cpu().numpy()

        zc_in = torch.cat([z1c[s:e], z2c[s:e]], dim=1)
        zc = model.z_combined_reward_head(zc_in).float().cpu().numpy()

        joint_chunks.append(joint)
        z1_chunks.append(z1_only)
        zc_chunks.append(zc)

    return {
        "joint_pos_reward": np.concatenate(joint_chunks, axis=0),     # (N, 7)
        "z1_only_reward":   np.concatenate(z1_chunks, axis=0),        # (N, 7)
        "z_combined_reward": np.concatenate(zc_chunks, axis=0),       # (N, 7)
    }


def make_intervention(name, z1_t, z2_t, rng, device):
    N = z1_t.size(0)
    if name == "baseline":               return z1_t, z2_t
    if name == "permute_z1":
        p = torch.randperm(N, generator=rng).to(device);  return z1_t[p], z2_t
    if name == "permute_z2":
        p = torch.randperm(N, generator=rng).to(device);  return z1_t, z2_t[p]
    if name == "permute_both":
        p1 = torch.randperm(N, generator=rng).to(device)
        p2 = torch.randperm(N, generator=rng).to(device)
        return z1_t[p1], z2_t[p2]
    if name == "zero_z1":  return torch.zeros_like(z1_t), z2_t
    if name == "zero_z2":  return z1_t, torch.zeros_like(z2_t)
    if name == "noise_z1":
        s = z1_t.std(dim=0, keepdim=True)
        return torch.randn(z1_t.shape, generator=rng).to(device) * s, z2_t
    if name == "noise_z2":
        s = z2_t.std(dim=0, keepdim=True)
        return z1_t, torch.randn(z2_t.shape, generator=rng).to(device) * s
    if name == "mean_z1":
        return z1_t.mean(dim=0, keepdim=True).expand_as(z1_t), z2_t
    if name == "mean_z2":
        return z1_t, z2_t.mean(dim=0, keepdim=True).expand_as(z2_t)
    raise ValueError(name)


INTERVENTIONS = [
    "baseline",
    "permute_z1", "permute_z2", "permute_both",
    "zero_z1", "zero_z2",
    "noise_z1", "noise_z2",
    "mean_z1", "mean_z2",
]
HEADS = ["joint_pos_reward", "z1_only_reward", "z_combined_reward"]


def run_variant(model, cache, device, seed: int) -> dict:
    z1_t = torch.from_numpy(cache["z1"]).to(device)
    z2_t = torch.from_numpy(cache["z2"]).to(device)
    pos_h = torch.from_numpy(cache["pos_h"]).to(device)
    target = cache["reward_target"]                       # (N, 7) numpy
    rng = torch.Generator(); rng.manual_seed(seed)

    out: dict[str, dict[str, dict]] = {}
    for name in INTERVENTIONS:
        z1p, z2p = make_intervention(name, z1_t, z2_t, rng, device)
        preds = predict_rewards(model, z1p, z2p, pos_h)
        per_head: dict[str, dict] = {}
        for head_name in HEADS:
            err = preds[head_name] - target            # (N, 7)
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            # variance explained
            tss = float(np.var(target) * target.size)
            rss = float(np.sum(err ** 2))
            r2 = 1.0 - rss / tss if tss > 0 else float("nan")
            per_head[head_name] = {"mae": mae, "rmse": rmse, "r2": r2}
        out[name] = per_head
        print(f"    {name:14s} | "
              f"joint MAE={per_head['joint_pos_reward']['mae']:.4f}  "
              f"z1-only MAE={per_head['z1_only_reward']['mae']:.4f}  "
              f"z-comb MAE={per_head['z_combined_reward']['mae']:.4f}",
              flush=True)
    return out


def render_summary(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = list(results.keys())
    md = ["# z-scramble intervention on reward predictions", "",
          "MAE between predicted and ground-truth reward (7-D, normalized).",
          "**Δ vs baseline** in parentheses. Larger Δ under intervention = "
          "head depends more on z.", ""]
    for head_name in HEADS:
        md += [f"## reward head: `{head_name}`", "",
               "| intervention | " + " | ".join(variants) + " |",
               "|---|" + "|".join(["---:"] * len(variants)) + "|"]
        baseline = {v: results[v]["baseline"][head_name]["mae"] for v in variants}
        for name in INTERVENTIONS:
            cells = []
            for v in variants:
                val = results[v][name][head_name]["mae"]
                if name == "baseline":
                    cells.append(f"{val:.4f}")
                else:
                    delta = val - baseline[v]
                    cells.append(f"{val:.4f} ({delta:+.4f})")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    (out_dir / "reward_intervention_table.md").write_text("\n".join(md))
    with (out_dir / "reward_intervention_summary.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def plot_intervention_reward(results, out_dir: Path):
    """Three-panel figure: one panel per reward head, MAE under each intervention."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14, "axes.labelweight": "bold",
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "legend.fontsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 11,
        "savefig.bbox": "tight", "savefig.dpi": 240,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    C = {"structured_bit": "#3A5F87",
         "flat_mental_summary": "#CC8A4C",
         "shuffled_mental": "#5DA39A"}
    PRETTY = {"structured_bit": "BIT (ours)",
              "flat_mental_summary": "flat summary",
              "shuffled_mental": "shuffled (control)"}
    HEAD_TITLE = {
        "joint_pos_reward":   "(a) joint head  [z1, z2, response_hidden]",
        "z1_only_reward":     "(b) z1-only head  [z1]",
        "z_combined_reward":  "(c) z-combined head  [z1, z2]",
    }
    HEAD_NOTE = {
        "joint_pos_reward":   "head sees response — can route around z",
        "z1_only_reward":     "head sees ONLY z1 — pure z-attribution",
        "z_combined_reward":  "head sees ONLY [z1, z2] — pure z-attribution",
    }
    variants = list(results.keys())
    n_int = len(INTERVENTIONS)
    x = np.arange(n_int)
    width = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=False)
    for ax, head in zip(axes, HEADS):
        for i, v in enumerate(variants):
            ys = [results[v][name][head]["mae"] for name in INTERVENTIONS]
            ax.bar(x + (i - 1) * width, ys, width=width, color=C[v],
                   edgecolor="#1F1F1F", linewidth=0.9, label=PRETTY[v])
            # mark baseline with horizontal line in same color
            base = results[v]["baseline"][head]["mae"]
            ax.axhline(base, color=C[v], linestyle=":", linewidth=1.0, alpha=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(INTERVENTIONS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("MAE  (lower = better)", fontweight="bold")
        ax.set_title(HEAD_TITLE[head], pad=8)
        ax.text(0.5, 0.97, HEAD_NOTE[head], transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color="#5C6066",
                fontstyle="italic")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#E5E8EB", linewidth=0.5, alpha=0.7)

    leg = axes[0].legend(loc="upper left", frameon=True, framealpha=0.95,
                         edgecolor="#888", fancybox=True)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.suptitle("Latent-collapse intervention: reward MAE under z scrambles",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "If z is causally informative, the z-only heads (b,c) must show large "
             "MAE jumps under z scrambles. The joint head (a) can route around z "
             "via response_hidden and is therefore not a reliable z-quality probe.",
             ha="center", fontsize=10, color="#444", fontstyle="italic")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_z_intervention_reward.png")
    plt.close(fig)
    print(f"wrote {out_dir/'fig_z_intervention_reward.png'}")


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

    results: dict[str, dict] = {}
    for spec in args.variants:
        tag, path = spec.split("=", 1)
        ckpt_dir = Path(path.strip())
        if not ckpt_dir.exists():
            print(f"[skip] {tag}", flush=True);  continue
        print(f"\n=== {tag} ===", flush=True)
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

        print("  caching features...", flush=True)
        cache = cache_features(model, val_loader, device)
        print(f"  cached: z1 {cache['z1'].shape}  reward_target {cache['reward_target'].shape}",
              flush=True)

        results[tag] = run_variant(model, cache, device, seed=args.seed)
        del model, tokenizer
        torch.cuda.empty_cache()

    render_summary(results, out_dir)
    plot_intervention_reward(results, out_dir)
    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
