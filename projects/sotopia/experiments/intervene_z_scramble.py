#!/usr/bin/env python3
"""z-scramble intervention test: confirm latent is not collapsed.

For each SOTOPIA Stage-1 variant, cache (z1, z2, pos_resp_hidden,
neg_resp_hidden, has_neg) on the val set, then run a battery of latent
interventions and recompute the joint-reward head's preference ranking.

Interventions:
  baseline         : real z1, z2
  permute_z1       : z1 randomly permuted across val samples
  permute_z2       : z2 randomly permuted across val samples
  permute_both     : both permuted (independently)
  zero_z1          : z1 := 0
  zero_z2          : z2 := 0
  noise_z1         : z1 := N(0, std(z1))
  noise_z2         : z2 := N(0, std(z2))
  mean_z1          : z1 := mean(z1) broadcast
  mean_z2          : z2 := mean(z2) broadcast

A *non-collapsed* latent should crash on permute_z1 (no longer aligned with
context) and degrade on noise_z1 / mean_z1. A *collapsed* latent (z is
constant or pure noise w.r.t. context) is unaffected by interventions.

Reports per-variant: preference accuracy and median margin under each
intervention, plus the relative drop vs baseline.

Output:
  runs/stage1/variant_eval/z_intervention/intervention_summary.json
  runs/stage1/variant_eval/z_intervention/intervention_table.md
  runs/stage1/variant_eval/z_intervention/fig_z_intervention.png
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
    """Run the model on val set, cache:
       z1, z2, pos_resp_hidden, neg_resp_hidden, has_neg
    so we can run interventions cheaply through joint_outcome_head."""
    z1_chunks, z2_chunks = [], []
    pos_h_chunks, neg_h_chunks = [], []
    has_neg_chunks = []
    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        has_neg = batch["has_negative"].to(device)

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            # Deterministic encoding (mu only)
            ctx_last = model._encode_context(ctx_ids, ctx_mask)
            mu1 = model.z1_mu(ctx_last.to(model.z1_mu.weight.dtype))
            z2_inp = torch.cat([ctx_last.to(mu1.dtype), mu1], dim=1)
            mu2 = model.z2_mu(z2_inp)

            # response hiddens (last non-pad position)
            def _last_hidden(ids, mask):
                out = model.transformer(
                    input_ids=ids, attention_mask=mask,
                    use_cache=False, return_dict=True,
                ).last_hidden_state
                last_idx = mask.sum(dim=1) - 1
                last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, out.size(-1))
                return out.gather(1, last_exp).squeeze(1)

            pos_h = _last_hidden(pos_ids, pos_mask)
            neg_h = _last_hidden(neg_ids, neg_mask)

        z1_chunks.append(mu1.float().cpu().numpy())
        z2_chunks.append(mu2.float().cpu().numpy())
        pos_h_chunks.append(pos_h.float().cpu().numpy())
        neg_h_chunks.append(neg_h.float().cpu().numpy())
        has_neg_chunks.append(has_neg.cpu().numpy())

    return {
        "z1": np.concatenate(z1_chunks, axis=0),
        "z2": np.concatenate(z2_chunks, axis=0),
        "pos_h": np.concatenate(pos_h_chunks, axis=0),
        "neg_h": np.concatenate(neg_h_chunks, axis=0),
        "has_neg": np.concatenate(has_neg_chunks, axis=0),
    }


@torch.no_grad()
def score_with_z(model, z1: torch.Tensor, z2: torch.Tensor,
                 pos_h: torch.Tensor, neg_h: torch.Tensor,
                 chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (margin_per_dim [N, R], pos_reward [N, R], neg_reward [N, R])."""
    N = z1.size(0)
    head_dtype = next(model.joint_outcome_head.parameters()).dtype
    z1c = z1.to(head_dtype)
    z2c = z2.to(head_dtype)
    pc = pos_h.to(head_dtype)
    nc = neg_h.to(head_dtype)
    margins = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        pos_in = torch.cat([z1c[s:e], z2c[s:e], pc[s:e]], dim=1)
        neg_in = torch.cat([z1c[s:e], z2c[s:e], nc[s:e]], dim=1)
        pos_r = model.joint_outcome_head(pos_in).float()
        neg_r = model.joint_outcome_head(neg_in).float()
        margins.append((pos_r - neg_r).cpu().numpy())
    return np.concatenate(margins, axis=0)


def make_intervention(name: str, z1_t: torch.Tensor, z2_t: torch.Tensor,
                       rng: torch.Generator, device: torch.device,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (z1', z2') for the named intervention."""
    N = z1_t.size(0)
    if name == "baseline":
        return z1_t, z2_t
    if name == "permute_z1":
        perm = torch.randperm(N, generator=rng).to(device)
        return z1_t[perm], z2_t
    if name == "permute_z2":
        perm = torch.randperm(N, generator=rng).to(device)
        return z1_t, z2_t[perm]
    if name == "permute_both":
        p1 = torch.randperm(N, generator=rng).to(device)
        p2 = torch.randperm(N, generator=rng).to(device)
        return z1_t[p1], z2_t[p2]
    if name == "zero_z1":
        return torch.zeros_like(z1_t), z2_t
    if name == "zero_z2":
        return z1_t, torch.zeros_like(z2_t)
    if name == "noise_z1":
        std1 = z1_t.std(dim=0, keepdim=True)
        return torch.randn(z1_t.shape, generator=rng, device="cpu").to(device) * std1, z2_t
    if name == "noise_z2":
        std2 = z2_t.std(dim=0, keepdim=True)
        return z1_t, torch.randn(z2_t.shape, generator=rng, device="cpu").to(device) * std2
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


def run_variant(model, cache: dict, device: torch.device, seed: int) -> dict:
    z1_t = torch.from_numpy(cache["z1"]).to(device)
    z2_t = torch.from_numpy(cache["z2"]).to(device)
    pos_h = torch.from_numpy(cache["pos_h"]).to(device)
    neg_h = torch.from_numpy(cache["neg_h"]).to(device)
    has_neg = cache["has_neg"]                    # numpy
    mask = has_neg > 0                            # filter

    rng = torch.Generator()
    rng.manual_seed(seed)

    out: dict[str, dict] = {}
    for name in INTERVENTIONS:
        z1p, z2p = make_intervention(name, z1_t, z2_t, rng, device)
        margin = score_with_z(model, z1p, z2p, pos_h, neg_h)   # (N, R)
        margin = margin[mask]
        per_sample = margin.mean(axis=1)
        per_dim_flat = margin.ravel()
        out[name] = {
            "median_margin":           float(np.median(per_sample)),
            "mean_margin":             float(per_sample.mean()),
            "p10_margin":              float(np.percentile(per_sample, 10)),
            "min_margin":              float(per_sample.min()),
            "frac_correct_per_sample": float((per_sample > 0).mean()),
            "frac_correct_per_dim":    float((per_dim_flat > 0).mean()),
        }
        print(f"    {name:15s} | "
              f"acc={out[name]['frac_correct_per_sample']:.4f}  "
              f"median={out[name]['median_margin']:+8.3f}  "
              f"p10={out[name]['p10_margin']:+8.3f}",
              flush=True)
    return out


def render_summary(results: dict[str, dict], out_dir: Path) -> None:
    """Write markdown table + bar/line figure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # markdown
    variants = list(results.keys())
    md_lines = ["# z-scramble intervention test", "",
                "Median per-sample preference margin (pos − neg reward) under each intervention.",
                "Baseline = real (z1, z2). Higher = more confidently correct ranking.",
                "Δ vs baseline shown in parentheses.", ""]
    for metric_name, metric_key in [
        ("Per-sample preference accuracy", "frac_correct_per_sample"),
        ("Median margin (pos − neg)", "median_margin"),
        ("p10 margin", "p10_margin"),
    ]:
        md_lines += [f"## {metric_name}", "",
                     "| intervention | " + " | ".join(variants) + " |",
                     "|---|" + "|".join(["---:"] * len(variants)) + "|"]
        baseline = {v: results[v]["baseline"][metric_key] for v in variants}
        for name in INTERVENTIONS:
            cells = []
            for v in variants:
                val = results[v][name][metric_key]
                if name == "baseline":
                    cells.append(f"{val:+.4f}" if "margin" in metric_key else f"{val:.4f}")
                else:
                    delta = val - baseline[v]
                    if "margin" in metric_key:
                        cells.append(f"{val:+.3f} ({delta:+.3f})")
                    else:
                        cells.append(f"{val:.4f} ({delta:+.4f})")
            md_lines.append(f"| {name} | " + " | ".join(cells) + " |")
        md_lines.append("")
    (out_dir / "intervention_table.md").write_text("\n".join(md_lines))

    with (out_dir / "intervention_summary.json").open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def plot_intervention_fig(results: dict[str, dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 15, "axes.labelweight": "bold",
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "legend.fontsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 12,
        "savefig.bbox": "tight", "savefig.dpi": 240,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    C = {"structured_bit": "#3A5F87",
         "flat_mental_summary": "#CC8A4C",
         "shuffled_mental": "#5DA39A"}
    PRETTY = {"structured_bit": "BIT (ours)",
              "flat_mental_summary": "flat summary",
              "shuffled_mental": "shuffled (control)"}

    fig, ax = plt.subplots(figsize=(13, 5.6))
    variants = list(results.keys())
    x = np.arange(len(INTERVENTIONS))
    width = 0.26

    for i, v in enumerate(variants):
        ys = [results[v][name]["median_margin"] for name in INTERVENTIONS]
        ax.bar(x + (i - 1) * width, ys, width=width, color=C[v],
               edgecolor="#1F1F1F", linewidth=1.0, label=PRETTY[v])

    ax.axhline(0, color="#444", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(INTERVENTIONS, rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel("median margin  (pos − neg reward)", fontweight="bold")
    ax.set_title("Latent-collapse intervention test: median preference margin under z scrambles",
                 pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.6, alpha=0.7)
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="#888", fancybox=True)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_z_intervention.png")
    plt.close(fig)
    print(f"wrote {out_dir/'fig_z_intervention.png'}")


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
    p.add_argument("--gpu", default="")
    args = p.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for spec in args.variants:
        tag, path = spec.split("=", 1)
        ckpt_dir = Path(path.strip())
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: missing {ckpt_dir}", flush=True)
            continue
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
        print(f"  cached: z1 {cache['z1'].shape}  z2 {cache['z2'].shape}  "
              f"pos_h {cache['pos_h'].shape}  has_neg sum={int(cache['has_neg'].sum())}",
              flush=True)

        print("  running interventions...", flush=True)
        results[tag] = run_variant(model, cache, device, seed=args.seed)

        del model, tokenizer
        torch.cuda.empty_cache()

    render_summary(results, out_dir)
    plot_intervention_fig(results, out_dir)
    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
