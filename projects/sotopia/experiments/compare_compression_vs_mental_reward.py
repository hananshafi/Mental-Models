#!/usr/bin/env python3
"""Compare compression-only VAE latents against mental-reward latents.

The comparison is deliberately latent-probing based:
  - mental reward: z1 + z2, 128 + 128 = 256 dimensions
  - compression VAE: z_compress, 256 dimensions

Both are evaluated on the same held-out records using the same ridge probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

SOTOPIA_DIR = Path(__file__).resolve().parents[1]
if str(SOTOPIA_DIR) not in sys.path:
    sys.path.insert(0, str(SOTOPIA_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_train_compression_vae import (  # noqa: E402
    COMPRESSION_CUSTOM_HEAD_NAMES,
    CompressionVAEModel,
)
from probe_recursive_decomposition_empirical import (  # noqa: E402
    fit_text_targets,
    regression_probe_cv,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_compression_model(
    model_name: str,
    checkpoint_dir: Path,
    device: torch.device,
    z_dim: int,
    num_memory_tokens: int,
    max_target_len: int,
    decoder_layers: int,
) -> tuple[AutoTokenizer, CompressionVAEModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    base = PeftModel.from_pretrained(base, checkpoint_dir / "lora_adapter")
    model = CompressionVAEModel(
        base_model=base,
        z_dim=z_dim,
        num_memory_tokens=num_memory_tokens,
        max_target_len=max_target_len,
        decoder_layers=decoder_layers,
    )
    for head_name in COMPRESSION_CUSTOM_HEAD_NAMES:
        path = checkpoint_dir / f"{head_name}.pth"
        if path.exists():
            getattr(model, head_name).load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def extract_compression_latents(
    records: list[dict[str, Any]],
    tokenizer,
    model: CompressionVAEModel,
    device: torch.device,
    batch_size: int,
    max_ctx_len: int,
) -> dict[str, np.ndarray]:
    z_chunks = []
    h_chunks = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        contexts = [r.get("context_text", "") for r in batch]
        enc = tokenizer(
            contexts,
            truncation=True,
            max_length=max_ctx_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            h = model._encode_context(input_ids, attention_mask)
            z = model.z_mu(h)
        z_chunks.append(z.float().cpu().numpy())
        h_chunks.append(h.float().cpu().numpy())
    return {
        "z_compress": np.concatenate(z_chunks, axis=0).astype(np.float32),
        "context_hidden_compression": np.concatenate(h_chunks, axis=0).astype(np.float32),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def plot_main(scores: list[dict[str, Any]], out_path: Path) -> None:
    target_order = ["context_text", "mental2_text", "reward_vec"]
    target_labels = ["Observed\ncontext", "Second-order\nmental state", "Reward\ncontrol"]
    model_order = ["mental_reward_z1z2", "compression_z"]
    model_labels = ["Mental-reward\nlatent", "Compression\nlatent"]
    colors = ["#4C78A8", "#4C9F8A"]

    lookup = {(r["feature"], r["target"]): float(r["r2"]) for r in scores}
    mat = np.array([[lookup[(feature, target)] for target in target_order] for feature in model_order])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#4a4a4a",
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(4.25, 1.95), dpi=300)
    x = np.arange(len(target_order))
    width = 0.34
    for i, (label, color) in enumerate(zip(model_labels, colors)):
        ax.bar(x + (i - 0.5) * width, mat[i], width, label=label, color=color, edgecolor="#555555", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_ylabel("Held-out probe R2", fontweight="bold")
    ax.set_xlabel("Probe target", fontweight="bold")
    ax.set_ylim(0, max(0.05, float(np.nanmax(mat))) * 1.22)
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.85)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4a4a4a")
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", width=0.6, length=2.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=6.7, handlelength=1.4)
    fig.tight_layout(pad=0.35)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def plot_gap(scores: list[dict[str, Any]], out_path: Path) -> None:
    target_order = ["context_text", "mental2_text", "reward_vec"]
    labels = ["Observed\ncontext", "Second-order\nmental state", "Reward\ncontrol"]
    lookup = {(r["feature"], r["target"]): float(r["r2"]) for r in scores}
    gaps = np.array([
        lookup[("mental_reward_z1z2", target)] - lookup[("compression_z", target)]
        for target in target_order
    ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "axes.labelweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(3.85, 1.7), dpi=300)
    colors = ["#7A7A7A" if g < 0 else "#4C78A8" for g in gaps]
    ax.bar(np.arange(len(labels)), gaps, color=colors, edgecolor="#555555", linewidth=0.35)
    ax.axhline(0, color="#4a4a4a", linewidth=0.75)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mental - compression R2", fontweight="bold")
    ax.set_xlabel("Probe target", fontweight="bold")
    ax.grid(True, axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4a4a4a")
        spine.set_linewidth(0.75)
    fig.tight_layout(pad=0.35)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression-checkpoint", required=True)
    parser.add_argument("--mental-arrays", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-ctx-len", type=int, default=1024)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--num-memory-tokens", type=int, default=24)
    parser.add_argument("--max-target-len", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--text-components", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute-compression-latents", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.records_jsonl)
    mental = np.load(args.mental_arrays)
    n = min(len(records), len(mental["z_concat"]))
    records = records[:n]
    mental_z = mental["z_concat"][:n].astype(np.float32)
    reward_target = StandardScaler().fit_transform(np.asarray([r["reward_vec"] for r in records], dtype=np.float32)).astype(np.float32)
    mental2_target = fit_text_targets([r.get("mental2_text", "") for r in records], args.text_components, args.seed)
    context_target = fit_text_targets([r.get("context_text", "") for r in records], args.text_components, args.seed)

    compression_cache = out_dir / "compression_latent_arrays.npz"
    if compression_cache.exists() and not args.recompute_compression_latents:
        comp_npz = np.load(compression_cache)
        compression_arrays = {key: comp_npz[key].astype(np.float32) for key in comp_npz.files}
    else:
        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        tokenizer, comp_model = load_compression_model(
            args.model_name,
            Path(args.compression_checkpoint),
            device,
            z_dim=args.z_dim,
            num_memory_tokens=args.num_memory_tokens,
            max_target_len=args.max_target_len,
            decoder_layers=args.decoder_layers,
        )
        compression_arrays = extract_compression_latents(records, tokenizer, comp_model, device, args.batch_size, args.max_ctx_len)
        np.savez_compressed(compression_cache, **compression_arrays)
        del comp_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    features = {
        "mental_reward_z1z2": mental_z,
        "compression_z": compression_arrays["z_compress"][:n],
    }
    targets = {
        "context_text": context_target,
        "mental2_text": mental2_target,
        "reward_vec": reward_target,
    }
    rows = []
    for feature_name, X in features.items():
        for target_name, Y in targets.items():
            metrics = regression_probe_cv(X, Y, folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
            rows.append({
                "feature": feature_name,
                "target": target_name,
                **metrics,
            })

    write_csv(out_dir / "compression_vs_mental_probe_scores.csv", rows)
    plot_main(rows, out_dir / "compression_vs_mental_probe_r2.png")
    plot_gap(rows, out_dir / "compression_vs_mental_probe_r2_gap.png")
    with (out_dir / "comparison_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Wrote compression-vs-mental comparison to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
