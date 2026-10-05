#!/usr/bin/env python3
"""Run and plot SOTOPIA counterfactual flip scoring.

The experiment keeps the visible dialogue context fixed while changing the
latent/state context used by the mental reward model. This produces the visual
test for whether the mental state acts as a decision-relevant variable rather
than a generic compression.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
SOTOPIA_DIMENSIONS = [
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
]
SIMPLE_REWARD_DIMS = ["goal", "relationship", "knowledge"]

PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_GREY = "#7A7A7A"
PLOT_LIGHT_GREY = "#D6D8DC"
PLOT_DARK_GREY = "#4B4B4B"
PLOT_PURPLE = "#7B61B5"
PLOT_RED = "#C84C4C"

MODEL_ORDER = [
    "simple_reward_observed",
    "compression_observed",
    "mental_observed_only",
    "mental_correct_z",
    "mental_swapped_z",
    "mental_shuffled_z",
]
MODEL_COLORS = {
    "simple_reward_observed": PLOT_GREY,
    "compression_observed": PLOT_PURPLE,
    "mental_observed_only": PLOT_GREEN,
    "mental_correct_z": PLOT_BLUE,
    "mental_swapped_z": PLOT_RED,
    "mental_shuffled_z": PLOT_DARK_GREY,
}
MODEL_LABELS = {
    "simple_reward_observed": "Simple reward",
    "compression_observed": "Compression VAE",
    "mental_observed_only": "Mental observed-only",
    "mental_correct_z": "Mental correct z",
    "mental_swapped_z": "Mental swapped z",
    "mental_shuffled_z": "Mental shuffled z",
}


def amp_context(device: str):
    if str(device).startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def style_plot_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C7CBD1", linewidth=0.9, alpha=0.9)
    ax.grid(True, which="minor", color="#E8EAED", linewidth=0.5, alpha=0.9)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.1, length=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def style_plot_legend(legend: Any) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(PLOT_LIGHT_GREY)
    legend.get_frame().set_alpha(0.94)
    for text in legend.get_texts():
        text.set_fontsize(9)


def ordered_models(model_names: list[str]) -> list[str]:
    return sorted(model_names, key=lambda name: (MODEL_ORDER.index(name) if name in MODEL_ORDER else 999, name))


class SimpleRewardModel(nn.Module):
    """Simple reward head used by stage2_grpo_ablation_no_mental.py."""

    def __init__(self, base_model: nn.Module, reward_dim: int = 3):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )

    def _get_transformer(self) -> nn.Module:
        base = self.base_model
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model"):
            base = base.model
        if hasattr(base, "model"):
            base = base.model
        return base

    def _encode_and_pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        transformer = self._get_transformer()
        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self._encode_and_pool(input_ids, attention_mask)
        return self.reward_head(pooled)


class FrozenSimpleReward:
    def __init__(
        self,
        model_name: str,
        reward_head_path: str,
        device: str,
        scoring_dim_indices: list[int] | None = None,
    ):
        self.device = device
        self.scoring_dim_indices = scoring_dim_indices
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        self.model = SimpleRewardModel(base, reward_dim=len(SIMPLE_REWARD_DIMS)).to(device)
        state = torch.load(reward_head_path, map_location=device, weights_only=True)
        self.model.reward_head.load_state_dict(state)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _reduce_reward(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        if self.scoring_dim_indices is not None:
            return reward_tensor[:, self.scoring_dim_indices].mean(dim=1)
        return reward_tensor.mean(dim=1)

    @torch.no_grad()
    def score(self, contexts: list[str], completions: list[str], max_len: int = 1280) -> list[float]:
        scores: list[float] = []
        texts = [f"{ctx} {comp}" for ctx, comp in zip(contexts, completions)]
        for i in range(0, len(texts), 8):
            enc = self.tokenizer(
                texts[i:i + 8],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device)
            with amp_context(self.device):
                reward = self.model(ids, mask)
            scores.extend(self._reduce_reward(reward.float()).cpu().tolist())
        return scores


def load_checkpoint_args(checkpoint_dir: str) -> dict[str, Any]:
    ckpt = Path(checkpoint_dir)
    candidates = [ckpt / "args.json", ckpt.parent / "args.json"]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


class FrozenCompressionReward:
    """Compression-only VAE encoder plus frozen-latent reward probe."""

    def __init__(
        self,
        model_name: str,
        checkpoint_dir: str,
        device: str,
        scoring_dim_indices: list[int] | None = None,
    ):
        self.device = device
        self.scoring_dim_indices = scoring_dim_indices
        config = load_checkpoint_args(checkpoint_dir)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from stage1_train_compression_vae import (  # noqa: WPS433
            COMPRESSION_CUSTOM_HEAD_NAMES,
            CompressionVAEModel,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        lora_path = Path(checkpoint_dir) / "lora_adapter"
        if lora_path.exists():
            base = PeftModel.from_pretrained(base, str(lora_path), torch_dtype=torch.bfloat16)
            base = base.merge_and_unload()
        base = base.to(device)

        self.model = CompressionVAEModel(
            base_model=base,
            z_dim=int(config.get("z_dim", 256)),
            reward_dim=len(SOTOPIA_DIMENSIONS),
            num_memory_tokens=int(config.get("num_memory_tokens", 24)),
            max_target_len=int(config.get("max_target_len", 256)),
            decoder_layers=int(config.get("decoder_layers", 2)),
        ).to(device)
        missing_heads = []
        for head_name in COMPRESSION_CUSTOM_HEAD_NAMES:
            path = Path(checkpoint_dir) / f"{head_name}.pth"
            if path.exists():
                getattr(self.model, head_name).load_state_dict(
                    torch.load(path, map_location=device, weights_only=True)
                )
            else:
                missing_heads.append(head_name)
        critical_missing = [name for name in missing_heads if name in {"z_mu", "reward_probe_head"}]
        if critical_missing:
            raise FileNotFoundError(
                f"Compression checkpoint {checkpoint_dir} is missing required heads {critical_missing}. "
                "For visual candidate scoring, train without --no_train_probe so reward_probe_head.pth exists."
            )
        for head_name in missing_heads:
            print(f"[Compression] WARNING: missing {Path(checkpoint_dir) / f'{head_name}.pth'}", flush=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _reduce_reward(self, reward_tensor: torch.Tensor) -> torch.Tensor:
        if self.scoring_dim_indices is not None:
            return reward_tensor[:, self.scoring_dim_indices].mean(dim=1)
        return reward_tensor.mean(dim=1)

    @torch.no_grad()
    def score(self, contexts: list[str], completions: list[str],
              max_ctx_len: int = 1024, max_resp_len: int = 256) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(contexts), 8):
            batch_contexts = contexts[i:i + 8]
            batch_completions = completions[i:i + 8]
            ctx = self.tokenizer(
                batch_contexts,
                truncation=True,
                max_length=max_ctx_len,
                padding=True,
                return_tensors="pt",
            )
            resp = self.tokenizer(
                batch_completions,
                truncation=True,
                max_length=max_resp_len,
                padding=True,
                return_tensors="pt",
            )
            ctx_ids = ctx.input_ids.to(self.device)
            ctx_mask = ctx.attention_mask.to(self.device)
            resp_ids = resp.input_ids.to(self.device)
            resp_mask = resp.attention_mask.to(self.device)
            with amp_context(self.device):
                z = self.model.encode_context_to_z(ctx_ids, ctx_mask)
                reward = self.model.forward_reward_probe_with_z(z, resp_ids, resp_mask)
            scores.extend(self._reduce_reward(reward.float()).cpu().tolist())
        return scores


def parse_scoring_dim_names(dims: str | None) -> list[str] | None:
    if not dims or dims.strip().lower() == "all":
        return None
    names = [d.strip() for d in dims.split(",") if d.strip()]
    invalid = [d for d in names if d not in SOTOPIA_DIMENSIONS]
    if invalid:
        raise ValueError(f"Invalid scoring dims {invalid}; valid dims are {SOTOPIA_DIMENSIONS}")
    return names


def scoring_indices_for(names: list[str] | None, valid_dims: list[str], model_name: str) -> list[int] | None:
    if names is None:
        return None
    unsupported = [name for name in names if name not in valid_dims]
    if unsupported:
        raise ValueError(
            f"{model_name} does not support scoring dims {unsupported}; "
            f"supported dims are {valid_dims}. Use --scoring-dims goal,relationship,knowledge "
            "when comparing against the simple no-mental baseline."
        )
    return [valid_dims.index(name) for name in names]


def load_pairs(path: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            required = ["pair_id", "observable_context", "candidate_a", "candidate_b", "state_a", "state_b"]
            missing = [key for key in required if key not in rec]
            if missing:
                raise ValueError(f"{path}:{line_no} missing required keys {missing}")
            for state_key in ["state_a", "state_b"]:
                state = rec[state_key]
                for key in ["z_context", "correct"]:
                    if key not in state:
                        raise ValueError(f"{path}:{line_no} {state_key} missing '{key}'")
            pairs.append(rec)
    return pairs


def score_two_candidates(model: Any, context: str, cand_a: str, cand_b: str) -> tuple[float, float]:
    scores = model.score([context, context], [cand_a, cand_b])
    return float(scores[0]), float(scores[1])


def correct_margin(score_a: float, score_b: float, correct: str) -> float:
    if correct == "a":
        return score_a - score_b
    if correct == "b":
        return score_b - score_a
    raise ValueError(f"correct must be 'a' or 'b', got {correct!r}")


def signed_ab_margin(score_a: float, score_b: float) -> float:
    return score_a - score_b


def add_row(rows: list[dict[str, Any]], pair_id: str, model_name: str, state_label: str,
            state_name: str, score_a: float, score_b: float, correct: str) -> None:
    rows.append({
        "pair_id": pair_id,
        "model": model_name,
        "state_label": state_label,
        "state_name": state_name,
        "score_candidate_a": score_a,
        "score_candidate_b": score_b,
        "signed_margin_a_minus_b": signed_ab_margin(score_a, score_b),
        "correct": correct,
        "correct_margin": correct_margin(score_a, score_b, correct),
        "is_correct": int(correct_margin(score_a, score_b, correct) > 0),
    })


def score_pairs(args: argparse.Namespace, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scoring_dim_names = parse_scoring_dim_names(args.scoring_dims)
    scoring_indices = scoring_indices_for(scoring_dim_names, SOTOPIA_DIMENSIONS, "mental/compression reward")

    mental = None
    if not args.skip_mental:
        sys.path.insert(0, str(SOTOPIA_ROOT))
        from stage2_grpo_agent_training_v3 import FrozenRewardModel  # noqa: WPS433

        mental = FrozenRewardModel(
            args.mental_model_name,
            args.mental_checkpoint,
            z_dim=args.z_dim,
            device=args.mental_device,
            scoring_dim_indices=scoring_indices,
            ensemble_weight=args.ensemble_weight,
        )

    simple = None
    if args.simple_reward_head:
        if scoring_dim_names is None:
            raise ValueError(
                "When using --simple-reward-head, pass an explicit shared subset such as "
                "--scoring-dims goal,relationship,knowledge. The simple baseline was trained "
                "only on goal/relationship/knowledge, while the mental and compression rewards have 7 dims."
            )
        simple_indices = scoring_indices_for(scoring_dim_names, SIMPLE_REWARD_DIMS, "simple reward baseline")
        simple = FrozenSimpleReward(
            args.simple_model_name,
            args.simple_reward_head,
            args.simple_device,
            scoring_dim_indices=simple_indices,
        )

    compression = None
    if args.compression_checkpoint:
        compression = FrozenCompressionReward(
            args.compression_model_name,
            args.compression_checkpoint,
            args.compression_device,
            scoring_dim_indices=scoring_indices,
        )

    shuffled_contexts = [p["state_a"]["z_context"] for p in pairs] + [p["state_b"]["z_context"] for p in pairs]
    rng = random.Random(args.seed)
    rng.shuffle(shuffled_contexts)
    shuffle_idx = 0

    for rec in pairs:
        pair_id = rec["pair_id"]
        cand_a = rec["candidate_a"]
        cand_b = rec["candidate_b"]
        state_a = rec["state_a"]
        state_b = rec["state_b"]

        if mental is not None:
            a_a, a_b = score_two_candidates(mental, state_a["z_context"], cand_a, cand_b)
            b_a, b_b = score_two_candidates(mental, state_b["z_context"], cand_a, cand_b)
            add_row(rows, pair_id, "mental_correct_z", "state_a", state_a.get("name", "state_a"), a_a, a_b, state_a["correct"])
            add_row(rows, pair_id, "mental_correct_z", "state_b", state_b.get("name", "state_b"), b_a, b_b, state_b["correct"])

            swap_a_a, swap_a_b = score_two_candidates(mental, state_b["z_context"], cand_a, cand_b)
            swap_b_a, swap_b_b = score_two_candidates(mental, state_a["z_context"], cand_a, cand_b)
            add_row(rows, pair_id, "mental_swapped_z", "state_a", state_a.get("name", "state_a"), swap_a_a, swap_a_b, state_a["correct"])
            add_row(rows, pair_id, "mental_swapped_z", "state_b", state_b.get("name", "state_b"), swap_b_a, swap_b_b, state_b["correct"])

            obs_a, obs_b = score_two_candidates(mental, rec["observable_context"], cand_a, cand_b)
            add_row(rows, pair_id, "mental_observed_only", "state_a", state_a.get("name", "state_a"), obs_a, obs_b, state_a["correct"])
            add_row(rows, pair_id, "mental_observed_only", "state_b", state_b.get("name", "state_b"), obs_a, obs_b, state_b["correct"])

            if args.include_shuffled_z and shuffled_contexts:
                shuf_a_ctx = shuffled_contexts[shuffle_idx % len(shuffled_contexts)]
                shuffle_idx += 1
                shuf_b_ctx = shuffled_contexts[shuffle_idx % len(shuffled_contexts)]
                shuffle_idx += 1
                shuf_a_a, shuf_a_b = score_two_candidates(mental, shuf_a_ctx, cand_a, cand_b)
                shuf_b_a, shuf_b_b = score_two_candidates(mental, shuf_b_ctx, cand_a, cand_b)
                add_row(rows, pair_id, "mental_shuffled_z", "state_a", state_a.get("name", "state_a"), shuf_a_a, shuf_a_b, state_a["correct"])
                add_row(rows, pair_id, "mental_shuffled_z", "state_b", state_b.get("name", "state_b"), shuf_b_a, shuf_b_b, state_b["correct"])

        if simple is not None:
            s_a, s_b = score_two_candidates(simple, rec["observable_context"], cand_a, cand_b)
            add_row(rows, pair_id, "simple_reward_observed", "state_a", state_a.get("name", "state_a"), s_a, s_b, state_a["correct"])
            add_row(rows, pair_id, "simple_reward_observed", "state_b", state_b.get("name", "state_b"), s_a, s_b, state_b["correct"])

        if compression is not None:
            c_a, c_b = score_two_candidates(compression, rec["observable_context"], cand_a, cand_b)
            add_row(rows, pair_id, "compression_observed", "state_a", state_a.get("name", "state_a"), c_a, c_b, state_a["correct"])
            add_row(rows, pair_id, "compression_observed", "state_b", state_b.get("name", "state_b"), c_a, c_b, state_b["correct"])

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    summary: dict[str, Any] = {"models": {}}
    for model, model_rows in by_model.items():
        pair_ids = sorted({row["pair_id"] for row in model_rows})
        flip_successes = []
        sensitivities = []
        for pair_id in pair_ids:
            pair_rows = [r for r in model_rows if r["pair_id"] == pair_id]
            state_a = next((r for r in pair_rows if r["state_label"] == "state_a"), None)
            state_b = next((r for r in pair_rows if r["state_label"] == "state_b"), None)
            if not state_a or not state_b:
                continue
            flip_successes.append(int(state_a["is_correct"] == 1 and state_b["is_correct"] == 1))
            sensitivities.append(abs(state_a["signed_margin_a_minus_b"] - state_b["signed_margin_a_minus_b"]))

        summary["models"][model] = {
            "num_pairs": len(pair_ids),
            "flip_accuracy": float(np.mean(flip_successes)) if flip_successes else None,
            "mean_correct_margin": float(np.mean([r["correct_margin"] for r in model_rows])) if model_rows else None,
            "mean_state_sensitivity": float(np.mean(sensitivities)) if sensitivities else None,
            "mean_signed_margin_state_a": float(np.mean([r["signed_margin_a_minus_b"] for r in model_rows if r["state_label"] == "state_a"])),
            "mean_signed_margin_state_b": float(np.mean([r["signed_margin_a_minus_b"] for r in model_rows if r["state_label"] == "state_b"])),
        }
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id", "model", "state_label", "state_name",
        "score_candidate_a", "score_candidate_b", "signed_margin_a_minus_b",
        "correct", "correct_margin", "is_correct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_crossover(rows: list[dict[str, Any]], out_path: Path) -> None:
    by_model: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {"state_a": [], "state_b": []})
        by_model[row["model"]][row["state_label"]].append(row["signed_margin_a_minus_b"])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xs = np.array([0, 1])
    for model in ordered_models(list(by_model.keys())):
        states = by_model[model]
        if not states["state_a"] or not states["state_b"]:
            continue
        means = [np.mean(states["state_a"]), np.mean(states["state_b"])]
        sems = [
            np.std(states["state_a"]) / max(1, np.sqrt(len(states["state_a"]))),
            np.std(states["state_b"]) / max(1, np.sqrt(len(states["state_b"]))),
        ]
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=MODEL_COLORS.get(model, PLOT_DARK_GREY),
            label=MODEL_LABELS.get(model, model),
        )

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xticks(xs, ["State A\n(candidate A correct)", "State B\n(candidate B correct)"])
    ax.set_title("Counterfactual Preference Crossover")
    style_plot_axes(ax, ylabel="score(candidate A) - score(candidate B)")
    style_plot_legend(ax.legend(frameon=True, fontsize=9))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_flip_accuracy(summary: dict[str, Any], out_path: Path) -> None:
    models = ordered_models(list(summary["models"].keys()))
    values = [summary["models"][m]["flip_accuracy"] for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, PLOT_DARK_GREY) for m in models]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(np.arange(len(models)), values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_title("Both Branches Correct Under State Flip")
    ax.set_xticks(np.arange(len(models)), labels, rotation=25, ha="right")
    style_plot_axes(ax, ylabel="flip accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_first_pair_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    mental_rows = [r for r in rows if r["model"] == "mental_correct_z"]
    if not mental_rows:
        return
    first_pair = mental_rows[0]["pair_id"]
    pair_rows = [r for r in mental_rows if r["pair_id"] == first_pair]
    state_a = next((r for r in pair_rows if r["state_label"] == "state_a"), None)
    state_b = next((r for r in pair_rows if r["state_label"] == "state_b"), None)
    if not state_a or not state_b:
        return
    mat = np.array([
        [state_a["score_candidate_a"], state_a["score_candidate_b"]],
        [state_b["score_candidate_a"], state_b["score_candidate_b"]],
    ])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks([0, 1], ["Candidate A", "Candidate B"])
    ax.set_yticks([0, 1], ["State A z", "State B z"])
    ax.set_title(f"Mental Reward Matrix: {first_pair}")
    for y in range(2):
        for x in range(2):
            ax.text(x, y, f"{mat[y, x]:.3f}", ha="center", va="center", color="white", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_plot_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mental-model-name", default=os.environ.get("SOTOPIA_REWARD_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--mental-checkpoint", default=os.environ.get("SOTOPIA_REWARD_CHECKPOINT"))
    parser.add_argument("--simple-model-name", default=os.environ.get("SOTOPIA_SIMPLE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--simple-reward-head", default=os.environ.get("SOTOPIA_SIMPLE_REWARD_HEAD"))
    parser.add_argument("--compression-model-name", default=os.environ.get("SOTOPIA_COMPRESSION_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--compression-checkpoint", default=os.environ.get("SOTOPIA_COMPRESSION_CHECKPOINT"))
    parser.add_argument("--mental-device", default=os.environ.get("SOTOPIA_MENTAL_DEVICE", "cuda:0"))
    parser.add_argument("--simple-device", default=os.environ.get("SOTOPIA_SIMPLE_DEVICE", "cuda:0"))
    parser.add_argument("--compression-device", default=os.environ.get("SOTOPIA_COMPRESSION_DEVICE", "cuda:0"))
    parser.add_argument("--scoring-dims", default=os.environ.get("SOTOPIA_COUNTERFACTUAL_DIMS", "goal,relationship,knowledge"))
    parser.add_argument("--ensemble-weight", type=float, default=0.7)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-mental", action="store_true")
    parser.add_argument("--include-shuffled-z", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_mental and not args.mental_checkpoint:
        raise ValueError("--mental-checkpoint is required unless --skip-mental is set")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_pairs(args.counterfactual_jsonl)
    rows = score_pairs(args, pairs)
    summary = summarize(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "counterfactual_scores.csv", rows)
    with open(out_dir / "counterfactual_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "counterfactual_run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    plot_crossover(rows, out_dir / "counterfactual_margin_crossover.png")
    plot_flip_accuracy(summary, out_dir / "counterfactual_flip_accuracy.png")
    plot_first_pair_heatmap(rows, out_dir / "counterfactual_heatmap_first_pair.png")

    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
