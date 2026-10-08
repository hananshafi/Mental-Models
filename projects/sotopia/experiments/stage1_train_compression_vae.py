#!/usr/bin/env python3
"""Train a pure compression VAE baseline for SOTOPIA.

This baseline mirrors the Stage-1 mental/reward training setup, but the latent
representation is trained only to reconstruct observed context information.
It does not use mental-state labels, reward regression, or preference loss.

For visual action-ranking comparisons, the script can optionally train a
frozen-latent reward probe after compression training. The probe is only an
evaluation head: gradients do not update the compression encoder/latent.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


SOTOPIA_DIMENSIONS = [
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
]

DIM_RANGES = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}

COMPRESSION_CUSTOM_HEAD_NAMES = [
    "z_mu",
    "z_logvar",
    "z_to_memory",
    "position_embed",
    "decoder",
    "out_norm",
    "reward_probe_head",
]


def normalize_score(dim: str, score: float) -> float:
    lo, hi = DIM_RANGES[dim]
    return (float(score) - lo) / (hi - lo + 1e-8)


def get_transformer_from_peft(peft_causal_lm):
    model = peft_causal_lm
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model"):
        return model.model
    raise RuntimeError("Could not find underlying transformer (.model).")


class SotopiaCompressionDataset(Dataset):
    """Per-turn SOTOPIA contexts for compression-only VAE training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_ctx_len: int = 1024,
        max_target_len: int = 256,
        max_resp_len: int = 256,
        target_mode: str = "summary",
        summary_history_turns: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_target_len = max_target_len
        self.max_resp_len = max_resp_len
        self.target_mode = target_mode
        self.summary_history_turns = summary_history_turns
        self.samples: list[dict[str, Any]] = []

        print(f"Loading SOTOPIA compression data from {data_path}...", flush=True)
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                    self._process_episode(episode)
                except json.JSONDecodeError:
                    parts = line.split("}{")
                    for idx, part in enumerate(parts):
                        obj_str = part
                        if idx > 0:
                            obj_str = "{" + obj_str
                        if idx < len(parts) - 1:
                            obj_str = obj_str + "}"
                        try:
                            self._process_episode(json.loads(obj_str))
                        except json.JSONDecodeError:
                            continue
        print(f"Created {len(self.samples)} compression samples.", flush=True)

    def _process_episode(self, episode: dict[str, Any]) -> None:
        pe = episode.get("parsed_episode", {})
        turns = pe.get("turns", [])
        turn_rewards = episode.get("turn_rewards", [])
        if not turns or not turn_rewards:
            return

        scenario = pe.get("scenario", "")
        agent_1_name = pe.get("agent_1_name", "Agent 1")
        agent_2_name = pe.get("agent_2_name", "Agent 2")
        agents = {
            agent_1_name: {
                "background": pe.get("agent_1_background", ""),
                "goal": pe.get("agent_1_goal", ""),
                "secret": pe.get("agent_1_secret", ""),
                "reward_key": "agent_1_rewards",
            },
            agent_2_name: {
                "background": pe.get("agent_2_background", ""),
                "goal": pe.get("agent_2_goal", ""),
                "secret": pe.get("agent_2_secret", ""),
                "reward_key": "agent_2_rewards",
            },
        }

        for tr in turn_rewards:
            turn_num = tr.get("turn")
            speaker = tr.get("agent")
            if speaker not in agents or turn_num is None or turn_num >= len(turns):
                continue
            pos_response = turns[turn_num].get("content", "")
            if not pos_response.strip():
                continue

            info = agents[speaker]
            history_lines = []
            for prev_t in turns[:turn_num]:
                prev_speaker = prev_t.get("agent", "Unknown")
                action = prev_t.get("action", "said")
                content = prev_t.get("content", "")
                history_lines.append(f"Turn {prev_t['turn'] + 1} | {prev_speaker} {action}: {content}")

            context_text = self._format_context(
                scenario=scenario,
                background=info["background"],
                goal=info["goal"],
                secret=info["secret"],
                history_lines=history_lines,
                turn_num=turn_num,
                speaker=speaker,
            )
            target_text = self._format_target(
                scenario=scenario,
                background=info["background"],
                goal=info["goal"],
                secret=info["secret"],
                history_lines=history_lines,
                turn_num=turn_num,
                speaker=speaker,
                context_text=context_text,
            )

            rewards_data = tr.get(info["reward_key"], {})
            reward_vec = []
            for dim in SOTOPIA_DIMENSIONS:
                dim_data = rewards_data.get(dim, {})
                score = dim_data.get("score", 0) if isinstance(dim_data, dict) else 0
                reward_vec.append(normalize_score(dim, score))

            mental_state = rewards_data.get("mental_state", {})
            hard_negative = str(mental_state.get("hard_negative_response", "") or "").strip()
            hard_negative = hard_negative.strip("\"'")
            if "(" in hard_negative:
                hard_negative = hard_negative[:hard_negative.rfind("(")].strip()

            self.samples.append({
                "context_text": context_text,
                "target_text": target_text,
                "pos_response": pos_response,
                "neg_response": hard_negative,
                "reward_vec": reward_vec,
            })

    @staticmethod
    def _format_context(
        scenario: str,
        background: str,
        goal: str,
        secret: str,
        history_lines: list[str],
        turn_num: int,
        speaker: str,
    ) -> str:
        secret_text = secret if secret else "None"
        history = "\n".join(history_lines)
        return (
            f"Scenario: {scenario}\n"
            f"Background: {background}\n"
            f"Goal: {goal}\n"
            f"Secret: {secret_text}\n"
            f"Dialogue History:\n{history}\n"
            f"Turn {turn_num + 1} | {speaker}:"
        )

    def _format_target(
        self,
        scenario: str,
        background: str,
        goal: str,
        secret: str,
        history_lines: list[str],
        turn_num: int,
        speaker: str,
        context_text: str,
    ) -> str:
        if self.target_mode == "context":
            return context_text
        if self.target_mode != "summary":
            raise ValueError(f"Unsupported target_mode={self.target_mode!r}")

        recent_history = history_lines[-self.summary_history_turns:] if self.summary_history_turns > 0 else []
        recent = "\n".join(recent_history) if recent_history else "No prior dialogue."
        secret_text = secret if secret else "None"
        return (
            f"Scenario: {scenario}\n"
            f"Speaker: {speaker}\n"
            f"Background: {background}\n"
            f"Goal: {goal}\n"
            f"Secret: {secret_text}\n"
            f"Recent dialogue:\n{recent}\n"
            f"Next turn: {turn_num + 1}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        ctx = self.tokenizer(
            sample["context_text"],
            truncation=True,
            max_length=self.max_ctx_len,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            sample["target_text"],
            truncation=True,
            max_length=self.max_target_len,
            padding="max_length",
            return_tensors="pt",
        )
        pos = self.tokenizer(
            sample["pos_response"],
            truncation=True,
            max_length=self.max_resp_len,
            padding="max_length",
            return_tensors="pt",
        )
        neg_text = sample["neg_response"] or sample["pos_response"]
        neg = self.tokenizer(
            neg_text,
            truncation=True,
            max_length=self.max_resp_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "ctx_input_ids": ctx.input_ids.squeeze(0),
            "ctx_attention_mask": ctx.attention_mask.squeeze(0),
            "target_input_ids": target.input_ids.squeeze(0),
            "target_attention_mask": target.attention_mask.squeeze(0),
            "pos_input_ids": pos.input_ids.squeeze(0),
            "pos_attention_mask": pos.attention_mask.squeeze(0),
            "neg_input_ids": neg.input_ids.squeeze(0),
            "neg_attention_mask": neg.attention_mask.squeeze(0),
            "reward_vec": torch.tensor(sample["reward_vec"], dtype=torch.float32),
            "has_negative": torch.tensor(1.0 if sample["neg_response"] else 0.0),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]], tokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id

    def pad(key: str, value: int) -> torch.Tensor:
        return nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True, padding_value=value)

    return {
        "ctx_input_ids": pad("ctx_input_ids", pad_id).long(),
        "ctx_attention_mask": pad("ctx_attention_mask", 0).long(),
        "target_input_ids": pad("target_input_ids", pad_id).long(),
        "target_attention_mask": pad("target_attention_mask", 0).long(),
        "pos_input_ids": pad("pos_input_ids", pad_id).long(),
        "pos_attention_mask": pad("pos_attention_mask", 0).long(),
        "neg_input_ids": pad("neg_input_ids", pad_id).long(),
        "neg_attention_mask": pad("neg_attention_mask", 0).long(),
        "reward_vec": torch.stack([item["reward_vec"] for item in batch], dim=0),
        "has_negative": torch.stack([item["has_negative"] for item in batch], dim=0),
    }


class CompressionVAEModel(nn.Module):
    """Single-latent VAE trained only as an observed-context compressor."""

    def __init__(
        self,
        base_model: nn.Module,
        z_dim: int = 256,
        reward_dim: int = len(SOTOPIA_DIMENSIONS),
        num_memory_tokens: int = 24,
        max_target_len: int = 256,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        self.transformer = get_transformer_from_peft(base_model)
        self.hidden_size = base_model.get_input_embeddings().embedding_dim
        self.z_dim = z_dim
        self.num_memory_tokens = num_memory_tokens
        self.max_target_len = max_target_len

        self.z_mu = nn.Linear(self.hidden_size, z_dim)
        self.z_logvar = nn.Linear(self.hidden_size, z_dim)
        self.z_to_memory = nn.Linear(z_dim, num_memory_tokens * self.hidden_size)
        self.position_embed = nn.Embedding(max_target_len, self.hidden_size)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=decoder_layers,
        )
        self.out_norm = nn.LayerNorm(self.hidden_size)
        self.reward_probe_head = nn.Sequential(
            nn.Linear(z_dim + self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.z_mu, self.z_logvar, self.z_to_memory]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.z_logvar.bias, -2.0)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        for layer in self.reward_probe_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def _encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)

    def _sample_z(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.z_mu(h)
        logvar = self.z_logvar(h).clamp(-10.0, 8.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar

    def encode_context_to_z(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self._encode_context(input_ids, attention_mask)
        return self.z_mu(h)

    def _decode_from_z(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = z.size(0)
        memory = self.z_to_memory(z).view(batch_size, self.num_memory_tokens, self.hidden_size)
        positions = torch.arange(seq_len, device=z.device).unsqueeze(0).expand(batch_size, seq_len)
        queries = self.position_embed(positions)
        decoded = self.decoder(tgt=queries, memory=memory)
        decoded = self.out_norm(decoded)
        output_embedding = self.base_model.get_output_embeddings()
        return F.linear(decoded, output_embedding.weight)

    def compression_loss(
        self,
        ctx_input_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = self._encode_context(ctx_input_ids, ctx_attention_mask)
        z, mu, logvar = self._sample_z(h)
        logits = self._decode_from_z(z, target_input_ids.size(1))
        labels = target_input_ids.clone()
        labels[target_attention_mask == 0] = -100
        recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"recon_loss": recon_loss, "mu": mu, "logvar": logvar}

    def encode_response(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)

    def forward_reward_probe_with_z(
        self,
        z: torch.Tensor,
        resp_input_ids: torch.Tensor,
        resp_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        resp_hidden = self.encode_response(resp_input_ids, resp_attention_mask)
        return self.reward_probe_head(torch.cat([z, resp_hidden], dim=-1))


def compute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    mu32 = mu.float()
    logvar32 = logvar.float().clamp(-10.0, 8.0)
    return -0.5 * torch.mean(1 + logvar32 - mu32.pow(2) - logvar32.exp())


def train_compression_epoch(
    model: CompressionVAEModel,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int,
    kl_weight: float,
    kl_anneal_steps: int,
    global_step_offset: int,
    max_grad_norm: float,
) -> tuple[float, dict[str, float], int]:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    opt_steps = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        current_step = global_step_offset + (batch_idx + 1) // grad_accum_steps
        anneal = min(1.0, current_step / kl_anneal_steps) if kl_anneal_steps > 0 else 1.0
        eff_kl = kl_weight * anneal

        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            out = model.compression_loss(ctx_ids, ctx_mask, target_ids, target_mask)
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                kl = compute_kl_loss(out["mu"], out["logvar"])
            raw_loss = out["recon_loss"] + eff_kl * kl
            loss = raw_loss / grad_accum_steps

        loss.backward()
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_steps += 1

        totals["loss"] += raw_loss.item()
        totals["recon"] += out["recon_loss"].item()
        totals["kl"] += kl.item()
        if (batch_idx + 1) % 10 == 0:
            n = batch_idx + 1
            print(
                f"  Epoch {epoch + 1} Step {n}: "
                f"loss={totals['loss'] / n:.4f} recon={totals['recon'] / n:.4f} "
                f"kl={totals['kl'] / n:.4f}(w={eff_kl:.4f}) lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    n = max(1, len(dataloader))
    metrics = {key: value / n for key, value in totals.items()}
    return metrics["loss"], metrics, global_step_offset + opt_steps


@torch.no_grad()
def evaluate_compression_epoch(
    model: CompressionVAEModel,
    dataloader: DataLoader,
    device: torch.device,
    kl_weight: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            out = model.compression_loss(ctx_ids, ctx_mask, target_ids, target_mask)
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                kl = compute_kl_loss(out["mu"], out["logvar"])
            raw_loss = out["recon_loss"] + kl_weight * kl
        totals["loss"] += raw_loss.item()
        totals["recon"] += out["recon_loss"].item()
        totals["kl"] += kl.item()
    n = max(1, len(dataloader))
    metrics = {key: value / n for key, value in totals.items()}
    return metrics["loss"], metrics


def train_probe_epoch(
    model: CompressionVAEModel,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    model.reward_probe_head.train()
    totals = {"loss": 0.0, "pref": 0.0, "reg": 0.0}

    for batch_idx, batch in enumerate(dataloader):
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        reward_targets = batch["reward_vec"].to(device)
        has_negative = batch["has_negative"].to(device)

        with torch.no_grad():
            z = model.encode_context_to_z(ctx_ids, ctx_mask)
            pos_hidden = model.encode_response(pos_ids, pos_mask)
            neg_hidden = model.encode_response(neg_ids, neg_mask)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            pos_reward = model.reward_probe_head(torch.cat([z, pos_hidden], dim=-1))
            neg_reward = model.reward_probe_head(torch.cat([z, neg_hidden], dim=-1))
            pref = F.softplus(neg_reward - pos_reward).mean(dim=1)
            pref_loss = (pref * has_negative).sum() / (has_negative.sum() + 1e-8)
            reg_loss = F.smooth_l1_loss(pos_reward, reward_targets)
            loss = pref_loss + 0.5 * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.reward_probe_head.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        totals["loss"] += loss.item()
        totals["pref"] += pref_loss.item()
        totals["reg"] += reg_loss.item()
        if (batch_idx + 1) % 50 == 0:
            n = batch_idx + 1
            print(
                f"  Probe Epoch {epoch + 1} Step {n}: "
                f"loss={totals['loss'] / n:.4f} pref={totals['pref'] / n:.4f} reg={totals['reg'] / n:.4f}",
                flush=True,
            )

    n = max(1, len(dataloader))
    metrics = {key: value / n for key, value in totals.items()}
    return metrics["loss"], metrics


@torch.no_grad()
def evaluate_probe_epoch(
    model: CompressionVAEModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals = {"loss": 0.0, "pref": 0.0, "reg": 0.0}
    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        reward_targets = batch["reward_vec"].to(device)
        has_negative = batch["has_negative"].to(device)
        z = model.encode_context_to_z(ctx_ids, ctx_mask)
        pos_hidden = model.encode_response(pos_ids, pos_mask)
        neg_hidden = model.encode_response(neg_ids, neg_mask)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=torch.bfloat16):
            pos_reward = model.reward_probe_head(torch.cat([z, pos_hidden], dim=-1))
            neg_reward = model.reward_probe_head(torch.cat([z, neg_hidden], dim=-1))
            pref = F.softplus(neg_reward - pos_reward).mean(dim=1)
            pref_loss = (pref * has_negative).sum() / (has_negative.sum() + 1e-8)
            reg_loss = F.smooth_l1_loss(pos_reward, reward_targets)
            loss = pref_loss + 0.5 * reg_loss
        totals["loss"] += loss.item()
        totals["pref"] += pref_loss.item()
        totals["reg"] += reg_loss.item()
    n = max(1, len(dataloader))
    metrics = {key: value / n for key, value in totals.items()}
    return metrics["loss"], metrics


def save_checkpoint(model: CompressionVAEModel, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model.base_model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    for head_name in COMPRESSION_CUSTOM_HEAD_NAMES:
        torch.save(getattr(model, head_name).state_dict(), os.path.join(save_dir, f"{head_name}.pth"))


def set_trainable_for_compression(model: CompressionVAEModel) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or any(head in name for head in COMPRESSION_CUSTOM_HEAD_NAMES if head != "reward_probe_head")
    for param in model.reward_probe_head.parameters():
        param.requires_grad = False


def set_trainable_for_probe(model: CompressionVAEModel) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reward_probe_head.parameters():
        param.requires_grad = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure compression VAE baseline for SOTOPIA.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument("--output_dir", type=str, default="projects/sotopia/experiments/runs/stage1/compression_vae_checkpoint")
    parser.add_argument("--target_mode", choices=["summary", "context"], default="summary")
    parser.add_argument("--summary_history_turns", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=256, help="Use 256 to match z1+z2 total capacity in the mental model.")
    parser.add_argument("--num_memory_tokens", type=int, default=24)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--kl_anneal_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_train_probe", action="store_true", help="Skip frozen-latent reward probe training.")
    parser.add_argument("--probe_epochs", type=int, default=3)
    parser.add_argument("--probe_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        config = vars(args).copy()
        config["baseline_type"] = "compression_only_vae"
        config["reward_dimensions"] = SOTOPIA_DIMENSIONS
        json.dump(config, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.model_name}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    base_model.config.use_cache = False

    num_layers = base_model.config.num_hidden_layers
    top_layers = list(range(num_layers - args.num_lora_layers, num_layers))
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

    model = CompressionVAEModel(
        base_model=base_model,
        z_dim=args.z_dim,
        reward_dim=len(SOTOPIA_DIMENSIONS),
        num_memory_tokens=args.num_memory_tokens,
        max_target_len=args.max_target_len,
        decoder_layers=args.decoder_layers,
    ).to(device)
    set_trainable_for_compression(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Compression trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    dataset = SotopiaCompressionDataset(
        args.data_path,
        tokenizer,
        max_ctx_len=args.max_ctx_len,
        max_target_len=args.max_target_len,
        max_resp_len=args.max_resp_len,
        target_mode=args.target_mode,
        summary_history_turns=args.summary_history_turns,
    )
    train_dataset = dataset
    val_dataset = None
    if args.val_ratio > 0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        split_gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)
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

    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(head in name for head in COMPRESSION_CUSTOM_HEAD_NAMES if head != "reward_probe_head"):
            head_params.append(param)
        else:
            lora_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": head_params, "lr": args.lr * args.head_lr_mult, "weight_decay": 0.01},
    ])
    total_steps = max(1, len(train_loader) * args.num_epochs // args.grad_accum_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_metric = float("inf")
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss, metrics, global_step = train_compression_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            grad_accum_steps=args.grad_accum_steps,
            kl_weight=args.kl_weight,
            kl_anneal_steps=args.kl_anneal_steps,
            global_step_offset=global_step,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"\nCompression Epoch {epoch + 1}/{args.num_epochs}: loss={avg_loss:.4f}", flush=True)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}", flush=True)

        monitor = avg_loss
        monitor_name = "train_loss"
        if val_loader is not None:
            val_loss, val_metrics = evaluate_compression_epoch(model, val_loader, device, kl_weight=args.kl_weight)
            print(f"  val_loss: {val_loss:.4f}", flush=True)
            for key, value in val_metrics.items():
                print(f"  val_{key}: {value:.4f}", flush=True)
            monitor = val_loss
            monitor_name = "val_loss"

        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        save_checkpoint(model, ckpt_dir)
        if monitor < best_metric:
            best_metric = monitor
            save_checkpoint(model, os.path.join(args.output_dir, "best"))
            print(f"  -> Best compression model saved ({monitor_name}={best_metric:.4f})", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    if not args.no_train_probe:
        print("\n=== Frozen-latent reward probe training ===", flush=True)
        set_trainable_for_probe(model)
        probe_optimizer = torch.optim.AdamW(model.reward_probe_head.parameters(), lr=args.probe_lr, weight_decay=0.01)
        best_probe = float("inf")
        for epoch in range(args.probe_epochs):
            probe_loss, probe_metrics = train_probe_epoch(model, train_loader, probe_optimizer, device, epoch)
            print(f"\nProbe Epoch {epoch + 1}/{args.probe_epochs}: loss={probe_loss:.4f}", flush=True)
            for key, value in probe_metrics.items():
                print(f"  {key}: {value:.4f}", flush=True)
            monitor = probe_loss
            if val_loader is not None:
                val_probe, val_probe_metrics = evaluate_probe_epoch(model, val_loader, device)
                print(f"  val_probe_loss: {val_probe:.4f}", flush=True)
                for key, value in val_probe_metrics.items():
                    print(f"  val_probe_{key}: {value:.4f}", flush=True)
                monitor = val_probe
            if monitor < best_probe:
                best_probe = monitor
                save_checkpoint(model, os.path.join(args.output_dir, "best"))
                print(f"  -> Best probe saved (loss={best_probe:.4f})", flush=True)

    print(f"\nCompression VAE training complete. Checkpoints: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
