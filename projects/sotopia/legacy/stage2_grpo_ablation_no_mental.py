#!/usr/bin/env python3
"""
Stage 2 Ablation: GRPO Agent Training WITHOUT Stage 1 Mental+Reward Model
==========================================================================
Ablation baseline: replaces the coupled mental+reward model (Stage 1) with
a simple reward model (frozen base LLM + MLP head) trained from scratch on
the same preference data. No VAE, no z-bottleneck, no mental decoder.

This proves the value of the Stage 1 mental model coupling.
"""

# CUDA_VISIBLE_DEVICES=0 python projects/sotopia/legacy/stage2_grpo_ablation_no_mental.py \
#   --policy_model_name Qwen/Qwen2.5-7B-Instruct \
#   --data_path projects/sotopia/data/sotopia_turn_rewards.jsonl \
#   --output_dir projects/sotopia/grpo_ablation_no_mental_checkpoint \
#   --group_size 8 \
#   --grpo_epochs 3 \
#   --prompts_per_step 4 \
#   --num_ppo_epochs 1 \
#   --clip_eps 0.2 \
#   --kl_coeff 0.04 \
#   --lr 5e-6 \
#   --temperature 0.8 \
#   --top_p 0.95 \
#   --max_gen_len 256 \
#   --max_ctx_len 1024 \
#   --sft_warmup \
#   --sft_epochs 1 \
#   --sft_lr 2e-5 \
#   --sft_batch_size 4 \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
#   --num_lora_layers 16 \
#   --reward_train_epochs 5 \
#   --reward_lr 1e-4 \
#   --reward_batch_size 8 \
#   --save_every 50 \
#   --seed 42


import os
import sys
import json
import gc
import time
import argparse
import random
import re
from typing import List, Optional

# Force unbuffered stdout/stderr BEFORE any other imports
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "fileno"):
    try:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    except Exception:
        pass

print(">> stage2_grpo_ablation_no_mental.py starting...", flush=True)
print(">> Importing torch...", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print(">> Importing transformers...", flush=True)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

print(">> All imports done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SOTOPIA_DIMENSIONS = [
    "goal",
    "relationship",
    "knowledge",
]

DIM_RANGES = {
    "goal":         (0, 10),
    "relationship": (-5, 5),
    "knowledge":    (0, 10),
}

REWARD_DIM = len(SOTOPIA_DIMENSIONS)  # 3


def normalize_score(dim: str, score: float) -> float:
    lo, hi = DIM_RANGES[dim]
    return (score - lo) / (hi - lo + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Simple Reward Model (NO mental model, NO VAE, NO z-bottleneck)
# Just: frozen LLM backbone → pool(context+response) → MLP → reward
# ──────────────────────────────────────────────────────────────────────────────
class SimpleRewardModel(nn.Module):
    """A basic reward model: frozen LLM encoder + MLP reward head."""

    def __init__(self, base_model: nn.Module, reward_dim: int = REWARD_DIM):
        super().__init__()
        self.base_model = base_model

        if hasattr(base_model, "config"):
            hidden_size = base_model.config.hidden_size
        else:
            hidden_size = base_model.get_input_embeddings().embedding_dim

        # Simple MLP reward head: pool(hidden) → reward_dim
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )

    def _get_transformer(self):
        base = self.base_model
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model"):
            base = base.model
        if hasattr(base, "model"):
            base = base.model
        return base

    def _encode_and_pool(self, input_ids, attention_mask):
        transformer = self._get_transformer()
        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def forward(self, input_ids, attention_mask):
        """Encode context+response concatenation and predict reward."""
        pooled = self._encode_and_pool(input_ids, attention_mask)
        return self.reward_head(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Reward Training Dataset (preference pairs from turn_rewards)
# ──────────────────────────────────────────────────────────────────────────────
class RewardPreferenceDataset(Dataset):
    """Builds (positive, negative) preference pairs from turn_rewards data."""

    def __init__(self, data_path: str, tokenizer, max_len: int = 1280):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []  # list of (pos_text, neg_text, reward_target)

        print(f"Loading reward preference data from {data_path}...", flush=True)
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                    self._process_episode(episode)
                except json.JSONDecodeError:
                    parts = line.split('}{')
                    for j, part in enumerate(parts):
                        obj_str = part
                        if j > 0:
                            obj_str = '{' + obj_str
                        if j < len(parts) - 1:
                            obj_str = obj_str + '}'
                        try:
                            episode = json.loads(obj_str)
                            self._process_episode(episode)
                        except json.JSONDecodeError:
                            pass
        print(f"Created {len(self.pairs)} reward preference pairs.", flush=True)

    def _process_episode(self, episode: dict):
        pe = episode["parsed_episode"]
        turns = pe.get("turns", [])
        turn_rewards = episode.get("turn_rewards", [])
        if not turns or not turn_rewards:
            return

        scenario = pe.get("scenario", "")
        agent_1_name = pe.get("agent_1_name", "Agent 1")
        agent_2_name = pe.get("agent_2_name", "Agent 2")

        agents_info = {
            agent_1_name: {
                "background": pe.get("agent_1_background", ""),
                "goal": pe.get("agent_1_goal", ""),
                "secret": pe.get("agent_1_secret", ""),
            },
            agent_2_name: {
                "background": pe.get("agent_2_background", ""),
                "goal": pe.get("agent_2_goal", ""),
                "secret": pe.get("agent_2_secret", ""),
            },
        }

        for tr in turn_rewards:
            turn_num = tr["turn"]
            speaker = tr["agent"]
            if speaker not in agents_info:
                continue
            info = agents_info[speaker]
            if turn_num >= len(turns):
                continue

            actual_utterance = turns[turn_num].get("content", "")
            if not actual_utterance.strip():
                continue

            # Get reward key for this agent
            reward_key = "agent_1_rewards" if speaker == agent_1_name else "agent_2_rewards"
            rewards_data = tr.get(reward_key, {})

            # Build reward target vector
            reward_target = []
            for dim in SOTOPIA_DIMENSIONS:
                dim_data = rewards_data.get(dim, {})
                score = dim_data.get("score", 0) if isinstance(dim_data, dict) else 0
                reward_target.append(normalize_score(dim, score))

            # Get hard negative
            mental_state = rewards_data.get("mental_state", {})
            hard_neg = mental_state.get("hard_negative_response", "")
            if not hard_neg.strip():
                continue

            # Build context
            history_lines = []
            for prev_t in turns[:turn_num]:
                spk = prev_t.get("agent", "Unknown")
                act = prev_t.get("action", "said")
                content = prev_t.get("content", "")
                history_lines.append(f"Turn {prev_t['turn']+1} | {spk} {act}: {content}")
            history_text = "\n".join(history_lines)
            secret_text = info["secret"] if info["secret"] else "None"

            context = (
                f"Scenario: {scenario}\n"
                f"Background: {info['background']}\n"
                f"Goal: {info['goal']}\n"
                f"Secret: {secret_text}\n"
                f"Dialogue History:\n{history_text}\n"
                f"Turn {turn_num+1} | {speaker}:"
            )

            pos_text = context + " " + actual_utterance
            neg_text = context + " " + hard_neg

            self.pairs.append((pos_text, neg_text, reward_target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_text, neg_text, reward_target = self.pairs[idx]

        pos_enc = self.tokenizer(
            pos_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            neg_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )

        return {
            "pos_input_ids": pos_enc.input_ids.squeeze(0),
            "pos_attention_mask": pos_enc.attention_mask.squeeze(0),
            "neg_input_ids": neg_enc.input_ids.squeeze(0),
            "neg_attention_mask": neg_enc.attention_mask.squeeze(0),
            "reward_target": torch.tensor(reward_target, dtype=torch.float32),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Train Simple Reward Model
# ──────────────────────────────────────────────────────────────────────────────
def train_simple_reward_model(reward_model, dataset, device,
                              num_epochs=5, lr=1e-4, batch_size=8):
    """Train the simple reward model with Bradley-Terry preference loss + regression."""
    print("\n=== Training Simple Reward Model (no mental model) ===", flush=True)

    # Freeze backbone, only train reward head
    for param in reward_model.base_model.parameters():
        param.requires_grad = False
    for param in reward_model.reward_head.parameters():
        param.requires_grad = True

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        reward_model.reward_head.parameters(), lr=lr, weight_decay=0.01
    )
    total_steps = len(dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    reward_model.train()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        total_pref_loss = 0
        total_reg_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)
            targets = batch["reward_target"].to(device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                pos_reward = reward_model(pos_ids, pos_mask)
                neg_reward = reward_model(neg_ids, neg_mask)

                # Bradley-Terry preference loss: positive > negative
                pref_loss = F.softplus(neg_reward - pos_reward).mean()

                # Regression loss: positive reward should match ground truth
                reg_loss = F.smooth_l1_loss(pos_reward, targets)

                loss = pref_loss + 0.5 * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reward_model.reward_head.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_pref_loss += pref_loss.item()
            total_reg_loss += reg_loss.item()

            if (batch_idx + 1) % 50 == 0:
                avg_pref = total_pref_loss / (batch_idx + 1)
                avg_reg = total_reg_loss / (batch_idx + 1)
                print(
                    f"  Reward Epoch {epoch+1} Step {batch_idx+1}: "
                    f"pref={avg_pref:.4f} reg={avg_reg:.4f}",
                    flush=True
                )

        avg_loss = (total_pref_loss + total_reg_loss) / len(dataloader)
        print(
            f"  Reward Epoch {epoch+1} done: "
            f"pref={total_pref_loss/len(dataloader):.4f} "
            f"reg={total_reg_loss/len(dataloader):.4f}",
            flush=True
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Freeze everything for inference
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    print(f"  Simple reward model trained. Best loss: {best_loss:.4f}", flush=True)
    return reward_model


# ──────────────────────────────────────────────────────────────────────────────
# Frozen Simple Reward Wrapper (same interface as FrozenRewardModel)
# ──────────────────────────────────────────────────────────────────────────────
class FrozenSimpleRewardModel:
    """Wraps the trained SimpleRewardModel with the same .score() interface."""

    def __init__(self, model: SimpleRewardModel, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def score(self, prompts: List[str], completions: List[str],
              max_ctx_len: int = 1024, max_resp_len: int = 256) -> List[float]:
        rewards = []
        max_len = max_ctx_len + max_resp_len

        for i in range(0, len(prompts), 8):
            batch_texts = [
                p + " " + c for p, c in
                zip(prompts[i:i+8], completions[i:i+8])
            ]

            enc = self.tokenizer(
                batch_texts, truncation=True, max_length=max_len,
                padding=True, return_tensors="pt"
            )
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                reward_vec = self.model(input_ids, attention_mask)

            # Mean across reward dimensions → scalar
            rewards.extend(reward_vec.float().mean(dim=1).cpu().tolist())

        return rewards


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Dataset (simplified — no trajectory rewards needed for ablation)
# ──────────────────────────────────────────────────────────────────────────────
class GRPOPromptDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_ctx_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.samples = []

        print(f"Loading GRPO prompts from {data_path}...", flush=True)
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                    self._process_episode(episode)
                except json.JSONDecodeError:
                    parts = line.split('}{')
                    for j, part in enumerate(parts):
                        obj_str = part
                        if j > 0:
                            obj_str = '{' + obj_str
                        if j < len(parts) - 1:
                            obj_str = obj_str + '}'
                        try:
                            episode = json.loads(obj_str)
                            self._process_episode(episode)
                        except json.JSONDecodeError:
                            pass
        print(f"Created {len(self.samples)} GRPO prompt samples.", flush=True)

    def _process_episode(self, episode: dict):
        pe = episode["parsed_episode"]
        turns = pe.get("turns", [])
        turn_rewards = episode.get("turn_rewards", [])
        if not turns or not turn_rewards:
            return

        scenario = pe.get("scenario", "")
        agent_1_name = pe.get("agent_1_name", "Agent 1")
        agent_2_name = pe.get("agent_2_name", "Agent 2")

        agents_info = {
            agent_1_name: {
                "background": pe.get("agent_1_background", ""),
                "goal": pe.get("agent_1_goal", ""),
                "secret": pe.get("agent_1_secret", ""),
            },
            agent_2_name: {
                "background": pe.get("agent_2_background", ""),
                "goal": pe.get("agent_2_goal", ""),
                "secret": pe.get("agent_2_secret", ""),
            },
        }

        for t_idx, tr in enumerate(turn_rewards):
            turn_num = tr["turn"]
            speaker = tr["agent"]
            if speaker not in agents_info:
                continue
            info = agents_info[speaker]
            if turn_num >= len(turns):
                continue
            actual_utterance = turns[turn_num].get("content", "")
            if not actual_utterance.strip():
                continue

            history_lines = []
            for prev_t in turns[:turn_num]:
                spk = prev_t.get("agent", "Unknown")
                act = prev_t.get("action", "said")
                content = prev_t.get("content", "")
                history_lines.append(f"Turn {prev_t['turn']+1} | {spk} {act}: {content}")

            history_text = "\n".join(history_lines)
            secret_text = info["secret"] if info["secret"] else "None"

            prompt = (
                f"Imagine you are {speaker}, your task is to act/speak as {speaker} would, "
                f"keeping in mind {speaker}'s social goal.\n"
                f"Here is the context of the interaction:\n"
                f"Scenario: {scenario}\n"
                f"Background: {info['background']}\n"
                f"Goal: {info['goal']}\n"
                f"Secret: {secret_text}\n"
                f"Dialogue History:\n{history_text}\n"
                f"You are at Turn #{turn_num+1}.\n"
                f"Generate the next natural response for {speaker}. "
                f"Stay in character and work towards your goal.\n"
                f"{speaker}:"
            )

            reward_context = (
                f"Scenario: {scenario}\n"
                f"Background: {info['background']}\n"
                f"Goal: {info['goal']}\n"
                f"Secret: {secret_text}\n"
                f"Dialogue History:\n{history_text}\n"
                f"Turn {turn_num+1} | {speaker}:"
            )

            self.samples.append({
                "prompt": prompt,
                "reward_context": reward_context,
                "reference": actual_utterance,
                "speaker": speaker,
                "turn_num": turn_num,
                "episode_id": episode.get("episode_id", ""),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# SFT Dataset & Warmup (identical to v2)
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, grpo_dataset, tokenizer, max_len: int = 1280):
        self.samples = grpo_dataset.samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        reference = sample["reference"]

        prompt_enc = self.tokenizer(prompt, add_special_tokens=True)
        prompt_len = len(prompt_enc["input_ids"])

        full_text = prompt + " " + reference
        full_enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )

        input_ids = full_enc.input_ids.squeeze(0)
        attention_mask = full_enc.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[:min(prompt_len, self.max_len)] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def run_sft_warmup(model, dataset, tokenizer, device, num_epochs=1, lr=2e-5,
                   batch_size=4, grad_accum=4):
    print("\n=== SFT Warm-up Phase ===", flush=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_steps = len(dataloader) * num_epochs // grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum

            if (batch_idx + 1) % 20 == 0:
                print(f"  SFT Epoch {epoch+1} Step {batch_idx+1}: loss={total_loss/(batch_idx+1):.4f}", flush=True)

        print(f"  SFT Epoch {epoch+1} done: avg_loss={total_loss/len(dataloader):.4f}", flush=True)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Trainer (identical to v2, but without trajectory bonus)
# ──────────────────────────────────────────────────────────────────────────────
class GRPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, tokenizer,
                 policy_device="cuda:0", ref_device="cuda:0",
                 group_size=8, max_gen_len=256, clip_eps=0.2,
                 kl_coeff=0.04, temperature=0.8, top_p=0.95,
                 num_ppo_epochs=1):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = policy_device
        self.ref_device = ref_device
        self.group_size = group_size
        self.max_gen_len = max_gen_len
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.temperature = temperature
        self.top_p = top_p
        self.num_ppo_epochs = num_ppo_epochs

    @torch.no_grad()
    def generate_candidates(self, prompts):
        self.policy.eval()
        all_prompts_expanded = []
        all_completions = []
        all_prompt_ids = []
        all_completion_ids = []

        for p_idx, prompt in enumerate(prompts):
            print(f"      gen prompt {p_idx+1}/{len(prompts)}...", flush=True)
            prompt_enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            prompt_ids = prompt_enc.input_ids

            outputs = self.policy.generate(
                **prompt_enc,
                max_new_tokens=self.max_gen_len,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.group_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for seq in outputs:
                completion_ids = seq[prompt_ids.shape[1]:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                for stop in ["\n", "Turn ", "\nTurn"]:
                    if stop in completion_text:
                        completion_text = completion_text[:completion_text.index(stop)]
                        break

                completion_text = completion_text.strip()
                if not completion_text:
                    completion_text = "..."

                all_prompts_expanded.append(prompt)
                all_completions.append(completion_text)
                all_prompt_ids.append(prompt_ids.squeeze(0))
                all_completion_ids.append(
                    self.tokenizer(completion_text, add_special_tokens=False,
                                   return_tensors="pt").input_ids.squeeze(0)
                )

        self.policy.train()
        return all_prompts_expanded, all_completions, all_prompt_ids, all_completion_ids

    def compute_log_probs(self, model, prompt_ids_list, completion_ids_list,
                          target_device=None):
        dev = target_device or self.device
        log_probs_list = []

        for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list):
            completion_ids = completion_ids.to(prompt_ids.device)
            full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(dev)
            attention_mask = torch.ones_like(full_ids)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=full_ids, attention_mask=attention_mask)
                logits = outputs.logits[0]

            prompt_len = prompt_ids.shape[0]
            completion_logits = logits[prompt_len - 1: prompt_len - 1 + completion_ids.shape[0]]
            completion_log_probs = F.log_softmax(completion_logits, dim=-1)
            token_log_probs = completion_log_probs.gather(
                1, completion_ids.unsqueeze(0).to(dev).T
            ).squeeze(-1)

            log_probs_list.append(token_log_probs.to(self.device))

        return log_probs_list

    def grpo_step(self, prompts, reward_contexts, optimizer, scheduler=None):
        t0 = time.time()

        # 1. Generate
        print(f"    [1/5] Generating {self.group_size}x{len(prompts)} candidates...", flush=True)
        expanded_prompts, completions, prompt_ids_list, completion_ids_list = \
            self.generate_candidates(prompts)
        t1 = time.time()
        print(f"    [1/5] Done ({t1-t0:.0f}s)", flush=True)

        # 2. Score with simple reward model
        print(f"    [2/5] Scoring {len(completions)} candidates...", flush=True)
        expanded_reward_contexts = []
        for ctx in reward_contexts:
            expanded_reward_contexts.extend([ctx] * self.group_size)
        rewards = self.reward_model.score(expanded_reward_contexts, completions)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        t2 = time.time()
        print(f"    [2/5] Done ({t2-t1:.0f}s)", flush=True)

        # 3. Advantages
        num_prompts = len(prompts)
        advantages = torch.zeros_like(rewards_tensor)
        for i in range(num_prompts):
            start = i * self.group_size
            end = start + self.group_size
            group_rewards = rewards_tensor[start:end]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std() + 1e-8
            advantages[start:end] = (group_rewards - mean_r) / std_r

        # 4. Old + ref log probs
        print(f"    [3/5] Old log probs ({len(prompt_ids_list)} samples)...", flush=True)
        with torch.no_grad():
            old_log_probs_list = self.compute_log_probs(
                self.policy, prompt_ids_list, completion_ids_list
            )
            ref_log_probs_list = None
            if self.ref_model is not None:
                print(f"    [4/5] Ref log probs...", flush=True)
                ref_log_probs_list = self.compute_log_probs(
                    self.ref_model, prompt_ids_list, completion_ids_list,
                    target_device=self.ref_device,
                )
        t3 = time.time()
        print(f"    [4/5] Done ({t3-t2:.0f}s)", flush=True)

        # 5. Policy gradient
        print(f"    [5/5] Gradient update...", flush=True)
        total_policy_loss = 0.0
        total_kl = 0.0

        for _ in range(self.num_ppo_epochs):
            all_loss = torch.tensor(0.0, device=self.device)
            all_kl = torch.tensor(0.0, device=self.device)

            for j in range(len(prompt_ids_list)):
                full_ids = torch.cat([
                    prompt_ids_list[j],
                    completion_ids_list[j].to(prompt_ids_list[j].device)
                ], dim=0).unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(full_ids)

                with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.policy(input_ids=full_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0]

                prompt_len = prompt_ids_list[j].shape[0]
                comp_logits = logits[prompt_len - 1: prompt_len - 1 + completion_ids_list[j].shape[0]]
                comp_log_probs = F.log_softmax(comp_logits, dim=-1)
                token_log_probs = comp_log_probs.gather(
                    1, completion_ids_list[j].unsqueeze(0).to(self.device).T
                ).squeeze(-1)

                old_token_log_probs = old_log_probs_list[j].detach()
                ratio = torch.exp(token_log_probs - old_token_log_probs)

                adv = advantages[j]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                kl = torch.tensor(0.0, device=self.device)
                if ref_log_probs_list is not None and self.kl_coeff > 0:
                    ref_lp = ref_log_probs_list[j].detach()
                    log_ratio = ref_lp - token_log_probs
                    kl = (torch.exp(log_ratio) - log_ratio - 1).mean()

                sample_loss = policy_loss + self.kl_coeff * kl
                all_loss = all_loss + sample_loss
                all_kl = all_kl + kl.detach()

            all_loss = all_loss / len(prompt_ids_list)
            all_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            total_policy_loss += all_loss.item()
            total_kl += all_kl.item() / len(prompt_ids_list)

        t4 = time.time()
        print(f"    [5/5] Done ({t4-t3:.0f}s) | Total: {t4-t0:.0f}s", flush=True)

        return {
            "policy_loss": total_policy_loss / self.num_ppo_epochs,
            "mean_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std().item(),
            "mean_kl": total_kl / self.num_ppo_epochs,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Ablation: GRPO without Stage 1 Mental Model"
    )
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str,
                        default="projects/sotopia/data/sotopia_turn_rewards.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="projects/sotopia/checkpoints/grpo_ablation_no_mental")

    # GRPO config (same as v2)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--grpo_epochs", type=int, default=3)
    parser.add_argument("--prompts_per_step", type=int, default=4)
    parser.add_argument("--num_ppo_epochs", type=int, default=1)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_ctx_len", type=int, default=1024)

    # SFT warmup (same as v2)
    parser.add_argument("--sft_warmup", action="store_true", default=True)
    parser.add_argument("--sft_checkpoint", type=str, default=None)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--sft_lr", type=float, default=2e-5)
    parser.add_argument("--sft_batch_size", type=int, default=4)

    # LoRA config (same as v2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)

    # Simple reward model training config
    parser.add_argument("--reward_train_epochs", type=int, default=5,
                        help="Epochs to train the simple reward model")
    parser.add_argument("--reward_lr", type=float, default=1e-4,
                        help="Learning rate for simple reward model head")
    parser.add_argument("--reward_batch_size", type=int, default=8,
                        help="Batch size for reward model training")

    parser.add_argument("--resume_from_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f">> Args parsed. GPU={args.gpu}", flush=True)

    num_gpus = torch.cuda.device_count()
    print(f">> Visible GPUs: {num_gpus}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args).copy()
    config["ablation"] = "no_mental_model"
    config["reward_type"] = "simple_mlp_head"
    config["reward_dim"] = REWARD_DIM
    config["reward_dimensions"] = SOTOPIA_DIMENSIONS
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2)

    policy_device = torch.device("cuda:0")
    frozen_device = torch.device(f"cuda:{num_gpus - 1}" if num_gpus > 1 else "cuda:0")
    print(f">> Devices: policy={policy_device}, frozen={frozen_device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Train Simple Reward Model (replaces Stage 1) ──
    print(">> Loading base model for simple reward model...", flush=True)
    reward_base_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_name, torch_dtype=torch.bfloat16,
    ).to(frozen_device)

    simple_reward = SimpleRewardModel(reward_base_model, reward_dim=REWARD_DIM).to(frozen_device)

    # Check if reward head already trained
    reward_save_dir = os.path.join(args.output_dir, "simple_reward_head")
    reward_head_path = os.path.join(reward_save_dir, "reward_head.pth")
    if os.path.exists(reward_head_path):
        print(f">> Loading existing simple reward head from {reward_head_path}", flush=True)
        simple_reward.reward_head.load_state_dict(torch.load(reward_head_path, map_location=frozen_device))
    else:
        reward_dataset = RewardPreferenceDataset(
            args.data_path, tokenizer, max_len=args.max_ctx_len + 256
        )
        simple_reward = train_simple_reward_model(
            simple_reward, reward_dataset, frozen_device,
            num_epochs=args.reward_train_epochs,
            lr=args.reward_lr,
            batch_size=args.reward_batch_size,
        )

        # Save reward model head
        os.makedirs(reward_save_dir, exist_ok=True)
        torch.save(simple_reward.reward_head.state_dict(), reward_head_path)
        print(f">> Simple reward head saved to {reward_save_dir}", flush=True)

    reward_model = FrozenSimpleRewardModel(simple_reward, tokenizer, str(frozen_device))

    # ── Reference Model (shares base weights with reward model to save memory) ──
    # Both are frozen copies of the same base model — no need to load twice.
    print(">> Reusing reward base model as reference model (shared weights)...", flush=True)
    ref_model = simple_reward.base_model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ── Load Policy Model ──
    print(">> Loading policy model...", flush=True)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_name, torch_dtype=torch.bfloat16,
    ).to(policy_device)
    policy_model.gradient_checkpointing_enable()
    policy_model.enable_input_require_grads()
    print(">> Policy model loaded.", flush=True)

    # Auto-detect existing SFT checkpoint in output_dir
    auto_sft_dir = os.path.join(args.output_dir, "sft_warmup")
    if not args.sft_checkpoint and args.resume_from_step == 0 and \
       os.path.exists(os.path.join(auto_sft_dir, "adapter_config.json")):
        args.sft_checkpoint = auto_sft_dir
        print(f">> Auto-detected existing SFT checkpoint: {auto_sft_dir}", flush=True)

    if args.resume_from_step > 0:
        resume_dir = os.path.join(args.output_dir, f"step_{args.resume_from_step}")
        print(f">> Resuming from {resume_dir}", flush=True)
        policy_model = PeftModel.from_pretrained(
            policy_model, resume_dir, torch_dtype=torch.bfloat16, is_trainable=True,
        )
        print(f">> LoRA loaded from step {args.resume_from_step}.", flush=True)
    elif args.sft_checkpoint:
        print(f">> Loading SFT checkpoint: {args.sft_checkpoint}", flush=True)
        policy_model = PeftModel.from_pretrained(
            policy_model, args.sft_checkpoint, torch_dtype=torch.bfloat16, is_trainable=True,
        )
        print(">> SFT LoRA loaded.", flush=True)
    else:
        num_layers = policy_model.config.num_hidden_layers
        top_layers = list(range(num_layers - args.num_lora_layers, num_layers))
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout, layers_to_transform=top_layers, layers_pattern="layers",
        )
        policy_model = get_peft_model(policy_model, lora_config)

    for name, param in policy_model.named_parameters():
        param.requires_grad = "lora_" in name
    policy_model.print_trainable_parameters()
    policy_model.config.use_cache = False

    print(">> Reference model ready (shared with reward base model).", flush=True)

    # ── Dataset ──
    grpo_dataset = GRPOPromptDataset(args.data_path, tokenizer, max_ctx_len=args.max_ctx_len)

    # ── SFT Warmup ──
    if args.resume_from_step > 0:
        print(f">> Skipping SFT (resuming from step {args.resume_from_step})", flush=True)
    elif not args.sft_checkpoint and args.sft_warmup:
        sft_dataset = SFTDataset(grpo_dataset, tokenizer, max_len=args.max_ctx_len + args.max_gen_len)
        policy_model = run_sft_warmup(
            policy_model, sft_dataset, tokenizer, policy_device,
            num_epochs=args.sft_epochs, lr=args.sft_lr, batch_size=args.sft_batch_size,
        )
        sft_dir = os.path.join(args.output_dir, "sft_warmup")
        os.makedirs(sft_dir, exist_ok=True)
        policy_model.save_pretrained(sft_dir)
        print(f">> SFT checkpoint saved to {sft_dir}", flush=True)

    # ── GRPO Training ──
    print("\n========================================", flush=True)
    print("  GRPO Training (ABLATION: simple reward, no mental model)", flush=True)
    print("========================================", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    grpo_trainer = GRPOTrainer(
        policy_model=policy_model, ref_model=ref_model, reward_model=reward_model,
        tokenizer=tokenizer, policy_device=str(policy_device), ref_device=str(frozen_device),
        group_size=args.group_size, max_gen_len=args.max_gen_len,
        clip_eps=args.clip_eps, kl_coeff=args.kl_coeff,
        temperature=args.temperature, top_p=args.top_p,
        num_ppo_epochs=args.num_ppo_epochs,
    )

    all_samples = [(s["prompt"], s["reward_context"]) for s in grpo_dataset.samples]
    steps_per_epoch = len(all_samples) // args.prompts_per_step

    global_step = 0
    resume_step = args.resume_from_step
    best_reward = -float("inf")

    print(f">> {len(all_samples)} samples, ~{steps_per_epoch} steps/epoch, {args.grpo_epochs} epochs", flush=True)
    if resume_step > 0:
        print(f">> Skipping first {resume_step} steps", flush=True)

    for epoch in range(args.grpo_epochs):
        print(f"\n--- GRPO Epoch {epoch+1}/{args.grpo_epochs} ---", flush=True)
        random.shuffle(all_samples)
        epoch_rewards = []

        for i in range(0, len(all_samples), args.prompts_per_step):
            batch = all_samples[i:i + args.prompts_per_step]
            if not batch:
                continue

            global_step += 1

            if global_step <= resume_step:
                if global_step % 50 == 0:
                    print(f"  Skipping step {global_step}/{resume_step}...", flush=True)
                continue

            batch_prompts = [b[0] for b in batch]
            batch_reward_contexts = [b[1] for b in batch]

            print(f"\n  === Step {global_step} (epoch {epoch+1}) ===", flush=True)
            metrics = grpo_trainer.grpo_step(batch_prompts, batch_reward_contexts, optimizer)
            epoch_rewards.append(metrics["mean_reward"])

            print(
                f"  Step {global_step}: "
                f"loss={metrics['policy_loss']:.4f} "
                f"reward={metrics['mean_reward']:.4f} "
                f"std={metrics['std_reward']:.4f} "
                f"kl={metrics['mean_kl']:.4f}",
                flush=True
            )

            if global_step % args.save_every == 0:
                step_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(step_dir, exist_ok=True)
                policy_model.save_pretrained(step_dir)
                print(f"  >> Checkpoint saved at step {global_step}", flush=True)

        mean_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        print(f"\nEpoch {epoch+1} done: mean_reward={mean_epoch_reward:.4f}", flush=True)

        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        policy_model.save_pretrained(epoch_dir)

        if mean_epoch_reward > best_reward:
            best_reward = mean_epoch_reward
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            policy_model.save_pretrained(best_dir)
            print(f"  >> Best model saved (reward={best_reward:.4f})", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n>> Training complete. Best reward: {best_reward:.4f}", flush=True)
    print(f">> Checkpoints: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
