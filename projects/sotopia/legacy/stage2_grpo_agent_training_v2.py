#!/usr/bin/env python3
"""
Stage 2 v2: GRPO Agent Policy Training Guided by Frozen Mental+Reward Model (v2)
==================================================================================
Fully self-contained — no imports from stage1 or stage2 v1.

Uses the v2 reward model (3 dimensions: goal, relationship, knowledge).
"""

# CUDA_VISIBLE_DEVICES=0 python projects/sotopia/legacy/stage2_grpo_agent_training_v2.py \
#   --policy_model_name Qwen/Qwen2.5-7B-Instruct \
#   --reward_model_name Qwen/Qwen2.5-7B-Instruct \
#   --reward_checkpoint_dir projects/sotopia/checkpoints/coupled_mental_reward_v3/best \
#   --data_path projects/sotopia/data/sotopia_turn_rewards.jsonl \
#   --output_dir projects/sotopia/checkpoints/grpo_agent_v3 \
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
#   --z_dim 128 \
#   --trajectory_weight 0.3 \
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

print(">> stage2_grpo_agent_training_v2.py starting...", flush=True)
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
# Constants (from stage1 v2)
# ──────────────────────────────────────────────────────────────────────────────
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
    "believability":                (0, 10),
    "relationship":                 (-5, 5),
    "knowledge":                    (0, 10),
    "secret":                       (-10, 0),
    "social_rules":                 (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal":                         (0, 10),
}

REWARD_DIM = len(SOTOPIA_DIMENSIONS)  # 7

CUSTOM_HEAD_NAMES = [
    "context_mu", "context_logvar",
    "z_to_hidden",
    "joint_outcome_head", "z_only_reward_head",
    "expl_cross_attn", "expl_reward_head",
    "mental_cross_attn", "mental_decode_ffn",
    "mental_decode_ln", "mental_decode_ln2",
    "z_belief_to_prefix", "z_intent_to_prefix", "z_thought_to_prefix",
]


def normalize_score(dim: str, score: float) -> float:
    lo, hi = DIM_RANGES[dim]
    return (score - lo) / (hi - lo + 1e-8)


def get_transformer_from_peft(peft_causal_lm):
    base = peft_causal_lm
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model
    if hasattr(base, "model"):
        base = base.model
    return base


# ──────────────────────────────────────────────────────────────────────────────
# CoupledMentalRewardModel (minimal version for inference only)
# ──────────────────────────────────────────────────────────────────────────────
class CoupledMentalRewardModel(nn.Module):
    """Minimal reproduction of Stage 1 v2 architecture for loading checkpoints."""

    def __init__(self, base_model: nn.Module, reward_dim: int = REWARD_DIM, z_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.z_dim = z_dim
        self.reward_dim = reward_dim

        if hasattr(base_model, "config"):
            hidden_size = base_model.config.hidden_size
        else:
            hidden_size = base_model.get_input_embeddings().embedding_dim

        # Context -> z
        self.context_mu = nn.Linear(hidden_size, z_dim)
        self.context_logvar = nn.Linear(hidden_size, z_dim)

        # z -> hidden for response conditioning
        self.z_to_hidden = nn.Linear(z_dim, hidden_size)

        # Joint outcome head: [z || response_hidden] -> reward_dim
        self.joint_outcome_head = nn.Sequential(
            nn.Linear(z_dim + hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )

        # z-only reward head
        self.z_only_reward_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, reward_dim),
        )

        # Explainable reward (cross-attention + head)
        self.expl_cross_attn = nn.MultiheadAttention(
            embed_dim=z_dim, num_heads=8, kdim=hidden_size, vdim=hidden_size,
            batch_first=True, dropout=0.1,
        )
        self.expl_reward_head = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, reward_dim),
        )

        # Mental decoder components (needed for checkpoint loading)
        self.mental_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.1
        )
        self.mental_decode_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.mental_decode_ln = nn.LayerNorm(hidden_size)
        self.mental_decode_ln2 = nn.LayerNorm(hidden_size)

        # z is split: belief=48, intent=40, thought=40 (sums to z_dim=128)
        Z_BELIEF_DIM, Z_INTENT_DIM, Z_THOUGHT_DIM = 48, 40, 40
        NUM_PREFIX_TOKENS = 8
        self.z_belief_to_prefix = nn.Linear(Z_BELIEF_DIM, NUM_PREFIX_TOKENS * hidden_size)
        self.z_intent_to_prefix = nn.Linear(Z_INTENT_DIM, NUM_PREFIX_TOKENS * hidden_size)
        self.z_thought_to_prefix = nn.Linear(Z_THOUGHT_DIM, NUM_PREFIX_TOKENS * hidden_size)

    def _get_transformer(self):
        return get_transformer_from_peft(self.base_model)

    def _encode_sequence(self, input_ids, attention_mask):
        transformer = self._get_transformer()
        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def encode_context_to_z(self, ctx_ids, ctx_mask, deterministic=True):
        hidden = self._encode_sequence(ctx_ids, ctx_mask)
        pooled = self._pool(hidden, ctx_mask)
        mu = self.context_mu(pooled)
        if deterministic:
            return mu
        logvar = self.context_logvar(pooled)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward_reward_with_z(self, z, resp_ids, resp_mask):
        resp_hidden = self._encode_sequence(resp_ids, resp_mask)
        resp_pooled = self._pool(resp_hidden, resp_mask)
        joint_input = torch.cat([z, resp_pooled], dim=-1)
        return self.joint_outcome_head(joint_input)


# ──────────────────────────────────────────────────────────────────────────────
# Frozen Reward Model Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class FrozenRewardModel:
    """Loads and freezes the Stage 1 v2 mental+reward model for scoring."""

    def __init__(self, base_model_name: str, checkpoint_dir: str,
                 z_dim: int = 128, device: str = "cuda",
                 scoring_dim_indices: list = None):
        self.device = device
        # Which dim indices to average for scalar reward (None = all)
        self.scoring_dim_indices = scoring_dim_indices
        print(f"  [Reward] Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  [Reward] Loading base model...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16,
        )

        lora_path = os.path.join(checkpoint_dir, "lora_adapter")
        if os.path.exists(lora_path):
            print(f"  [Reward] Loading LoRA adapter...", flush=True)
            base_model = PeftModel.from_pretrained(
                base_model, lora_path, torch_dtype=torch.bfloat16,
            )
            base_model = base_model.merge_and_unload()
            print(f"  [Reward] LoRA merged.", flush=True)

        print(f"  [Reward] Moving to {device}...", flush=True)
        base_model = base_model.to(device)

        self.model = CoupledMentalRewardModel(base_model, reward_dim=REWARD_DIM, z_dim=z_dim)

        for head_name in CUSTOM_HEAD_NAMES:
            path = os.path.join(checkpoint_dir, f"{head_name}.pth")
            if os.path.exists(path):
                getattr(self.model, head_name).load_state_dict(
                    torch.load(path, map_location=device, weights_only=True)
                )

        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"  [Reward] Loaded: {REWARD_DIM} dims ({SOTOPIA_DIMENSIONS})", flush=True)

    @torch.no_grad()
    def score(self, prompts: List[str], completions: List[str],
              max_ctx_len: int = 1024, max_resp_len: int = 256) -> List[float]:
        unique_prompts = list(dict.fromkeys(prompts))
        prompt_to_z = {}

        for i in range(0, len(unique_prompts), 8):
            batch_unique = unique_prompts[i:i+8]
            ctx_enc = self.tokenizer(
                batch_unique, truncation=True, max_length=max_ctx_len,
                padding=True, return_tensors="pt"
            )
            ctx_ids = ctx_enc.input_ids.to(self.device)
            ctx_mask = ctx_enc.attention_mask.to(self.device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                z_batch = self.model.encode_context_to_z(ctx_ids, ctx_mask, deterministic=True)

            for j, p in enumerate(batch_unique):
                prompt_to_z[p] = z_batch[j]

        rewards = []
        for i in range(0, len(prompts), 8):
            batch_prompts = prompts[i:i+8]
            batch_completions = completions[i:i+8]

            z = torch.stack([prompt_to_z[p] for p in batch_prompts], dim=0)
            resp_enc = self.tokenizer(
                batch_completions, truncation=True, max_length=max_resp_len,
                padding=True, return_tensors="pt"
            )
            resp_ids = resp_enc.input_ids.to(self.device)
            resp_mask = resp_enc.attention_mask.to(self.device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                joint_reward = self.model.forward_reward_with_z(z, resp_ids, resp_mask)

            if self.scoring_dim_indices is not None:
                scalar = joint_reward.float()[:, self.scoring_dim_indices].mean(dim=1)
            else:
                scalar = joint_reward.float().mean(dim=1)
            rewards.extend(scalar.cpu().tolist())

        return rewards


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Dataset
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

        # Extract episode-level trajectory rewards from original_rewards
        # original_rewards is a list of [overall_score, {dim: score, ...}] per agent
        original_rewards = episode.get("original_rewards", [])
        trajectory_reward_by_agent = {}
        agent_names_ordered = [agent_1_name, agent_2_name]
        for ag_idx, ag_name in enumerate(agent_names_ordered):
            if ag_idx < len(original_rewards):
                or_entry = original_rewards[ag_idx]
                if isinstance(or_entry, list) and len(or_entry) >= 2:
                    dim_scores = or_entry[1]
                    # Normalize and average across all available dimensions
                    norm_scores = []
                    for dim_name, score in dim_scores.items():
                        if dim_name == "overall_score":
                            continue
                        if dim_name in DIM_RANGES:
                            norm_scores.append(normalize_score(dim_name, score))
                        else:
                            # For dims not in DIM_RANGES, use raw/10 as rough norm
                            norm_scores.append(score / 10.0)
                    trajectory_reward_by_agent[ag_name] = (
                        sum(norm_scores) / len(norm_scores) if norm_scores else 0.0
                    )

        # Determine last turn index per agent for trajectory bonus
        last_turn_by_agent = {}
        for tr in turn_rewards:
            speaker = tr["agent"]
            last_turn_by_agent[speaker] = tr["turn"]

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

            is_last_turn = (turn_num == last_turn_by_agent.get(speaker, -1))

            self.samples.append({
                "prompt": prompt,
                "reward_context": reward_context,
                "reference": actual_utterance,
                "speaker": speaker,
                "turn_num": turn_num,
                "episode_id": episode.get("episode_id", ""),
                "is_last_turn": is_last_turn,
                "trajectory_reward": trajectory_reward_by_agent.get(speaker, 0.0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# SFT Dataset & Warmup
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
# GRPO Trainer (self-contained with verbose logging)
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

    def grpo_step(self, prompts, reward_contexts, optimizer, scheduler=None,
                  trajectory_rewards=None, trajectory_weight=0.0):
        t0 = time.time()

        # 1. Generate
        print(f"    [1/5] Generating {self.group_size}x{len(prompts)} candidates...", flush=True)
        expanded_prompts, completions, prompt_ids_list, completion_ids_list = \
            self.generate_candidates(prompts)
        t1 = time.time()
        print(f"    [1/5] Done ({t1-t0:.0f}s)", flush=True)

        # 2. Score
        print(f"    [2/5] Scoring {len(completions)} candidates...", flush=True)
        expanded_reward_contexts = []
        for ctx in reward_contexts:
            expanded_reward_contexts.extend([ctx] * self.group_size)
        rewards = self.reward_model.score(expanded_reward_contexts, completions)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Add trajectory bonus for last-turn samples (Option A)
        if trajectory_rewards is not None and trajectory_weight > 0:
            for i, traj_r in enumerate(trajectory_rewards):
                if traj_r is not None:  # None means not a last turn
                    start = i * self.group_size
                    end = start + self.group_size
                    rewards_tensor[start:end] += trajectory_weight * traj_r

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
        description="Stage 2 v2: GRPO Agent Training with Frozen Reward Model (3-dim)"
    )
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reward_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reward_checkpoint_dir", type=str,
                        default="projects/sotopia/checkpoints/coupled_mental_reward_v2/best")
    parser.add_argument("--data_path", type=str,
                        default="projects/sotopia/data/sotopia_turn_rewards.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="projects/sotopia/checkpoints/grpo_agent_v2")

    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--grpo_epochs", type=int, default=3)
    parser.add_argument("--prompts_per_step", type=int, default=4)
    parser.add_argument("--num_ppo_epochs", type=int, default=1)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_ctx_len", type=int, default=1024)

    parser.add_argument("--sft_warmup", action="store_true", default=True)
    parser.add_argument("--sft_checkpoint", type=str, default=None)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--sft_lr", type=float, default=2e-5)
    parser.add_argument("--sft_batch_size", type=int, default=4)
    parser.add_argument("--extra_sft_data", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--trajectory_weight", type=float, default=0.0,
                        help="Weight for episode-level trajectory reward bonus on last turn (0=off, recommended 0.3-0.5)")
    parser.add_argument("--resume_from_step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--reward_scoring_dims", type=str, default=None,
                        help="Comma-separated subset of dims to use for GRPO reward scoring. "
                             "E.g. 'goal,relationship,knowledge'. Defaults to all dims.")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f">> Args parsed. GPU={args.gpu}, resume_from_step={args.resume_from_step}", flush=True)

    num_gpus = torch.cuda.device_count()
    print(f">> Visible GPUs: {num_gpus}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse reward_scoring_dims
    if args.reward_scoring_dims:
        scoring_dims = [d.strip() for d in args.reward_scoring_dims.split(",")]
        invalid = [d for d in scoring_dims if d not in SOTOPIA_DIMENSIONS]
        if invalid:
            raise ValueError(f"Invalid reward_scoring_dims: {invalid}. Valid: {SOTOPIA_DIMENSIONS}")
        scoring_dim_indices = [SOTOPIA_DIMENSIONS.index(d) for d in scoring_dims]
        print(f">> Reward scoring dims: {scoring_dims} (indices {scoring_dim_indices})", flush=True)
    else:
        scoring_dims = SOTOPIA_DIMENSIONS
        scoring_dim_indices = list(range(REWARD_DIM))
        print(f">> Reward scoring dims: all {REWARD_DIM} dims", flush=True)

    config = vars(args).copy()
    config["reward_dim"] = REWARD_DIM
    config["reward_dimensions"] = SOTOPIA_DIMENSIONS
    config["reward_scoring_dims"] = scoring_dims
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2)

    policy_device = torch.device("cuda:0")
    frozen_device = torch.device(f"cuda:{num_gpus - 1}" if num_gpus > 1 else "cuda:0")
    print(f">> Devices: policy={policy_device}, frozen={frozen_device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reward model
    print(">> Loading frozen reward model...", flush=True)
    reward_model = FrozenRewardModel(
        args.reward_model_name, args.reward_checkpoint_dir,
        z_dim=args.z_dim, device=str(frozen_device),
        scoring_dim_indices=scoring_dim_indices,
    )

    # Load policy model
    print(">> Loading policy model...", flush=True)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_name, torch_dtype=torch.bfloat16,
    ).to(policy_device)
    policy_model.gradient_checkpointing_enable()
    policy_model.enable_input_require_grads()
    print(">> Policy model loaded.", flush=True)

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

    # Reference model
    print(">> Loading reference model...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_name, torch_dtype=torch.bfloat16,
    ).to(frozen_device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print(">> Reference model loaded.", flush=True)

    # Dataset
    grpo_dataset = GRPOPromptDataset(args.data_path, tokenizer, max_ctx_len=args.max_ctx_len)

    # SFT warmup
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

    # GRPO Training
    print("\n========================================", flush=True)
    print("  GRPO Training Phase (v2: 3-dim reward)", flush=True)
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

    all_samples = [
        (s["prompt"], s["reward_context"],
         s.get("trajectory_reward") if s.get("is_last_turn") else None)
        for s in grpo_dataset.samples
    ]
    steps_per_epoch = len(all_samples) // args.prompts_per_step

    global_step = 0
    resume_step = args.resume_from_step
    best_reward = -float("inf")

    n_last_turn = sum(1 for s in all_samples if s[2] is not None)
    print(f">> {len(all_samples)} samples, ~{steps_per_epoch} steps/epoch, {args.grpo_epochs} epochs", flush=True)
    print(f">> Trajectory bonus: weight={args.trajectory_weight}, last-turn samples={n_last_turn}/{len(all_samples)}", flush=True)
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
            batch_trajectory_rewards = [b[2] for b in batch]

            print(f"\n  === Step {global_step} (epoch {epoch+1}) ===", flush=True)
            metrics = grpo_trainer.grpo_step(
                batch_prompts, batch_reward_contexts, optimizer,
                trajectory_rewards=batch_trajectory_rewards,
                trajectory_weight=args.trajectory_weight,
            )
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
