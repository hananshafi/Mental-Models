#!/usr/bin/env python3
"""
Stage 2 policy training for ToM-SB with the learned structured reward model.

This script is standalone inside the ToM-SB folder. It does not import Sotopia.

Training loop:
  1. Sample ToM-SB prompts.
  2. Generate a group of defender responses from a LoRA policy.
  3. Score each response with the frozen structured ToM-SB reward checkpoint.
  4. Apply group-relative PPO/GRPO updates to the policy LoRA.
  5. Optionally add a small SFT anchor on the synthetic safe response.

The default command is intentionally conservative for a single 48GB GPU:
one prompt per update, four sampled responses, no reference model loaded.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_tom_sb_mental_reward_standalone import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    _dtype_from_arg,
    _load_reward_schema,
    _read_json_or_jsonl,
    _save_json,
    validate_records,
)
from train_tom_sb_mental_reward_structured import (  # noqa: E402
    StructuredRecursiveToMModel,
    StructuredToMSpec,
    _load_checkpoint_metadata,
    _resolve_checkpoint_dir,
    _validate_resume_metadata,
    load_custom_heads,
)


DEFAULT_TRAIN_PATH = "projects/tom_sb/data/tom_sb_belief_only_5k_train.jsonl"
DEFAULT_REWARD_CKPT = "projects/tom_sb/checkpoints/tom_sb_reward_structured_tom_v1_plus2/best"
DEFAULT_OUTPUT_DIR = "projects/tom_sb/checkpoints/tom_sb_policy_grpo_v1"
DEFAULT_SCALAR_WEIGHTS = {
    "naturalness": 0.0,
    "trust_maintenance": 0.15,
    "attacker_belief_tracking": 0.15,
    "privacy_preservation": 0.35,
    "format_and_non_refusal": 0.10,
    "decoy_plausibility": 0.05,
    "defender_goal_success": 0.20,
}
SYSTEM_PROMPT = (
    "You are a careful double-agent defender. You see private ground-truth "
    "information, but your response must protect any field the attacker has not "
    "already demonstrated they know. Maintain trust, avoid over-refusal, and "
    "give only a concise defender response."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_scalar_weights(value: str | None, reward_schema: list[str]) -> torch.Tensor:
    weights = dict(DEFAULT_SCALAR_WEIGHTS)
    if value:
        weights = {name: 0.0 for name in reward_schema}
        for part in value.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"Expected name=value in --scalar_weights, got {part!r}")
            name, raw_weight = part.split("=", 1)
            name = name.strip()
            if name not in reward_schema:
                raise ValueError(f"Unknown reward dimension {name!r}; schema={reward_schema}")
            weights[name] = float(raw_weight)
    missing = [name for name in reward_schema if name not in weights]
    if missing:
        raise ValueError(f"Missing scalar weights for reward dimensions: {missing}")
    tensor = torch.tensor([weights[name] for name in reward_schema], dtype=torch.float32)
    total = float(tensor.sum().item())
    if abs(total) < 1e-12:
        raise ValueError("Scalar reward weights sum to zero.")
    return tensor / total


def build_policy_prompt(record: dict[str, Any], tokenizer, *, use_chat_template: bool) -> str:
    context = str(record.get("context_text", "")).strip()
    instruction = (
        f"{context}\n\n"
        "Write only the defender's next response. Do not reveal missing private "
        "ground-truth fields. Do not include analysis or labels."
    )
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{instruction}\n\nDefender response:"


def clean_completion(text: str) -> str:
    text = text.strip()
    for prefix in ["Defender:", "defender:", "Assistant:", "assistant:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    stop_markers = ["\nUser:", "\nAttacker:", "\nDefender:", "\n##", "<|im_end|>"]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].strip()
    return text


def truth_leak(record: dict[str, Any], completion: str) -> bool:
    unknown = (record.get("metadata", {}) or {}).get("unknown_fields", {}) or {}
    lowered = completion.lower()
    for value in unknown.values():
        value_text = str(value).strip().lower()
        if value_text and value_text in lowered:
            return True
    return False


def trim_completion_ids(ids: torch.Tensor, *, eos_token_id: int | None, pad_token_id: int | None) -> torch.Tensor:
    kept: list[int] = []
    for token in ids.detach().cpu().tolist():
        if pad_token_id is not None and token == pad_token_id:
            break
        kept.append(int(token))
        if eos_token_id is not None and token == eos_token_id:
            break
    if kept:
        return torch.tensor(kept, dtype=torch.long)
    fallback = eos_token_id if eos_token_id is not None else (pad_token_id if pad_token_id is not None else 0)
    return torch.tensor([int(fallback)], dtype=torch.long)


def tokenize_policy_prompt(text: str, tokenizer, *, max_prompt_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,
        )
    finally:
        tokenizer.truncation_side = old_side
    return {key: value.to(device) for key, value in enc.items()}


def completion_log_probs(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(full_ids)
    with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=dtype):
        logits = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False).logits[0]
    prompt_len = int(prompt_ids.numel())
    comp_logits = logits[prompt_len - 1 : prompt_len - 1 + completion_ids.numel()]
    log_probs = F.log_softmax(comp_logits.float(), dim=-1)
    return log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)


def sft_anchor_loss(
    model: nn.Module,
    records: list[dict[str, Any]],
    tokenizer,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_prompt_len: int,
    max_response_len: int,
    use_chat_template: bool,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for record in records:
        prompt = build_policy_prompt(record, tokenizer, use_chat_template=use_chat_template)
        prompt_enc = tokenize_policy_prompt(prompt, tokenizer, max_prompt_len=max_prompt_len, device=device)
        response = str(record.get("pos_response", "")).strip()
        if tokenizer.eos_token:
            response = response + tokenizer.eos_token
        resp_enc = tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=max_response_len,
            add_special_tokens=False,
        )
        resp_ids = resp_enc.input_ids.squeeze(0).to(device)
        if resp_ids.numel() == 0:
            continue
        prompt_ids = prompt_enc["input_ids"].squeeze(0)
        full_ids = torch.cat([prompt_ids, resp_ids], dim=0).unsqueeze(0)
        labels = full_ids.clone()
        labels[:, : prompt_ids.numel()] = -100
        attention_mask = torch.ones_like(full_ids)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=dtype):
            out = model(input_ids=full_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        losses.append(out.loss.float())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def set_deterministic_latents(model: StructuredRecursiveToMModel) -> None:
    def sample_z_mean(mu_proj: nn.Linear, logvar_proj: nn.Linear, hidden: torch.Tensor):
        mu = mu_proj(hidden)
        logvar = logvar_proj(hidden).clamp(-10.0, 10.0)
        return mu, mu, logvar

    model._sample_z = sample_z_mean  # type: ignore[method-assign]


class LearnedToMSBRewardScorer:
    def __init__(
        self,
        *,
        model_name: str,
        checkpoint: str,
        reward_schema: list[str],
        scalar_weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        max_ctx_len: int,
        max_resp_len: int,
        activation: str,
        trust_remote_code: bool,
        attn_implementation: str | None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.reward_schema = reward_schema
        self.scalar_weights = scalar_weights.to(device)
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        self.activation = activation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        ckpt_dir = _resolve_checkpoint_dir(checkpoint)
        metadata = _load_checkpoint_metadata(ckpt_dir)
        spec_payload = metadata.get("structured_tom_spec")
        if not isinstance(spec_payload, dict):
            raise ValueError(f"Reward checkpoint is missing structured_tom_spec: {ckpt_dir}")
        spec = StructuredToMSpec(**spec_payload)
        _validate_resume_metadata(
            ckpt_dir=ckpt_dir,
            metadata=metadata,
            reward_schema=reward_schema,
            spec=spec,
            use_expl_reward=False,
            z_dim=int(metadata.get("z_dim", 128)),
        )

        model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        print(f"Loading frozen reward base: {model_name} on {device}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
        base.config.use_cache = False
        if getattr(base.config, "pad_token_id", None) is None:
            base.config.pad_token_id = self.tokenizer.pad_token_id
        base = PeftModel.from_pretrained(base, ckpt_dir / "lora_adapter", is_trainable=False)
        self.model = StructuredRecursiveToMModel(
            base,
            reward_dim=len(reward_schema),
            spec=spec,
            z_dim=int(metadata.get("z_dim", 128)),
            use_expl_reward=False,
        ).to(device)
        load_custom_heads(self.model, ckpt_dir, device)
        set_deterministic_latents(self.model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"Loaded frozen learned reward from: {ckpt_dir}", flush=True)

    def _score_pos(self, ctx_ids: torch.Tensor, ctx_mask: torch.Tensor, resp_ids: torch.Tensor, resp_mask: torch.Tensor) -> torch.Tensor:
        context_hidden, z1, _, _, z2, _, _ = self.model.encode_z1_z2(ctx_ids, ctx_mask, stop_grad_z1=False)
        resp_hidden = self.model.transformer(
            input_ids=resp_ids,
            attention_mask=resp_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        last_idx = (resp_mask.sum(dim=1) - 1).clamp(min=0)
        gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, resp_hidden.size(-1))
        pos_hidden = resp_hidden.gather(1, gather_idx).squeeze(1)
        return self.model.joint_outcome_head(torch.cat([z1, z2, pos_hidden], dim=1))

    @torch.no_grad()
    def score(self, records: list[dict[str, Any]], completions: list[str], *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores: list[torch.Tensor] = []
        reward_vecs: list[torch.Tensor] = []
        for start in range(0, len(completions), batch_size):
            chunk_records = records[start : start + batch_size]
            chunk_completions = [text.strip() if text.strip() else "." for text in completions[start : start + batch_size]]
            contexts = [str(record.get("context_text", "")) for record in chunk_records]
            ctx = self.tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_ctx_len,
            ).to(self.device)
            resp = self.tokenizer(
                chunk_completions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_resp_len,
            ).to(self.device)
            with torch.amp.autocast(enabled=self.device.type == "cuda", device_type="cuda", dtype=self.dtype):
                raw_reward = self._score_pos(ctx.input_ids, ctx.attention_mask, resp.input_ids, resp.attention_mask)
            raw_reward = raw_reward.float()
            reward_for_scalar = torch.sigmoid(raw_reward) if self.activation == "sigmoid" else raw_reward
            scalar = (reward_for_scalar * self.scalar_weights).sum(dim=1)
            scores.append(scalar.detach().cpu())
            reward_vecs.append(raw_reward.detach().cpu())
        return torch.cat(scores, dim=0), torch.cat(reward_vecs, dim=0)


class TomSBPolicyGRPOTrainer:
    def __init__(
        self,
        *,
        policy: nn.Module,
        ref_model: nn.Module | None,
        reward_scorer: LearnedToMSBRewardScorer,
        tokenizer,
        device: torch.device,
        ref_device: torch.device | None,
        dtype: torch.dtype,
        group_size: int,
        max_prompt_len: int,
        max_new_tokens: int,
        max_response_len: int,
        reward_batch_size: int,
        temperature: float,
        top_p: float,
        clip_eps: float,
        ref_kl_coeff: float,
        sft_weight: float,
        leak_penalty: float,
        empty_penalty: float,
        short_penalty: float,
        min_response_chars: int,
        max_grad_norm: float,
        use_chat_template: bool,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.reward_scorer = reward_scorer
        self.tokenizer = tokenizer
        self.device = device
        self.ref_device = ref_device
        self.dtype = dtype
        self.group_size = group_size
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.max_response_len = max_response_len
        self.reward_batch_size = reward_batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.clip_eps = clip_eps
        self.ref_kl_coeff = ref_kl_coeff
        self.sft_weight = sft_weight
        self.leak_penalty = leak_penalty
        self.empty_penalty = empty_penalty
        self.short_penalty = short_penalty
        self.min_response_chars = min_response_chars
        self.max_grad_norm = max_grad_norm
        self.use_chat_template = use_chat_template

    @torch.no_grad()
    def generate_candidates(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        self.policy.eval()
        all_records: list[dict[str, Any]] = []
        all_prompts: list[str] = []
        all_prompt_ids: list[torch.Tensor] = []
        all_completion_ids: list[torch.Tensor] = []
        all_completions: list[str] = []

        for record in records:
            prompt = build_policy_prompt(record, self.tokenizer, use_chat_template=self.use_chat_template)
            enc = tokenize_policy_prompt(prompt, self.tokenizer, max_prompt_len=self.max_prompt_len, device=self.device)
            prompt_ids = enc["input_ids"][0].detach().cpu()
            outputs = self.policy.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.group_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            for seq in outputs:
                continuation = seq[enc["input_ids"].shape[1] :]
                completion_ids = trim_completion_ids(
                    continuation,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                completion_text = clean_completion(completion_text)
                if not completion_text:
                    completion_text = "."
                all_records.append(record)
                all_prompts.append(prompt)
                all_prompt_ids.append(prompt_ids)
                all_completion_ids.append(completion_ids)
                all_completions.append(completion_text)

        self.policy.train()
        return {
            "records": all_records,
            "prompts": all_prompts,
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "completions": all_completions,
        }

    def _apply_reward_penalties(
        self,
        records: list[dict[str, Any]],
        completions: list[str],
        learned_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rewards = learned_rewards.clone()
        leak_flags: list[float] = []
        empty_flags: list[float] = []
        short_flags: list[float] = []
        for idx, (record, completion) in enumerate(zip(records, completions)):
            leaked = truth_leak(record, completion)
            empty = not completion.strip() or completion.strip() == "."
            short = len(completion.strip()) < self.min_response_chars
            leak_flags.append(float(leaked))
            empty_flags.append(float(empty))
            short_flags.append(float(short))
            if leaked:
                rewards[idx] -= self.leak_penalty
            if empty:
                rewards[idx] -= self.empty_penalty
            elif short:
                rewards[idx] -= self.short_penalty
        metrics = {
            "leak_rate": sum(leak_flags) / max(1, len(leak_flags)),
            "empty_rate": sum(empty_flags) / max(1, len(empty_flags)),
            "short_rate": sum(short_flags) / max(1, len(short_flags)),
        }
        return rewards, metrics

    def _advantages(self, rewards: torch.Tensor, num_prompts: int) -> tuple[torch.Tensor, float]:
        advantages = torch.zeros_like(rewards)
        zero_std_groups = 0
        for i in range(num_prompts):
            start = i * self.group_size
            end = start + self.group_size
            group = rewards[start:end]
            std = group.std(unbiased=False)
            if std.item() < 1e-6:
                zero_std_groups += 1
                continue
            advantages[start:end] = (group - group.mean()) / (std + 1e-6)
        return advantages.clamp(-3.0, 3.0), zero_std_groups / max(1, num_prompts)

    def grpo_step(self, records: list[dict[str, Any]], optimizer, scheduler=None) -> dict[str, Any]:
        t0 = time.time()
        generated = self.generate_candidates(records)
        t1 = time.time()
        learned_rewards, raw_reward_vecs = self.reward_scorer.score(
            generated["records"],
            generated["completions"],
            batch_size=self.reward_batch_size,
        )
        final_rewards, penalty_metrics = self._apply_reward_penalties(
            generated["records"], generated["completions"], learned_rewards
        )
        rewards = final_rewards.to(self.device)
        advantages, zero_std_group_rate = self._advantages(rewards, len(records))
        t2 = time.time()

        was_training = self.policy.training
        self.policy.eval()
        ref_log_probs = None
        if self.ref_model is not None and self.ref_kl_coeff > 0:
            with torch.no_grad():
                ref_log_probs = [
                    completion_log_probs(
                        self.ref_model,
                        prompt_ids,
                        completion_ids,
                        device=self.ref_device or self.device,
                        dtype=self.dtype,
                    ).detach().to(self.device)
                    for prompt_ids, completion_ids in zip(generated["prompt_ids"], generated["completion_ids"])
                ]
        t3 = time.time()

        optimizer.zero_grad(set_to_none=True)
        pg_loss = torch.tensor(0.0, device=self.device)
        kl_loss = torch.tensor(0.0, device=self.device)
        active_items = 0
        for idx, (prompt_ids, completion_ids) in enumerate(zip(generated["prompt_ids"], generated["completion_ids"])):
            if completion_ids.numel() == 0:
                continue
            token_log_probs = completion_log_probs(
                self.policy,
                prompt_ids,
                completion_ids,
                device=self.device,
                dtype=self.dtype,
            )
            usable = token_log_probs.numel()
            if usable == 0:
                continue
            old_lp = token_log_probs.detach()
            ratio = torch.exp(token_log_probs - old_lp)
            adv = advantages[idx].detach()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            pg_loss = pg_loss - torch.min(surr1, surr2).mean()
            if ref_log_probs is not None and self.ref_kl_coeff > 0:
                ref_lp = ref_log_probs[idx][:usable].to(self.device)
                log_ratio = ref_lp - token_log_probs
                kl_loss = kl_loss + (torch.exp(log_ratio) - log_ratio - 1.0).mean()
            active_items += 1

        if active_items == 0:
            raise RuntimeError("No active generated completions for GRPO update.")
        pg_loss = pg_loss / active_items
        kl_loss = kl_loss / active_items
        if was_training:
            self.policy.train()
        sft_loss = torch.tensor(0.0, device=self.device)
        if self.sft_weight > 0:
            sft_loss = sft_anchor_loss(
                self.policy,
                records,
                self.tokenizer,
                device=self.device,
                dtype=self.dtype,
                max_prompt_len=self.max_prompt_len,
                max_response_len=self.max_response_len,
                use_chat_template=self.use_chat_template,
            )
        total_loss = pg_loss + self.ref_kl_coeff * kl_loss + self.sft_weight * sft_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in self.policy.parameters() if param.requires_grad],
            self.max_grad_norm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        t4 = time.time()

        completion_lengths = [len(text) for text in generated["completions"]]
        best_idx = int(torch.argmax(final_rewards).item())
        worst_idx = int(torch.argmin(final_rewards).item())
        metrics = {
            "loss": float(total_loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "sft_loss": float(sft_loss.detach().item()),
            "learned_reward_mean": float(learned_rewards.mean().item()),
            "learned_reward_std": float(learned_rewards.std(unbiased=False).item()),
            "reward_mean": float(final_rewards.mean().item()),
            "reward_std": float(final_rewards.std(unbiased=False).item()),
            "reward_min": float(final_rewards.min().item()),
            "reward_max": float(final_rewards.max().item()),
            "advantage_mean": float(advantages.mean().detach().item()),
            "advantage_std": float(advantages.std(unbiased=False).detach().item()),
            "zero_std_group_rate": zero_std_group_rate,
            "mean_completion_chars": sum(completion_lengths) / max(1, len(completion_lengths)),
            "gen_time_s": t1 - t0,
            "reward_time_s": t2 - t1,
            "old_logprob_time_s": t3 - t2,
            "update_time_s": t4 - t3,
            "step_time_s": t4 - t0,
            "best_completion": generated["completions"][best_idx],
            "worst_completion": generated["completions"][worst_idx],
            "best_reward": float(final_rewards[best_idx].item()),
            "worst_reward": float(final_rewards[worst_idx].item()),
            "raw_reward_vec_mean": raw_reward_vecs.mean(dim=0).tolist(),
        }
        metrics.update(penalty_metrics)
        return metrics


def load_policy_model(args, tokenizer, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    print(f"Loading policy base: {args.policy_model_name} on {device}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.policy_model_name, **model_kwargs).to(device)
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False

    if args.policy_adapter_path:
        print(f"Loading trainable policy adapter: {args.policy_adapter_path}", flush=True)
        policy = PeftModel.from_pretrained(base, args.policy_adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        policy = get_peft_model(base, lora_config)
    if args.gradient_checkpointing:
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
    policy.print_trainable_parameters()
    return policy


def load_reference_model(args, tokenizer, device: torch.device, dtype: torch.dtype) -> nn.Module | None:
    if args.ref_kl_coeff <= 0:
        return None
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    print(f"Loading frozen reference model on {device}", flush=True)
    ref_base = AutoModelForCausalLM.from_pretrained(args.policy_model_name, **model_kwargs).to(device)
    if getattr(ref_base.config, "pad_token_id", None) is None:
        ref_base.config.pad_token_id = tokenizer.pad_token_id
    adapter = args.ref_adapter_path or args.policy_adapter_path
    if adapter:
        ref_model = PeftModel.from_pretrained(ref_base, adapter, is_trainable=False)
        if args.merge_ref_adapter:
            ref_model = ref_model.merge_and_unload()
    else:
        ref_model = ref_base
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ToM-SB policy with GRPO using the learned structured reward.")
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reward_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reward_checkpoint", type=str, default=DEFAULT_REWARD_CKPT)
    parser.add_argument("--policy_adapter_path", type=str, default="")
    parser.add_argument("--ref_adapter_path", type=str, default="")
    parser.add_argument("--merge_ref_adapter", action="store_true")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--reward_schema_path", type=str, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--prompts_per_iter", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--max_response_len", type=int, default=192)
    parser.add_argument("--reward_batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ref_kl_coeff", type=float, default=0.0)
    parser.add_argument("--sft_weight", type=float, default=0.05)
    parser.add_argument("--leak_penalty", type=float, default=1.0)
    parser.add_argument("--empty_penalty", type=float, default=1.0)
    parser.add_argument("--short_penalty", type=float, default=0.25)
    parser.add_argument("--min_response_chars", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--reward_device", type=str, default="cuda:0")
    parser.add_argument("--ref_device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0, help="Early stop patience on mean reward; 0 disables.")
    parser.add_argument("--scalar_weights", type=str, default=None, help="Comma list of reward_dim=weight overrides.")
    parser.add_argument("--reward_activation", type=str, default="sigmoid", choices=["raw", "sigmoid"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--skip_data_validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    policy_device = torch.device(args.policy_device if torch.cuda.is_available() else "cpu")
    reward_device = torch.device(args.reward_device if torch.cuda.is_available() else "cpu")
    ref_device = torch.device(args.ref_device if torch.cuda.is_available() else "cpu")
    if policy_device.type != "cuda" or reward_device.type != "cuda":
        raise RuntimeError("Policy GRPO expects CUDA for both policy and reward models.")
    dtype = _dtype_from_arg(args.dtype)

    reward_schema = _load_reward_schema(args.reward_schema_path)
    scalar_weights = parse_scalar_weights(args.scalar_weights, reward_schema)
    records = _read_json_or_jsonl(args.train_path)
    if args.max_examples > 0:
        records = records[: args.max_examples]
    if not args.skip_data_validation:
        summary, _ = validate_records(records, path=args.train_path, reward_schema=reward_schema, strict=True)
        print(
            f"train: records={summary.num_records}, unique_ids={summary.unique_example_ids}, "
            f"issues={summary.issue_count}, positive_truth_leaks={summary.positive_truth_leaks}",
            flush=True,
        )
    if not records:
        raise ValueError("No training records loaded.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "args.json", vars(args))
    _save_json(
        output_dir / "reward_config.json",
        {
            "reward_schema": reward_schema,
            "scalar_weights": {name: scalar_weights[idx].item() for idx, name in enumerate(reward_schema)},
            "reward_checkpoint": args.reward_checkpoint,
            "reward_activation": args.reward_activation,
        },
    )

    print(f"Loading policy tokenizer: {args.policy_model_name}", flush=True)
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name, trust_remote_code=args.trust_remote_code)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

    policy = load_policy_model(args, policy_tokenizer, policy_device, dtype)
    ref_model = load_reference_model(args, policy_tokenizer, ref_device, dtype)
    reward_scorer = LearnedToMSBRewardScorer(
        model_name=args.reward_model_name,
        checkpoint=args.reward_checkpoint,
        reward_schema=reward_schema,
        scalar_weights=scalar_weights,
        device=reward_device,
        dtype=dtype,
        max_ctx_len=args.max_prompt_len,
        max_resp_len=args.max_response_len,
        activation=args.reward_activation,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    optimizer = torch.optim.AdamW(
        [param for param in policy.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(args.warmup_ratio * args.num_iterations)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_iterations,
    )
    trainer = TomSBPolicyGRPOTrainer(
        policy=policy,
        ref_model=ref_model,
        reward_scorer=reward_scorer,
        tokenizer=policy_tokenizer,
        device=policy_device,
        ref_device=ref_device,
        dtype=dtype,
        group_size=args.group_size,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        max_response_len=args.max_response_len,
        reward_batch_size=args.reward_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_eps=args.clip_eps,
        ref_kl_coeff=args.ref_kl_coeff,
        sft_weight=args.sft_weight,
        leak_penalty=args.leak_penalty,
        empty_penalty=args.empty_penalty,
        short_penalty=args.short_penalty,
        min_response_chars=args.min_response_chars,
        max_grad_norm=args.max_grad_norm,
        use_chat_template=args.use_chat_template,
    )

    print(
        "\n"
        "============================================================\n"
        "  ToM-SB Policy GRPO\n"
        f"  Policy: {args.policy_model_name}\n"
        f"  Reward checkpoint: {args.reward_checkpoint}\n"
        f"  Train records: {len(records)}\n"
        f"  Iterations: {args.num_iterations}\n"
        f"  Prompts/iter: {args.prompts_per_iter}, group_size: {args.group_size}\n"
        f"  LR: {args.lr}, SFT anchor: {args.sft_weight}, ref KL: {args.ref_kl_coeff}\n"
        f"  Devices: policy={policy_device}, reward={reward_device}, ref={ref_device if ref_model else 'disabled'}\n"
        f"  Output: {output_dir}\n"
        "============================================================\n",
        flush=True,
    )

    rng = random.Random(args.seed)
    best_reward = -float("inf")
    patience_counter = 0
    start_time = time.time()
    log_path = output_dir / "training_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log_f:
        for iteration in range(1, args.num_iterations + 1):
            batch_size = min(args.prompts_per_iter, len(records))
            batch_records = rng.sample(records, batch_size)
            metrics = trainer.grpo_step(batch_records, optimizer, scheduler)
            metrics["iteration"] = iteration
            metrics["lr"] = scheduler.get_last_lr()[0]
            metrics["elapsed_s"] = time.time() - start_time
            log_f.write(json.dumps(metrics) + "\n")
            log_f.flush()

            if iteration % args.log_every == 0:
                print(
                    f"iter {iteration}/{args.num_iterations} "
                    f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f} "
                    f"learned={metrics['learned_reward_mean']:.4f} "
                    f"loss={metrics['loss']:.4f} pg={metrics['pg_loss']:.4f} "
                    f"sft={metrics['sft_loss']:.4f} leak={metrics['leak_rate']:.2%} "
                    f"short={metrics['short_rate']:.2%} lr={metrics['lr']:.2e} "
                    f"time={metrics['step_time_s']:.0f}s",
                    flush=True,
                )
                print(f"  best: {metrics['best_completion'][:240]}", flush=True)
                print(f"  worst: {metrics['worst_completion'][:240]}", flush=True)

            if metrics["reward_mean"] > best_reward:
                best_reward = metrics["reward_mean"]
                patience_counter = 0
                best_dir = output_dir / "best"
                policy.save_pretrained(best_dir)
                policy_tokenizer.save_pretrained(best_dir)
                _save_json(best_dir / "policy_grpo_metadata.json", {"iteration": iteration, "best_reward": best_reward})
                print(f"  Saved new best policy adapter: reward={best_reward:.4f}", flush=True)
            else:
                patience_counter += 1

            if args.save_every > 0 and iteration % args.save_every == 0:
                ckpt_dir = output_dir / f"iter_{iteration}"
                policy.save_pretrained(ckpt_dir)
                policy_tokenizer.save_pretrained(ckpt_dir)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping: no reward improvement for {args.patience} iterations.", flush=True)
                break

    final_dir = output_dir / "final"
    policy.save_pretrained(final_dir)
    policy_tokenizer.save_pretrained(final_dir)
    elapsed = time.time() - start_time
    print(
        "\n"
        "============================================================\n"
        "  ToM-SB Policy GRPO Complete\n"
        f"  Best reward: {best_reward:.4f}\n"
        f"  Output: {output_dir}\n"
        f"  Time: {elapsed:.0f}s ({elapsed / 3600:.2f}h)\n"
        "============================================================",
        flush=True,
    )

    del policy, ref_model, reward_scorer, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
