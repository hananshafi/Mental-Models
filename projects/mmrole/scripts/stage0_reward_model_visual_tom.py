#!/usr/bin/env python3
"""
Stage 0: Visual Recursive ToM Reward Model (VAE)
=================================================
Adapts the Sotopia-style learned reward model
(stage1_train_coupled_mental_reward_v3.py) to a VLM backbone for MMRole.

Architecture:
  VLM (Qwen2.5-VL / Qwen-VL-Chat) with LoRA on the language decoder
  + z1 (128-dim) VAE encoder: (image ⊕ context) -> 1st-order ToM latent
      z1_belief (48) | z1_intent (40) | z1_thought (40)
  + z2 (128-dim) VAE encoder: (context ⊕ z1) -> 2nd-order ToM latent
  + Joint reward head:   [z1 || z2 || resp_hidden] -> 8-dim official MMRole reward
  + z1-only reward head:                       z1 -> 3-dim ToM auxiliary reward
  + z-combined reward head:             [z1 || z2] -> 3-dim ToM auxiliary reward
  + Mental decoders (z1-bottlenecked for 1st-order belief text,
                     z2-bottlenecked for 2nd-order belief text)

Training signal (per MMRole sample):
  * preference loss: pos response (current_utterance) > rejected response
    (from preference_pairs.jsonl, joined by example_id)
  * reward regression: joint_reward ≈ 8-dim MMRole response scores
  * optional negative regression: neg_joint ≈ rejected-response 8-dim scores
  * z-only regression (anti-bypass): z1_only / z_combined match 3-dim ToM auxiliaries
  * mental generation: z1 reconstructs 1st-order belief text,
                       z2 reconstructs 2nd-order belief text
  * KL regularization on both z1 and z2 (annealed; z2 delayed)
  * future-token prediction from [z1||z2] to ground the latents

The trained checkpoint is consumed by stage2_grpo_learned_reward.py
via a FrozenRewardModel wrapper that loads the LoRA adapter + custom heads.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python stage0_reward_model_visual_tom.py \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --output_dir projects/mmrole/checkpoints/stage0_reward_v3

    # Multi-GPU with DDP
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 stage0_reward_model_visual_tom.py \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --output_dir projects/mmrole/checkpoints/stage0_reward_v3
"""

import os
import sys
import json
import gc
import time
import atexit
import argparse
import random
from typing import Optional, List, Dict

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "fileno"):
    try:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    except Exception:
        pass

print(">> stage0_reward_model_visual_tom.py starting...", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    detect_model_type, load_base_model, get_tokenizer,
    resolve_image, load_and_resize_image, default_lora_target_modules,
)

print(">> All imports done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Reward schema
# ──────────────────────────────────────────────────────────────────────────────
MMROLE_REWARD_DIMS = [
    "instruction_adherence",
    "fluency",
    "coherency",
    "image_text_relevance",
    "response_accuracy",
    "personality_consistency",
    "knowledge_consistency",
    "tone_consistency",
]
REWARD_DIM = len(MMROLE_REWARD_DIMS)

TOM_AUX_REWARD_DIMS = [
    "first_order_tom",
    "second_order_tom",
    "belief_divergence",
]
TOM_AUX_REWARD_DIM = len(TOM_AUX_REWARD_DIMS)

TOM_LEVEL_MAP = {"high": 1.0, "moderate": 0.7, "medium": 0.7, "low": 0.4, "none": 0.1}


def derive_tom_aux_targets(example: dict) -> List[float]:
    """Derive the 3-dim auxiliary ToM target from MMRole annotations.

    These auxiliary targets supervise the latent space directly. They are kept
    separate from the 8 official MMRole response-quality dimensions, which are
    consumed by the response-dependent joint reward head.
    """
    tom_rel = TOM_LEVEL_MAP.get(str(example.get("tom_relevance", "moderate")).lower(), 0.7)
    bel_div = TOM_LEVEL_MAP.get(str(example.get("belief_divergence", "moderate")).lower(), 0.7)
    vis_asym = 1.0 if example.get("visual_asymmetry", False) else 0.5
    return [
        tom_rel,                    # first_order_tom
        (tom_rel + bel_div) / 2.0,  # second_order_tom
        bel_div * vis_asym,         # belief_divergence
    ]


def _normalize_mmrole_score(value) -> float:
    score = float(value)
    if score < 1.0 or score > 10.0:
        raise ValueError(f"MMRole reward score must be in [1, 10], got {score}")
    return (score - 1.0) / 9.0


def load_mmrole_reward_targets(record: dict, score_key: str = "mmrole_reward_scores") -> List[float]:
    scores = record.get(score_key)
    if not isinstance(scores, dict):
        raise KeyError(
            f"Missing '{score_key}' in example_id={record.get('example_id', '<unknown>')}. "
            "Run step5c_annotate_mmrole_reward_dims_openai.py (or another 8-dim "
            "annotation pipeline) before Stage 0 training."
        )
    missing = [dim for dim in MMROLE_REWARD_DIMS if dim not in scores]
    if missing:
        raise KeyError(
            f"Missing MMRole reward dimensions {missing} under '{score_key}' for "
            f"example_id={record.get('example_id', '<unknown>')}."
        )
    return [_normalize_mmrole_score(scores[dim]) for dim in MMROLE_REWARD_DIMS]


ANNOTATED_FILENAME_MAP = {
    "belief_prediction.jsonl": "belief_prediction_mmrole_reward_openai.jsonl",
    "preference_pairs.jsonl": "preference_pairs_mmrole_reward_openai.jsonl",
}


def prefer_annotated_path(path: str, log_fn=print) -> str:
    basename = os.path.basename(path)
    annotated_name = ANNOTATED_FILENAME_MAP.get(basename)
    if not annotated_name:
        return path
    annotated_path = os.path.join(os.path.dirname(path), annotated_name)
    if os.path.exists(annotated_path):
        log_fn(f"  Using annotated reward file: {annotated_path}", flush=True)
        return annotated_path
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Mental decoder (z-bottlenecked, autoregressive via base-model LM head)
#
# The decoder projects each z sub-component to a short learned prefix of
# embeddings, which is prepended to the mental-text embeddings. The combined
# sequence is then fed through the base VLM (which applies a causal mask by
# construction) and decoded via the base model's own LM head. This replaces
# the previous non-causal cross-attention block, which saw future tokens and
# therefore trivialised the generation loss.
# ──────────────────────────────────────────────────────────────────────────────
def _build_mental_decoder(hidden_size, z_dim, z_sub_dims, num_prefix):
    z_belief_dim, z_intent_dim, z_thought_dim = z_sub_dims
    return nn.ModuleDict({
        "z_belief_to_prefix": nn.Linear(z_belief_dim, num_prefix * hidden_size),
        "z_intent_to_prefix": nn.Linear(z_intent_dim, num_prefix * hidden_size),
        "z_thought_to_prefix": nn.Linear(z_thought_dim, num_prefix * hidden_size),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Visual Recursive ToM Reward Model
# ──────────────────────────────────────────────────────────────────────────────
class VisualRecursiveToMRewardModel(nn.Module):
    """
    Recursive ToM VAE reward model on a VLM backbone.

    z1: pooled(image+context hidden) -> mu1, logvar1 -> z1
    z2: [pooled || z1]               -> mu2, logvar2 -> z2

    For Qwen2.5-VL the context forward pass receives pixel_values.
    For Qwen-VL-Chat the image is already embedded via <img>path</img> tags
    inside the text, so only input_ids/attention_mask are passed.
    """

    Z_BELIEF_DIM = 48
    Z_INTENT_DIM = 40
    Z_THOUGHT_DIM = 40
    NUM_PREFIX_TOKENS = 8

    def __init__(self, base_model: nn.Module, model_type: str,
                 reward_dim: int = REWARD_DIM,
                 aux_reward_dim: int = TOM_AUX_REWARD_DIM,
                 z_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        self.reward_dim = reward_dim
        self.aux_reward_dim = aux_reward_dim
        self.z_dim = z_dim

        hidden_size = self._infer_hidden_size(base_model)
        self.hidden_size = hidden_size

        assert z_dim == self.Z_BELIEF_DIM + self.Z_INTENT_DIM + self.Z_THOUGHT_DIM

        self.z1_mu = nn.Linear(hidden_size, z_dim)
        self.z1_logvar = nn.Linear(hidden_size, z_dim)
        self.z2_mu = nn.Linear(hidden_size + z_dim, z_dim)
        self.z2_logvar = nn.Linear(hidden_size + z_dim, z_dim)

        self.joint_outcome_head = nn.Sequential(
            nn.Linear(2 * z_dim + hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )
        self.z1_only_reward_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, aux_reward_dim),
        )
        self.z_combined_reward_head = nn.Sequential(
            nn.Linear(2 * z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, aux_reward_dim),
        )
        self.z_to_hidden = nn.Linear(2 * z_dim, hidden_size)

        z_sub_dims = (self.Z_BELIEF_DIM, self.Z_INTENT_DIM, self.Z_THOUGHT_DIM)
        self.mental1_decoder = _build_mental_decoder(
            hidden_size, z_dim, z_sub_dims, self.NUM_PREFIX_TOKENS
        )
        self.mental2_decoder = _build_mental_decoder(
            hidden_size, z_dim, z_sub_dims, self.NUM_PREFIX_TOKENS
        )

        self._init_weights()

    @staticmethod
    def _infer_hidden_size(base_model):
        cfg = getattr(base_model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "hidden_size") and cfg.hidden_size:
                return cfg.hidden_size
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
                return cfg.text_config.hidden_size
        return base_model.get_input_embeddings().embedding_dim

    def _init_weights(self):
        for m in [self.z1_mu, self.z1_logvar, self.z2_mu, self.z2_logvar]:
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            nn.init.zeros_(m.bias)
        nn.init.constant_(self.z1_logvar.bias, -2.0)
        nn.init.constant_(self.z2_logvar.bias, -2.0)

        nn.init.xavier_uniform_(self.z_to_hidden.weight)
        nn.init.zeros_(self.z_to_hidden.bias)

        for decoder in [self.mental1_decoder, self.mental2_decoder]:
            for key in ["z_belief_to_prefix", "z_intent_to_prefix", "z_thought_to_prefix"]:
                nn.init.xavier_uniform_(decoder[key].weight)
                nn.init.zeros_(decoder[key].bias)

        for group in [self.joint_outcome_head, self.z1_only_reward_head,
                      self.z_combined_reward_head]:
            for layer in group:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ── Backbone forward helpers ──
    def _backbone_hidden(self, input_ids, attention_mask,
                         pixel_values=None, image_grid_thw=None):
        """Run the VLM base model and return last hidden layer."""
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        out = self.base_model(**kwargs)
        return out.hidden_states[-1]

    @staticmethod
    def _pool(hidden, mask):
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def _sample_z(self, mu_proj, logvar_proj, h):
        mu = mu_proj(h)
        logvar = logvar_proj(h).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar).clamp(min=1e-8)
        eps = torch.randn_like(std, dtype=torch.float32)
        z = mu + std * eps
        return z, mu, logvar

    def encode_context_z1_z2(self, ctx_input_ids, ctx_attention_mask,
                             pixel_values=None, image_grid_thw=None,
                             stop_grad_z1: bool = False):
        hidden = self._backbone_hidden(
            ctx_input_ids, ctx_attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        )
        pooled = self._pool(hidden, ctx_attention_mask).float()

        z1, mu1, logvar1 = self._sample_z(self.z1_mu, self.z1_logvar, pooled)

        z1_for_z2 = z1.detach() if stop_grad_z1 else z1
        z2_input = torch.cat([pooled, z1_for_z2], dim=1)
        z2, mu2, logvar2 = self._sample_z(self.z2_mu, self.z2_logvar, z2_input)

        return pooled, z1, mu1, logvar1, z2, mu2, logvar2

    @torch.no_grad()
    def encode_context_deterministic(self, ctx_input_ids, ctx_attention_mask,
                                     pixel_values=None, image_grid_thw=None):
        hidden = self._backbone_hidden(
            ctx_input_ids, ctx_attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        )
        pooled = self._pool(hidden, ctx_attention_mask).float()
        mu1 = self.z1_mu(pooled)
        z2_input = torch.cat([pooled, mu1], dim=1)
        mu2 = self.z2_mu(z2_input)
        return mu1, mu2

    def encode_response(self, resp_input_ids, resp_attention_mask,
                        pixel_values=None, image_grid_thw=None):
        """Multimodal forward pass for response (image + text).

        Qwen2.5-VL needs image placeholder tokens in resp_input_ids that
        correspond to the supplied pixel_values / image_grid_thw. Qwen-VL-Chat
        embeds the <img>path</img> tag inside the text, so pixel_values is not
        used (pass None).
        """
        hidden = self._backbone_hidden(
            resp_input_ids, resp_attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        )
        return self._pool(hidden, resp_attention_mask).float()

    # ── Mental text decoding ──
    def _expand_z_to_prefix(self, z, decoder):
        b = z.size(0)
        zb = z[:, :self.Z_BELIEF_DIM]
        zi = z[:, self.Z_BELIEF_DIM:self.Z_BELIEF_DIM + self.Z_INTENT_DIM]
        zt = z[:, self.Z_BELIEF_DIM + self.Z_INTENT_DIM:]
        pb = decoder["z_belief_to_prefix"](zb).view(b, self.NUM_PREFIX_TOKENS, self.hidden_size)
        pi = decoder["z_intent_to_prefix"](zi).view(b, self.NUM_PREFIX_TOKENS, self.hidden_size)
        pt = decoder["z_thought_to_prefix"](zt).view(b, self.NUM_PREFIX_TOKENS, self.hidden_size)
        return torch.cat([pb, pi, pt], dim=1)

    def _decode_mental(self, z, decoder, mental_input_ids, mental_attention_mask):
        """Autoregressive mental-text LM loss conditioned on a z-prefix.

        The z sub-components are projected to a short continuous prefix, then
        concatenated with mental-text token embeddings and passed through the
        base VLM's causal LM. Logits for the mental tokens come from the
        base model's own LM head, so the loss is properly autoregressive.
        """
        embed_layer = self.base_model.get_input_embeddings()
        param_dtype = embed_layer.weight.dtype

        z_prefix = self._expand_z_to_prefix(z, decoder).to(dtype=param_dtype)
        mental_embeds = embed_layer(mental_input_ids).to(dtype=param_dtype)

        inputs_embeds = torch.cat([z_prefix, mental_embeds], dim=1)

        prefix_len = z_prefix.size(1)
        batch = mental_input_ids.size(0)
        prefix_mask = torch.ones(
            batch, prefix_len,
            dtype=mental_attention_mask.dtype,
            device=mental_attention_mask.device,
        )
        full_mask = torch.cat([prefix_mask, mental_attention_mask], dim=1)

        lm = self.base_model.get_base_model() if hasattr(self.base_model, "get_base_model") else self.base_model
        out = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits  # [B, prefix_len + L, V]

        # Align: logit at position p predicts token at position p+1.
        # Logit at (prefix_len - 1) predicts mental_token[0]; logit at
        # (prefix_len + k - 1) predicts mental_token[k].
        # We supervise tokens 0..L-2 using logits at prefix_len-1..prefix_len+L-3,
        # paired with labels mental_input_ids[:, :-1].
        mental_logits = logits[:, prefix_len - 1: prefix_len - 1 + mental_input_ids.size(1) - 1, :]

        shift_labels = mental_input_ids[:, :-1].clone().contiguous()
        shift_mask = mental_attention_mask[:, :-1].contiguous()
        shift_labels[shift_mask == 0] = -100

        return F.cross_entropy(
            mental_logits.contiguous().view(-1, mental_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ── Inference-time reward scoring ──
    def forward_reward_with_z(self, z1, z2, resp_input_ids, resp_attention_mask,
                              pixel_values=None, image_grid_thw=None):
        resp_pooled = self.encode_response(
            resp_input_ids, resp_attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        )
        joint_input = torch.cat([z1, z2, resp_pooled], dim=1)
        joint_reward = self.joint_outcome_head(joint_input)
        z1_reward = self.z1_only_reward_head(z1)
        z_combined_reward = self.z_combined_reward_head(torch.cat([z1, z2], dim=1))
        return joint_reward, z1_reward, z_combined_reward

    def forward(self,
                ctx_input_ids, ctx_attention_mask,
                pos_input_ids, pos_attention_mask,
                neg_input_ids, neg_attention_mask,
                mental1_input_ids, mental1_attention_mask,
                mental2_input_ids, mental2_attention_mask,
                reward_vec, aux_reward_vec, has_negative, first_pos_token,
                neg_reward_vec=None, has_negative_reward=None,
                ctx_pixel_values=None, ctx_image_grid_thw=None,
                pos_pixel_values=None, pos_image_grid_thw=None,
                neg_pixel_values=None, neg_image_grid_thw=None,
                kl_weight=0.1, z_only_weight=0.5,
                mental1_weight=0.3, mental2_weight=0.3, future_weight=0.3,
                kl_anneal_steps=200, z2_kl_delay_steps=100, z2_warmup_steps=100,
                current_opt_step=0):
        stop_grad_z1 = current_opt_step < z2_warmup_steps

        pooled, z1, mu1, logvar1, z2, mu2, logvar2 = self.encode_context_z1_z2(
            ctx_input_ids, ctx_attention_mask,
            pixel_values=ctx_pixel_values, image_grid_thw=ctx_image_grid_thw,
            stop_grad_z1=stop_grad_z1,
        )

        pos_pooled = self.encode_response(
            pos_input_ids, pos_attention_mask,
            pixel_values=pos_pixel_values, image_grid_thw=pos_image_grid_thw,
        )
        neg_pooled = self.encode_response(
            neg_input_ids, neg_attention_mask,
            pixel_values=neg_pixel_values, image_grid_thw=neg_image_grid_thw,
        )

        z1f = z1.to(pos_pooled.dtype)
        z2f = z2.to(pos_pooled.dtype)
        pos_joint = self.joint_outcome_head(
            torch.cat([z1f, z2f, pos_pooled], dim=1).float()
        )
        neg_joint = self.joint_outcome_head(
            torch.cat([z1f, z2f, neg_pooled], dim=1).float()
        )

        z1_only = self.z1_only_reward_head(z1)
        z_comb = self.z_combined_reward_head(torch.cat([z1, z2], dim=1))

        pref_per = F.softplus(neg_joint - pos_joint).mean(dim=1)
        pref_loss = (pref_per * has_negative).sum() / (has_negative.sum() + 1e-8)

        pos_reward_reg_loss = F.smooth_l1_loss(pos_joint, reward_vec)
        neg_reward_reg_loss = torch.tensor(0.0, device=pos_joint.device)
        has_any_negative_reward = False
        if neg_reward_vec is not None and has_negative_reward is not None:
            per_example_neg_reg = F.smooth_l1_loss(
                neg_joint, neg_reward_vec, reduction="none"
            ).mean(dim=1)
            neg_reward_reg_loss = (
                per_example_neg_reg * has_negative_reward
            ).sum() / (has_negative_reward.sum() + 1e-8)
            has_any_negative_reward = bool(has_negative_reward.sum().item() > 0)
        reward_reg_loss = (
            0.5 * (pos_reward_reg_loss + neg_reward_reg_loss)
            if has_any_negative_reward else pos_reward_reg_loss
        )

        z1_only_loss = F.smooth_l1_loss(z1_only, aux_reward_vec)
        z_comb_loss = F.smooth_l1_loss(z_comb, aux_reward_vec)

        amp_device = ctx_input_ids.device.type
        with torch.amp.autocast(device_type=amp_device, enabled=False):
            kl1 = compute_kl_loss(mu1, logvar1)
            kl2 = compute_kl_loss(mu2, logvar2)

        if kl_anneal_steps > 0:
            anneal1 = min(1.0, current_opt_step / kl_anneal_steps)
            anneal2 = min(
                1.0,
                max(0.0, (current_opt_step - z2_kl_delay_steps)) / kl_anneal_steps,
            )
        else:
            anneal1, anneal2 = 1.0, 1.0
        eff_kl1 = kl_weight * anneal1
        eff_kl2 = kl_weight * anneal2

        m1_loss = self._decode_mental(
            z1, self.mental1_decoder,
            mental1_input_ids, mental1_attention_mask,
        )
        m2_loss = self._decode_mental(
            z2, self.mental2_decoder,
            mental2_input_ids, mental2_attention_mask,
        )

        z_cat = torch.cat([z1, z2], dim=1)
        cond_hidden = (pooled + self.z_to_hidden(z_cat)).to(
            self.base_model.get_input_embeddings().weight.dtype
        )
        out_embed = self.base_model.get_output_embeddings()
        next_logits = F.linear(
            cond_hidden, out_embed.weight,
            out_embed.bias if hasattr(out_embed, "bias") and out_embed.bias is not None else None,
        )
        future_loss = F.cross_entropy(next_logits, first_pos_token)

        loss = (
            pref_loss
            + reward_reg_loss
            + z_only_weight * z1_only_loss
            + z_only_weight * z_comb_loss
            + eff_kl1 * kl1
            + eff_kl2 * kl2
            + mental1_weight * m1_loss
            + mental2_weight * m2_loss
            + future_weight * future_loss
        )

        return {
            "loss": loss,
            "pref_loss": pref_loss.detach(),
            "reward_reg_loss": reward_reg_loss.detach(),
            "pos_reward_reg_loss": pos_reward_reg_loss.detach(),
            "neg_reward_reg_loss": neg_reward_reg_loss.detach(),
            "z1_only_loss": z1_only_loss.detach(),
            "z_comb_loss": z_comb_loss.detach(),
            "kl1": kl1.detach(),
            "kl2": kl2.detach(),
            "m1_loss": m1_loss.detach(),
            "m2_loss": m2_loss.detach(),
            "future_loss": future_loss.detach(),
            "eff_kl1": float(eff_kl1),
            "eff_kl2": float(eff_kl2),
            "stop_grad_z1": bool(stop_grad_z1),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def format_reward_context(example: dict) -> str:
    """Text side of reward context (image is passed separately for Qwen2.5-VL,
    or embedded via <img> tags for Qwen-VL-Chat)."""
    speaker = example.get("speaker_name", "Speaker")
    partner = example.get("partner_name", "Partner")
    speaker_profile = example.get("speaker_profile", "")[:1200]
    partner_profile = example.get("partner_profile", "")[:1200]

    history_lines = []
    for t in example.get("dialogue_history", [])[-6:]:
        history_lines.append(f"[{t.get('speaker','?')}]: {t.get('utterance','')}")
    history_text = "\n".join(history_lines)

    return (
        f"Scene context for role-play between {speaker} and {partner}.\n\n"
        f"## {speaker} (speaker)\n{speaker_profile}\n\n"
        f"## {partner} (partner)\n{partner_profile}\n\n"
        f"## Dialogue so far\n{history_text if history_text else '(none)'}\n\n"
        f"Turn: {speaker} is about to respond."
    )


def format_first_order_mental(example: dict) -> str:
    d = example.get("target_speaker_belief", {}) or {}
    parts = []
    if d.get("partner_visual_focus"):
        parts.append(f"Partner Visual Focus: {d['partner_visual_focus']}")
    if d.get("partner_intent"):
        parts.append(f"Partner Intent: {d['partner_intent']}")
    if d.get("partner_knowledge"):
        parts.append(f"Partner Knowledge: {d['partner_knowledge']}")
    if d.get("partner_emotion"):
        parts.append(f"Partner Emotion: {d['partner_emotion']}")
    return " | ".join(parts) if parts else "N/A"


def format_second_order_mental(example: dict) -> str:
    d = example.get("target_speaker_2nd_order", {}) or {}
    parts = []
    if d.get("partner_thinks_i_see"):
        parts.append(f"Partner thinks I see: {d['partner_thinks_i_see']}")
    if d.get("partner_thinks_i_want"):
        parts.append(f"Partner thinks I want: {d['partner_thinks_i_want']}")
    if d.get("partner_thinks_i_know"):
        parts.append(f"Partner thinks I know: {d['partner_thinks_i_know']}")
    return " | ".join(parts) if parts else "N/A"


def _parse_paths_arg(paths: str) -> List[str]:
    return [p.strip() for p in str(paths or "").split(",") if p.strip()]


def build_reward_samples(belief_paths: List[str], image_dir: str,
                         preference_pairs_paths: Optional[List[str]] = None,
                         max_examples: int = -1,
                         log_fn=print) -> List[dict]:
    """Build Stage 0 reward samples from one or more split files."""
    belief_examples = []
    seen_example_ids = set()

    for belief_path in belief_paths:
        belief_path = prefer_annotated_path(belief_path, log_fn=log_fn)
        log_fn(f"Loading belief data from {belief_path}...", flush=True)
        file_count = 0
        with open(belief_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                ex_id = ex.get("example_id", "")
                if ex_id and ex_id in seen_example_ids:
                    continue
                if ex_id:
                    seen_example_ids.add(ex_id)
                belief_examples.append(ex)
                file_count += 1
        log_fn(f"  Loaded {file_count} belief examples", flush=True)

    if max_examples > 0:
        belief_examples = belief_examples[:max_examples]
    log_fn(f"  Total belief examples: {len(belief_examples)}", flush=True)

    pair_by_id: Dict[str, Dict[str, object]] = {}
    for pref_path in preference_pairs_paths or []:
        pref_path = prefer_annotated_path(pref_path, log_fn=log_fn)
        if not os.path.exists(pref_path):
            continue
        log_fn(f"Loading preference pairs from {pref_path}...", flush=True)
        file_pairs = 0
        with open(pref_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                ex_id = pair.get("example_id", "")
                if ex_id and ex_id not in pair_by_id:
                    neg_reward_vec = None
                    if isinstance(pair.get("rejected_mmrole_reward_scores"), dict):
                        neg_reward_vec = load_mmrole_reward_targets(
                            pair, score_key="rejected_mmrole_reward_scores"
                        )
                    pair_by_id[ex_id] = {
                        "rejected_response": pair.get("rejected_response", ""),
                        "neg_reward_vec": neg_reward_vec,
                    }
                    file_pairs += 1
        log_fn(f"  Unique negatives added: {file_pairs}", flush=True)
    if pair_by_id:
        log_fn(f"  Total unique negatives: {len(pair_by_id)}", flush=True)

    samples = []
    for ex in belief_examples:
        pos = ex.get("current_utterance", "").strip()
        if not pos:
            continue
        example_id = ex.get("example_id", "")
        pair_info = pair_by_id.get(example_id, {})
        hard_neg = str(pair_info.get("rejected_response", "")).strip()
        neg_reward_vec = pair_info.get("neg_reward_vec")
        samples.append({
            "example": ex,
            "context_text": format_reward_context(ex),
            "image_path": resolve_image(ex, image_dir),
            "pos_response": pos,
            "hard_negative": hard_neg,
            "mental1_text": format_first_order_mental(ex),
            "mental2_text": format_second_order_mental(ex),
            "reward_vec": load_mmrole_reward_targets(ex),
            "aux_reward_vec": derive_tom_aux_targets(ex),
            "neg_reward_vec": neg_reward_vec,
        })

    log_fn(f"  Reward samples built: {len(samples)}", flush=True)
    return samples


def split_reward_samples(samples: List[dict], val_size: int, seed: int) -> tuple[list, list]:
    if val_size <= 0 or val_size >= len(samples):
        return samples, []
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_idx = set(indices[:val_size])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    return train_samples, val_samples


class VisualToMRewardDataset(Dataset):
    """Dataset wrapper over already-built Stage 0 reward samples."""

    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Collate: context uses VLM processor (with image), text-only pieces use tokenizer
# ──────────────────────────────────────────────────────────────────────────────
def _build_context_text_for_processor(sample, model_type, processor):
    """Build the chat text for the context forward pass. For Qwen2.5-VL we
    return chat-templated text with an <image> placeholder; for Qwen-VL-Chat
    we embed the image path directly via from_list_format."""
    ctx = sample["context_text"]
    img_path = sample["image_path"]

    if model_type == "qwen2.5-vl":
        content = []
        if img_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": ctx})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    # qwen-vl-chat: image tag embedded in text
    if img_path:
        query = processor.from_list_format([
            {"image": img_path}, {"text": ctx}
        ])
    else:
        query = ctx
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_response_text_for_processor(response: str, img_path: Optional[str],
                                       model_type: str, processor) -> str:
    """Build the chat text for a response forward pass (multimodal)."""
    text = (response or " ").strip() or " "
    if model_type == "qwen2.5-vl":
        content = []
        if img_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": f"Response: {text}"})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    if img_path:
        query = processor.from_list_format([
            {"image": img_path}, {"text": f"Response: {text}"}
        ])
    else:
        query = f"Response: {text}"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def create_collate_fn(processor, model_type: str,
                      max_ctx_len: int = 1024,
                      max_resp_len: int = 256,
                      max_mental_len: int = 256):
    tokenizer = get_tokenizer(processor, model_type)
    pad_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or getattr(tokenizer, "eod_id", None)
        or getattr(tokenizer, "im_end_id", None)
        or 0
    )

    def _run_processor(texts, images, max_len):
        """Run the VLM processor on (texts, images); returns an inputs dict
        with input_ids / attention_mask and pixel_values / image_grid_thw
        if images are present.

        Truncation is disabled when images are present because the image
        placeholder tokens are expanded inline by the processor; truncating
        the resulting sequence can sever the image-token block and cause
        a "Mismatch in image token count" error.
        """
        if model_type == "qwen2.5-vl":
            has_images = [im is not None for im in images]
            if any(has_images):
                if not all(has_images):
                    missing = sum(1 for has_image in has_images if not has_image)
                    raise ValueError(
                        "Mixed image availability in a Qwen2.5-VL batch "
                        f"({missing}/{len(images)} missing). Resolve image paths "
                        "before reward-model tokenization."
                    )
                return processor(
                    text=texts, images=images,
                    return_tensors="pt", padding=True,
                )
            return processor(
                text=texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
        # qwen-vl-chat: image already embedded in text
        all_ids = []
        for t in texts:
            enc = tokenizer(
                t, return_tensors="pt", truncation=True, max_length=max_len,
            )
            all_ids.append(enc.input_ids.squeeze(0))
        mx = max(ids.shape[0] for ids in all_ids)
        ids_t = torch.full((len(all_ids), mx), pad_id, dtype=torch.long)
        mask_t = torch.zeros_like(ids_t)
        for i, ids in enumerate(all_ids):
            ids_t[i, :ids.shape[0]] = ids
            mask_t[i, :ids.shape[0]] = 1
        return {"input_ids": ids_t, "attention_mask": mask_t}

    def _find_first_content_token(ids: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
        """Pick the first real content token (skip BOS / padding) as the
        "future token" target. Previously we used ids[:, 0] which is typically
        BOS (same token every example) and made future-token loss trivial."""
        batch = ids.size(0)
        out = torch.zeros(batch, dtype=torch.long)
        bos = getattr(tokenizer, "bos_token_id", None)
        for i in range(batch):
            picked = None
            for t in range(ids.size(1)):
                if mask[i, t].item() == 0:
                    continue
                tok = ids[i, t].item()
                if bos is not None and tok == bos:
                    continue
                if tok == pad_id:
                    continue
                picked = tok
                break
            if picked is None and ids.size(1) > 1:
                picked = int(ids[i, 1].item())
            out[i] = int(picked if picked is not None else ids[i, 0].item())
        return out

    def collate(batch):
        # 1. Context inputs (with image)
        ctx_texts, ctx_images = [], []
        for sample in batch:
            ctx_texts.append(
                _build_context_text_for_processor(sample, model_type, processor)
            )
            img = load_and_resize_image(sample["image_path"]) if sample["image_path"] else None
            ctx_images.append(img)
        ctx_inputs = _run_processor(ctx_texts, ctx_images, max_ctx_len)

        # 2. Response inputs (with image) — multimodal encode_response
        pos_texts_raw = [s["pos_response"] for s in batch]
        neg_texts_raw = [
            s["hard_negative"] if s["hard_negative"] else s["pos_response"]
            for s in batch
        ]
        pos_texts = [
            _build_response_text_for_processor(
                t, s["image_path"], model_type, processor,
            )
            for t, s in zip(pos_texts_raw, batch)
        ]
        neg_texts = [
            _build_response_text_for_processor(
                t, s["image_path"], model_type, processor,
            )
            for t, s in zip(neg_texts_raw, batch)
        ]
        pos_inputs = _run_processor(pos_texts, list(ctx_images), max_resp_len)
        neg_inputs = _run_processor(neg_texts, list(ctx_images), max_resp_len)

        # 3. Mental texts (text-only: these are raw ToM labels, not dialog)
        def _tok_text_only(texts, max_len):
            if model_type == "qwen-vl-chat":
                all_ids = []
                for text in texts:
                    enc = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=max_len,
                    )
                    all_ids.append(enc.input_ids.squeeze(0))
                mx = max(ids.shape[0] for ids in all_ids)
                ids_t = torch.full((len(all_ids), mx), pad_id, dtype=torch.long)
                mask_t = torch.zeros_like(ids_t)
                for i, ids in enumerate(all_ids):
                    ids_t[i, :ids.shape[0]] = ids
                    mask_t[i, :ids.shape[0]] = 1
                return ids_t, mask_t
            enc = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
            return enc.input_ids, enc.attention_mask

        mental1_texts = [s["mental1_text"] for s in batch]
        mental2_texts = [s["mental2_text"] for s in batch]
        m1_ids, m1_mask = _tok_text_only(mental1_texts, max_mental_len)
        m2_ids, m2_mask = _tok_text_only(mental2_texts, max_mental_len)

        has_neg = torch.tensor(
            [1.0 if s["hard_negative"] else 0.0 for s in batch], dtype=torch.float32,
        )
        reward_vec = torch.tensor(
            [s["reward_vec"] for s in batch], dtype=torch.float32,
        )
        aux_reward_vec = torch.tensor(
            [s["aux_reward_vec"] for s in batch], dtype=torch.float32,
        )
        neg_reward_vec = torch.tensor(
            [
                s["neg_reward_vec"] if s.get("neg_reward_vec") is not None
                else [0.0] * REWARD_DIM
                for s in batch
            ],
            dtype=torch.float32,
        )
        has_negative_reward = torch.tensor(
            [1.0 if s.get("neg_reward_vec") is not None else 0.0 for s in batch],
            dtype=torch.float32,
        )

        pos_ids = pos_inputs["input_ids"].long()
        pos_mask = pos_inputs["attention_mask"].long()
        neg_ids = neg_inputs["input_ids"].long()
        neg_mask = neg_inputs["attention_mask"].long()

        first_pos_token = _find_first_content_token(pos_ids, pos_mask)

        out = {
            "ctx_input_ids": ctx_inputs["input_ids"].long(),
            "ctx_attention_mask": ctx_inputs["attention_mask"].long(),
            "pos_input_ids": pos_ids,
            "pos_attention_mask": pos_mask,
            "neg_input_ids": neg_ids,
            "neg_attention_mask": neg_mask,
            "mental1_input_ids": m1_ids.long(),
            "mental1_attention_mask": m1_mask.long(),
            "mental2_input_ids": m2_ids.long(),
            "mental2_attention_mask": m2_mask.long(),
            "reward_vec": reward_vec,
            "aux_reward_vec": aux_reward_vec,
            "neg_reward_vec": neg_reward_vec,
            "has_negative": has_neg,
            "has_negative_reward": has_negative_reward,
            "first_pos_token": first_pos_token,
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in ctx_inputs:
                out[f"ctx_{key}"] = ctx_inputs[key]
            if key in pos_inputs:
                out[f"pos_{key}"] = pos_inputs[key]
            if key in neg_inputs:
                out[f"neg_{key}"] = neg_inputs[key]
        return out

    return collate


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def compute_kl_loss(mu, logvar):
    mu32 = mu.float()
    lv32 = logvar.float().clamp(-10.0, 10.0)
    return -0.5 * torch.mean(1 + lv32 - mu32.pow(2) - lv32.exp())


CUSTOM_HEAD_NAMES = [
    "z1_mu", "z1_logvar", "z2_mu", "z2_logvar",
    "joint_outcome_head", "z1_only_reward_head", "z_combined_reward_head",
    "z_to_hidden",
    "mental1_decoder", "mental2_decoder",
]


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_metrics(metrics: Dict[str, float], device: torch.device,
                   world_size: int) -> Dict[str, float]:
    if world_size <= 1:
        return metrics
    keys = list(metrics.keys())
    vec = torch.tensor([metrics[k] for k in keys], dtype=torch.float32, device=device)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vec /= world_size
    return {k: float(v) for k, v in zip(keys, vec.tolist())}


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, loader, optimizer, scheduler, device, epoch,
                kl_weight=0.1, z_only_weight=0.5,
                mental1_weight=0.3, mental2_weight=0.3, future_weight=0.3,
                kl_anneal_steps=200, z2_kl_delay_steps=100, z2_warmup_steps=100,
                grad_accum=1, global_step_offset=0, max_grad_norm=5.0,
                world_size: int = 1, is_main_process: bool = True,
                save_every_steps: int = 0, on_checkpoint_step=None):
    model.train()
    metrics = {
        "loss": 0.0, "pref": 0.0, "reward_reg": 0.0, "z1_only": 0.0,
        "z_comb": 0.0, "kl1": 0.0, "kl2": 0.0, "m1_gen": 0.0, "m2_gen": 0.0,
        "future": 0.0,
    }
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        current_opt_step = global_step_offset + (batch_idx + 1) // grad_accum

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                **batch,
                kl_weight=kl_weight,
                z_only_weight=z_only_weight,
                mental1_weight=mental1_weight,
                mental2_weight=mental2_weight,
                future_weight=future_weight,
                kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps,
                z2_warmup_steps=z2_warmup_steps,
                current_opt_step=current_opt_step,
            )
            loss = outputs["loss"] / grad_accum

        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=max_grad_norm,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        metrics["loss"] += loss.item() * grad_accum
        metrics["pref"] += outputs["pref_loss"].item()
        metrics["reward_reg"] += outputs["reward_reg_loss"].item()
        metrics["z1_only"] += outputs["z1_only_loss"].item()
        metrics["z_comb"] += outputs["z_comb_loss"].item()
        metrics["kl1"] += outputs["kl1"].item()
        metrics["kl2"] += outputs["kl2"].item()
        metrics["m1_gen"] += outputs["m1_loss"].item()
        metrics["m2_gen"] += outputs["m2_loss"].item()
        metrics["future"] += outputs["future_loss"].item()

        if (
            (batch_idx + 1) % grad_accum == 0
            and on_checkpoint_step is not None
            and save_every_steps > 0
            and current_opt_step > 0
            and current_opt_step % save_every_steps == 0
        ):
            on_checkpoint_step(
                current_step=current_opt_step,
                checkpoint_name=f"step_{current_opt_step:06d}",
                epoch_idx=epoch,
                batch_progress=batch_idx + 1,
                running_train_metrics=metrics,
            )

        if is_main_process and (batch_idx + 1) % 10 == 0:
            n = batch_idx + 1
            lr_now = scheduler.get_last_lr()[0] if scheduler else 0.0
            sg = "SG" if outputs["stop_grad_z1"] else ""
            print(
                f"  Epoch {epoch+1} Step {n}/{n_batches}: "
                f"loss={metrics['loss']/n:.4f} pref={metrics['pref']/n:.4f} "
                f"reg={metrics['reward_reg']/n:.4f} "
                f"z1={metrics['z1_only']/n:.4f} zc={metrics['z_comb']/n:.4f} "
                f"kl1={metrics['kl1']/n:.4f}(w={outputs['eff_kl1']:.3f}) "
                f"kl2={metrics['kl2']/n:.4f}(w={outputs['eff_kl2']:.3f}) "
                f"m1={metrics['m1_gen']/n:.4f} m2={metrics['m2_gen']/n:.4f} "
                f"fut={metrics['future']/n:.4f} lr={lr_now:.2e} {sg}",
                flush=True,
            )

    for k in metrics:
        metrics[k] /= n_batches
    metrics = reduce_metrics(metrics, device, world_size)
    final_step = global_step_offset + n_batches // grad_accum
    return metrics, final_step


@torch.no_grad()
def eval_epoch(model, loader, device,
               kl_weight=0.1, z_only_weight=0.5,
               mental1_weight=0.3, mental2_weight=0.3, future_weight=0.3,
               kl_anneal_steps=200, z2_kl_delay_steps=100, z2_warmup_steps=100,
               current_opt_step=0, world_size: int = 1):
    model.eval()
    metrics = {
        "loss": 0.0, "pref": 0.0, "reward_reg": 0.0, "z1_only": 0.0,
        "z_comb": 0.0, "kl1": 0.0, "kl2": 0.0, "m1_gen": 0.0, "m2_gen": 0.0,
        "future": 0.0,
    }
    n_batches = len(loader)
    if n_batches == 0:
        return metrics

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                **batch,
                kl_weight=kl_weight,
                z_only_weight=z_only_weight,
                mental1_weight=mental1_weight,
                mental2_weight=mental2_weight,
                future_weight=future_weight,
                kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps,
                z2_warmup_steps=z2_warmup_steps,
                current_opt_step=current_opt_step,
            )

        metrics["loss"] += outputs["loss"].item()
        metrics["pref"] += outputs["pref_loss"].item()
        metrics["reward_reg"] += outputs["reward_reg_loss"].item()
        metrics["z1_only"] += outputs["z1_only_loss"].item()
        metrics["z_comb"] += outputs["z_comb_loss"].item()
        metrics["kl1"] += outputs["kl1"].item()
        metrics["kl2"] += outputs["kl2"].item()
        metrics["m1_gen"] += outputs["m1_loss"].item()
        metrics["m2_gen"] += outputs["m2_loss"].item()
        metrics["future"] += outputs["future_loss"].item()

    for k in metrics:
        metrics[k] /= n_batches
    return reduce_metrics(metrics, device, world_size)


def save_checkpoint(model, processor, save_dir: str):
    model = unwrap_model(model)
    os.makedirs(save_dir, exist_ok=True)
    model.base_model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    try:
        processor.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    except Exception:
        pass
    for head_name in CUSTOM_HEAD_NAMES:
        head = getattr(model, head_name)
        torch.save(head.state_dict(), os.path.join(save_dir, f"{head_name}.pth"))
    with open(os.path.join(save_dir, "reward_schema.json"), "w") as f:
        json.dump({
            "reward_dim": REWARD_DIM,
            "reward_dimensions": MMROLE_REWARD_DIMS,
            "aux_reward_dim": TOM_AUX_REWARD_DIM,
            "aux_reward_dimensions": TOM_AUX_REWARD_DIMS,
            "z_dim": model.z_dim,
            "z_belief_dim": model.Z_BELIEF_DIM,
            "z_intent_dim": model.Z_INTENT_DIM,
            "z_thought_dim": model.Z_THOUGHT_DIM,
            "num_prefix_tokens": model.NUM_PREFIX_TOKENS,
            "hidden_size": model.hidden_size,
        }, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Visual Recursive ToM Reward Model (VAE)"
    )
    # Model
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat"])
    # Data
    parser.add_argument("--train_path", type=str,
                        default=",".join([
                            "projects/mmrole/training_data/train/belief_prediction.jsonl",
                            "projects/mmrole/training_data/val/belief_prediction.jsonl",
                            "projects/mmrole/training_data/test_in/belief_prediction.jsonl",
                        ]),
                        help="Comma-separated belief_prediction.jsonl paths. "
                             "By default Stage 0 uses the full non-official pool "
                             "(train + val + test_in).")
    parser.add_argument("--preference_pairs_path", type=str,
                        default=",".join([
                            "projects/mmrole/training_data/train/preference_pairs.jsonl",
                            "projects/mmrole/training_data/val/preference_pairs.jsonl",
                            "projects/mmrole/training_data/test_in/preference_pairs.jsonl",
                        ]),
                        help="Comma-separated preference_pairs.jsonl paths aligned "
                             "with --train_path.")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--val_holdout_size", type=int, default=600,
                        help="Number of non-official samples to hold out for "
                             "validation/model selection. Set 0 to disable.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Deterministic seed for the Stage 0 train/val holdout.")
    # Training
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=1024,
                        help="Must be large enough to fit image tokens (~256) "
                             "plus response text. Too small → truncated image "
                             "token sequence → processor mismatch.")
    parser.add_argument("--max_mental_len", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every_steps", type=int, default=50,
                        help="Save a checkpoint every N optimizer steps. "
                             "Validation and best-model selection follow the "
                             "same cadence.")
    # Loss weights
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--z_only_weight", type=float, default=0.5)
    parser.add_argument("--mental1_weight", type=float, default=0.3)
    parser.add_argument("--mental2_weight", type=float, default=0.3)
    parser.add_argument("--future_weight", type=float, default=0.3)
    parser.add_argument("--kl_anneal_steps", type=int, default=200)
    parser.add_argument("--z2_kl_delay_steps", type=int, default=100)
    parser.add_argument("--z2_warmup_steps", type=int, default=100)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="Comma-separated LoRA target module names; defaults depend on model_type")
    parser.add_argument("--z_dim", type=int, default=128)
    # Output
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/checkpoints/stage0_reward_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Optional local rank for distributed launchers; "
                             "torchrun usually provides LOCAL_RANK via env.")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = os.environ.get("LOCAL_RANK")
    local_rank = int(env_local_rank) if env_local_rank is not None else args.local_rank
    distributed = world_size > 1

    if distributed and local_rank < 0:
        raise ValueError(
            "Distributed launch detected but LOCAL_RANK is missing. "
            "Launch with torchrun, e.g. torchrun --nproc_per_node=2 ..."
        )

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        atexit.register(cleanup_distributed)

    if torch.cuda.is_available():
        if distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    def maybe_barrier():
        if not distributed:
            return
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()

    is_main_process = rank == 0
    seed = args.seed + (rank if distributed else 0)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def log(msg: str, flush: bool = True):
        if is_main_process:
            print(msg, flush=flush)

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    maybe_barrier()

    # Model
    model_type = args.model_type or detect_model_type(args.base_model)
    log(f"Model type: {model_type}")
    base_model, processor, model_type = load_base_model(args.base_model, model_type)

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if distributed:
        log(
            f"Distributed training enabled: world_size={world_size}, "
            f"rank={rank}, local_rank={local_rank}, device={device}"
        )
    elif visible_gpus > 1:
        log(
            f"Visible GPUs: {visible_gpus}. This trainer runs in a single process, "
            f"so the full backbone will stay on {device}. Use torchrun for real "
            "multi-GPU Stage 0 training."
        )
    base_model = base_model.to(device)

    try:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log("Gradient checkpointing: enabled with use_reentrant=False")
    except TypeError:
        base_model.gradient_checkpointing_enable()
        log("Gradient checkpointing: enabled with default settings")
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    else:
        target_modules = default_lora_target_modules(model_type)
    log(f"LoRA target modules: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_" in name
    if is_main_process:
        base_model.print_trainable_parameters()
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    reward_model = VisualRecursiveToMRewardModel(
        base_model, model_type, reward_dim=REWARD_DIM, z_dim=args.z_dim,
    ).to(device)

    for name, param in reward_model.named_parameters():
        if any(k in name for k in CUSTOM_HEAD_NAMES):
            param.requires_grad = True

    n_train = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in reward_model.parameters())
    log(f"Trainable: {n_train:,}/{n_total:,} ({100*n_train/n_total:.2f}%)")

    if distributed:
        reward_model = DDP(
            reward_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
            static_graph=True,
        )
        log("DDP: static_graph=True")

    belief_paths = _parse_paths_arg(args.train_path)
    preference_paths = _parse_paths_arg(args.preference_pairs_path)
    all_samples = build_reward_samples(
        belief_paths,
        args.image_dir,
        preference_pairs_paths=preference_paths,
        max_examples=args.max_examples,
        log_fn=log,
    )
    train_samples, val_samples = split_reward_samples(
        all_samples, args.val_holdout_size, args.split_seed,
    )

    train_dataset = VisualToMRewardDataset(train_samples)
    val_dataset = VisualToMRewardDataset(val_samples)
    collate = create_collate_fn(
        processor, model_type,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
    )
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if len(val_dataset) > 0:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )

    # Optimizer: separate LR for LoRA vs custom heads
    head_params, lora_params = [], []
    for name, p in reward_model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in CUSTOM_HEAD_NAMES):
            head_params.append(p)
        else:
            lora_params.append(p)
    head_lr = args.lr * args.head_lr_mult
    log(f"Param groups: LoRA lr={args.lr}, Head lr={head_lr}")
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay},
    ])

    total_steps = (len(loader) * args.num_epochs) // args.grad_accum
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_metric = float("inf")
    best_source = ""

    def save_and_maybe_validate(current_step: int, checkpoint_name: str, epoch_idx: int,
                                batch_progress: Optional[int] = None,
                                running_train_metrics: Optional[Dict[str, float]] = None,
                                train_metrics_are_averages: bool = False,
                                run_validation: bool = True):
        nonlocal best_metric, best_source

        if is_main_process:
            checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
            save_checkpoint(reward_model, processor, checkpoint_dir)
        maybe_barrier()

        val_metrics = None
        if run_validation and val_loader is not None:
            val_metrics = eval_epoch(
                reward_model, val_loader, device,
                kl_weight=args.kl_weight, z_only_weight=args.z_only_weight,
                mental1_weight=args.mental1_weight, mental2_weight=args.mental2_weight,
                future_weight=args.future_weight,
                kl_anneal_steps=args.kl_anneal_steps,
                z2_kl_delay_steps=args.z2_kl_delay_steps,
                z2_warmup_steps=args.z2_warmup_steps,
                current_opt_step=current_step,
                world_size=world_size,
            )

        if is_main_process:
            if checkpoint_name.startswith("step_"):
                print(
                    f"\nCheckpoint {checkpoint_name} saved "
                    f"(epoch {epoch_idx+1}, optimizer step {current_step})",
                    flush=True,
                )
            else:
                print(
                    f"\nEpoch {epoch_idx+1}/{args.num_epochs} checkpoint saved "
                    f"as {checkpoint_name}",
                    flush=True,
                )

            if running_train_metrics is not None and batch_progress:
                print("  Train so far:", flush=True)
                for k, v in running_train_metrics.items():
                    train_value = v if train_metrics_are_averages else (v / batch_progress)
                    print(f"  {k}: {train_value:.4f}", flush=True)

            selection_metric = None
            metric_name = None
            if val_metrics is not None:
                print("  Val:", flush=True)
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.4f}", flush=True)
                selection_metric = val_metrics["loss"]
                metric_name = "val_loss"
            elif val_loader is None and running_train_metrics is not None and batch_progress:
                selection_metric = (
                    running_train_metrics["loss"]
                    if train_metrics_are_averages
                    else (running_train_metrics["loss"] / batch_progress)
                )
                metric_name = "train_loss"

            if selection_metric is not None and selection_metric < best_metric:
                best_metric = selection_metric
                best_source = checkpoint_name
                best_dir = os.path.join(args.output_dir, "best")
                save_checkpoint(reward_model, processor, best_dir)
                print(
                    f"  -> New best saved ({metric_name}={best_metric:.4f}, "
                    f"source={checkpoint_name})",
                    flush=True,
                )
            elif not run_validation and val_loader is not None:
                print(
                    f"  Validation already ran at optimizer step {current_step}; "
                    f"best checkpoint tracking already used that result.",
                    flush=True,
                )

        maybe_barrier()
        reward_model.train()

        return val_metrics

    if is_main_process:
        global_batch = args.batch_size * args.grad_accum * world_size
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage 0: Visual Recursive ToM Reward Model", flush=True)
        print(f"  Base model: {args.base_model}", flush=True)
        print(f"  Train samples: {len(train_dataset)}  val samples: {len(val_dataset)}", flush=True)
        print(f"  Batch x accum x world: {args.batch_size} x {args.grad_accum} x {world_size}",
              flush=True)
        print(f"  Effective global batch: {global_batch}", flush=True)
        print(f"  Total opt steps/rank: {total_steps}", flush=True)
        print(f"  Save/validate every: {args.save_every_steps} optimizer steps", flush=True)
        print(f"  Reward dims ({REWARD_DIM}): {MMROLE_REWARD_DIMS}", flush=True)
        print(f"  Output: {args.output_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

    global_step = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        metrics, global_step = train_epoch(
            reward_model, loader, optimizer, scheduler, device, epoch,
            kl_weight=args.kl_weight, z_only_weight=args.z_only_weight,
            mental1_weight=args.mental1_weight, mental2_weight=args.mental2_weight,
            future_weight=args.future_weight,
            kl_anneal_steps=args.kl_anneal_steps,
            z2_kl_delay_steps=args.z2_kl_delay_steps,
            z2_warmup_steps=args.z2_warmup_steps,
            grad_accum=args.grad_accum,
            global_step_offset=global_step,
            max_grad_norm=args.max_grad_norm,
            world_size=world_size,
            is_main_process=is_main_process,
            save_every_steps=args.save_every_steps,
            on_checkpoint_step=save_and_maybe_validate,
        )
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.num_epochs} complete:", flush=True)
            print("  Train:", flush=True)
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}", flush=True)

        should_validate_epoch = (
            val_loader is not None
            and args.save_every_steps > 0
            and global_step > 0
            and global_step % args.save_every_steps == 0
        )
        save_and_maybe_validate(
            current_step=global_step,
            checkpoint_name=f"epoch_{epoch}",
            epoch_idx=epoch,
            batch_progress=len(loader),
            running_train_metrics=metrics,
            train_metrics_are_averages=True,
            run_validation=not should_validate_epoch,
        )

        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    if is_main_process:
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage 0 reward training complete", flush=True)
        label = "Best val loss" if len(val_dataset) > 0 else "Best train loss"
        print(f"  {label}: {best_metric:.4f}", flush=True)
        if best_source:
            print(f"  Best source checkpoint: {best_source}", flush=True)
        print(f"  Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
        print(f"  Checkpoint: {args.output_dir}/best", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"\nNext: Stage 2 GRPO with learned reward", flush=True)
        print(f"  python stage2_grpo_learned_reward.py \\", flush=True)
        print(f"    --base_model {args.base_model} \\", flush=True)
        print(f"    --reward_checkpoint_dir {os.path.join(args.output_dir, 'best')} \\", flush=True)
        print(f"    --sft_checkpoint <stage1_sft_best>", flush=True)

    maybe_barrier()


if __name__ == "__main__":
    main()
