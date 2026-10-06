"""
Stage 1: Train Coupled Mental-State + Reward Model (v2)
=======================================================
Changes from v1:
  - LoRA rank: 16 -> 32 (alpha: 32 -> 64)
  - LoRA layers: top 8 -> top 24 (of 28 total in Qwen2.5-7B)
  - Reward dimensions: 7 -> 3 (goal, relationship, knowledge only)

Uses per-turn reward data from sotopia_turn_rewards.jsonl containing:
  - Per-turn reward scores with reasoning explanations
  - Mental state: partner_belief, strategic_intent, thought_process
  - Hard negative responses (socially poor alternatives)

Architecture:
  Base LLM (Qwen2.5-7B-Instruct) with LoRA
  + Latent z (128-dim) via VAE with 3 structured sub-spaces:
      z_belief (48d) | z_intent (40d) | z_thought (40d)
  + Joint outcome head: [z || response_hidden] -> 3-dim reward
  + True z-bottlenecked mental decoder: z is expanded into prefix tokens,
    mental text generation attends ONLY to z-prefix via cross-attention
    (NOT through the backbone's self-attention on mental tokens)
  + Explanation-conditioned reward: z cross-attends to reasoning text
  + Contrastive loss: positive utterance > hard_negative in reward space

Loss = L_preference + L_reward_regression + L_z_only_reward (anti-bypass)
       + L_kl + L_future + L_mental_gen (z-bottlenecked)
       + L_expl_reward (cross-attention)
"""

# CUDA_VISIBLE_DEVICES=0 python projects/sotopia/legacy/stage1_train_coupled_mental_reward_v2.py \
#   --model_name Qwen/Qwen2.5-7B-Instruct \
#   --data_path projects/sotopia/data/sotopia_turn_rewards.jsonl \
#   --output_dir projects/sotopia/checkpoints/coupled_mental_reward_v3 \
#   --batch_size 4 \
#   --grad_accum_steps 8 \
#   --num_epochs 10 \
#   --lr 2e-5 \
#   --warmup_ratio 0.03 \
#   --max_ctx_len 1024 \
#   --max_resp_len 256 \
#   --z_dim 128 \
#   --lora_r 16 \
#   --lora_alpha 32 \
#   --lora_dropout 0.05 \
#   --num_lora_layers 16 \
#   --kl_weight 0.1 \
#   --future_weight 0.5 \
#   --mental_weight 0.3 \
#   --expl_weight 0.3 \
#   --z_only_weight 0.5 \
#   --kl_anneal_steps 200 \
#   --mental_prewarm_data projects/sotopia/data/mental_model_persona_dataset.jsonl \
#   --mental_prewarm_epochs 1 \
#   --mental_prewarm_lr 2e-4 \
#   --head_lr_mult 10.0 \
#   --max_grad_norm 5.0 \
#   --seed 42
#   --gpu 7


import os
import json
import gc
import math
import argparse
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SOTOPIA_DIMENSIONS = [
    "believability", "relationship", "knowledge", "secret",
    "social_rules", "financial_and_material_benefits", "goal",
]

# Score ranges for normalization (from official SOTOPIA)
DIM_RANGES = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}

REWARD_DIM = len(SOTOPIA_DIMENSIONS)  # 3


def normalize_score(dim: str, score: float) -> float:
    """Normalize a SOTOPIA dimension score to [0, 1]."""
    lo, hi = DIM_RANGES[dim]
    return (score - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────
def get_transformer_from_peft(peft_causal_lm):
    m = peft_causal_lm
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    if hasattr(m, "model"):
        return m.model
    raise RuntimeError("Couldn't find underlying transformer (.model).")


class CoupledMentalRewardModel(nn.Module):
    """
    Coupled Mental-State + Reward model.

    Components:
      1. LoRA-tuned LLM backbone (Qwen2.5-7B)
      2. VAE latent z from context (encodes partner mental state)
         z is 128-dim, partitioned into 3 sub-spaces:
           z_belief (48d), z_intent (40d), z_thought (40d)
      3. Joint outcome head: [z || response_hidden] -> 3-dim reward prediction
      4. z_to_hidden: projects z back into hidden space for next-token conditioning
      5. Mental text decoder: z -> partner_belief + strategic_intent + thought_process
         Uses TRUE z-bottleneck: z is expanded into prefix tokens, and mental text
         generation can ONLY attend to these z-derived prefix tokens (not to each
         other via the backbone's self-attention on the mental text).
      6. Explanation-conditioned reward head: uses reasoning text hidden states to
         inform reward prediction via cross-attention from z
    """

    # Sub-space sizes for the 3 mental state fields
    Z_BELIEF_DIM = 48
    Z_INTENT_DIM = 40
    Z_THOUGHT_DIM = 40
    NUM_PREFIX_TOKENS = 8  # Number of virtual prefix tokens expanded from z

    def __init__(self, base_model: nn.Module, reward_dim: int = REWARD_DIM, z_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.transformer = get_transformer_from_peft(self.base_model)
        hidden_size = base_model.get_input_embeddings().embedding_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        assert z_dim == self.Z_BELIEF_DIM + self.Z_INTENT_DIM + self.Z_THOUGHT_DIM, \
            f"z_dim ({z_dim}) must equal sum of sub-space dims"

        # VAE latent projections
        self.context_mu = nn.Linear(hidden_size, z_dim)
        self.context_logvar = nn.Linear(hidden_size, z_dim)

        # Joint outcome head: [z || response_hidden_isolated] -> reward_dim
        # response_hidden is encoded IN ISOLATION (separate forward pass without
        # context) so it cannot short-circuit z by carrying context information.
        self.joint_outcome_head = nn.Sequential(
            nn.Linear(z_dim + hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )

        # z-only reward head: forces z to independently carry reward-relevant info.
        # Without this, joint_outcome_head could still learn to extract residual
        # context information from response tokens (since natural language responses
        # implicitly reveal context). This head makes z directly accountable.
        self.z_only_reward_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, reward_dim),
        )

        # Latent-to-hidden for conditioning next-token prediction
        self.z_to_hidden = nn.Linear(z_dim, hidden_size)

        # ── Mental Text Decoder (True z-Bottleneck) ──
        # Expands z into NUM_PREFIX_TOKENS virtual tokens in hidden space.
        # During mental text decoding, the mental tokens can ONLY attend to these
        # z-derived prefix tokens. The LM is NOT run on the mental tokens directly —
        # instead we use a lightweight cross-attention decoder where:
        #   Q = mental token embeddings (position-encoded)
        #   K,V = z-prefix tokens
        # Then project to vocab logits via the LM head.
        #
        # Each mental sub-field (belief, intent, thought) has its own z->prefix
        # projection, so z is forced to partition information across sub-spaces.
        self.z_belief_to_prefix = nn.Linear(
            self.Z_BELIEF_DIM, self.NUM_PREFIX_TOKENS * hidden_size
        )
        self.z_intent_to_prefix = nn.Linear(
            self.Z_INTENT_DIM, self.NUM_PREFIX_TOKENS * hidden_size
        )
        self.z_thought_to_prefix = nn.Linear(
            self.Z_THOUGHT_DIM, self.NUM_PREFIX_TOKENS * hidden_size
        )

        # Cross-attention decoder for mental text generation:
        # mental token embeddings attend to z-prefix tokens
        self.mental_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8,
            kdim=hidden_size, vdim=hidden_size,
            batch_first=True, dropout=0.1,
        )
        # LayerNorm + feed-forward after cross-attention
        self.mental_decode_ln = nn.LayerNorm(hidden_size)
        self.mental_decode_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.mental_decode_ln2 = nn.LayerNorm(hidden_size)

        # ── Explanation-Aware Reward Refinement ──
        # Cross-attention: z attends over explanation hidden states to extract
        # reasoning-grounded features for reward prediction.
        # Q = z projected, K/V = explanation hidden states
        self.expl_cross_attn = nn.MultiheadAttention(
            embed_dim=z_dim, num_heads=8, kdim=hidden_size, vdim=hidden_size,
            batch_first=True, dropout=0.1,
        )
        # Refined reward head that uses explanation-attended z
        self.expl_reward_head = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, reward_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # mu/logvar projections: tiny init so z starts near N(0, ~0.14I).
        # Xavier gives scale ~0.04, but pretrained hidden states have magnitude
        # ~30-50 per dim, producing |mu| ~ 50 and KL ~ 1400. Scale 0.001
        # keeps |mu| ~ 1-2, KL ~ 1, and downstream heads in a learnable regime.
        for module in [self.context_mu, self.context_logvar]:
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            nn.init.zeros_(module.bias)
        # logvar bias = -2 → initial std = exp(-1) ≈ 0.37, keeping z tight
        nn.init.constant_(self.context_logvar.bias, -2.0)

        for module in [self.z_to_hidden,
                       self.z_belief_to_prefix, self.z_intent_to_prefix,
                       self.z_thought_to_prefix]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for layer_group in [self.joint_outcome_head, self.z_only_reward_head,
                            self.expl_reward_head, self.mental_decode_ffn]:
            for layer in layer_group:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _encode_context(self, input_ids, attention_mask):
        """Encode context to get hidden state at last token."""
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        final_hidden = out.last_hidden_state
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            last_idx_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, final_hidden.size(-1))
            context_hidden = final_hidden.gather(1, last_idx_exp).squeeze(1)
        else:
            context_hidden = final_hidden[:, -1, :]
        return context_hidden

    def _sample_z(self, context_hidden):
        """Sample latent z via reparameterization trick."""
        mu = self.context_mu(context_hidden)
        logvar = self.context_logvar(context_hidden).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar).clamp(min=1e-8)
        eps = torch.randn_like(std, dtype=torch.float32)
        z = mu + std * eps
        return z, mu, logvar

    def forward_reward(self, ctx_input_ids, ctx_attention_mask,
                       resp_input_ids, resp_attention_mask):
        """
        Forward pass for reward prediction with isolated response encoding.

        CRITICAL DESIGN: context and response are encoded in SEPARATE forward
        passes through the transformer. This prevents response_hidden from
        carrying context information via causal self-attention, which would
        allow the reward head to bypass z entirely.

        Returns:
            joint_reward: [batch, 3] from joint_outcome_head([z || resp_hidden])
            z_only_reward: [batch, 3] from z_only_reward_head(z)
            mu, logvar: VAE parameters
        """
        # 1. Encode context -> z
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        z, mu, logvar = self._sample_z(context_hidden)

        # 2. Encode response IN ISOLATION (no context in this forward pass)
        resp_out = self.transformer(
            input_ids=resp_input_ids,
            attention_mask=resp_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        resp_hidden = resp_out.last_hidden_state
        if resp_attention_mask is not None:
            resp_last_idx = resp_attention_mask.sum(dim=1) - 1
            resp_last_exp = resp_last_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, resp_hidden.size(-1)
            )
            response_hidden = resp_hidden.gather(1, resp_last_exp).squeeze(1)
        else:
            response_hidden = resp_hidden[:, -1, :]

        # 3. Joint reward: z provides context understanding, response_hidden
        #    provides response-level features. Neither alone has both.
        joint_input = torch.cat([z, response_hidden], dim=1)
        joint_reward = self.joint_outcome_head(joint_input)

        # 4. z-only reward: forces z to independently encode reward signal
        z_only_reward = self.z_only_reward_head(z)

        return joint_reward, z_only_reward, mu, logvar

    def forward_next_token(self, input_ids, attention_mask):
        """
        Forward pass for next-token prediction conditioned on latent z.
        Used for the future prediction auxiliary loss.
        """
        context_hidden = self._encode_context(input_ids, attention_mask)
        z, mu, logvar = self._sample_z(context_hidden)

        conditioned_hidden = context_hidden + self.z_to_hidden(z)

        output_embedding = self.base_model.get_output_embeddings()
        if hasattr(output_embedding, 'weight'):
            logits = F.linear(
                conditioned_hidden, output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, 'bias') and output_embedding.bias is not None else None
            )
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]

        return logits, mu, logvar

    def _expand_z_to_prefix(self, z):
        """
        Split z into 3 sub-spaces and expand each into prefix tokens.

        z: [batch, z_dim=128]
        Returns: [batch, 3*NUM_PREFIX_TOKENS, hidden_size] — concatenated prefix
                 tokens from all 3 sub-fields.
        """
        batch_size = z.size(0)
        z_belief = z[:, :self.Z_BELIEF_DIM]
        z_intent = z[:, self.Z_BELIEF_DIM:self.Z_BELIEF_DIM + self.Z_INTENT_DIM]
        z_thought = z[:, self.Z_BELIEF_DIM + self.Z_INTENT_DIM:]

        # Each sub-field -> NUM_PREFIX_TOKENS tokens in hidden_size
        prefix_belief = self.z_belief_to_prefix(z_belief).view(
            batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size
        )
        prefix_intent = self.z_intent_to_prefix(z_intent).view(
            batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size
        )
        prefix_thought = self.z_thought_to_prefix(z_thought).view(
            batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size
        )

        # Concatenate: [batch, 3*NUM_PREFIX_TOKENS, hidden_size]
        return torch.cat([prefix_belief, prefix_intent, prefix_thought], dim=1)

    def forward_mental_decode(self, ctx_input_ids, ctx_attention_mask,
                               mental_input_ids, mental_attention_mask):
        """
        True z-bottlenecked mental text decoding.

        The mental text tokens DO NOT go through the backbone transformer.
        Instead:
          1. Context -> z via backbone + VAE
          2. z is split into 3 sub-spaces and expanded into prefix tokens
          3. Mental token embeddings (from the embedding layer only) are used as
             queries in a cross-attention decoder where K,V = z-prefix tokens
          4. The cross-attended representations are projected to vocab logits
             via the LM head for next-token prediction

        This ensures z is the ONLY source of content information for
        reconstructing the mental state text. The LM's autoregressive
        self-attention over mental tokens is NOT used — z cannot be bypassed.
        """
        # Encode context to get z
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        z, mu, logvar = self._sample_z(context_hidden)

        # Expand z into prefix tokens: [batch, 3*NUM_PREFIX_TOKENS, hidden_size]
        z_prefix_tokens = self._expand_z_to_prefix(z)

        # Get mental token embeddings (embedding layer only, NO transformer)
        embedding_layer = self.base_model.get_input_embeddings()
        mental_embeds = embedding_layer(mental_input_ids)  # [batch, seq, hidden_size]

        # Cross-attention: mental embeds (Q) attend to z-prefix tokens (K,V)
        # This is the key: mental tokens can ONLY access information through z
        attended, _ = self.mental_cross_attn(
            query=mental_embeds,
            key=z_prefix_tokens,
            value=z_prefix_tokens,
        )

        # Residual + LayerNorm + FFN (lightweight decoder block)
        h = self.mental_decode_ln(mental_embeds + attended)
        h = h + self.mental_decode_ffn(h)
        h = self.mental_decode_ln2(h)

        # Project to vocab logits via LM head
        output_embedding = self.base_model.get_output_embeddings()
        if hasattr(output_embedding, 'weight'):
            logits = F.linear(
                h, output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, 'bias') and output_embedding.bias is not None else None
            )
        else:
            logits = h  # fallback

        # Shift for next-token prediction: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = mental_input_ids[:, 1:].clone().contiguous()

        # Mask out padding positions
        shift_mask = mental_attention_mask[:, 1:].contiguous()
        shift_labels[shift_mask == 0] = -100

        mental_gen_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return mental_gen_loss, mu, logvar

    def forward_expl_reward(self, ctx_input_ids, ctx_attention_mask,
                            expl_input_ids, expl_attention_mask):
        """
        Explanation-conditioned reward prediction.

        Encodes context → z, encodes explanation text → hidden states,
        then z cross-attends to explanation hidden states to produce a
        reasoning-grounded reward prediction.

        This forces the model to align z with the textual reasoning that
        justifies each dimension's score.
        """
        # Encode context → z
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        z, mu, logvar = self._sample_z(context_hidden)

        # Encode explanation text
        expl_out = self.transformer(
            input_ids=expl_input_ids,
            attention_mask=expl_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        expl_hidden = expl_out.last_hidden_state  # [batch, expl_seq, hidden]

        # Cross-attention: z (query) attends to explanation hidden states (key/value)
        z_query = z.unsqueeze(1)  # [batch, 1, z_dim]

        # Build key padding mask for cross-attention (True = ignore)
        key_padding_mask = (expl_attention_mask == 0)

        attended_z, _ = self.expl_cross_attn(
            query=z_query, key=expl_hidden, value=expl_hidden,
            key_padding_mask=key_padding_mask,
        )
        attended_z = attended_z.squeeze(1)  # [batch, z_dim]

        # Predict reward from explanation-attended z
        expl_reward_pred = self.expl_reward_head(attended_z)

        return expl_reward_pred, mu, logvar

    def encode_context_to_z(self, ctx_input_ids, ctx_attention_mask,
                            deterministic: bool = True):
        """Encode context to z. Use deterministic=True (mu only) at inference
        to ensure consistent scoring across candidates for the same prompt."""
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        mu = self.context_mu(context_hidden)
        if deterministic:
            return mu
        logvar = self.context_logvar(context_hidden).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar).clamp(min=1e-8)
        eps = torch.randn_like(std, dtype=torch.float32)
        return mu + std * eps

    def forward_reward_with_z(self, z, resp_input_ids, resp_attention_mask):
        """Reward prediction given a pre-computed z and isolated response.

        Used at inference (Stage 2/3) so the same z is reused across all
        G candidate responses for a given prompt, ensuring fair comparison.

        Returns: joint_reward [batch, 3]
        """
        # Encode response in isolation
        resp_out = self.transformer(
            input_ids=resp_input_ids,
            attention_mask=resp_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        resp_hidden = resp_out.last_hidden_state
        if resp_attention_mask is not None:
            resp_last_idx = resp_attention_mask.sum(dim=1) - 1
            resp_last_exp = resp_last_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, 1, resp_hidden.size(-1)
            )
            response_hidden = resp_hidden.gather(1, resp_last_exp).squeeze(1)
        else:
            response_hidden = resp_hidden[:, -1, :]

        joint_input = torch.cat([z, response_hidden], dim=1)
        return self.joint_outcome_head(joint_input)

    def forward_all(self, ctx_input_ids, ctx_attention_mask,
                    pos_input_ids, pos_attention_mask,
                    neg_input_ids, neg_attention_mask,
                    mental_input_ids, mental_attention_mask,
                    expl_input_ids, expl_attention_mask,
                    first_pos_token):
        """
        Unified forward pass that encodes context ONCE and batches isolated
        encodings (pos, neg, expl) into a single transformer call.

        Reduces transformer forward passes from ~7 to ~2 per training step.

        Returns dict with all loss components.
        """
        batch_size = ctx_input_ids.size(0)

        # ── 1. Encode context ONCE → z ──
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        z, mu, logvar = self._sample_z(context_hidden)

        # ── 2. Batch pos + neg + expl into ONE transformer forward pass ──
        # Concatenate along batch dimension: [3*B, seq_len]
        # Pad to same seq length first
        max_resp_len = max(pos_input_ids.size(1), neg_input_ids.size(1))
        max_expl_len = expl_input_ids.size(1)
        max_isolated_len = max(max_resp_len, max_expl_len)

        def _pad_to(ids, mask, target_len):
            pad_len = target_len - ids.size(1)
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=self.base_model.config.pad_token_id
                            if hasattr(self.base_model.config, 'pad_token_id') and self.base_model.config.pad_token_id is not None
                            else 0)
                mask = F.pad(mask, (0, pad_len), value=0)
            return ids, mask

        pos_ids_p, pos_mask_p = _pad_to(pos_input_ids, pos_attention_mask, max_isolated_len)
        neg_ids_p, neg_mask_p = _pad_to(neg_input_ids, neg_attention_mask, max_isolated_len)
        expl_ids_p, expl_mask_p = _pad_to(expl_input_ids, expl_attention_mask, max_isolated_len)

        batched_ids = torch.cat([pos_ids_p, neg_ids_p, expl_ids_p], dim=0)    # [3B, L]
        batched_mask = torch.cat([pos_mask_p, neg_mask_p, expl_mask_p], dim=0)  # [3B, L]

        batched_out = self.transformer(
            input_ids=batched_ids, attention_mask=batched_mask,
            use_cache=False, return_dict=True,
        )
        batched_hidden = batched_out.last_hidden_state  # [3B, L, H]

        # Extract last-token hidden states for pos and neg
        def _extract_last(hidden, mask, start, end):
            h = hidden[start:end]
            m = mask[start:end]
            last_idx = m.sum(dim=1) - 1
            last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
            return h.gather(1, last_exp).squeeze(1)

        pos_response_hidden = _extract_last(batched_hidden, batched_mask, 0, batch_size)
        neg_response_hidden = _extract_last(batched_hidden, batched_mask, batch_size, 2 * batch_size)
        expl_hidden = batched_hidden[2 * batch_size:]  # [B, L, H] — full sequence for cross-attn

        # ── 3. Joint reward predictions (reusing z and isolated response hiddens) ──
        pos_joint_input = torch.cat([z, pos_response_hidden], dim=1)
        pos_joint_reward = self.joint_outcome_head(pos_joint_input)
        pos_z_only_reward = self.z_only_reward_head(z)

        neg_joint_input = torch.cat([z, neg_response_hidden], dim=1)
        neg_joint_reward = self.joint_outcome_head(neg_joint_input)

        # ── 4. Next-token prediction from z (no extra transformer call) ──
        conditioned_hidden = context_hidden + self.z_to_hidden(z)
        output_embedding = self.base_model.get_output_embeddings()
        if hasattr(output_embedding, 'weight'):
            next_logits = F.linear(
                conditioned_hidden, output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, 'bias') and output_embedding.bias is not None else None
            )
        else:
            next_logits = conditioned_hidden  # fallback
        future_loss = F.cross_entropy(next_logits, first_pos_token)

        # ── 5. Mental text decode (z-bottlenecked, no transformer on mental tokens) ──
        z_prefix_tokens = self._expand_z_to_prefix(z)
        embedding_layer = self.base_model.get_input_embeddings()
        mental_embeds = embedding_layer(mental_input_ids)
        attended, _ = self.mental_cross_attn(
            query=mental_embeds, key=z_prefix_tokens, value=z_prefix_tokens,
        )
        h = self.mental_decode_ln(mental_embeds + attended)
        h = h + self.mental_decode_ffn(h)
        h = self.mental_decode_ln2(h)
        if hasattr(output_embedding, 'weight'):
            mental_logits = F.linear(
                h, output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, 'bias') and output_embedding.bias is not None else None
            )
        else:
            mental_logits = h
        shift_logits = mental_logits[:, :-1, :].contiguous()
        shift_labels = mental_input_ids[:, 1:].clone().contiguous()
        shift_mask = mental_attention_mask[:, 1:].contiguous()
        shift_labels[shift_mask == 0] = -100
        mental_gen_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

        # ── 6. Explanation-conditioned reward (reusing expl_hidden from batched pass) ──
        z_query = z.unsqueeze(1)
        key_padding_mask = (expl_mask_p == 0)
        attended_z, _ = self.expl_cross_attn(
            query=z_query, key=expl_hidden, value=expl_hidden,
            key_padding_mask=key_padding_mask,
        )
        attended_z = attended_z.squeeze(1)
        expl_reward_pred = self.expl_reward_head(attended_z)

        return {
            "pos_joint_reward": pos_joint_reward,
            "neg_joint_reward": neg_joint_reward,
            "pos_z_only_reward": pos_z_only_reward,
            "expl_reward_pred": expl_reward_pred,
            "mu": mu, "logvar": logvar,
            "future_loss": future_loss,
            "mental_gen_loss": mental_gen_loss,
        }

    def predict_reward(self, ctx_input_ids, ctx_attention_mask,
                       resp_input_ids, resp_attention_mask):
        """Inference-time reward prediction (no grad). Returns [batch, 3] reward.
        Uses deterministic z (mu) + joint reward head."""
        with torch.no_grad():
            z = self.encode_context_to_z(ctx_input_ids, ctx_attention_mask,
                                          deterministic=True)
            reward = self.forward_reward_with_z(z, resp_input_ids, resp_attention_mask)
        return reward


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: Per-Turn Coupled Mental + Reward
# ──────────────────────────────────────────────────────────────────────────────
class CoupledMentalRewardDataset(Dataset):
    """
    Loads sotopia_turn_rewards.jsonl and creates per-turn training samples.

    Each sample consists of:
      - context: scenario + agent background + goal + dialogue history up to turn t
      - positive response: the actual utterance at turn t
      - hard negative: the hard_negative_response from mental_state
      - reward vector: 3-dim normalized scores (goal, relationship, knowledge)
      - mental state text: partner_belief + strategic_intent + thought_process
      - reward explanations: concatenated reasoning across the 3 dims
    """

    def __init__(self, data_path: str, tokenizer, max_ctx_len: int = 1024,
                 max_resp_len: int = 256, max_mental_len: int = 256):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        self.max_mental_len = max_mental_len
        self.samples = []

        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Handle lines with multiple concatenated JSON objects
                try:
                    episode = json.loads(line)
                    self._process_episode(episode)
                except json.JSONDecodeError:
                    # Split on }{ boundary and reconstruct
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
                            print(f"Warning: skipping malformed JSON segment")
        print(f"Created {len(self.samples)} per-turn training samples.")

    def _process_episode(self, episode: dict):
        pe = episode["parsed_episode"]
        turn_rewards = episode.get("turn_rewards", [])
        turns = pe.get("turns", [])

        if not turns or not turn_rewards:
            return

        scenario = pe.get("scenario", "")
        agent_1_name = pe.get("agent_1_name", "Agent 1")
        agent_2_name = pe.get("agent_2_name", "Agent 2")
        agent_1_bg = pe.get("agent_1_background", "")
        agent_2_bg = pe.get("agent_2_background", "")
        agent_1_goal = pe.get("agent_1_goal", "")
        agent_2_goal = pe.get("agent_2_goal", "")

        for t_idx, tr in enumerate(turn_rewards):
            turn_num = tr["turn"]
            speaker = tr["agent"]

            # Determine which agent is speaking
            if speaker == agent_1_name:
                agent_idx = 1
                rewards_key = "agent_1_rewards"
                bg = agent_1_bg
                goal = agent_1_goal
                secret = pe.get("agent_1_secret", "")
            elif speaker == agent_2_name:
                agent_idx = 2
                rewards_key = "agent_2_rewards"
                bg = agent_2_bg
                goal = agent_2_goal
                secret = pe.get("agent_2_secret", "")
            else:
                continue

            agent_rewards = tr.get(rewards_key, {})
            if not agent_rewards:
                continue

            # Extract 3-dim reward vector (normalized to [0,1])
            reward_vec = []
            reward_explanations = []
            for dim in SOTOPIA_DIMENSIONS:
                dim_data = agent_rewards.get(dim, {})
                if isinstance(dim_data, dict):
                    score = dim_data.get("score", 0)
                    reasoning = dim_data.get("reasoning", "")
                else:
                    score = 0
                    reasoning = ""
                reward_vec.append(normalize_score(dim, score))
                if reasoning:
                    reward_explanations.append(f"{dim}: {reasoning}")

            # Extract mental state
            mental_state = agent_rewards.get("mental_state", {})
            partner_belief = mental_state.get("partner_belief", "")
            strategic_intent = mental_state.get("strategic_intent", "")
            thought_process = mental_state.get("thought_process", "")
            hard_negative = mental_state.get("hard_negative_response", "")

            # Get actual utterance (positive response)
            if turn_num < len(turns):
                pos_response = turns[turn_num].get("content", "")
                action_type = turns[turn_num].get("action", "said")
            else:
                continue

            if not pos_response.strip():
                continue

            # Build dialogue history up to this turn
            history_lines = []
            for prev_t in turns[:turn_num]:
                spk = prev_t.get("agent", "Unknown")
                act = prev_t.get("action", "said")
                content = prev_t.get("content", "")
                history_lines.append(f"Turn {prev_t['turn']+1} | {spk} {act}: {content}")

            # Format mental state text for auxiliary supervision
            mental_text_parts = []
            if partner_belief:
                mental_text_parts.append(f"Partner Belief: {partner_belief}")
            if strategic_intent:
                mental_text_parts.append(f"Strategic Intent: {strategic_intent}")
            if thought_process:
                mental_text_parts.append(f"Thought Process: {thought_process}")
            mental_text = " | ".join(mental_text_parts)

            # Clean hard negative
            if hard_negative:
                hard_negative = hard_negative.strip().strip('"').strip("'")
                # Remove parenthetical explanation if present
                if "(" in hard_negative:
                    hard_negative = hard_negative[:hard_negative.rfind("(")].strip()

            self.samples.append({
                "scenario": scenario,
                "speaker": speaker,
                "agent_background": bg,
                "goal": goal,
                "secret": secret,
                "history": "\n".join(history_lines),
                "pos_response": pos_response,
                "hard_negative": hard_negative if hard_negative else "",
                "reward_vec": reward_vec,
                "mental_text": mental_text,
                "reward_explanations": "\n".join(reward_explanations),
                "turn_num": turn_num,
            })

    def _format_context(self, sample: dict) -> str:
        """Format context in a way consistent with SOTOPIA agent prompt."""
        secret_text = sample["secret"] if sample["secret"] else "None"
        ctx = (
            f"Scenario: {sample['scenario']}\n"
            f"Background: {sample['agent_background']}\n"
            f"Goal: {sample['goal']}\n"
            f"Secret: {secret_text}\n"
            f"Dialogue History:\n{sample['history']}\n"
            f"Turn {sample['turn_num']+1} | {sample['speaker']}:"
        )
        return ctx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context_text = self._format_context(sample)
        pos_text = sample["pos_response"]
        neg_text = sample["hard_negative"]
        mental_text = sample["mental_text"]

        # Tokenize context
        ctx_enc = self.tokenizer(
            context_text, truncation=True, max_length=self.max_ctx_len,
            padding="max_length", return_tensors="pt",
        )

        # Tokenize positive response
        pos_enc = self.tokenizer(
            pos_text, truncation=True, max_length=self.max_resp_len,
            padding="max_length", return_tensors="pt",
        )

        # Tokenize hard negative (if available)
        if neg_text:
            neg_enc = self.tokenizer(
                neg_text, truncation=True, max_length=self.max_resp_len,
                padding="max_length", return_tensors="pt",
            )
        else:
            neg_enc = pos_enc  # fallback: will be masked out in loss

        # Tokenize mental state text for auxiliary loss
        mental_enc = self.tokenizer(
            mental_text if mental_text else "N/A",
            truncation=True, max_length=self.max_mental_len,
            padding="max_length", return_tensors="pt",
        )

        # Tokenize reward explanations as auxiliary text
        reward_expl = sample["reward_explanations"]
        expl_enc = self.tokenizer(
            reward_expl if reward_expl else "N/A",
            truncation=True, max_length=self.max_mental_len,
            padding="max_length", return_tensors="pt",
        )

        return {
            "ctx_input_ids": ctx_enc.input_ids.squeeze(0),
            "ctx_attention_mask": ctx_enc.attention_mask.squeeze(0),
            "pos_input_ids": pos_enc.input_ids.squeeze(0),
            "pos_attention_mask": pos_enc.attention_mask.squeeze(0),
            "neg_input_ids": neg_enc.input_ids.squeeze(0),
            "neg_attention_mask": neg_enc.attention_mask.squeeze(0),
            "mental_input_ids": mental_enc.input_ids.squeeze(0),
            "mental_attention_mask": mental_enc.attention_mask.squeeze(0),
            "expl_input_ids": expl_enc.input_ids.squeeze(0),
            "expl_attention_mask": expl_enc.attention_mask.squeeze(0),
            "reward_vec": torch.tensor(sample["reward_vec"], dtype=torch.float32),
            "has_negative": torch.tensor(1.0 if neg_text else 0.0),
        }


def collate_fn(batch, tokenizer):
    """Collate with SEPARATE context and response tokens (no combined sequences).

    forward_reward now takes ctx and resp separately to prevent response_hidden
    from encoding context info via causal self-attention (which would let the
    reward head bypass z).
    """
    pad_id = tokenizer.pad_token_id

    rewards_list = []
    has_neg_list = []
    # We also need the first response token for future_loss (next-token prediction)
    first_pos_token_list = []

    for item in batch:
        rewards_list.append(item["reward_vec"])
        has_neg_list.append(item["has_negative"])
        # First real token of the positive response (for future prediction loss)
        pos_len = int(item["pos_attention_mask"].sum().item())
        first_pos_token_list.append(item["pos_input_ids"][0] if pos_len > 0 else item["pos_input_ids"][0])

    def _pad(key):
        return nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=pad_id
        )

    def _pad_mask(key):
        return nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=0
        )

    return {
        # Context tokens (used by forward_reward, forward_next_token, forward_mental_decode, forward_expl_reward)
        "ctx_input_ids": _pad("ctx_input_ids").long(),
        "ctx_attention_mask": _pad_mask("ctx_attention_mask").long(),
        # Positive response tokens (encoded in isolation by forward_reward)
        "pos_input_ids": _pad("pos_input_ids").long(),
        "pos_attention_mask": _pad_mask("pos_attention_mask").long(),
        # Negative response tokens (encoded in isolation by forward_reward)
        "neg_input_ids": _pad("neg_input_ids").long(),
        "neg_attention_mask": _pad_mask("neg_attention_mask").long(),
        # Mental state text tokens (for z-bottlenecked generation)
        "mental_input_ids": _pad("mental_input_ids").long(),
        "mental_attention_mask": _pad_mask("mental_attention_mask").long(),
        # Reward explanation tokens (for cross-attention from z)
        "expl_input_ids": _pad("expl_input_ids").long(),
        "expl_attention_mask": _pad_mask("expl_attention_mask").long(),
        # First positive response token (target for next-token/future prediction)
        "first_pos_token": torch.stack(first_pos_token_list, dim=0).long(),
        "reward_vec": torch.stack(rewards_list, dim=0),
        "has_negative": torch.stack(has_neg_list, dim=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mental Pre-Warmup Dataset (Persona Data)
# ──────────────────────────────────────────────────────────────────────────────
class MentalPrewarmDataset(Dataset):
    """
    Loads persona-based mental reasoning data for pre-warming the z-encoder
    and mental decoder before main training.

    Each sample has: persona + context + observation -> mental reasoning text
    (thought, goal, belief_state). No reward scores or responses needed.
    """

    def __init__(self, data_path: str, tokenizer, max_ctx_len: int = 1024,
                 max_mental_len: int = 256):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_mental_len = max_mental_len
        self.samples = []

        print(f"Loading mental pre-warmup data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                persona = entry.get("persona", "")
                context = entry.get("context", "")
                observation = entry.get("observation", "")
                mental = entry.get("target_mental_reasoning", {})

                thought = mental.get("thought", "")
                goal = mental.get("goal", "")
                belief = mental.get("belief_state", "")

                # Format context similar to SOTOPIA style
                ctx = (
                    f"Persona: {persona}\n"
                    f"Context: {context}\n"
                    f"Observation: {observation}"
                )

                # Format mental text to match stage 1's mental text format
                mental_parts = []
                if belief:
                    mental_parts.append(f"Partner Belief: {belief}")
                if goal:
                    mental_parts.append(f"Strategic Intent: {goal}")
                if thought:
                    mental_parts.append(f"Thought Process: {thought}")
                mental_text = " | ".join(mental_parts)

                if not mental_text.strip():
                    continue

                self.samples.append({
                    "context": ctx,
                    "mental_text": mental_text,
                })

        print(f"Loaded {len(self.samples)} mental pre-warmup samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        ctx_enc = self.tokenizer(
            sample["context"], truncation=True, max_length=self.max_ctx_len,
            padding="max_length", return_tensors="pt",
        )
        mental_enc = self.tokenizer(
            sample["mental_text"], truncation=True, max_length=self.max_mental_len,
            padding="max_length", return_tensors="pt",
        )

        return {
            "ctx_input_ids": ctx_enc.input_ids.squeeze(0),
            "ctx_attention_mask": ctx_enc.attention_mask.squeeze(0),
            "mental_input_ids": mental_enc.input_ids.squeeze(0),
            "mental_attention_mask": mental_enc.attention_mask.squeeze(0),
        }


def run_mental_prewarm(model, dataset, device, num_epochs=3, lr=2e-4,
                       batch_size=4, kl_weight=0.05, max_grad_norm=5.0):
    """
    Pre-warm the z-encoder + mental decoder on persona mental reasoning data.

    Only trains: LoRA (z-encoder pathway), context_mu, context_logvar,
    z_belief/intent/thought_to_prefix, mental_cross_attn, mental_decode_*.
    """
    print(f"\n=== Mental Pre-Warmup Phase ({num_epochs} epochs, {len(dataset)} samples) ===")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Only optimize params relevant to mental decoding
    mental_head_names = [
        "context_mu", "context_logvar",
        "z_belief_to_prefix", "z_intent_to_prefix", "z_thought_to_prefix",
        "mental_cross_attn", "mental_decode_ln", "mental_decode_ffn",
        "mental_decode_ln2",
    ]
    prewarm_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name or any(k in name for k in mental_head_names):
            prewarm_params.append(param)

    print(f"  Pre-warmup params: {sum(p.numel() for p in prewarm_params):,}")

    optimizer = torch.optim.AdamW(prewarm_params, lr=lr, weight_decay=0.01)

    model.train()
    for epoch in range(num_epochs):
        total_mental_loss = 0
        total_kl_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            ctx_ids = batch["ctx_input_ids"].to(device)
            ctx_mask = batch["ctx_attention_mask"].to(device)
            mental_ids = batch["mental_input_ids"].to(device)
            mental_mask = batch["mental_attention_mask"].to(device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                mental_gen_loss, mu, logvar = model.forward_mental_decode(
                    ctx_ids, ctx_mask, mental_ids, mental_mask,
                )

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    kl_loss = compute_kl_loss(mu, logvar)

                loss = mental_gen_loss + kl_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(prewarm_params, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            total_mental_loss += mental_gen_loss.item()
            total_kl_loss += kl_loss.item()

        n = len(dataloader)
        print(f"  Pre-warmup Epoch {epoch+1}/{num_epochs}: "
              f"mental_gen={total_mental_loss/n:.4f} kl={total_kl_loss/n:.4f}")

    print("=== Mental Pre-Warmup Complete ===\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def compute_kl_loss(mu, logvar):
    """KL divergence D_KL(q(z|x) || N(0,I))."""
    mu32 = mu.float()
    logvar32 = logvar.float().clamp(-10.0, 10.0)
    kl_per_dim = 1 + logvar32 - mu32.pow(2) - logvar32.exp()
    return -0.5 * torch.mean(kl_per_dim)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                kl_weight=0.1, future_weight=0.5, mental_weight=0.3,
                expl_weight=0.3, z_only_weight=0.5, grad_accum_steps=1,
                kl_anneal_steps=200, global_step_offset=0,
                max_grad_norm=5.0):
    """
    Args:
        kl_anneal_steps: number of optimizer steps over which kl_weight ramps
            linearly from 0 to its target value. Prevents KL from dominating
            early training when mu/logvar projections are randomly initialized.
        global_step_offset: cumulative optimizer steps from previous epochs.
    """
    model.train()
    total_loss = 0.0
    metrics = {"preference": 0, "reward_reg": 0, "z_only_reg": 0,
               "kl": 0, "future": 0, "mental_gen": 0, "expl_reward": 0}
    step = 0

    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        reward_targets = batch["reward_vec"].to(device)
        has_neg = batch["has_negative"].to(device)
        mental_ids = batch["mental_input_ids"].to(device)
        mental_mask = batch["mental_attention_mask"].to(device)
        expl_ids = batch["expl_input_ids"].to(device)
        expl_mask = batch["expl_attention_mask"].to(device)
        first_pos_token = batch["first_pos_token"].to(device)

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            # Unified forward: encodes context ONCE, batches pos/neg/expl into
            # a single transformer call. ~2 fwd passes instead of ~7.
            out = model.forward_all(
                ctx_ids, ctx_mask, pos_ids, pos_mask,
                neg_ids, neg_mask, mental_ids, mental_mask,
                expl_ids, expl_mask, first_pos_token,
            )

            pos_joint_reward = out["pos_joint_reward"]
            neg_joint_reward = out["neg_joint_reward"]
            pos_z_only_reward = out["pos_z_only_reward"]
            expl_reward_pred = out["expl_reward_pred"]
            pos_mu, pos_logvar = out["mu"], out["logvar"]
            future_loss = out["future_loss"]
            mental_gen_loss = out["mental_gen_loss"]

            expl_reward_loss = F.smooth_l1_loss(expl_reward_pred, reward_targets)

            # === Core Loss Components ===

            # Preference loss: pos reward should be higher than neg reward (joint head)
            pref_diff = neg_joint_reward - pos_joint_reward  # [batch, 3]
            pref_loss_per_sample = F.softplus(pref_diff).mean(dim=1)  # [batch]
            pref_loss = (pref_loss_per_sample * has_neg).sum() / (has_neg.sum() + 1e-8)

            # Reward regression loss (joint head)
            reward_reg_loss = F.smooth_l1_loss(pos_joint_reward, reward_targets)

            # z-only reward regression
            z_only_reg_loss = F.smooth_l1_loss(pos_z_only_reward, reward_targets)

            # KL divergence
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kl_loss = compute_kl_loss(pos_mu, pos_logvar)

            # KL annealing: linearly ramp kl_weight from 0 to target
            current_opt_step = global_step_offset + (batch_idx + 1) // grad_accum_steps
            if kl_anneal_steps > 0:
                anneal_factor = min(1.0, current_opt_step / kl_anneal_steps)
            else:
                anneal_factor = 1.0
            effective_kl_weight = kl_weight * anneal_factor

            # Total loss
            loss = (
                pref_loss                              # hard_negative preference
                + reward_reg_loss                      # joint head regression
                + z_only_weight * z_only_reg_loss      # z-only head regression (anti-bypass)
                + effective_kl_weight * kl_loss        # regularizes z (annealed)
                + future_weight * future_loss          # z predicts next response token
                + mental_weight * mental_gen_loss       # z decodes partner beliefs/thoughts
                + expl_weight * expl_reward_loss       # z + reasoning text -> reward
            )

            loss = loss / grad_accum_steps

        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        total_loss += loss.item() * grad_accum_steps
        metrics["preference"] += pref_loss.item()
        metrics["reward_reg"] += reward_reg_loss.item()
        metrics["z_only_reg"] += z_only_reg_loss.item()
        metrics["kl"] += kl_loss.item()
        metrics["future"] += future_loss.item()
        metrics["mental_gen"] += mental_gen_loss.item()
        metrics["expl_reward"] += expl_reward_loss.item()

        if (batch_idx + 1) % 10 == 0:
            n = batch_idx + 1
            print(
                f"  Epoch {epoch+1} Step {n}: "
                f"loss={total_loss/n:.4f} pref={metrics['preference']/n:.4f} "
                f"joint_reg={metrics['reward_reg']/n:.4f} z_only={metrics['z_only_reg']/n:.4f} "
                f"kl={metrics['kl']/n:.4f}(w={effective_kl_weight:.4f}) "
                f"future={metrics['future']/n:.4f} "
                f"mental_gen={metrics['mental_gen']/n:.4f} "
                f"expl_rwd={metrics['expl_reward']/n:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    n = len(dataloader)
    for k in metrics:
        metrics[k] /= n
    # Return updated global step count for KL annealing across epochs
    final_global_step = global_step_offset + num_batches // grad_accum_steps
    return total_loss / n, metrics, final_global_step


CUSTOM_HEAD_NAMES = [
    "context_mu", "context_logvar", "joint_outcome_head", "z_only_reward_head",
    "z_to_hidden",
    "z_belief_to_prefix", "z_intent_to_prefix", "z_thought_to_prefix",
    "mental_cross_attn", "mental_decode_ln", "mental_decode_ffn",
    "mental_decode_ln2", "expl_cross_attn", "expl_reward_head",
]


def _save_checkpoint(model, save_dir):
    """Save LoRA adapter + all custom heads."""
    os.makedirs(save_dir, exist_ok=True)
    model.base_model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    for head_name in CUSTOM_HEAD_NAMES:
        head = getattr(model, head_name)
        torch.save(head.state_dict(), os.path.join(save_dir, f"{head_name}.pth"))


def main():
    parser = argparse.ArgumentParser(description="Stage 1 v2: Train Coupled Mental+Reward Model (3-dim, stronger LoRA)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str,
                        default="projects/sotopia/data/sotopia_turn_rewards.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="projects/sotopia/checkpoints/coupled_mental_reward_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16,
                        help="Number of top transformer layers to apply LoRA to")
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--future_weight", type=float, default=0.5)
    parser.add_argument("--mental_weight", type=float, default=0.3)
    parser.add_argument("--expl_weight", type=float, default=0.3)
    parser.add_argument("--z_only_weight", type=float, default=0.5,
                        help="Weight for z-only reward head loss (anti-bypass)")
    parser.add_argument("--kl_anneal_steps", type=int, default=200,
                        help="Optimizer steps to linearly ramp kl_weight from 0 to target")
    parser.add_argument("--mental_prewarm_data", type=str, default=None,
                        help="Path to persona mental reasoning JSONL for pre-warming z-encoder + mental decoder")
    parser.add_argument("--mental_prewarm_epochs", type=int, default=3,
                        help="Epochs for mental pre-warmup phase")
    parser.add_argument("--mental_prewarm_lr", type=float, default=2e-4,
                        help="Learning rate for mental pre-warmup")
    parser.add_argument("--head_lr_mult", type=float, default=10.0,
                        help="Multiplier for custom head learning rate vs LoRA lr")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="7")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base Model + LoRA ──
    print(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

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

    # Freeze all base params except LoRA
    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_" in name

    base_model.print_trainable_parameters()
    base_model.config.use_cache = False

    # ── Combined Model ──
    model = CoupledMentalRewardModel(
        base_model, reward_dim=REWARD_DIM, z_dim=args.z_dim
    ).to(device)

    # Ensure custom heads are trainable
    for name, param in model.named_parameters():
        if any(k in name for k in CUSTOM_HEAD_NAMES):
            param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    # ── Mental Pre-Warmup (optional) ──
    if args.mental_prewarm_data:
        prewarm_dataset = MentalPrewarmDataset(
            args.mental_prewarm_data, tokenizer,
            max_ctx_len=args.max_ctx_len,
        )
        model = run_mental_prewarm(
            model, prewarm_dataset, device,
            num_epochs=args.mental_prewarm_epochs,
            lr=args.mental_prewarm_lr,
            batch_size=args.batch_size,
        )
        del prewarm_dataset
        gc.collect()
        torch.cuda.empty_cache()

    # ── Dataset ──
    dataset = CoupledMentalRewardDataset(
        args.data_path, tokenizer,
        max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Optimizer + Scheduler ──
    # Separate param groups: custom heads train from scratch and need higher lr
    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in CUSTOM_HEAD_NAMES):
            head_params.append(param)
        else:
            lora_params.append(param)

    head_lr = args.lr * args.head_lr_mult
    print(f"Param groups: LoRA lr={args.lr}, Head lr={head_lr} ({len(lora_params)} LoRA, {len(head_params)} head tensors)")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
    ])
    total_steps = len(dataloader) * args.num_epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training ──
    best_loss = float("inf")
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss, metrics, global_step = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch,
            kl_weight=args.kl_weight, future_weight=args.future_weight,
            mental_weight=args.mental_weight, expl_weight=args.expl_weight,
            z_only_weight=args.z_only_weight,
            grad_accum_steps=args.grad_accum_steps,
            kl_anneal_steps=args.kl_anneal_steps,
            global_step_offset=global_step,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"\nEpoch {epoch+1}/{args.num_epochs}: avg_loss={avg_loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save checkpoint
        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        _save_checkpoint(model, ckpt_dir)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = os.path.join(args.output_dir, "best")
            _save_checkpoint(model, best_dir)
            print(f"  -> Best model saved (loss={best_loss:.4f})")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nStage 1 v2 training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
