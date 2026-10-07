"""
Stage 1: Train Coupled Mental-State + Reward Model (v3 — Recursive ToM)
========================================================================
Changes from v2:
  - Recursive z encoding: z1 (1st-order ToM) + z2 (2nd-order ToM)
  - z2 is conditioned on [hidden_state || z1], encoding "what does the
    partner think I believe/intend/think?"
  - Same single LLM backbone, two z-spaces with separate projection heads
  - Joint reward head: [z1 || z2 || response_hidden] -> 7-dim reward
  - Stability: stop-gradient on z1 for first z2_warmup_steps,
    delayed KL annealing for z2 (starts after z1's KL is annealed)

Uses per-turn reward data from sotopia_turn_rewards_v3.jsonl containing:
  - Per-turn 7-dim reward scores with reasoning explanations
  - 1st-order mental state: partner_belief, strategic_intent, thought_process
  - 2nd-order mental state: second_order_belief, second_order_intent,
    second_order_thought
  - Hard negative responses (socially poor alternatives)

Architecture:
  Base LLM with LoRA
  + z1 (128-dim) via VAE: 3 structured sub-spaces for 1st-order ToM
      z1_belief (48d) | z1_intent (40d) | z1_thought (40d)
  + z2 (128-dim) via VAE conditioned on z1: 3 sub-spaces for 2nd-order ToM
      z2_belief (48d) | z2_intent (40d) | z2_thought (40d)
  + Joint outcome head: [z1 || z2 || response_hidden] -> 7-dim reward
  + z-bottlenecked mental decoders (one for z1, one for z2)
  + Explanation-conditioned reward: [z1 || z2] cross-attends to reasoning
  + Contrastive loss: positive utterance > hard_negative in reward space

Loss = L_preference + L_reward_regression + L_z_only_reward (anti-bypass)
       + L_kl1 + L_kl2 + L_future
       + L_mental_gen_1st (z1-bottlenecked)
       + L_mental_gen_2nd (z2-bottlenecked)
       + L_expl_reward (cross-attention)
"""

import os
import sys
import json
import gc
import math
import argparse
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resume_state import (  # noqa: E402
    check_resume_args, epoch_order, load_lora_weights, load_resume_state,
    require_resume_dir, restore_rng_state, save_resume_dir,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SOTOPIA_DIMENSIONS = [
    "believability", "relationship", "knowledge", "secret",
    "social_rules", "financial_and_material_benefits", "goal",
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

REWARD_DIM = len(SOTOPIA_DIMENSIONS)


def normalize_score(dim: str, score: float) -> float:
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


def _build_mental_decoder(hidden_size, z_dim, z_sub_dims, num_prefix):
    """Build a set of z-bottlenecked mental decoder components.

    Returns a dict of nn.Modules for: z_to_prefix (3 sub-spaces),
    cross_attn, ln, ffn, ln2.
    """
    z_belief_dim, z_intent_dim, z_thought_dim = z_sub_dims
    return nn.ModuleDict({
        "z_belief_to_prefix": nn.Linear(z_belief_dim, num_prefix * hidden_size),
        "z_intent_to_prefix": nn.Linear(z_intent_dim, num_prefix * hidden_size),
        "z_thought_to_prefix": nn.Linear(z_thought_dim, num_prefix * hidden_size),
        "cross_attn": nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8,
            kdim=hidden_size, vdim=hidden_size,
            batch_first=True, dropout=0.1,
        ),
        "ln": nn.LayerNorm(hidden_size),
        "ffn": nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        ),
        "ln2": nn.LayerNorm(hidden_size),
    })


class RecursiveToMModel(nn.Module):
    """
    Coupled Mental-State + Reward model with Recursive Theory of Mind.

    Two latent spaces from the same LLM backbone:
      z1: context_hidden -> mu1, logvar1 -> z1  (1st-order ToM)
      z2: [context_hidden || z1] -> mu2, logvar2 -> z2  (2nd-order ToM)

    z2 is conditioned on z1, so 2nd-order beliefs are explicitly built
    on top of 1st-order beliefs, creating a recursive belief hierarchy.
    """

    Z_BELIEF_DIM = 48
    Z_INTENT_DIM = 40
    Z_THOUGHT_DIM = 40
    NUM_PREFIX_TOKENS = 8

    def __init__(self, base_model: nn.Module, reward_dim: int = REWARD_DIM, z_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.transformer = get_transformer_from_peft(self.base_model)
        hidden_size = base_model.get_input_embeddings().embedding_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        assert z_dim == self.Z_BELIEF_DIM + self.Z_INTENT_DIM + self.Z_THOUGHT_DIM

        # ── z1: 1st-order ToM (context -> z1) ──
        self.z1_mu = nn.Linear(hidden_size, z_dim)
        self.z1_logvar = nn.Linear(hidden_size, z_dim)

        # ── z2: 2nd-order ToM (context || z1 -> z2) ──
        # z2 is conditioned on BOTH context hidden states and z1
        self.z2_mu = nn.Linear(hidden_size + z_dim, z_dim)
        self.z2_logvar = nn.Linear(hidden_size + z_dim, z_dim)

        # ── Joint outcome head: [z1 || z2 || response_hidden] -> reward ──
        self.joint_outcome_head = nn.Sequential(
            nn.Linear(2 * z_dim + hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )

        # ── z-only reward heads (anti-bypass) ──
        # z1-only: forces z1 to carry reward info independently
        self.z1_only_reward_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, reward_dim),
        )
        # z1+z2 combined (no response): forces both z's to jointly encode reward
        self.z_combined_reward_head = nn.Sequential(
            nn.Linear(2 * z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, reward_dim),
        )

        # ── z1 -> hidden for next-token prediction ──
        self.z_to_hidden = nn.Linear(2 * z_dim, hidden_size)

        # ── Mental decoder for 1st-order (z1-bottlenecked) ──
        z_sub_dims = (self.Z_BELIEF_DIM, self.Z_INTENT_DIM, self.Z_THOUGHT_DIM)
        self.mental1_decoder = _build_mental_decoder(
            hidden_size, z_dim, z_sub_dims, self.NUM_PREFIX_TOKENS
        )

        # ── Mental decoder for 2nd-order (z2-bottlenecked) ──
        self.mental2_decoder = _build_mental_decoder(
            hidden_size, z_dim, z_sub_dims, self.NUM_PREFIX_TOKENS
        )

        # ── Explanation-conditioned reward ──
        # Query is [z1 || z2], cross-attends to explanation hidden states
        self.expl_cross_attn = nn.MultiheadAttention(
            embed_dim=2 * z_dim, num_heads=8, kdim=hidden_size, vdim=hidden_size,
            batch_first=True, dropout=0.1,
        )
        self.expl_reward_head = nn.Sequential(
            nn.Linear(2 * z_dim, 128),
            nn.GELU(),
            nn.Linear(128, reward_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.z1_mu, self.z1_logvar, self.z2_mu, self.z2_logvar]:
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.z1_logvar.bias, -2.0)
        nn.init.constant_(self.z2_logvar.bias, -2.0)

        for module in [self.z_to_hidden]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for decoder in [self.mental1_decoder, self.mental2_decoder]:
            for key in ["z_belief_to_prefix", "z_intent_to_prefix", "z_thought_to_prefix"]:
                nn.init.xavier_uniform_(decoder[key].weight)
                nn.init.zeros_(decoder[key].bias)
            for layer in decoder["ffn"]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        for layer_group in [self.joint_outcome_head, self.z1_only_reward_head,
                            self.z_combined_reward_head, self.expl_reward_head]:
            for layer in layer_group:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ── Encoding helpers ──
    def _encode_context(self, input_ids, attention_mask):
        out = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        final_hidden = out.last_hidden_state
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, final_hidden.size(-1))
            return final_hidden.gather(1, last_exp).squeeze(1)
        return final_hidden[:, -1, :]

    def _sample_z(self, mu_proj, logvar_proj, h):
        mu = mu_proj(h)
        logvar = logvar_proj(h).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar).clamp(min=1e-8)
        eps = torch.randn_like(std, dtype=torch.float32)
        z = mu + std * eps
        return z, mu, logvar

    def encode_z1_z2(self, ctx_input_ids, ctx_attention_mask,
                     stop_grad_z1: bool = False):
        """Encode context -> z1 -> z2 (recursive)."""
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)

        z1, mu1, logvar1 = self._sample_z(self.z1_mu, self.z1_logvar, context_hidden)

        # z2 is conditioned on context_hidden and z1
        z1_for_z2 = z1.detach() if stop_grad_z1 else z1
        z2_input = torch.cat([context_hidden, z1_for_z2], dim=1)
        z2, mu2, logvar2 = self._sample_z(self.z2_mu, self.z2_logvar, z2_input)

        return context_hidden, z1, mu1, logvar1, z2, mu2, logvar2

    def encode_z1_z2_deterministic(self, ctx_input_ids, ctx_attention_mask):
        """Deterministic encoding (mu only) for inference."""
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        mu1 = self.z1_mu(context_hidden)
        z2_input = torch.cat([context_hidden, mu1], dim=1)
        mu2 = self.z2_mu(z2_input)
        return mu1, mu2

    # ── Mental text decoding (z-bottlenecked) ──
    def _expand_z_to_prefix(self, z, decoder):
        batch_size = z.size(0)
        z_b = z[:, :self.Z_BELIEF_DIM]
        z_i = z[:, self.Z_BELIEF_DIM:self.Z_BELIEF_DIM + self.Z_INTENT_DIM]
        z_t = z[:, self.Z_BELIEF_DIM + self.Z_INTENT_DIM:]

        prefix_b = decoder["z_belief_to_prefix"](z_b).view(batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size)
        prefix_i = decoder["z_intent_to_prefix"](z_i).view(batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size)
        prefix_t = decoder["z_thought_to_prefix"](z_t).view(batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size)
        return torch.cat([prefix_b, prefix_i, prefix_t], dim=1)

    def _decode_mental(self, z, decoder, mental_input_ids, mental_attention_mask):
        """Z-bottlenecked mental text generation loss."""
        z_prefix = self._expand_z_to_prefix(z, decoder)

        embedding_layer = self.base_model.get_input_embeddings()
        mental_embeds = embedding_layer(mental_input_ids)

        attended, _ = decoder["cross_attn"](
            query=mental_embeds, key=z_prefix, value=z_prefix,
        )
        h = decoder["ln"](mental_embeds + attended)
        h = h + decoder["ffn"](h)
        h = decoder["ln2"](h)

        output_embedding = self.base_model.get_output_embeddings()
        if hasattr(output_embedding, 'weight'):
            logits = F.linear(
                h, output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, 'bias') and output_embedding.bias is not None else None
            )
        else:
            logits = h

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = mental_input_ids[:, 1:].clone().contiguous()
        shift_mask = mental_attention_mask[:, 1:].contiguous()
        shift_labels[shift_mask == 0] = -100

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

    # ── Inference methods ──
    def forward_reward_with_z(self, z1, z2, resp_input_ids, resp_attention_mask):
        """Reward prediction given pre-computed z1, z2 and isolated response."""
        resp_out = self.transformer(
            input_ids=resp_input_ids, attention_mask=resp_attention_mask,
            use_cache=False, return_dict=True,
        )
        resp_hidden = resp_out.last_hidden_state
        if resp_attention_mask is not None:
            last_idx = resp_attention_mask.sum(dim=1) - 1
            last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, resp_hidden.size(-1))
            response_hidden = resp_hidden.gather(1, last_exp).squeeze(1)
        else:
            response_hidden = resp_hidden[:, -1, :]

        joint_input = torch.cat([z1, z2, response_hidden], dim=1)
        return self.joint_outcome_head(joint_input)

    def predict_reward(self, ctx_input_ids, ctx_attention_mask,
                       resp_input_ids, resp_attention_mask):
        """Inference-time reward prediction (no grad). Returns [batch, reward_dim]."""
        with torch.no_grad():
            z1, z2 = self.encode_z1_z2_deterministic(ctx_input_ids, ctx_attention_mask)
            reward = self.forward_reward_with_z(z1, z2, resp_input_ids, resp_attention_mask)
        return reward

    # ── Unified forward (training) ──
    def forward_all(self, ctx_input_ids, ctx_attention_mask,
                    pos_input_ids, pos_attention_mask,
                    neg_input_ids, neg_attention_mask,
                    mental1_input_ids, mental1_attention_mask,
                    mental2_input_ids, mental2_attention_mask,
                    expl_input_ids, expl_attention_mask,
                    first_pos_token,
                    stop_grad_z1: bool = False):
        """
        Unified forward pass: encodes context ONCE, batches isolated encodings.

        Returns dict with all loss components.
        """
        batch_size = ctx_input_ids.size(0)

        # ── 1. Encode context ONCE → z1, z2 ──
        context_hidden, z1, mu1, logvar1, z2, mu2, logvar2 = \
            self.encode_z1_z2(ctx_input_ids, ctx_attention_mask, stop_grad_z1=stop_grad_z1)

        # ── 2. Batch pos + neg + expl into ONE transformer forward ──
        max_resp_len = max(pos_input_ids.size(1), neg_input_ids.size(1))
        max_expl_len = expl_input_ids.size(1)
        max_isolated_len = max(max_resp_len, max_expl_len)

        def _pad_to(ids, mask, target_len):
            pad_len = target_len - ids.size(1)
            if pad_len > 0:
                pad_val = self.base_model.config.pad_token_id \
                    if hasattr(self.base_model.config, 'pad_token_id') and self.base_model.config.pad_token_id is not None \
                    else 0
                ids = F.pad(ids, (0, pad_len), value=pad_val)
                mask = F.pad(mask, (0, pad_len), value=0)
            return ids, mask

        pos_ids_p, pos_mask_p = _pad_to(pos_input_ids, pos_attention_mask, max_isolated_len)
        neg_ids_p, neg_mask_p = _pad_to(neg_input_ids, neg_attention_mask, max_isolated_len)
        expl_ids_p, expl_mask_p = _pad_to(expl_input_ids, expl_attention_mask, max_isolated_len)

        batched_ids = torch.cat([pos_ids_p, neg_ids_p, expl_ids_p], dim=0)
        batched_mask = torch.cat([pos_mask_p, neg_mask_p, expl_mask_p], dim=0)

        batched_out = self.transformer(
            input_ids=batched_ids, attention_mask=batched_mask,
            use_cache=False, return_dict=True,
        )
        batched_hidden = batched_out.last_hidden_state

        def _extract_last(hidden, mask, start, end):
            h = hidden[start:end]
            m = mask[start:end]
            last_idx = m.sum(dim=1) - 1
            last_exp = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
            return h.gather(1, last_exp).squeeze(1)

        pos_resp_hidden = _extract_last(batched_hidden, batched_mask, 0, batch_size)
        neg_resp_hidden = _extract_last(batched_hidden, batched_mask, batch_size, 2 * batch_size)
        expl_hidden = batched_hidden[2 * batch_size:]

        # ── 3. Reward predictions ──
        z_cat = torch.cat([z1, z2], dim=1)  # [B, 2*z_dim]

        pos_joint_input = torch.cat([z1, z2, pos_resp_hidden], dim=1)
        pos_joint_reward = self.joint_outcome_head(pos_joint_input)

        neg_joint_input = torch.cat([z1, z2, neg_resp_hidden], dim=1)
        neg_joint_reward = self.joint_outcome_head(neg_joint_input)

        z1_only_reward = self.z1_only_reward_head(z1)
        z_combined_reward = self.z_combined_reward_head(z_cat)

        # ── 4. Next-token prediction from z1+z2 ──
        conditioned_hidden = context_hidden + self.z_to_hidden(z_cat)
        output_embedding = self.base_model.get_output_embeddings()
        if hasattr(output_embedding, 'weight'):
            next_logits = F.linear(
                conditioned_hidden, output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, 'bias') and output_embedding.bias is not None else None
            )
        else:
            next_logits = conditioned_hidden
        future_loss = F.cross_entropy(next_logits, first_pos_token)

        # ── 5. Mental text decode: 1st-order (z1-bottlenecked) ──
        mental1_gen_loss = self._decode_mental(
            z1, self.mental1_decoder, mental1_input_ids, mental1_attention_mask,
        )

        # ── 6. Mental text decode: 2nd-order (z2-bottlenecked) ──
        mental2_gen_loss = self._decode_mental(
            z2, self.mental2_decoder, mental2_input_ids, mental2_attention_mask,
        )

        # ── 7. Explanation-conditioned reward ──
        z_query = z_cat.unsqueeze(1)
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
            "z1_only_reward": z1_only_reward,
            "z_combined_reward": z_combined_reward,
            "expl_reward_pred": expl_reward_pred,
            "mu1": mu1, "logvar1": logvar1,
            "mu2": mu2, "logvar2": logvar2,
            "future_loss": future_loss,
            "mental1_gen_loss": mental1_gen_loss,
            "mental2_gen_loss": mental2_gen_loss,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class RecursiveToMDataset(Dataset):
    """
    Loads sotopia_turn_rewards_v3.jsonl with both 1st and 2nd order mental states.

    Each sample:
      - context, positive response, hard negative, reward vector
      - mental1_text: 1st-order (partner_belief | strategic_intent | thought_process)
      - mental2_text: 2nd-order (second_order_belief | second_order_intent | second_order_thought)
      - reward explanations
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

            if speaker == agent_1_name:
                rewards_key = "agent_1_rewards"
                bg = agent_1_bg
                goal = agent_1_goal
                secret = pe.get("agent_1_secret", "")
            elif speaker == agent_2_name:
                rewards_key = "agent_2_rewards"
                bg = agent_2_bg
                goal = agent_2_goal
                secret = pe.get("agent_2_secret", "")
            else:
                continue

            agent_rewards = tr.get(rewards_key, {})
            if not agent_rewards:
                continue

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

            mental_state = agent_rewards.get("mental_state", {})
            partner_belief = mental_state.get("partner_belief", "")
            strategic_intent = mental_state.get("strategic_intent", "")
            thought_process = mental_state.get("thought_process", "")
            hard_negative = mental_state.get("hard_negative_response", "")

            # 2nd-order fields
            second_order_belief = mental_state.get("second_order_belief", "")
            second_order_intent = mental_state.get("second_order_intent", "")
            second_order_thought = mental_state.get("second_order_thought", "")

            if turn_num < len(turns):
                pos_response = turns[turn_num].get("content", "")
            else:
                continue

            if not pos_response.strip():
                continue

            # Build dialogue history
            history_lines = []
            for prev_t in turns[:turn_num]:
                spk = prev_t.get("agent", "Unknown")
                act = prev_t.get("action", "said")
                content = prev_t.get("content", "")
                history_lines.append(f"Turn {prev_t['turn']+1} | {spk} {act}: {content}")

            # 1st-order mental text
            mental1_parts = []
            if partner_belief:
                mental1_parts.append(f"Partner Belief: {partner_belief}")
            if strategic_intent:
                mental1_parts.append(f"Strategic Intent: {strategic_intent}")
            if thought_process:
                mental1_parts.append(f"Thought Process: {thought_process}")
            mental1_text = " | ".join(mental1_parts)

            # 2nd-order mental text
            mental2_parts = []
            if second_order_belief:
                mental2_parts.append(f"Second-Order Belief: {second_order_belief}")
            if second_order_intent:
                mental2_parts.append(f"Second-Order Intent: {second_order_intent}")
            if second_order_thought:
                mental2_parts.append(f"Second-Order Thought: {second_order_thought}")
            mental2_text = " | ".join(mental2_parts)

            # Clean hard negative
            if hard_negative:
                hard_negative = hard_negative.strip().strip('"').strip("'")
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
                "mental1_text": mental1_text,
                "mental2_text": mental2_text,
                "reward_explanations": "\n".join(reward_explanations),
                "turn_num": turn_num,
            })

    def _format_context(self, sample: dict) -> str:
        secret_text = sample["secret"] if sample["secret"] else "None"
        return (
            f"Scenario: {sample['scenario']}\n"
            f"Background: {sample['agent_background']}\n"
            f"Goal: {sample['goal']}\n"
            f"Secret: {secret_text}\n"
            f"Dialogue History:\n{sample['history']}\n"
            f"Turn {sample['turn_num']+1} | {sample['speaker']}:"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context_text = self._format_context(sample)

        ctx_enc = self.tokenizer(
            context_text, truncation=True, max_length=self.max_ctx_len,
            padding="max_length", return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            sample["pos_response"], truncation=True, max_length=self.max_resp_len,
            padding="max_length", return_tensors="pt",
        )
        neg_text = sample["hard_negative"]
        neg_enc = self.tokenizer(
            neg_text if neg_text else sample["pos_response"],
            truncation=True, max_length=self.max_resp_len,
            padding="max_length", return_tensors="pt",
        )
        mental1_enc = self.tokenizer(
            sample["mental1_text"] if sample["mental1_text"] else "N/A",
            truncation=True, max_length=self.max_mental_len,
            padding="max_length", return_tensors="pt",
        )
        mental2_enc = self.tokenizer(
            sample["mental2_text"] if sample["mental2_text"] else "N/A",
            truncation=True, max_length=self.max_mental_len,
            padding="max_length", return_tensors="pt",
        )
        expl_enc = self.tokenizer(
            sample["reward_explanations"] if sample["reward_explanations"] else "N/A",
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
            "mental1_input_ids": mental1_enc.input_ids.squeeze(0),
            "mental1_attention_mask": mental1_enc.attention_mask.squeeze(0),
            "mental2_input_ids": mental2_enc.input_ids.squeeze(0),
            "mental2_attention_mask": mental2_enc.attention_mask.squeeze(0),
            "expl_input_ids": expl_enc.input_ids.squeeze(0),
            "expl_attention_mask": expl_enc.attention_mask.squeeze(0),
            "reward_vec": torch.tensor(sample["reward_vec"], dtype=torch.float32),
            "has_negative": torch.tensor(1.0 if neg_text else 0.0),
        }


def collate_fn(batch, tokenizer):
    pad_id = tokenizer.pad_token_id
    rewards_list = []
    has_neg_list = []
    first_pos_token_list = []

    for item in batch:
        rewards_list.append(item["reward_vec"])
        has_neg_list.append(item["has_negative"])
        pos_len = int(item["pos_attention_mask"].sum().item())
        first_pos_token_list.append(item["pos_input_ids"][0])

    def _pad(key):
        return nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=pad_id)

    def _pad_mask(key):
        return nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=0)

    return {
        "ctx_input_ids": _pad("ctx_input_ids").long(),
        "ctx_attention_mask": _pad_mask("ctx_attention_mask").long(),
        "pos_input_ids": _pad("pos_input_ids").long(),
        "pos_attention_mask": _pad_mask("pos_attention_mask").long(),
        "neg_input_ids": _pad("neg_input_ids").long(),
        "neg_attention_mask": _pad_mask("neg_attention_mask").long(),
        "mental1_input_ids": _pad("mental1_input_ids").long(),
        "mental1_attention_mask": _pad_mask("mental1_attention_mask").long(),
        "mental2_input_ids": _pad("mental2_input_ids").long(),
        "mental2_attention_mask": _pad_mask("mental2_attention_mask").long(),
        "expl_input_ids": _pad("expl_input_ids").long(),
        "expl_attention_mask": _pad_mask("expl_attention_mask").long(),
        "first_pos_token": torch.stack(first_pos_token_list, dim=0).long(),
        "reward_vec": torch.stack(rewards_list, dim=0),
        "has_negative": torch.stack(has_neg_list, dim=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mental Pre-Warmup Dataset (Persona Data)
# ──────────────────────────────────────────────────────────────────────────────
class MentalPrewarmDataset(Dataset):
    """Persona-based mental reasoning data for pre-warming z1-encoder."""

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

                ctx = f"Persona: {persona}\nContext: {context}\nObservation: {observation}"

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

                self.samples.append({"context": ctx, "mental_text": mental_text})
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


def run_mental_prewarm(model, dataset, device, num_epochs=1, lr=2e-4,
                       batch_size=4, kl_weight=0.05, max_grad_norm=5.0):
    """Pre-warm z1-encoder + mental1 decoder on persona data."""
    print(f"\n=== Mental Pre-Warmup Phase ({num_epochs} epochs, {len(dataset)} samples) ===")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    prewarm_params = []
    prewarm_names = [
        "z1_mu", "z1_logvar",
        "mental1_decoder",
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name or any(k in name for k in prewarm_names):
            prewarm_params.append(param)

    print(f"  Pre-warmup params: {sum(p.numel() for p in prewarm_params):,}")
    optimizer = torch.optim.AdamW(prewarm_params, lr=lr, weight_decay=0.01)

    model.train()
    for epoch in range(num_epochs):
        total_mental = 0
        total_kl = 0
        for batch in dataloader:
            ctx_ids = batch["ctx_input_ids"].to(device)
            ctx_mask = batch["ctx_attention_mask"].to(device)
            mental_ids = batch["mental_input_ids"].to(device)
            mental_mask = batch["mental_attention_mask"].to(device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                context_hidden = model._encode_context(ctx_ids, ctx_mask)
                z1, mu1, logvar1 = model._sample_z(model.z1_mu, model.z1_logvar, context_hidden)

                mental_gen_loss = model._decode_mental(
                    z1, model.mental1_decoder, mental_ids, mental_mask,
                )
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    kl_loss = compute_kl_loss(mu1, logvar1)
                loss = mental_gen_loss + kl_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(prewarm_params, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            total_mental += mental_gen_loss.item()
            total_kl += kl_loss.item()

        n = len(dataloader)
        print(f"  Pre-warmup Epoch {epoch+1}/{num_epochs}: "
              f"mental_gen={total_mental/n:.4f} kl={total_kl/n:.4f}")

    print("=== Mental Pre-Warmup Complete ===\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def compute_kl_loss(mu, logvar):
    mu32 = mu.float()
    logvar32 = logvar.float().clamp(-10.0, 10.0)
    kl_per_dim = 1 + logvar32 - mu32.pow(2) - logvar32.exp()
    return -0.5 * torch.mean(kl_per_dim)


PREF_WEIGHT = float(os.environ.get("PREF_WEIGHT", "1.0"))  # ablation hook for hard-negative preference


def compute_objective_components(out, reward_targets, has_neg, current_opt_step,
                                 kl_weight=0.1, future_weight=0.5,
                                 mental1_weight=0.3, mental2_weight=0.3,
                                 expl_weight=0.3, z_only_weight=0.5,
                                 kl_anneal_steps=200, z2_kl_delay_steps=100):
    pos_joint_reward = out["pos_joint_reward"]
    neg_joint_reward = out["neg_joint_reward"]
    z1_only_reward = out["z1_only_reward"]
    z_combined_reward = out["z_combined_reward"]
    expl_reward_pred = out["expl_reward_pred"]

    pref_diff = neg_joint_reward - pos_joint_reward
    pref_loss_per_sample = F.softplus(pref_diff).mean(dim=1)
    pref_loss = (pref_loss_per_sample * has_neg).sum() / (has_neg.sum() + 1e-8)

    reward_reg_loss = F.smooth_l1_loss(pos_joint_reward, reward_targets)
    z1_only_reg_loss = F.smooth_l1_loss(z1_only_reward, reward_targets)
    z_combined_reg_loss = F.smooth_l1_loss(z_combined_reward, reward_targets)
    expl_reward_loss = F.smooth_l1_loss(expl_reward_pred, reward_targets)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        kl1_loss = compute_kl_loss(out["mu1"], out["logvar1"])
        kl2_loss = compute_kl_loss(out["mu2"], out["logvar2"])

    if kl_anneal_steps > 0:
        anneal1 = min(1.0, current_opt_step / kl_anneal_steps)
        z2_start = kl_anneal_steps + z2_kl_delay_steps
        anneal2 = min(1.0, max(0.0, (current_opt_step - z2_start)) / kl_anneal_steps)
    else:
        anneal1 = 1.0
        anneal2 = 1.0

    eff_kl1_w = kl_weight * anneal1
    eff_kl2_w = kl_weight * anneal2

    m1_loss = out["mental1_gen_loss"]
    m2_loss = out["mental2_gen_loss"]
    future_loss = out["future_loss"]

    total_loss = (
        PREF_WEIGHT * pref_loss
        + reward_reg_loss
        + z_only_weight * z1_only_reg_loss
        + z_only_weight * z_combined_reg_loss
        + eff_kl1_w * kl1_loss
        + eff_kl2_w * kl2_loss
        + future_weight * future_loss
        + mental1_weight * m1_loss
        + mental2_weight * m2_loss
        + expl_weight * expl_reward_loss
    )

    metrics = {
        "preference": pref_loss.item(),
        "reward_reg": reward_reg_loss.item(),
        "z1_only_reg": z1_only_reg_loss.item(),
        "z_combined_reg": z_combined_reg_loss.item(),
        "kl1": kl1_loss.item(),
        "kl2": kl2_loss.item(),
        "future": future_loss.item(),
        "mental1_gen": m1_loss.item(),
        "mental2_gen": m2_loss.item(),
        "expl_reward": expl_reward_loss.item(),
    }

    return total_loss, metrics, eff_kl1_w, eff_kl2_w


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                kl_weight=0.1, future_weight=0.5,
                mental1_weight=0.3, mental2_weight=0.3,
                expl_weight=0.3, z_only_weight=0.5,
                grad_accum_steps=1,
                kl_anneal_steps=200, z2_kl_delay_steps=100,
                z2_warmup_steps=100,
                global_step_offset=0,
                max_grad_norm=5.0,
                start_batch=0,
                totals=None,
                on_optimizer_step=None):
    """
    Training epoch with recursive z support.

    z2_kl_delay_steps: z2's KL annealing starts this many steps AFTER z1's
    z2_warmup_steps: stop-gradient on z1→z2 for this many steps (lets z1
        stabilize before z2 learns from it)
    start_batch / totals: resume mid-epoch; dataloader then yields only the
        remaining batches and totals carries the running loss sums.
    on_optimizer_step: called as (next_batch, opt_step, totals) after each
        optimizer step, e.g. to save resume state.
    """
    model.train()
    total_loss = 0.0
    metrics = {
        "preference": 0, "reward_reg": 0, "z1_only_reg": 0, "z_combined_reg": 0,
        "kl1": 0, "kl2": 0, "future": 0,
        "mental1_gen": 0, "mental2_gen": 0, "expl_reward": 0,
    }
    if totals is not None:
        total_loss = totals["loss"]
        metrics.update(totals["metrics"])
    step = 0
    num_batches = start_batch + len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=start_batch):
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        reward_targets = batch["reward_vec"].to(device)
        has_neg = batch["has_negative"].to(device)
        mental1_ids = batch["mental1_input_ids"].to(device)
        mental1_mask = batch["mental1_attention_mask"].to(device)
        mental2_ids = batch["mental2_input_ids"].to(device)
        mental2_mask = batch["mental2_attention_mask"].to(device)
        expl_ids = batch["expl_input_ids"].to(device)
        expl_mask = batch["expl_attention_mask"].to(device)
        first_pos_token = batch["first_pos_token"].to(device)

        current_opt_step = global_step_offset + (batch_idx + 1) // grad_accum_steps

        # Stop-gradient on z1→z2 for the first z2_warmup_steps
        stop_grad = current_opt_step < z2_warmup_steps

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            out = model.forward_all(
                ctx_ids, ctx_mask, pos_ids, pos_mask,
                neg_ids, neg_mask,
                mental1_ids, mental1_mask,
                mental2_ids, mental2_mask,
                expl_ids, expl_mask,
                first_pos_token,
                stop_grad_z1=stop_grad,
            )

            raw_loss, batch_metrics, eff_kl1_w, eff_kl2_w = compute_objective_components(
                out, reward_targets, has_neg, current_opt_step,
                kl_weight=kl_weight, future_weight=future_weight,
                mental1_weight=mental1_weight, mental2_weight=mental2_weight,
                expl_weight=expl_weight, z_only_weight=z_only_weight,
                kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps,
            )
            loss = raw_loss / grad_accum_steps

        loss.backward()

        stepped = (batch_idx + 1) % grad_accum_steps == 0
        if stepped:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        total_loss += raw_loss.item()
        for key, value in batch_metrics.items():
            metrics[key] += value
        if stepped and on_optimizer_step is not None:
            on_optimizer_step(
                batch_idx + 1, current_opt_step,
                {"loss": total_loss, "metrics": dict(metrics)},
            )

        if (batch_idx + 1) % 10 == 0:
            n = batch_idx + 1
            sg = "SG" if stop_grad else ""
            print(
                f"  Epoch {epoch+1} Step {n}: "
                f"loss={total_loss/n:.4f} pref={metrics['preference']/n:.4f} "
                f"jnt_reg={metrics['reward_reg']/n:.4f} "
                f"z1_only={metrics['z1_only_reg']/n:.4f} z_comb={metrics['z_combined_reg']/n:.4f} "
                f"kl1={metrics['kl1']/n:.4f}(w={eff_kl1_w:.4f}) "
                f"kl2={metrics['kl2']/n:.4f}(w={eff_kl2_w:.4f}) "
                f"future={metrics['future']/n:.4f} "
                f"m1_gen={metrics['mental1_gen']/n:.4f} m2_gen={metrics['mental2_gen']/n:.4f} "
                f"expl={metrics['expl_reward']/n:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} {sg}"
            )

    n = num_batches
    for k in metrics:
        metrics[k] /= n
    final_global_step = global_step_offset + num_batches // grad_accum_steps
    return total_loss / n, metrics, final_global_step


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, current_opt_step,
                   kl_weight=0.1, future_weight=0.5,
                   mental1_weight=0.3, mental2_weight=0.3,
                   expl_weight=0.3, z_only_weight=0.5,
                   kl_anneal_steps=200, z2_kl_delay_steps=100):
    model.eval()
    total_loss = 0.0
    metrics = {
        "preference": 0, "reward_reg": 0, "z1_only_reg": 0, "z_combined_reg": 0,
        "kl1": 0, "kl2": 0, "future": 0,
        "mental1_gen": 0, "mental2_gen": 0, "expl_reward": 0,
    }

    for batch in dataloader:
        ctx_ids = batch["ctx_input_ids"].to(device)
        ctx_mask = batch["ctx_attention_mask"].to(device)
        pos_ids = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)
        reward_targets = batch["reward_vec"].to(device)
        has_neg = batch["has_negative"].to(device)
        mental1_ids = batch["mental1_input_ids"].to(device)
        mental1_mask = batch["mental1_attention_mask"].to(device)
        mental2_ids = batch["mental2_input_ids"].to(device)
        mental2_mask = batch["mental2_attention_mask"].to(device)
        expl_ids = batch["expl_input_ids"].to(device)
        expl_mask = batch["expl_attention_mask"].to(device)
        first_pos_token = batch["first_pos_token"].to(device)

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            out = model.forward_all(
                ctx_ids, ctx_mask, pos_ids, pos_mask,
                neg_ids, neg_mask,
                mental1_ids, mental1_mask,
                mental2_ids, mental2_mask,
                expl_ids, expl_mask,
                first_pos_token,
                stop_grad_z1=False,
            )
            raw_loss, batch_metrics, _, _ = compute_objective_components(
                out, reward_targets, has_neg, current_opt_step,
                kl_weight=kl_weight, future_weight=future_weight,
                mental1_weight=mental1_weight, mental2_weight=mental2_weight,
                expl_weight=expl_weight, z_only_weight=z_only_weight,
                kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps,
            )

        total_loss += raw_loss.item()
        for key, value in batch_metrics.items():
            metrics[key] += value

    n = len(dataloader)
    for key in metrics:
        metrics[key] /= n
    return total_loss / n, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_HEAD_NAMES = [
    "z1_mu", "z1_logvar", "z2_mu", "z2_logvar",
    "joint_outcome_head", "z1_only_reward_head", "z_combined_reward_head",
    "z_to_hidden",
    "mental1_decoder", "mental2_decoder",
    "expl_cross_attn", "expl_reward_head",
]


def _write_weights(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.base_model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    for head_name in CUSTOM_HEAD_NAMES:
        head = getattr(model, head_name)
        torch.save(head.state_dict(), os.path.join(save_dir, f"{head_name}.pth"))


def _save_checkpoint(model, save_dir):
    _write_weights(model, save_dir)


# Arguments that must match between an interrupted run and its resumption.
RESUME_INVARIANT_ARGS = (
    "model_name", "data_path", "batch_size", "grad_accum_steps", "num_epochs", "lr",
    "max_ctx_len", "max_resp_len", "z_dim", "lora_r", "lora_alpha", "num_lora_layers",
    "val_ratio", "seed",
)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 v3: Recursive ToM Coupled Mental+Reward Model"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str,
                        default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="projects/sotopia/checkpoints/coupled_mental_reward_v3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--future_weight", type=float, default=0.5)
    parser.add_argument("--mental1_weight", type=float, default=0.3,
                        help="Weight for 1st-order mental generation loss")
    parser.add_argument("--mental2_weight", type=float, default=0.3,
                        help="Weight for 2nd-order mental generation loss")
    parser.add_argument("--expl_weight", type=float, default=0.3)
    parser.add_argument("--z_only_weight", type=float, default=0.5)
    parser.add_argument("--kl_anneal_steps", type=int, default=200,
                        help="Steps to linearly ramp kl_weight for z1")
    parser.add_argument("--z2_kl_delay_steps", type=int, default=100,
                        help="Extra steps before z2 KL annealing starts (after z1's)")
    parser.add_argument("--z2_warmup_steps", type=int, default=100,
                        help="Stop-gradient z1->z2 for this many steps")
    parser.add_argument("--mental_prewarm_data", type=str, default=None)
    parser.add_argument("--mental_prewarm_epochs", type=int, default=1)
    parser.add_argument("--mental_prewarm_lr", type=float, default=2e-4)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Hold out this fraction of data for validation-based checkpoint selection")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <output_dir>/last after an interruption.")
    parser.add_argument("--resume_every", type=int, default=100,
                        help="Refresh <output_dir>/last every N optimizer steps (and every epoch).")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

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

    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_" in name

    base_model.print_trainable_parameters()
    base_model.config.use_cache = False

    # ── Combined Model ──
    model = RecursiveToMModel(
        base_model, reward_dim=REWARD_DIM, z_dim=args.z_dim
    ).to(device)

    for name, param in model.named_parameters():
        if any(k in name for k in CUSTOM_HEAD_NAMES):
            param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    resume_state = None
    if args.resume:
        resume_dir = require_resume_dir(args.output_dir)
        resume_state = load_resume_state(resume_dir)
        check_resume_args(resume_state["args"], args, RESUME_INVARIANT_ARGS)
        print(f"Resuming from {resume_dir}", flush=True)

    # ── Mental Pre-Warmup (optional, z1 only; its effect is in the resumed weights) ──
    if args.mental_prewarm_data and resume_state is None:
        prewarm_dataset = MentalPrewarmDataset(
            args.mental_prewarm_data, tokenizer, max_ctx_len=args.max_ctx_len,
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
    dataset = RecursiveToMDataset(
        args.data_path, tokenizer,
        max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len,
    )
    train_dataset = dataset
    val_dataset = None
    if args.val_ratio > 0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        split_gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)
        print(f"Dataset split: train={train_size}, val={val_size}")
    else:
        print(f"Dataset split: train={len(dataset)}, val=0 (validation disabled)")

    def make_train_loader(indices):
        # Deterministic per-epoch order (see epoch_order) makes mid-epoch resume exact;
        # the private generator keeps worker seeding off the global RNG stream.
        return DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=indices,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            num_workers=args.num_workers, pin_memory=True,
            generator=torch.Generator(),
        )

    batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
            generator=torch.Generator(),
        )

    # ── Optimizer + Scheduler ──
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
    print(f"Param groups: LoRA lr={args.lr}, Head lr={head_lr}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
    ])
    total_steps = batches_per_epoch * args.num_epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training ──
    best_metric = float("inf")
    global_step = 0
    start_epoch, start_batch, epoch_start_step, epoch_totals = 0, 0, 0, None
    if resume_state is not None:
        for head_name in CUSTOM_HEAD_NAMES:
            getattr(model, head_name).load_state_dict(torch.load(
                os.path.join(resume_dir, f"{head_name}.pth"), map_location=device, weights_only=True,
            ))
        load_lora_weights(model.base_model, os.path.join(resume_dir, "lora_adapter"))
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        best_metric = resume_state["best_metric"]
        global_step = resume_state["global_step"]
        start_epoch, start_batch = resume_state["epoch"], resume_state["next_batch"]
        epoch_start_step, epoch_totals = resume_state["epoch_start_step"], resume_state["totals"]
        restore_rng_state(resume_state["rng"])
        print(f"Resumed at epoch {start_epoch}, batch {start_batch}, step {global_step}", flush=True)

    def save_resume(epoch, next_batch, epoch_start, step, totals):
        save_resume_dir(args.output_dir, lambda d: _write_weights(model, str(d)), {
            "args": vars(args), "epoch": epoch, "next_batch": next_batch,
            "epoch_start_step": epoch_start, "global_step": step, "totals": totals,
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        })

    for epoch in range(start_epoch, args.num_epochs):
        first_batch = start_batch if epoch == start_epoch else 0
        totals = epoch_totals if epoch == start_epoch else None
        epoch_offset = epoch_start_step if epoch == start_epoch else global_step
        train_dataloader = make_train_loader(
            epoch_order(len(train_dataset), args.seed, epoch)[first_batch * args.batch_size:]
        )

        def on_optimizer_step(next_batch, opt_step, running, epoch=epoch, epoch_offset=epoch_offset):
            if args.resume_every > 0 and opt_step % args.resume_every == 0:
                save_resume(epoch, next_batch, epoch_offset, opt_step, running)

        avg_loss, metrics, global_step = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch,
            kl_weight=args.kl_weight, future_weight=args.future_weight,
            mental1_weight=args.mental1_weight, mental2_weight=args.mental2_weight,
            expl_weight=args.expl_weight, z_only_weight=args.z_only_weight,
            grad_accum_steps=args.grad_accum_steps,
            kl_anneal_steps=args.kl_anneal_steps,
            z2_kl_delay_steps=args.z2_kl_delay_steps,
            z2_warmup_steps=args.z2_warmup_steps,
            global_step_offset=epoch_offset,
            max_grad_norm=args.max_grad_norm,
            start_batch=first_batch,
            totals=totals,
            on_optimizer_step=on_optimizer_step,
        )
        print(f"\nEpoch {epoch+1}/{args.num_epochs}: avg_loss={avg_loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        monitor_metric = avg_loss
        monitor_name = "train_loss"
        if val_dataloader is not None:
            val_loss, val_metrics = evaluate_epoch(
                model, val_dataloader, device, current_opt_step=max(global_step, 1),
                kl_weight=args.kl_weight, future_weight=args.future_weight,
                mental1_weight=args.mental1_weight, mental2_weight=args.mental2_weight,
                expl_weight=args.expl_weight, z_only_weight=args.z_only_weight,
                kl_anneal_steps=args.kl_anneal_steps,
                z2_kl_delay_steps=args.z2_kl_delay_steps,
            )
            print(f"  val_loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"  val_{k}: {v:.4f}")
            monitor_metric = val_loss
            monitor_name = "val_loss"

        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        _save_checkpoint(model, ckpt_dir)

        if monitor_metric < best_metric:
            best_metric = monitor_metric
            best_dir = os.path.join(args.output_dir, "best")
            _save_checkpoint(model, best_dir)
            print(f"  -> Best model saved ({monitor_name}={best_metric:.4f})")
        save_resume(epoch + 1, 0, global_step, global_step, None)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nStage 1 v3 training complete. Best metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
