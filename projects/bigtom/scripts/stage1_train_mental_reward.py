"""
BigToM Stage 1 — Coupled recursive mental encoder + reward model.

Mirrors the Sotopia stage1_train_coupled_mental_reward_v3.py structure but
adapted to BigToM's single-agent belief-tracking setting.

Architecture
------------
  context_hidden = base_lm(story + percept_sentence)[last token]
  z1, mu1, logvar1 = VAE_z1(context_hidden)                      # 1st order
  z2, mu2, logvar2 = VAE_z2([context_hidden || z1])               # 2nd order
  mental1_decoder(z1 prefix)   -> first_order_belief text         (CE loss)
  mental2_decoder(z2 prefix)   -> second_order_belief text        (CE loss)
  joint_outcome_head([z1||z2||response_hidden]) -> scalar reward  (preference + regression)
  z1_only_reward_head(z1)         -> scalar                       (regularizer)
  z_combined_reward_head([z1||z2]) -> scalar                      (regularizer)

Losses
------
  L = L_pref + L_reward_reg
      + z_only_w  * (L_z1_only + L_zcomb)
      + kl1_w * KL(q(z1|x) || N(0,I))     # linearly annealed
      + kl2_w * KL(q(z2|x,z1) || N(0,I))  # delayed anneal
      + m1_w * L_mental1_gen + m2_w * L_mental2_gen
      + future_w * L_future_ntp

Positive response = the gold action matching the current aware/not_aware branch.
Negative response = the gold action of the OTHER branch (false-vs-true belief).

Usage
-----
  python stage1_train_mental_reward.py \\
      --data ../data/bigtom_qwen_5k_annotated.jsonl \\
      --base_model Qwen/Qwen2.5-7B-Instruct \\
      --out ../checkpoints/stage1 --epochs 3 --batch_size 4

Every checkpoint also refreshes <out>/last/, which holds the optimizer,
scheduler, data position, and RNG state. After a crash, rerun the same command
with --resume to continue exactly where the last checkpoint left off.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
from safetensors.torch import load_file


def split_story_sentences(story: str) -> List[str]:
    return [s.strip() for s in story.split(".") if s.strip()]


def build_presented_story(story: str, init_belief_flag: int) -> str:
    parts = split_story_sentences(story)
    if len(parts) < 5:
        return story.strip()
    shown = parts[:5] if init_belief_flag else parts[:3] + [parts[4]]
    return ". ".join(shown) + "."


def build_task_context(
    story: str,
    init_belief_flag: int,
    *,
    percept: Optional[str] = None,
    action: Optional[str] = None,
) -> str:
    context = build_presented_story(story, init_belief_flag)
    extras = [x.strip() for x in (percept, action) if x and x.strip()]
    if extras:
        context = f"{context} {' '.join(extras)}"
    return context.strip()


def build_encoder_context(context: str, question: str) -> str:
    return f"{context}\nQuestion: {question.strip()}".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
def get_transformer_from_peft(peft_causal_lm):
    """Walk through a PEFT-wrapped CausalLM to the underlying transformer."""
    base = peft_causal_lm
    seen = set()
    while hasattr(base, "base_model"):
        if id(base) in seen:
            break
        seen.add(id(base))
        next_base = base.base_model
        if next_base is base:
            break
        base = next_base
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model
    if hasattr(base, "model"):
        return base.model
    return base


def _build_mental_decoder(hidden_size: int, z_dim: int, num_prefix: int = 8, num_layers: int = 2):
    """A small transformer decoder that takes a z-expanded prefix and predicts tokens."""
    bundle = nn.ModuleDict({
        "z_to_prefix": nn.Linear(z_dim, num_prefix * hidden_size),
        "decoder": nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=0.1, batch_first=True, norm_first=True,
            ),
            num_layers=num_layers,
        ),
        "out_norm": nn.LayerNorm(hidden_size),
    })
    bundle.register_buffer("num_prefix", torch.tensor(num_prefix), persistent=True)
    return bundle


class RecursiveToMModel(nn.Module):
    def __init__(self, base_model: nn.Module, z_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.transformer = get_transformer_from_peft(base_model)
        hidden_size = base_model.config.hidden_size
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        # z1: context -> (mu1, logvar1)
        self.z1_mu = nn.Linear(hidden_size, z_dim)
        self.z1_logvar = nn.Linear(hidden_size, z_dim)

        # z2: [context || z1] -> (mu2, logvar2)
        self.z2_mu = nn.Linear(hidden_size + z_dim, z_dim)
        self.z2_logvar = nn.Linear(hidden_size + z_dim, z_dim)

        # Reward heads
        self.joint_outcome_head = nn.Sequential(
            nn.Linear(z_dim * 2 + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.z1_only_reward_head = nn.Sequential(
            nn.Linear(z_dim, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.z_combined_reward_head = nn.Sequential(
            nn.Linear(z_dim * 2, 128), nn.GELU(), nn.Linear(128, 1),
        )

        # Binary belief classifier on z1 (aux: BigToM true/false belief label)
        self.belief_classifier = nn.Sequential(
            nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, 2),
        )

        # z to hidden (for future-token prediction regularizer)
        self.z_to_hidden = nn.Linear(z_dim * 2, hidden_size)

        # Mental decoders (small transformers, z-bottlenecked)
        self.mental1_decoder = _build_mental_decoder(hidden_size, z_dim)
        self.mental2_decoder = _build_mental_decoder(hidden_size, z_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.z1_mu, self.z1_logvar, self.z2_mu, self.z2_logvar]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)
        nn.init.constant_(self.z1_logvar.bias, -2.0)
        nn.init.constant_(self.z2_logvar.bias, -2.0)
        for group in [self.joint_outcome_head, self.z1_only_reward_head,
                      self.z_combined_reward_head, self.belief_classifier]:
            for m in group:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _encode(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        hidden = out.last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1), hidden

    def _sample_z(self, mu_proj, logvar_proj, h) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = h.to(mu_proj.weight.dtype)
        mu = mu_proj(h)
        logvar = logvar_proj(h).clamp(min=-8.0, max=6.0)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar

    def encode_z1_z2(self, ctx_ids, ctx_mask, stop_grad_z1: bool = False):
        ctx_last, _ = self._encode(ctx_ids, ctx_mask)
        z1, mu1, logvar1 = self._sample_z(self.z1_mu, self.z1_logvar, ctx_last)
        z1_for_z2 = z1.detach() if stop_grad_z1 else z1
        z2_input = torch.cat([ctx_last, z1_for_z2], dim=1)
        z2, mu2, logvar2 = self._sample_z(self.z2_mu, self.z2_logvar, z2_input)
        return ctx_last, z1, mu1, logvar1, z2, mu2, logvar2

    def encode_z1_z2_deterministic(self, ctx_ids, ctx_mask):
        ctx_last, _ = self._encode(ctx_ids, ctx_mask)
        mu1 = self.z1_mu(ctx_last.to(self.z1_mu.weight.dtype))
        mu2 = self.z2_mu(torch.cat([ctx_last.to(mu1.dtype), mu1], dim=1))
        return mu1, mu2

    def _decode_mental(self, z, decoder_bundle, mental_ids, mental_mask) -> torch.Tensor:
        B = z.size(0)
        num_prefix = int(decoder_bundle.num_prefix.item())
        prefix = decoder_bundle["z_to_prefix"](
            z.to(decoder_bundle["z_to_prefix"].weight.dtype)
        ).view(B, num_prefix, self.hidden_size)
        pad_id = self.base_model.config.pad_token_id
        if pad_id is None:
            pad_id = 0

        # Token embeddings for teacher forcing
        emb = self.base_model.get_input_embeddings()(mental_ids).to(prefix.dtype)   # [B, L, H]
        memory = torch.cat([prefix, emb[:, :-1]], dim=1)           # input seq
        tgt_len = memory.size(1)

        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=z.device, dtype=torch.bool), diagonal=1
        )
        out = decoder_bundle["decoder"](
            tgt=memory, memory=memory, tgt_mask=causal_mask,
        )
        out = decoder_bundle["out_norm"](out)

        # Only predict the mental tokens (positions after the prefix)
        lm_weight = self.base_model.get_input_embeddings().weight
        pred_logits = F.linear(
            out[:, num_prefix - 1: num_prefix - 1 + mental_ids.size(1)].to(lm_weight.dtype),
            lm_weight,
        )
        target = mental_ids.masked_fill(mental_mask == 0, pad_id)
        shift_logits = pred_logits[:, :target.size(1), :]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            target.reshape(-1),
            ignore_index=pad_id,
        )
        return loss

    def forward_all(self, ctx_ids, ctx_mask,
                    pos_ids, pos_mask, neg_ids, neg_mask,
                    m1_ids, m1_mask, m2_ids, m2_mask,
                    first_pos_token: torch.Tensor,
                    belief_label: torch.Tensor,
                    stop_grad_z1: bool = False):
        B = ctx_ids.size(0)
        ctx_last, z1, mu1, logvar1, z2, mu2, logvar2 = \
            self.encode_z1_z2(ctx_ids, ctx_mask, stop_grad_z1=stop_grad_z1)

        # Batched encoding of pos + neg through the transformer
        max_len = max(pos_ids.size(1), neg_ids.size(1))
        pad_id = self.base_model.config.pad_token_id or 0
        def pad_to(ids, mask, L):
            p = L - ids.size(1)
            if p > 0:
                ids = F.pad(ids, (0, p), value=pad_id)
                mask = F.pad(mask, (0, p), value=0)
            return ids, mask
        pos_ids_p, pos_mask_p = pad_to(pos_ids, pos_mask, max_len)
        neg_ids_p, neg_mask_p = pad_to(neg_ids, neg_mask, max_len)

        batched_ids = torch.cat([pos_ids_p, neg_ids_p], dim=0)
        batched_mask = torch.cat([pos_mask_p, neg_mask_p], dim=0)
        batched_out = self.transformer(
            input_ids=batched_ids, attention_mask=batched_mask,
            use_cache=False, return_dict=True,
        )
        batched_hidden = batched_out.last_hidden_state

        def last_hidden(hidden, mask):
            idx = mask.sum(dim=1) - 1
            idx_e = idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            return hidden.gather(1, idx_e).squeeze(1)

        pos_resp_h = last_hidden(batched_hidden[:B], batched_mask[:B])
        neg_resp_h = last_hidden(batched_hidden[B:], batched_mask[B:]).to(z1.dtype)
        pos_resp_h = pos_resp_h.to(z1.dtype)

        z_cat = torch.cat([z1, z2], dim=1)
        pos_r = self.joint_outcome_head(
            torch.cat([z1, z2, pos_resp_h], dim=1).to(self.joint_outcome_head[0].weight.dtype)
        )
        neg_r = self.joint_outcome_head(
            torch.cat([z1, z2, neg_resp_h], dim=1).to(self.joint_outcome_head[0].weight.dtype)
        )
        z1_only_r = self.z1_only_reward_head(z1.to(self.z1_only_reward_head[0].weight.dtype))
        zc_r = self.z_combined_reward_head(z_cat.to(self.z_combined_reward_head[0].weight.dtype))

        belief_logits = self.belief_classifier(z1.to(self.belief_classifier[0].weight.dtype))

        # Next-token prediction from z1+z2 (future regularizer)
        conditioned = (
            ctx_last.to(self.z_to_hidden.weight.dtype)
            + self.z_to_hidden(z_cat.to(self.z_to_hidden.weight.dtype))
        )
        out_emb = self.base_model.get_output_embeddings()
        next_logits = F.linear(conditioned.to(out_emb.weight.dtype), out_emb.weight)
        future_loss = F.cross_entropy(next_logits, first_pos_token)

        m1_loss = self._decode_mental(z1, self.mental1_decoder, m1_ids, m1_mask)
        m2_loss = self._decode_mental(z2, self.mental2_decoder, m2_ids, m2_mask)

        return {
            "pos_r": pos_r, "neg_r": neg_r,
            "z1_only_r": z1_only_r, "zc_r": zc_r,
            "belief_logits": belief_logits,
            "mu1": mu1, "logvar1": logvar1,
            "mu2": mu2, "logvar2": logvar2,
            "future_loss": future_loss,
            "m1_loss": m1_loss, "m2_loss": m2_loss,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class BigToMRecursiveDataset(Dataset):
    """
    Each row of bigtom_qwen_annotated.jsonl becomes ONE sample. We also
    need the opposite branch's gold action as the negative response, so
    we index by scenario_id and pair {aware, not_aware} together at build.
    """
    def __init__(self, path: str, tokenizer,
                 max_ctx_len: int = 768,
                 max_resp_len: int = 128,
                 max_mental_len: int = 96):
        self.tok = tokenizer
        self.max_ctx = max_ctx_len
        self.max_resp = max_resp_len
        self.max_mental = max_mental_len

        by_sid: Dict[int, Dict[str, dict]] = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r

        self.samples: List[dict] = []
        dropped = 0
        for sid, d in by_sid.items():
            if "aware" not in d or "not_aware" not in d:
                dropped += 1
                continue
            for cond in ("aware", "not_aware"):
                pos = d[cond]
                neg = d["not_aware" if cond == "aware" else "aware"]
                branch_label = 0 if cond == "aware" else 1
                for init_belief_flag in (0, 1):
                    fwd_ctx = build_task_context(
                        pos["story"], init_belief_flag, percept=pos["percept"],
                    )
                    bwd_ctx = build_task_context(
                        pos["story"], init_belief_flag, action=pos["gold_action"],
                    )
                    task_rows = [
                        {
                            "task": "forward_action",
                            "context": fwd_ctx,
                            "question": pos["action_question"],
                            "pos_answer": pos["gold_action"],
                            "neg_answer": neg["gold_action"],
                        },
                        {
                            "task": "forward_belief",
                            "context": fwd_ctx,
                            "question": pos["belief_question"],
                            "pos_answer": pos["gold_belief"],
                            "neg_answer": neg["gold_belief"],
                        },
                        {
                            "task": "backward_belief",
                            "context": bwd_ctx,
                            "question": pos["belief_question"],
                            "pos_answer": pos["gold_belief"],
                            "neg_answer": neg["gold_belief"],
                        },
                    ]
                    for task_row in task_rows:
                        self.samples.append({
                            "sid": pos.get("scenario_uid", sid),
                            "condition": cond,
                            "task": task_row["task"],
                            "context": build_encoder_context(task_row["context"], task_row["question"]),
                            "pos_answer": task_row["pos_answer"],
                            "neg_answer": task_row["neg_answer"],
                            "first_order_belief": pos["first_order_belief"],
                            "second_order_belief": pos["second_order_belief"],
                            "belief_label": branch_label,
                        })
        print(f"Loaded {len(self.samples)} samples ({dropped} scenarios dropped for missing branch)")

    def __len__(self):
        return len(self.samples)

    def _enc(self, text, max_len):
        out = self.tok(text, truncation=True, max_length=max_len,
                       return_tensors="pt", add_special_tokens=True)
        return out["input_ids"][0], out["attention_mask"][0]

    def __getitem__(self, i):
        s = self.samples[i]
        ctx_ids, ctx_mask = self._enc(s["context"], self.max_ctx)
        pos_ids, pos_mask = self._enc(s["pos_answer"], self.max_resp)
        neg_ids, neg_mask = self._enc(s["neg_answer"], self.max_resp)
        m1_ids, m1_mask = self._enc(s["first_order_belief"], self.max_mental)
        m2_ids, m2_mask = self._enc(s["second_order_belief"], self.max_mental)

        # first token of the positive response (for future-NTP regularizer)
        first_pos = pos_ids[0].item()

        return {
            "ctx_ids": ctx_ids, "ctx_mask": ctx_mask,
            "pos_ids": pos_ids, "pos_mask": pos_mask,
            "neg_ids": neg_ids, "neg_mask": neg_mask,
            "m1_ids": m1_ids, "m1_mask": m1_mask,
            "m2_ids": m2_ids, "m2_mask": m2_mask,
            "first_pos": torch.tensor(first_pos, dtype=torch.long),
            "belief_label": torch.tensor(s["belief_label"], dtype=torch.long),
        }


def collate(batch, pad_id: int = 0):
    def pad_stack(key, mask_key=None):
        seqs = [b[key] for b in batch]
        masks = [b[mask_key] for b in batch] if mask_key else None
        L = max(s.size(0) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mk = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : s.size(0)] = s
            mk[i, : s.size(0)] = masks[i] if masks else 1
        return ids, mk
    ctx_ids, ctx_mask = pad_stack("ctx_ids", "ctx_mask")
    pos_ids, pos_mask = pad_stack("pos_ids", "pos_mask")
    neg_ids, neg_mask = pad_stack("neg_ids", "neg_mask")
    m1_ids, m1_mask = pad_stack("m1_ids", "m1_mask")
    m2_ids, m2_mask = pad_stack("m2_ids", "m2_mask")
    return {
        "ctx_ids": ctx_ids, "ctx_mask": ctx_mask,
        "pos_ids": pos_ids, "pos_mask": pos_mask,
        "neg_ids": neg_ids, "neg_mask": neg_mask,
        "m1_ids": m1_ids, "m1_mask": m1_mask,
        "m2_ids": m2_ids, "m2_mask": m2_mask,
        "first_pos": torch.stack([b["first_pos"] for b in batch]),
        "belief_label": torch.stack([b["belief_label"] for b in batch]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────
def kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def compute_loss(model, batch, device,
                 kl_weight: float, m1_w: float, m2_w: float,
                 future_w: float, z_only_w: float,
                 step: int, kl_anneal_steps: int, z2_kl_delay_steps: int,
                 stop_grad_z1: bool):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model.forward_all(
        ctx_ids=batch["ctx_ids"], ctx_mask=batch["ctx_mask"],
        pos_ids=batch["pos_ids"], pos_mask=batch["pos_mask"],
        neg_ids=batch["neg_ids"], neg_mask=batch["neg_mask"],
        m1_ids=batch["m1_ids"], m1_mask=batch["m1_mask"],
        m2_ids=batch["m2_ids"], m2_mask=batch["m2_mask"],
        first_pos_token=batch["first_pos"],
        belief_label=batch["belief_label"],
        stop_grad_z1=stop_grad_z1,
    )

    # Preference loss: pos reward > neg reward
    pref = F.softplus(out["neg_r"] - out["pos_r"]).mean()

    # Reward regression: pos should map to 1.0, neg to 0.0
    ones = torch.ones_like(out["pos_r"])
    zeros = torch.zeros_like(out["neg_r"])
    reward_reg = (F.smooth_l1_loss(out["pos_r"], ones)
                  + F.smooth_l1_loss(out["neg_r"], zeros))
    branch_target = (1.0 - batch["belief_label"].float()).unsqueeze(-1)
    z1_only_reg = F.smooth_l1_loss(out["z1_only_r"], branch_target)
    zc_reg = F.smooth_l1_loss(out["zc_r"], branch_target)

    # Belief classifier
    belief_ce = F.cross_entropy(out["belief_logits"], batch["belief_label"])
    belief_acc = (out["belief_logits"].argmax(-1) == batch["belief_label"]).float().mean()

    # KL with annealing
    with torch.amp.autocast(device_type="cuda", enabled=False):
        kl1 = kl_loss(out["mu1"], out["logvar1"])
        kl2 = kl_loss(out["mu2"], out["logvar2"])
    if kl_anneal_steps > 0:
        a1 = min(1.0, step / kl_anneal_steps)
        z2_start = kl_anneal_steps + z2_kl_delay_steps
        a2 = min(1.0, max(0.0, (step - z2_start)) / kl_anneal_steps)
    else:
        a1 = a2 = 1.0

    total = (
        pref
        + reward_reg
        + z_only_w * (z1_only_reg + zc_reg)
        + belief_ce
        + kl_weight * a1 * kl1
        + kl_weight * a2 * kl2
        + m1_w * out["m1_loss"]
        + m2_w * out["m2_loss"]
        + future_w * out["future_loss"]
    )
    metrics = {
        "total": total.item(),
        "pref": pref.item(),
        "reward_reg": reward_reg.item(),
        "belief_ce": belief_ce.item(),
        "belief_acc": belief_acc.item(),
        "kl1": kl1.item(), "kl2": kl2.item(),
        "m1": out["m1_loss"].item(), "m2": out["m2_loss"].item(),
        "future": out["future_loss"].item(),
    }
    return total, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
RESUME_DIR = "last"
# Arguments that must match between the interrupted run and its resumption.
RESUME_INVARIANT_ARGS = (
    "data", "base_model", "epochs", "batch_size", "grad_accum", "lr", "z_dim",
    "lora_r", "lora_alpha", "max_ctx_len", "seed",
)


def epoch_order(num_samples: int, seed: int, epoch: int) -> List[int]:
    """Deterministic per-epoch shuffle, so a resumed run sees the same batches."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(num_samples, generator=generator).tolist()


def find_resume_dir(out: Path) -> Optional[Path]:
    # "last.old" only survives if a crash interrupted the swap in save_resume_state.
    for name in (RESUME_DIR, f"{RESUME_DIR}.old"):
        candidate = out / name
        if (candidate / "trainer_state.pt").is_file():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=str, default="projects/bigtom/checkpoints/stage1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--kl_weight", type=float, default=0.1)
    ap.add_argument("--kl_anneal_steps", type=int, default=500)
    ap.add_argument("--z2_kl_delay_steps", type=int, default=500)
    ap.add_argument("--z1_stop_grad_steps", type=int, default=300)
    ap.add_argument("--m1_weight", type=float, default=0.5)
    ap.add_argument("--m2_weight", type=float, default=0.3)
    ap.add_argument("--future_weight", type=float, default=0.3)
    ap.add_argument("--z_only_weight", type=float, default=0.3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_ctx_len", type=int, default=768)
    ap.add_argument("--log_every_steps", type=int, default=10)
    ap.add_argument("--save_every_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="Continue from <out>/last, written at every checkpoint.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading tokenizer and base model: {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    base.config.pad_token_id = tok.pad_token_id
    base.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    )
    base = get_peft_model(base, lora)
    base.print_trainable_parameters()

    model = RecursiveToMModel(base, z_dim=args.z_dim).to(device)
    for name, p in model.named_parameters():
        if not name.startswith("base_model.") and not name.startswith("transformer."):
            p.data = p.data.float()

    dataset = BigToMRecursiveDataset(args.data, tok, max_ctx_len=args.max_ctx_len)
    batches_per_epoch = len(dataset) // args.batch_size

    def make_loader(indices: List[int]) -> DataLoader:
        # A private generator keeps worker seeding off the global RNG stream.
        return DataLoader(
            dataset, batch_size=args.batch_size, sampler=indices,
            collate_fn=lambda b: collate(b, pad_id=tok.pad_token_id),
            num_workers=2, drop_last=True, generator=torch.Generator(),
        )

    total_steps = max(1, (batches_per_epoch // args.grad_accum) * args.epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps,
    )

    os.makedirs(args.out, exist_ok=True)
    print(
        f"Training setup: samples={len(dataset)} batches_per_epoch={batches_per_epoch} "
        f"total_optimizer_steps={total_steps} grad_accum={args.grad_accum}",
        flush=True,
    )

    def filtered_state_dict():
        return {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("base_model.") and not k.startswith("transformer.")
        }

    def save_heads(out_dir: Path, epoch: int, global_step: int, checkpoint_type: str, avg_metrics, best_metric, best_step, source_checkpoint=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": filtered_state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_type": checkpoint_type,
                "avg_metrics": avg_metrics,
                "best_metric": best_metric,
                "best_step": best_step,
                "source_checkpoint": source_checkpoint,
            },
            out_dir / "heads.pt",
        )
        # heads.pt excludes the backbone, so the trained encoder LoRA is saved
        # separately; Stage 2/3 and evaluation load both.
        model.base_model.save_pretrained(out_dir / "lora")
        tok.save_pretrained(out_dir / "lora")

    def avg_metrics(rows):
        if not rows:
            return {}
        keys = rows[0].keys()
        return {k: sum(r[k] for r in rows) / len(rows) for k in keys}

    best_metric = None
    best_step = None
    global_step = 0
    start_epoch = 0
    start_batch = 0
    resumed_epoch_window = []

    def save_resume_state(epoch: int, next_batch: int, epoch_window):
        """Atomically refresh <out>/last with everything needed to continue.

        Saved right after an optimizer step (or at the end of an epoch), so no
        partial gradient accumulation is pending except the epoch's trailing
        remainder batches, which a resume from an epoch boundary drops.
        """
        out = Path(args.out)
        tmp, final, old = out / f"{RESUME_DIR}.tmp", out / RESUME_DIR, out / f"{RESUME_DIR}.old"
        if tmp.exists():
            shutil.rmtree(tmp)
        save_heads(tmp, epoch, global_step, "resume", {}, best_metric, best_step)
        torch.save(
            {
                "args": vars(args),
                "epoch": epoch,
                "next_batch": next_batch,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_step": best_step,
                "epoch_window": epoch_window,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": {
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            tmp / "trainer_state.pt",
        )
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            final.rename(old)
        tmp.rename(final)
        if old.exists():
            shutil.rmtree(old)

    if args.resume:
        resume_dir = find_resume_dir(Path(args.out))
        if resume_dir is None:
            raise FileNotFoundError(
                f"--resume was given but {Path(args.out) / RESUME_DIR} has no trainer_state.pt."
            )
        state = torch.load(resume_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        changed = {
            k: (state["args"].get(k), getattr(args, k))
            for k in RESUME_INVARIANT_ARGS if state["args"].get(k) != getattr(args, k)
        }
        if changed:
            raise ValueError(f"Cannot resume with different training arguments: {changed}")
        heads = torch.load(resume_dir / "heads.pt", map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(heads, strict=False)
        missing_heads = [k for k in missing if not k.startswith(("base_model.", "transformer."))]
        if missing_heads or unexpected:
            raise RuntimeError(f"Resume heads mismatch: missing={missing_heads[:5]} unexpected={unexpected[:5]}")
        lora_result = set_peft_model_state_dict(
            model.base_model, load_file(str(resume_dir / "lora" / "adapter_model.safetensors")),
        )
        if lora_result.unexpected_keys:
            raise RuntimeError(f"Resume LoRA mismatch: unexpected={lora_result.unexpected_keys[:5]}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = state["global_step"]
        best_metric, best_step = state["best_metric"], state["best_step"]
        start_epoch, start_batch = state["epoch"], state["next_batch"]
        resumed_epoch_window = state["epoch_window"]
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        print(
            f"Resumed from {resume_dir}: global_step={global_step} "
            f"epoch={start_epoch} next_batch={start_batch}",
            flush=True,
        )

    train_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}", flush=True)
        model.train()
        log_window = []
        accum_window = []
        first_batch = start_batch if epoch == start_epoch else 0
        epoch_window = resumed_epoch_window if epoch == start_epoch else []
        order = epoch_order(len(dataset), args.seed, epoch)
        loader = make_loader(order[first_batch * args.batch_size:])

        for i, batch in enumerate(loader, start=first_batch):
            stop_grad_z1 = global_step < args.z1_stop_grad_steps
            loss, metrics = compute_loss(
                model, batch, device,
                kl_weight=args.kl_weight,
                m1_w=args.m1_weight, m2_w=args.m2_weight,
                future_w=args.future_weight, z_only_w=args.z_only_weight,
                step=global_step,
                kl_anneal_steps=args.kl_anneal_steps,
                z2_kl_delay_steps=args.z2_kl_delay_steps,
                stop_grad_z1=stop_grad_z1,
            )
            (loss / args.grad_accum).backward()
            accum_window.append(metrics)

            is_optimizer_step = (i + 1) % args.grad_accum == 0
            if not is_optimizer_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_metrics = avg_metrics(accum_window)
            accum_window = []
            log_window.append(step_metrics)
            epoch_window.append(step_metrics)

            if global_step == 1 or global_step % args.log_every_steps == 0:
                avg = avg_metrics(log_window)
                elapsed_min = (time.time() - train_start) / 60.0
                pct = 100.0 * global_step / max(total_steps, 1)
                print(
                    f"[epoch {epoch} step {global_step}/{total_steps}] "
                    f"{pct:.1f}% elapsed_min={elapsed_min:.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} "
                    + " ".join(f"{k}={v:.3f}" for k, v in avg.items()),
                    flush=True,
                )
                log_window = []

            if global_step % args.save_every_steps == 0:
                ckpt = Path(args.out) / f"step_{global_step}"
                save_heads(
                    ckpt, epoch, global_step, "step", step_metrics,
                    best_metric, best_step, str(ckpt),
                )
                print(f"Saved checkpoint to {ckpt}", flush=True)
                metric = step_metrics.get("total")
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_step = global_step
                    best_dir = Path(args.out) / "best_ckpt"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    save_heads(
                        best_dir, epoch, global_step, "best", step_metrics,
                        best_metric, best_step, str(ckpt),
                    )
                    print(f"Updated best checkpoint: {best_dir} (total={metric:.4f})", flush=True)
                save_resume_state(epoch, i + 1, epoch_window)

        epoch_metrics = avg_metrics(epoch_window)
        ckpt = Path(args.out) / f"epoch_{epoch}"
        save_heads(
            ckpt, epoch, global_step, "epoch", epoch_metrics,
            best_metric, best_step, str(ckpt),
        )
        print(f"Saved epoch checkpoint to {ckpt}", flush=True)
        metric = epoch_metrics.get("total")
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_step = global_step
            best_dir = Path(args.out) / "best_ckpt"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            save_heads(
                best_dir, epoch, global_step, "best", epoch_metrics,
                best_metric, best_step, str(ckpt),
            )
            print(f"Updated best checkpoint: {best_dir} (total={metric:.4f})", flush=True)
        save_resume_state(epoch + 1, 0, [])


if __name__ == "__main__":
    main()
