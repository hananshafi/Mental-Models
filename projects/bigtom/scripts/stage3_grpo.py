"""
BigToM Stage 3 — GRPO on the z-conditioned QA policy.

Starting from the Stage 2 SFT checkpoint, run GRPO where:

  - context + question -> (z1, z2) via frozen Stage-1 encoder
  - mental prefix injected as input embeddings in front of the prompt
  - candidates sampled via `generate(inputs_embeds=..)`
  - reward = frozen Stage-1 joint_outcome_head(z1, z2, completion_hidden)
  - KL penalty vs. the frozen Stage-2 SFT reference

The prompt/task format matches Stage 2 exactly.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel

from stage2_policy_sft import (
    MENTAL_PREFIX_LEN,
    MentalPrefixProjector,
    build_prompt,
    build_target,
    iter_training_examples,
    load_stage1_encoder,
)
from stage1_train_mental_reward import build_encoder_context


# ──────────────────────────────────────────────────────────────────────────────
# Frozen reward model
# ──────────────────────────────────────────────────────────────────────────────
class FrozenBigToMRewardModel:
    def __init__(self, encoder, tokenizer, device: torch.device, ensemble_weight: float = 1.0, max_resp_len: int = 256):
        self.model = encoder
        self.tok = tokenizer
        self.device = device
        self.ensemble_weight = ensemble_weight  # kept for CLI compatibility
        self.max_resp_len = max_resp_len
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_context(self, ctx_ids: torch.Tensor, ctx_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            z1, z2 = self.model.encode_z1_z2_deterministic(
                ctx_ids.to(self.device), ctx_mask.to(self.device),
            )
        return z1.float(), z2.float()

    @torch.no_grad()
    def score(self, z1: torch.Tensor, z2: torch.Tensor, completions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(
            completions, return_tensors="pt",
            truncation=True, max_length=self.max_resp_len, padding=True,
        )
        resp_ids = enc["input_ids"].to(self.device)
        resp_mask = enc["attention_mask"].to(self.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.transformer(
                input_ids=resp_ids, attention_mask=resp_mask,
                use_cache=False, return_dict=True,
            )
            hidden = out.last_hidden_state
            last_idx = resp_mask.sum(dim=1) - 1
            idx_e = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            resp_h = hidden.gather(1, idx_e).squeeze(1)

            joint = self.model.joint_outcome_head(
                torch.cat([z1.to(hidden.dtype), z2.to(hidden.dtype), resp_h], dim=-1)
            ).float().squeeze(-1)
            prior = self.model.z_combined_reward_head(
                torch.cat([z1.to(hidden.dtype), z2.to(hidden.dtype)], dim=-1)
            ).float().squeeze(-1)
        return joint, prior


# ──────────────────────────────────────────────────────────────────────────────
# GRPO prompt dataset
# ──────────────────────────────────────────────────────────────────────────────
class GRPOPromptDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if not (row.get("gold_action") and row.get("gold_belief")):
                    continue
                for ex in iter_training_examples(row):
                    self.samples.append({
                        "scenario_uid": row.get("scenario_uid", str(row.get("scenario_id"))),
                        "task": ex["task"],
                        "context": ex["context"],
                        "question": ex["question"],
                        "answer": ex["answer"],
                        "encoder_context": build_encoder_context(ex["context"], ex["question"]),
                    })
        print(f"Loaded {len(self.samples)} GRPO prompts from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ──────────────────────────────────────────────────────────────────────────────
# Generation + scoring helpers
# ──────────────────────────────────────────────────────────────────────────────
def policy_logprobs_with_prefix(
    model,
    mental_prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Per-token log-probs of the completion tokens, with mental prefix injected."""
    embed = model.get_input_embeddings()
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
    tok_embeds = embed(full_ids)

    mental = mental_prefix_embeds.to(device=device, dtype=tok_embeds.dtype)
    full_embeds = torch.cat([mental, tok_embeds], dim=1)
    full_mask = torch.ones(full_embeds.size()[:2], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
        logits = out.logits[0]

    prefix_len = mental.size(1)
    start = prefix_len + prompt_ids.size(0) - 1
    end = start + completion_ids.size(0)
    comp_logits = logits[start:end]
    log_probs = F.log_softmax(comp_logits.float(), dim=-1)
    comp_ids_dev = completion_ids.to(device)
    return log_probs.gather(1, comp_ids_dev.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate_with_prefix(
    model,
    mental_prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    tokenizer,
    device: torch.device,
    group_size: int = 8,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Sample `group_size` completions conditioned on the mental prefix."""
    was_training = model.training
    model.eval()

    embed = model.get_input_embeddings()
    prompt_ids_b = prompt_ids.unsqueeze(0).expand(group_size, -1).to(device)
    tok_embeds = embed(prompt_ids_b)
    mental = mental_prefix_embeds.expand(group_size, -1, -1).to(device=device, dtype=tok_embeds.dtype)
    if not torch.isfinite(mental).all():
        raise RuntimeError("Non-finite mental prefix before generation.")
    if not torch.isfinite(tok_embeds).all():
        raise RuntimeError("Non-finite token embeddings before generation.")
    full_embeds = torch.cat([mental, tok_embeds], dim=1)
    attn = torch.ones(full_embeds.size()[:2], dtype=torch.long, device=device)

    out = model.generate(
        inputs_embeds=full_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remove_invalid_values=True,
        renormalize_logits=True,
        use_cache=True,
    )
    completion_ids_list = [out[i] for i in range(group_size)]
    completion_texts = [
        tokenizer.decode(c, skip_special_tokens=True).strip() or "..."
        for c in completion_ids_list
    ]
    model.train(was_training)
    return completion_ids_list, completion_texts


# ──────────────────────────────────────────────────────────────────────────────
# GRPO trainer
# ──────────────────────────────────────────────────────────────────────────────
class BigToMGRPO:
    def __init__(self, policy, ref, projector, reward_model, tokenizer, policy_device, ref_device, *,
                 group_size=8, max_gen_len=200, clip_eps=0.2, kl_coeff=0.06,
                 temperature=0.8, top_p=0.95, max_grad_norm=1.0, log_ratio_clip=10.0):
        self.policy = policy
        self.ref = ref
        self.projector = projector
        self.rm = reward_model
        self.tok = tokenizer
        self.policy_device = policy_device
        self.ref_device = ref_device
        self.group_size = group_size
        self.max_gen_len = max_gen_len
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.temperature = temperature
        self.top_p = top_p
        self.max_grad_norm = max_grad_norm
        self.log_ratio_clip = log_ratio_clip

    def step(self, batch_samples, optimizer, scheduler=None):
        t0 = time.time()
        was_training = self.policy.training
        self.policy.eval()
        self.ref.eval()

        ctx_texts = [s["encoder_context"] for s in batch_samples]
        ctx_enc = self.tok(
            ctx_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=768,
        )
        z1_ctx, z2_ctx = self.rm.encode_context(ctx_enc["input_ids"], ctx_enc["attention_mask"])

        all_prompt_ids = []
        all_completion_ids = []
        all_completion_texts = []
        all_mental_prefixes = []
        all_z1_rm = []
        all_z2_rm = []

        for i, sample in enumerate(batch_samples):
            prompt = build_prompt(sample["context"], sample["question"], sample["task"])
            prompt_ids = self.tok(
                prompt, return_tensors="pt", truncation=True,
                max_length=1024, add_special_tokens=True,
            )["input_ids"][0]

            z1_pol = z1_ctx[i:i + 1].to(self.policy_device, dtype=torch.float32)
            z2_pol = z2_ctx[i:i + 1].to(self.policy_device, dtype=torch.float32)
            mental = self.projector(z1_pol, z2_pol).detach()

            comp_ids, comp_texts = generate_with_prefix(
                self.policy,
                mental,
                prompt_ids,
                self.tok,
                self.policy_device,
                group_size=self.group_size,
                max_new_tokens=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            all_prompt_ids.extend([prompt_ids] * self.group_size)
            all_completion_ids.extend(comp_ids)
            all_completion_texts.extend(comp_texts)
            all_mental_prefixes.extend([mental] * self.group_size)
            all_z1_rm.append(z1_ctx[i:i + 1].expand(self.group_size, -1))
            all_z2_rm.append(z2_ctx[i:i + 1].expand(self.group_size, -1))

        z1_all = torch.cat(all_z1_rm, dim=0)
        z2_all = torch.cat(all_z2_rm, dim=0)
        joint_r, prior_r = self.rm.score(z1_all, z2_all, all_completion_texts)

        group_size = self.group_size
        num_prompts = len(batch_samples)
        rewards_view = joint_r.view(num_prompts, group_size)
        baseline = rewards_view.mean(dim=1, keepdim=True)
        std = rewards_view.std(dim=1, unbiased=False, keepdim=True) + 1e-8
        advantages = ((rewards_view - baseline) / std).flatten().clamp(-3.0, 3.0)

        with torch.no_grad():
            old_lp = [
                policy_logprobs_with_prefix(
                    self.policy,
                    all_mental_prefixes[j],
                    all_prompt_ids[j],
                    all_completion_ids[j],
                    self.policy_device,
                )
                for j in range(len(all_prompt_ids))
            ]
            ref_lp = None
            if self.kl_coeff > 0:
                ref_lp = [
                    policy_logprobs_with_prefix(
                        self.ref,
                        all_mental_prefixes[j].to(self.ref_device),
                        all_prompt_ids[j],
                        all_completion_ids[j],
                        self.ref_device,
                    )
                    for j in range(len(all_prompt_ids))
                ]

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_kl = 0.0
        total_log_ratio = 0.0
        total_clip_frac = 0.0
        total_completion_len = 0.0
        max_abs_log_ratio = 0.0
        num_items = len(all_prompt_ids)

        for j in range(num_items):
            new_lp = policy_logprobs_with_prefix(
                self.policy,
                all_mental_prefixes[j],
                all_prompt_ids[j],
                all_completion_ids[j],
                self.policy_device,
            )
            old_lp_j = old_lp[j].detach().to(self.policy_device)
            raw_log_ratio = new_lp - old_lp_j
            log_ratio = raw_log_ratio.clamp(-self.log_ratio_clip, self.log_ratio_clip)
            ratio = torch.exp(log_ratio)
            adv = advantages[j].to(self.policy_device)
            clipped_ratio = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
            s1 = ratio * adv
            s2 = clipped_ratio * adv
            pg = -torch.min(s1, s2).mean()

            kl = torch.tensor(0.0, device=self.policy_device)
            if ref_lp is not None:
                lr = (ref_lp[j].detach().to(self.policy_device) - new_lp).clamp(
                    -self.log_ratio_clip, self.log_ratio_clip,
                )
                kl = (torch.exp(lr) - lr - 1).mean()

            loss_j = (pg + self.kl_coeff * kl) / num_items
            loss_j.backward()

            total_loss += float(loss_j.detach().item())
            total_kl += float(kl.detach().item())
            total_log_ratio += float(log_ratio.mean().detach().item())
            total_clip_frac += float((ratio != clipped_ratio).float().mean().detach().item())
            total_completion_len += float(new_lp.numel())
            max_abs_log_ratio = max(
                max_abs_log_ratio,
                float(raw_log_ratio.abs().max().detach().item()),
            )

        trainable = [p for p in self.policy.parameters() if p.requires_grad] + \
                    [p for p in self.projector.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        self.policy.train(was_training)
        t1 = time.time()
        return {
            "policy_loss": total_loss,
            "mean_reward": joint_r.mean().item(),
            "std_reward": joint_r.std().item(),
            "mean_joint": joint_r.mean().item(),
            "mean_prior": prior_r.mean().item(),
            "mean_kl": total_kl / max(num_items, 1),
            "mean_log_ratio": total_log_ratio / max(num_items, 1),
            "clip_frac": total_clip_frac / max(num_items, 1),
            "mean_completion_len": total_completion_len / max(num_items, 1),
            "max_abs_log_ratio": max_abs_log_ratio,
            "step_time_s": t1 - t0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def _load_stage2_policy(base_model_name, stage2_ckpt: Path, device):
    print(f"Loading policy base: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    lora_dir = stage2_ckpt / "policy_lora"
    if not lora_dir.exists():
        raise FileNotFoundError(f"No policy_lora in {stage2_ckpt}")
    policy = PeftModel.from_pretrained(base, str(lora_dir), is_trainable=True)
    policy.config.pad_token_id = base.config.pad_token_id
    policy.gradient_checkpointing_disable()
    return policy


def _load_ref_policy(base_model_name, stage2_ckpt: Path, device):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    lora_dir = stage2_ckpt / "policy_lora"
    ref = PeftModel.from_pretrained(base, str(lora_dir), is_trainable=False)
    ref = ref.merge_and_unload()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default="projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stage1_ckpt", type=str,
                    default="projects/bigtom/checkpoints/stage1/epoch_2")
    ap.add_argument("--stage2_ckpt", type=str,
                    default="projects/bigtom/checkpoints/stage2/epoch_1")
    ap.add_argument("--out", type=str,
                    default="projects/bigtom/checkpoints/stage3")
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_gen_len", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--kl_coeff", type=float, default=0.06)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--ensemble_weight", type=float, default=1.0)
    ap.add_argument("--log_ratio_clip", type=float, default=10.0)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ngpus = torch.cuda.device_count()
    encoder_device = torch.device("cuda:0")
    policy_device = torch.device(f"cuda:{1 if ngpus > 1 else 0}")
    ref_device = torch.device(f"cuda:{2 if ngpus > 2 else (1 if ngpus > 1 else 0)}")
    print(f"Devices: encoder={encoder_device} policy={policy_device} ref={ref_device}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    encoder = load_stage1_encoder(
        args.base_model, Path(args.stage1_ckpt), args.z_dim, encoder_device,
    )
    reward_model = FrozenBigToMRewardModel(
        encoder, tok, encoder_device, ensemble_weight=args.ensemble_weight,
    )

    policy = _load_stage2_policy(args.base_model, Path(args.stage2_ckpt), policy_device)

    proj_pt = Path(args.stage2_ckpt) / "projector.pt"
    projector = MentalPrefixProjector(
        z_dim=args.z_dim, hidden_size=policy.config.hidden_size, num_prefix=MENTAL_PREFIX_LEN,
    ).to(policy_device).float()
    projector.load_state_dict(torch.load(proj_pt, map_location=policy_device)["projector"])
    for p in projector.parameters():
        p.requires_grad = True

    ref = _load_ref_policy(args.base_model, Path(args.stage2_ckpt), ref_device)

    dataset = GRPOPromptDataset(args.data)
    loader = DataLoader(
        dataset, batch_size=args.prompts_per_step, shuffle=True,
        collate_fn=lambda batch: batch,
    )

    trainable = [p for p in policy.parameters() if p.requires_grad] + \
                [p for p in projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * args.max_steps), max(1, args.max_steps),
    )

    trainer = BigToMGRPO(
        policy=policy,
        ref=ref,
        projector=projector,
        reward_model=reward_model,
        tokenizer=tok,
        policy_device=policy_device,
        ref_device=ref_device,
        group_size=args.group_size,
        max_gen_len=args.max_gen_len,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        temperature=args.temperature,
        top_p=args.top_p,
        log_ratio_clip=args.log_ratio_clip,
    )

    os.makedirs(args.out, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for batch_samples in loader:
            metrics = trainer.step(batch_samples, optimizer, scheduler)
            step += 1
            print(
                f"[epoch {epoch} step {step}/{args.max_steps}] "
                + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()),
                flush=True,
            )
            if step % args.save_every == 0 or step >= args.max_steps:
                ckpt = Path(args.out) / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt / "policy_lora")
                tok.save_pretrained(ckpt / "policy_lora")
                torch.save(
                    {"projector": projector.state_dict(), "args": vars(args)},
                    ckpt / "projector.pt",
                )
                print(f"Saved {ckpt}")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
