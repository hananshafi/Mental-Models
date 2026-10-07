#!/usr/bin/env python3
"""
Stage 2: GRPO with Composite ToM-Reward
=========================================
Group Relative Policy Optimization with multi-component ToM-aware reward:
  R_total = α₁·R_belief + α₂·R_perspective + α₃·R_format + α₄·R_roleplay

Generates groups of candidates, scores them with a composite reward, computes
group-normalized advantages, and applies a PPO-clip update.

Starts from Stage 1 SFT checkpoint (LoRA).

Supported models:
  - Qwen/Qwen2.5-VL-7B-Instruct  (model_type: qwen2.5-vl, default)
  - Qwen/Qwen-VL-Chat             (model_type: qwen-vl-chat)

Usage:
    # Qwen2.5-VL
    CUDA_VISIBLE_DEVICES=0,1 python stage2_grpo_tom_reward.py \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --sft_checkpoint projects/mmrole/checkpoints/stage1_sft/best

    # Qwen-VL-Chat
    CUDA_VISIBLE_DEVICES=0 python stage2_grpo_tom_reward.py \
        --base_model Qwen/Qwen-VL-Chat \
        --sft_checkpoint projects/mmrole/checkpoints/stage1_sft_qwenvl/best
"""

import os
import sys
import json
import gc
import time
import argparse
import random
import re
from typing import List, Optional, Dict, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "fileno"):
    try:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    except Exception:
        pass

print(">> stage2_grpo_tom_reward.py starting...", flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import PeftModel
from PIL import Image

# Shared model utilities (supports Qwen2.5-VL + Qwen-VL-Chat)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    detect_model_type, load_base_model, get_tokenizer,
    resolve_image, load_and_resize_image,
    prepare_generation_inputs,
)
from resume_state import (
    check_resume_args, load_lora_weights, load_resume_state,
    require_resume_dir, restore_rng_state, save_resume_dir,
)

# Arguments that must match between an interrupted run and its resumption.
RESUME_INVARIANT_ARGS = (
    "base_model", "model_type", "sft_checkpoint", "train_path", "image_dir",
    "max_examples", "num_iterations", "prompts_per_iter", "group_size",
    "max_gen_len", "lr", "kl_coeff", "clip_eps", "seed",
)

print(">> All imports done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chain-of-Belief parsing utilities
# ──────────────────────────────────────────────────────────────────────────────

def parse_chain_of_belief(text: str) -> Dict[str, str]:
    """Parse chain-of-belief formatted text into components."""
    result = {
        "perception": "",
        "belief_1st": "",
        "belief_2nd": "",
        "response": "",
    }
    for tag in result:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[tag] = match.group(1).strip()
    return result


def has_valid_format(text: str) -> bool:
    """Check if text has all required chain-of-belief tags."""
    required = ["<perception>", "</perception>",
                 "<belief_1st>", "</belief_1st>",
                 "<belief_2nd>", "</belief_2nd>",
                 "<response>", "</response>"]
    return all(tag in text for tag in required)


# ──────────────────────────────────────────────────────────────────────────────
# ToM Reward Functions (rule-based, no external LLM needed)
# ──────────────────────────────────────────────────────────────────────────────

class ToMRewardModel:
    """
    Composite ToM-aware reward model.
    R_total = α₁·R_belief + α₂·R_perspective + α₃·R_format + α₄·R_roleplay

    All rewards are rule-based for fast scoring during GRPO.
    """

    def __init__(self,
                 alpha_belief: float = 0.35,
                 alpha_perspective: float = 0.25,
                 alpha_format: float = 0.15,
                 alpha_roleplay: float = 0.25):
        self.alpha_belief = alpha_belief
        self.alpha_perspective = alpha_perspective
        self.alpha_format = alpha_format
        self.alpha_roleplay = alpha_roleplay

    def score(self, completions: List[str], references: List[Dict]) -> List[float]:
        """Score a batch of completions against reference annotations."""
        rewards = []
        for comp, ref in zip(completions, references):
            r = self._score_single(comp, ref)
            rewards.append(r)
        return rewards

    def _score_single(self, completion: str, reference: Dict) -> float:
        """Score a single completion."""
        parsed = parse_chain_of_belief(completion)

        r_format = self._score_format(completion, parsed)
        r_belief = self._score_belief(parsed, reference)
        r_perspective = self._score_perspective(parsed, reference)
        r_roleplay = self._score_roleplay(parsed, reference)

        total = (self.alpha_belief * r_belief +
                 self.alpha_perspective * r_perspective +
                 self.alpha_format * r_format +
                 self.alpha_roleplay * r_roleplay)
        return total

    def _score_format(self, raw_text: str, parsed: Dict) -> float:
        """R_format: reward for correct chain-of-belief structure."""
        score = 0.0
        # Check all 4 tags present
        if has_valid_format(raw_text):
            score += 0.5
        # Check each section is non-empty
        for section in ["perception", "belief_1st", "belief_2nd", "response"]:
            if len(parsed.get(section, "")) > 10:
                score += 0.125
        # Penalize if response is too short or too long
        resp_len = len(parsed.get("response", ""))
        if resp_len > 20:
            score += 0.0  # already counted above
        elif resp_len > 0:
            score -= 0.1
        return min(max(score, 0.0), 1.0)

    def _score_belief(self, parsed: Dict, reference: Dict) -> float:
        """R_belief: reward for belief accuracy via keyword overlap with reference."""
        score = 0.0
        belief_text = parsed.get("belief_1st", "").lower()
        if not belief_text:
            return 0.0

        ref_belief = reference.get("target_speaker_belief", {})

        # Check coverage of key belief dimensions
        for key in ["partner_visual_focus", "partner_intent", "partner_knowledge", "partner_emotion"]:
            ref_val = ref_belief.get(key, "").lower()
            if not ref_val:
                continue
            # Extract key nouns/concepts from reference
            ref_words = set(w for w in ref_val.split() if len(w) > 4)
            if not ref_words:
                continue
            # Check overlap
            belief_words = set(belief_text.split())
            overlap = len(ref_words & belief_words)
            coverage = overlap / max(len(ref_words), 1)
            score += coverage * 0.25  # 4 dimensions, each worth 0.25

        return min(max(score, 0.0), 1.0)

    def _score_perspective(self, parsed: Dict, reference: Dict) -> float:
        """R_perspective: reward for visual perspective-taking accuracy."""
        score = 0.0
        perception_text = parsed.get("perception", "").lower()
        belief_2nd_text = parsed.get("belief_2nd", "").lower()

        if not perception_text:
            return 0.0

        # Check scene object coverage
        ref_objects = reference.get("target_scene_objects", [])
        if ref_objects:
            mentioned = 0
            for obj in ref_objects:
                obj_name = obj.get("object", "").lower()
                if obj_name and any(w in perception_text for w in obj_name.split() if len(w) > 3):
                    mentioned += 1
            obj_coverage = mentioned / max(len(ref_objects), 1)
            score += obj_coverage * 0.4

        # Check salience awareness (mentions high/medium/low)
        salience_keywords = ["high", "medium", "low", "salience", "notice", "focus", "attention"]
        if any(kw in perception_text for kw in salience_keywords):
            score += 0.2

        # Check 2nd-order visual perspective
        ref_2nd = reference.get("target_speaker_2nd_order", {})
        if belief_2nd_text and ref_2nd:
            ref_2nd_text = " ".join(str(v) for v in ref_2nd.values()).lower()
            ref_words = set(w for w in ref_2nd_text.split() if len(w) > 4)
            if ref_words:
                belief_words = set(belief_2nd_text.split())
                overlap = len(ref_words & belief_words) / max(len(ref_words), 1)
                score += overlap * 0.4

        return min(max(score, 0.0), 1.0)

    def _score_roleplay(self, parsed: Dict, reference: Dict) -> float:
        """R_roleplay: reward for in-character response quality."""
        response = parsed.get("response", "")
        if not response or len(response) < 10:
            return 0.0

        score = 0.0

        # Length appropriateness (20-500 chars is good)
        if 20 <= len(response) <= 500:
            score += 0.3
        elif len(response) > 500:
            score += 0.15  # too long
        else:
            score += 0.1  # too short

        # Coherence: response should reference dialogue context
        ref_utterance = reference.get("current_utterance", "")
        if ref_utterance:
            ref_words = set(w.lower() for w in ref_utterance.split() if len(w) > 4)
            resp_words = set(w.lower() for w in response.split() if len(w) > 4)
            if ref_words:
                thematic_overlap = len(ref_words & resp_words) / max(len(ref_words), 1)
                score += min(thematic_overlap, 0.4) * 0.5

        # Character name reference
        speaker = reference.get("speaker_name", "")
        partner = reference.get("partner_name", "")
        if speaker.lower() in response.lower() or partner.lower() in response.lower():
            score += 0.2

        return min(max(score, 0.0), 1.0)

    def score_detailed(self, completion: str, reference: Dict) -> Dict[str, float]:
        """Return per-component scores for logging."""
        parsed = parse_chain_of_belief(completion)
        return {
            "r_format": self._score_format(completion, parsed),
            "r_belief": self._score_belief(parsed, reference),
            "r_perspective": self._score_perspective(parsed, reference),
            "r_roleplay": self._score_roleplay(parsed, reference),
        }


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GRPOBeliefDataset(Dataset):
    """Dataset for GRPO: provides prompts and reference annotations."""

    def __init__(self, data_path: str, image_dir: str, max_examples: int = -1):
        self.image_dir = image_dir
        self.examples = []

        print(f"Loading GRPO data from {data_path}...", flush=True)
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        if max_examples > 0:
            self.examples = self.examples[:max_examples]
        random.shuffle(self.examples)

        print(f"  GRPO examples: {len(self.examples)}", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_grpo_prompt(example: dict) -> str:
    """Build the prompt for GRPO generation (same as SFT but without target)."""
    speaker = example.get("speaker_name", "Speaker")
    partner = example.get("partner_name", "Partner")
    speaker_profile = example.get("speaker_profile", "")[:1500]
    partner_profile = example.get("partner_profile", "")[:1500]

    history_lines = []
    for t in example.get("dialogue_history", []):
        history_lines.append(f"[{t['speaker']}]: {t['utterance']}")
    history_text = "\n".join(history_lines[-6:])

    prompt = (
        f"You are {speaker}, engaging in a conversation with {partner} about an image.\n\n"
        f"## {speaker}'s Profile\n{speaker_profile}\n\n"
        f"## {partner}'s Profile\n{partner_profile}\n\n"
    )
    if history_text:
        prompt += f"## Dialogue History\n{history_text}\n\n"

    prompt += (
        f"Analyze the image from {speaker}'s perspective and generate a structured "
        f"chain-of-belief response.\n\n"
        f"Use the format:\n"
        f"<perception>...</perception>\n"
        f"<belief_1st>...</belief_1st>\n"
        f"<belief_2nd>...</belief_2nd>\n"
        f"<response>...</response>"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Trainer
# ──────────────────────────────────────────────────────────────────────────────

class GRPOToMTrainer:
    """GRPO trainer with composite ToM reward for Visual ToM."""

    def __init__(self, policy_model, ref_model, reward_model: ToMRewardModel,
                 processor, model_type: str, image_dir: str,
                 policy_device="cuda:0", ref_device="cuda:0",
                 group_size: int = 8, max_gen_len: int = 512,
                 clip_eps: float = 0.2, kl_coeff: float = 0.05,
                 temperature: float = 0.8, top_p: float = 0.95,
                 max_grad_norm: float = 1.0):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.processor = processor
        self.model_type = model_type
        self.tokenizer = get_tokenizer(processor, model_type)
        self.image_dir = image_dir
        self.device = policy_device
        self.ref_device = ref_device
        self.group_size = group_size
        self.max_gen_len = max_gen_len
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.temperature = temperature
        self.top_p = top_p
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def generate_candidates(self, examples: List[Dict]) -> Tuple:
        """Generate group_size candidates for each example."""
        self.policy.eval()
        all_completions = []
        all_prompt_ids = []
        all_prompt_aux = []   # per-candidate {pixel_values, image_grid_thw}
        all_completion_ids = []
        all_references = []

        for ex_idx, example in enumerate(examples):
            prompt_text = build_grpo_prompt(example)
            image_path = resolve_image(example, self.image_dir)

            # Build generation input (model-agnostic via model_utils)
            inputs, prompt_len = prepare_generation_inputs(
                prompt_text, image_path, self.processor,
                self.model_type, device=str(self.device)
            )
            prompt_ids = inputs["input_ids"][0]
            aux = {}
            if "pixel_values" in inputs:
                aux["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                aux["image_grid_thw"] = inputs["image_grid_thw"]

            # Generate group_size candidates
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=self.max_gen_len,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.group_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for seq in outputs:
                completion_ids = seq[prompt_ids.shape[0]:]
                completion_text = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )
                completion_text = completion_text.strip()
                if not completion_text:
                    completion_text = "<perception>\n</perception>\n<belief_1st>\n</belief_1st>\n<belief_2nd>\n</belief_2nd>\n<response>\n...\n</response>"

                all_completions.append(completion_text)
                all_prompt_ids.append(prompt_ids)
                all_prompt_aux.append(aux)
                all_completion_ids.append(
                    self.tokenizer(
                        completion_text, add_special_tokens=False,
                        return_tensors="pt"
                    ).input_ids.squeeze(0)
                )
                all_references.append(example)

        self.policy.train()
        return (all_completions, all_prompt_ids, all_prompt_aux,
                all_completion_ids, all_references)

    def compute_log_probs(self, model, prompt_ids_list, prompt_aux_list,
                          completion_ids_list, target_device=None):
        """Compute per-token log probs for completions.

        When the VLM needs pixel_values (Qwen2.5-VL), we forward the full
        prompt+completion sequence together with the image tensors captured
        at generation time so the model actually attends to the image.
        """
        dev = target_device or self.device
        log_probs_list = []

        for prompt_ids, aux, completion_ids in zip(
            prompt_ids_list, prompt_aux_list, completion_ids_list
        ):
            completion_ids = completion_ids.to(prompt_ids.device)
            full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(dev)
            attention_mask = torch.ones_like(full_ids)

            fwd_kwargs = dict(input_ids=full_ids, attention_mask=attention_mask)
            if aux.get("pixel_values") is not None:
                fwd_kwargs["pixel_values"] = aux["pixel_values"].to(dev)
            if aux.get("image_grid_thw") is not None:
                fwd_kwargs["image_grid_thw"] = aux["image_grid_thw"].to(dev)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**fwd_kwargs)
                logits = outputs.logits[0]

            prompt_len = prompt_ids.shape[0]
            comp_logits = logits[prompt_len - 1: prompt_len - 1 + completion_ids.shape[0]]
            comp_log_probs = F.log_softmax(comp_logits, dim=-1)
            token_log_probs = comp_log_probs.gather(
                1, completion_ids.unsqueeze(0).to(dev).T
            ).squeeze(-1)

            log_probs_list.append(token_log_probs.to(self.device))

        return log_probs_list

    def grpo_step(self, examples: List[Dict], optimizer, scheduler=None):
        """One GRPO step: generate, score, advantage, gradient update."""
        t0 = time.time()
        num_examples = len(examples)

        # 1. Generate candidates
        print(f"    [1/5] Generating {self.group_size}x{num_examples} candidates...",
              flush=True)
        (completions, prompt_ids_list, prompt_aux_list,
         completion_ids_list, references) = self.generate_candidates(examples)
        t1 = time.time()
        print(f"    [1/5] Done ({t1-t0:.0f}s)", flush=True)

        # 2. Score with composite ToM reward
        print(f"    [2/5] Scoring {len(completions)} candidates...", flush=True)
        rewards = self.reward_model.score(completions, references)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        t2 = time.time()
        print(f"    [2/5] Done ({t2-t1:.0f}s)", flush=True)

        # 3. Group-normalized advantages
        advantages = torch.zeros_like(rewards_tensor)
        for i in range(num_examples):
            start = i * self.group_size
            end = start + self.group_size
            group_rewards = rewards_tensor[start:end]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std() + 1e-8
            advantages[start:end] = (group_rewards - mean_r) / std_r
        advantages = advantages.clamp(-3.0, 3.0)

        # 4. Compute old + ref log probs
        print(f"    [3/5] Computing old log probs...", flush=True)
        with torch.no_grad():
            old_log_probs_list = self.compute_log_probs(
                self.policy, prompt_ids_list, prompt_aux_list, completion_ids_list
            )
            ref_log_probs_list = None
            if self.ref_model is not None:
                print(f"    [4/5] Computing ref log probs...", flush=True)
                ref_log_probs_list = self.compute_log_probs(
                    self.ref_model, prompt_ids_list, prompt_aux_list,
                    completion_ids_list, target_device=self.ref_device,
                )
        t3 = time.time()
        print(f"    [4/5] Done ({t3-t2:.0f}s)", flush=True)

        # 5. Policy gradient with PPO-clip
        print(f"    [5/5] Gradient update...", flush=True)
        total_policy_loss = 0.0
        total_kl = 0.0

        all_loss = torch.tensor(0.0, device=self.device)
        all_kl = torch.tensor(0.0, device=self.device)

        for j in range(len(prompt_ids_list)):
            full_ids = torch.cat([
                prompt_ids_list[j],
                completion_ids_list[j].to(prompt_ids_list[j].device)
            ], dim=0).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(full_ids)

            fwd_kwargs = dict(input_ids=full_ids, attention_mask=attention_mask)
            aux_j = prompt_aux_list[j]
            if aux_j.get("pixel_values") is not None:
                fwd_kwargs["pixel_values"] = aux_j["pixel_values"].to(self.device)
            if aux_j.get("image_grid_thw") is not None:
                fwd_kwargs["image_grid_thw"] = aux_j["image_grid_thw"].to(self.device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                outputs = self.policy(**fwd_kwargs)
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
            [p for p in self.policy.parameters() if p.requires_grad],
            max_norm=self.max_grad_norm,
        )
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        total_policy_loss = all_loss.item()
        total_kl = all_kl.item() / len(prompt_ids_list)

        t4 = time.time()
        print(f"    [5/5] Done ({t4-t3:.0f}s) | Total: {t4-t0:.0f}s", flush=True)

        # Log detailed reward breakdown for first few samples
        detailed_rewards = []
        for k in range(min(3, len(completions))):
            detailed_rewards.append(
                self.reward_model.score_detailed(completions[k], references[k])
            )

        return {
            "policy_loss": total_policy_loss,
            "mean_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std().item(),
            "mean_kl": total_kl,
            "mean_advantage": advantages.mean().item(),
            "format_rate": sum(1 for c in completions if has_valid_format(c)) / len(completions),
            "detailed_rewards": detailed_rewards,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {n_gpus}", flush=True)

    ref_device = f"cuda:{n_gpus - 1}" if n_gpus > 1 else "cuda:0"

    # Detect model type
    model_type = args.model_type or detect_model_type(args.base_model)
    print(f"Model type: {model_type}", flush=True)

    # Load model + processor/tokenizer
    base_model, processor, model_type = load_base_model(args.base_model, model_type)

    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        print(f"Loading SFT LoRA from {args.sft_checkpoint}...", flush=True)
        policy_model = PeftModel.from_pretrained(
            base_model, args.sft_checkpoint, is_trainable=True,
        )
        print("  SFT LoRA loaded (trainable).", flush=True)
    else:
        print("WARNING: No SFT checkpoint provided, using base model.", flush=True)
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_model, lora_config)

    policy_model = policy_model.to(device)
    # Without checkpointing one 7B VLM update needs more than a 48 GB GPU holds;
    # non-reentrant checkpointing preserves the RNG stream, so updates are unchanged.
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy_model.enable_input_require_grads()
    policy_model.print_trainable_parameters()

    # Load reference model (frozen SFT or base)
    print(f"Loading reference model...", flush=True)
    ref_base, _, _ = load_base_model(args.base_model, model_type)
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        ref_model = PeftModel.from_pretrained(ref_base, args.sft_checkpoint)
        ref_model = ref_model.merge_and_unload()
    else:
        ref_model = ref_base
    ref_model = ref_model.to(ref_device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  Reference model on {ref_device}", flush=True)

    # Reward model
    reward_model = ToMRewardModel(
        alpha_belief=args.alpha_belief,
        alpha_perspective=args.alpha_perspective,
        alpha_format=args.alpha_format,
        alpha_roleplay=args.alpha_roleplay,
    )

    # Dataset
    dataset = GRPOBeliefDataset(args.train_path, args.image_dir,
                                max_examples=args.max_examples)

    # Optimizer + scheduler
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = args.num_iterations
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Trainer
    trainer = GRPOToMTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        processor=processor,
        model_type=model_type,
        image_dir=args.image_dir,
        policy_device=str(device),
        ref_device=ref_device,
        group_size=args.group_size,
        max_gen_len=args.max_gen_len,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 2: GRPO with ToM-Reward", flush=True)
    print(f"  Base model: {args.base_model}", flush=True)
    print(f"  SFT checkpoint: {args.sft_checkpoint}", flush=True)
    print(f"  Train examples: {len(dataset)}", flush=True)
    print(f"  Iterations: {args.num_iterations}", flush=True)
    print(f"  Group size: {args.group_size}", flush=True)
    print(f"  Batch (prompts/iter): {args.prompts_per_iter}", flush=True)
    print(f"  LR: {args.lr}, KL coeff: {args.kl_coeff}", flush=True)
    print(f"  Reward weights: belief={args.alpha_belief} perspective={args.alpha_perspective} "
          f"format={args.alpha_format} roleplay={args.alpha_roleplay}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args)
    config["total_steps"] = total_steps
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_reward = -float("inf")
    reward_history = []
    patience_counter = 0
    start_iteration = 1
    if args.resume:
        resume_dir = require_resume_dir(args.output_dir)
        state = load_resume_state(resume_dir)
        check_resume_args(state["args"], args, RESUME_INVARIANT_ARGS)
        load_lora_weights(policy_model, os.path.join(resume_dir, "policy"))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_reward, patience_counter = state["best_reward"], state["patience_counter"]
        reward_history = state["reward_history"]
        start_iteration = (
            args.num_iterations + 1 if state["stopped_early"] else state["next_iteration"]
        )
        restore_rng_state(state["rng"])
        print(f"  Resumed from {resume_dir} at iteration {start_iteration}", flush=True)

    def save_resume(next_iteration, stopped_early):
        save_resume_dir(
            args.output_dir,
            lambda d: policy_model.save_pretrained(os.path.join(d, "policy")),
            {
                "args": config, "next_iteration": next_iteration,
                "stopped_early": stopped_early, "best_reward": best_reward,
                "patience_counter": patience_counter, "reward_history": reward_history,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            },
        )

    start_time = time.time()
    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "a")

    for iteration in range(start_iteration, args.num_iterations + 1):
        # Sample batch of prompts
        batch_indices = random.sample(range(len(dataset)), min(args.prompts_per_iter, len(dataset)))
        batch_examples = [dataset[i] for i in batch_indices]

        print(f"\n  Iteration {iteration}/{args.num_iterations}", flush=True)

        metrics = trainer.grpo_step(batch_examples, optimizer, scheduler)

        # Log
        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  reward={metrics['mean_reward']:.4f}±{metrics['std_reward']:.4f} "
            f"loss={metrics['policy_loss']:.4f} kl={metrics['mean_kl']:.4f} "
            f"format={metrics['format_rate']:.0%} lr={lr_now:.2e} "
            f"time={elapsed:.0f}s",
            flush=True,
        )
        if metrics.get("detailed_rewards"):
            for k, dr in enumerate(metrics["detailed_rewards"][:2]):
                print(f"    sample {k}: {dr}", flush=True)

        log_entry = {
            "iteration": iteration,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "policy_loss": metrics["policy_loss"],
            "mean_kl": metrics["mean_kl"],
            "format_rate": metrics["format_rate"],
            "lr": lr_now,
            "elapsed": elapsed,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Track best
        reward_history.append(metrics["mean_reward"])
        if metrics["mean_reward"] > best_reward:
            best_reward = metrics["mean_reward"]
            patience_counter = 0
            best_dir = os.path.join(args.output_dir, "best")
            print(f"  New best reward! Saving to {best_dir}", flush=True)
            policy_model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
        else:
            patience_counter += 1

        # Save checkpoint
        if args.save_every > 0 and iteration % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"iter_{iteration}")
            policy_model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)

        # Early stopping
        stop = args.patience > 0 and patience_counter >= args.patience
        if args.resume_every > 0 and (
            iteration % args.resume_every == 0 or stop or iteration == args.num_iterations
        ):
            log_file.flush()
            save_resume(iteration + 1, stop)
        if stop:
            print(f"\n  Early stopping: no improvement for {args.patience} iterations",
                  flush=True)
            break

    log_file.close()

    # Final save
    final_dir = os.path.join(args.output_dir, "final")
    policy_model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 2 GRPO Complete!", flush=True)
    print(f"  Iterations: {iteration}", flush=True)
    print(f"  Best reward: {best_reward:.4f}", flush=True)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nNext: Stage 3 DPO", flush=True)
    print(f"  python stage3_dpo_contrastive.py \\", flush=True)
    print(f"    --grpo_checkpoint {os.path.join(args.output_dir, 'best')} \\", flush=True)
    print(f"    --base_model {args.base_model}", flush=True)

    del policy_model, ref_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Stage 2: GRPO with ToM-Reward")
    # Model
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat"],
                        help="Model type (auto-detected from base_model if empty)")
    parser.add_argument("--sft_checkpoint", type=str,
                        default="projects/mmrole/checkpoints/stage1_sft/best")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    # Data
    parser.add_argument("--train_path", type=str,
                        default="projects/mmrole/training_data/train/belief_prediction.jsonl")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--max_examples", type=int, default=-1)
    # GRPO
    parser.add_argument("--num_iterations", type=int, default=600)
    parser.add_argument("--prompts_per_iter", type=int, default=4,
                        help="Number of unique prompts per GRPO iteration")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6)
    # Reward weights
    parser.add_argument("--alpha_belief", type=float, default=0.35)
    parser.add_argument("--alpha_perspective", type=float, default=0.25)
    parser.add_argument("--alpha_format", type=float, default=0.15)
    parser.add_argument("--alpha_roleplay", type=float, default=0.25)
    # Training
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (0=disable)")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/checkpoints/stage2_grpo")
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <output_dir>/last after an interruption.")
    parser.add_argument("--resume_every", type=int, default=10,
                        help="Refresh <output_dir>/last every N iterations.")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
