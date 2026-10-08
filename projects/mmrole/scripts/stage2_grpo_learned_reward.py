#!/usr/bin/env python3
"""
Stage 2 (Learned-Reward variant): GRPO Guided by Frozen Visual ToM Reward Model
================================================================================
Sotopia-style training pipeline for MMRole:

  Stage 0 -> stage0_reward_model_visual_tom.py (learned VAE reward)
  Stage 1 -> stage1_sft_visual_tom.py           (chain-of-belief SFT)
  Stage 2 -> THIS FILE                          (GRPO on frozen reward)
  Stage 3 -> stage3_dpo_contrastive.py          (contrastive DPO pairs)

It consumes a frozen Stage 0 checkpoint that scores policy candidates
through the recursive ToM VAE (z1/z2 + joint/z-only ensemble).

Supported backbones: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen-VL-Chat

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python stage2_grpo_learned_reward.py \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --sft_checkpoint projects/mmrole/checkpoints/stage1_sft/best \
        --reward_checkpoint_dir projects/mmrole/checkpoints/stage0_reward_v3/best \
        --output_dir projects/mmrole/checkpoints/stage2_grpo_learned
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

print(">> stage2_grpo_learned_reward.py starting...", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    detect_model_type, load_base_model, get_tokenizer,
    resolve_image, load_and_resize_image,
    prepare_generation_inputs, default_lora_target_modules,
)
from mental_prefix_utils import FrozenMentalPrefixModel
from resume_state import (
    check_resume_args, load_lora_weights, load_resume_state,
    require_resume_dir, restore_rng_state, save_resume_dir,
)

# Arguments that must match between an interrupted run and its resumption.
RESUME_INVARIANT_ARGS = (
    "base_model", "model_type", "sft_checkpoint", "reward_base_model",
    "reward_checkpoint_dir", "mental_prefix_checkpoint_dir", "train_path", "val_path",
    "image_dir", "max_examples", "num_iterations", "prompts_per_iter", "group_size",
    "max_gen_len", "lr", "kl_coeff", "clip_eps", "lora_r", "lora_alpha", "seed",
)

# Stage 0 reward architecture is re-used from the reward training script
from stage0_reward_model_visual_tom import (
    VisualRecursiveToMRewardModel,
    CUSTOM_HEAD_NAMES as REWARD_HEAD_NAMES,
    MMROLE_REWARD_DIMS, REWARD_DIM,
    TOM_AUX_REWARD_DIM,
    format_reward_context,
    _build_context_text_for_processor,
    _build_response_text_for_processor,
)

print(">> All imports done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chain-of-belief utilities (shared with rule-based Stage 2)
# ──────────────────────────────────────────────────────────────────────────────
def parse_chain_of_belief(text: str) -> Dict[str, str]:
    result = {"perception": "", "belief_1st": "", "belief_2nd": "", "response": ""}
    for tag in result:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            result[tag] = m.group(1).strip()
    return result


def has_valid_format(text: str) -> bool:
    required = ["<perception>", "</perception>",
                "<belief_1st>", "</belief_1st>",
                "<belief_2nd>", "</belief_2nd>",
                "<response>", "</response>"]
    return all(tag in text for tag in required)


def completion_to_reward_text(text: str) -> str:
    """Canonicalize a chain-of-belief completion for reward scoring.

    Stage 0's reward model was trained on response text, but Stage 2 should not
    discard the generated mental chain. We therefore fold the structured chain
    back into a plain-text summary so the reward model can see the reasoning
    content instead of only the final <response>.
    """
    parsed = parse_chain_of_belief(text)
    if not any(v.strip() for v in parsed.values()):
        return text.strip()

    parts = []
    if parsed["perception"]:
        parts.append(f"Perception: {parsed['perception']}")
    if parsed["belief_1st"]:
        parts.append(f"First-order belief: {parsed['belief_1st']}")
    if parsed["belief_2nd"]:
        parts.append(f"Second-order belief: {parsed['belief_2nd']}")
    if parsed["response"]:
        parts.append(f"Response: {parsed['response']}")
    else:
        parts.append(f"Response: {text.strip()}")
    return "\n".join(parts).strip()


def chain_structure_bonus(text: str) -> float:
    """Reward complete structured reasoning instead of only tag presence."""
    if not has_valid_format(text):
        return 0.0
    parsed = parse_chain_of_belief(text)
    filled = sum(1 for value in parsed.values() if value.strip())
    # Max bonus matches the old 0.05 scale, but now it is graded and rewards
    # actually populating the mental chain rather than just emitting empty tags.
    return min(0.05, 0.0125 * filled)


_MISSING_IMAGE_WARNINGS = set()
_TRUNCATED_COMPLETION_WARNINGS = set()
_NONFINITE_LOGPROB_WARNINGS = set()


def resolve_image_with_warning(example: Dict, image_dir: str) -> Optional[str]:
    image_path = resolve_image(example, image_dir)
    if image_path:
        return image_path

    example_id = example.get("example_id", "<unknown>")
    raw_image = example.get("image_local") or example.get("image") or "<missing>"
    key = (example_id, raw_image)
    if key not in _MISSING_IMAGE_WARNINGS:
        _MISSING_IMAGE_WARNINGS.add(key)
        print(
            "  [WARN] Image not found for "
            f"example_id={example_id} image={raw_image}. "
            "Falling back to text-only.",
            flush=True,
        )
    return None


def _warn_nonfinite_logprob(kind: str, candidate_idx: int, detail: str) -> None:
    key = (kind, candidate_idx, detail)
    if key in _NONFINITE_LOGPROB_WARNINGS:
        return
    _NONFINITE_LOGPROB_WARNINGS.add(key)
    print(
        f"  [WARN] Non-finite {kind} log-probs at candidate {candidate_idx}; "
        f"{detail}. Applying a safe fallback for this candidate.",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Frozen Visual ToM Reward Model (mirrors Sotopia's FrozenRewardModel)
# ──────────────────────────────────────────────────────────────────────────────
class FrozenVisualToMRewardModel:
    """
    Loads the Stage 0 checkpoint (LoRA adapter + custom VAE heads) and
    exposes a `.score(samples, completions)` method for GRPO candidate
    ranking. Encodes each unique context once and reuses the (z1, z2) pair
    for all candidates from that prompt.

    Candidate ranking uses only the response-dependent joint reward head.
    The z-only heads are still trained in Stage 0 as anti-bypass regularizers,
    but they are constant within a prompt group and therefore do not affect
    GRPO ranking after per-group normalization.
    """

    def __init__(self, base_model_name: str, checkpoint_dir: str,
                 model_type: Optional[str] = None,
                 z_dim: int = 128, device: str = "cuda:0",
                 image_dir: str = "",
                 scoring_dim_indices: Optional[List[int]] = None,
                 ensemble_weight: float = 0.7,
                 max_ctx_len: int = 1024, max_resp_len: int = 256):
        self.device = device
        self.image_dir = image_dir
        self.scoring_dim_indices = scoring_dim_indices
        self.ensemble_weight = ensemble_weight
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        required = ["lora_adapter"] + [
            f"{name}.pth" for name in ("z1_mu", "z2_mu", "joint_outcome_head")
        ]
        missing = [name for name in required if not os.path.exists(os.path.join(checkpoint_dir, name))]
        if missing:
            raise FileNotFoundError(f"Reward checkpoint {checkpoint_dir} is missing {missing}.")

        print(f"  [Reward] Loading base VLM ({base_model_name})...", flush=True)
        base_model, processor, detected_type = load_base_model(
            base_model_name, model_type,
        )
        self.model_type = detected_type
        self.processor = processor
        self.tokenizer = get_tokenizer(processor, detected_type)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_path = os.path.join(checkpoint_dir, "lora_adapter")
        print(f"  [Reward] Merging LoRA adapter from {lora_path}...", flush=True)
        base_model = PeftModel.from_pretrained(
            base_model, lora_path, torch_dtype=torch.bfloat16,
        )
        base_model = base_model.merge_and_unload()

        base_model = base_model.to(device)

        self.model = VisualRecursiveToMRewardModel(
            base_model, detected_type, reward_dim=REWARD_DIM, z_dim=z_dim,
        ).to(device)

        schema_path = os.path.join(checkpoint_dir, "reward_schema.json")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                schema = json.load(f)
            ckpt_reward_dim = schema.get("reward_dim")
            ckpt_reward_dims = schema.get("reward_dimensions")
            if ckpt_reward_dim != REWARD_DIM or ckpt_reward_dims != MMROLE_REWARD_DIMS:
                raise ValueError(
                    "Stage 0 checkpoint schema mismatch. "
                    f"Expected {REWARD_DIM} official MMRole dims {MMROLE_REWARD_DIMS}, "
                    f"but checkpoint reports reward_dim={ckpt_reward_dim} "
                    f"reward_dimensions={ckpt_reward_dims}. "
                    "This usually means the checkpoint was trained with the older "
                    "3-dim ToM reward and must be retrained for the 8-dim setup."
                )

        # Load custom head weights
        for head_name in REWARD_HEAD_NAMES:
            path = os.path.join(checkpoint_dir, f"{head_name}.pth")
            if os.path.exists(path):
                try:
                    getattr(self.model, head_name).load_state_dict(
                        torch.load(path, map_location=device, weights_only=True)
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Failed to load Stage 0 head '{head_name}' from {path}. "
                        "This often means the checkpoint was trained with the older "
                        "3-dim reward schema and is incompatible with the new 8-dim "
                        "MMRole reward setup."
                    ) from exc
            else:
                print(f"  [Reward] WARNING: {head_name}.pth missing", flush=True)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        print(
            f"  [Reward] Loaded: joint_dims={REWARD_DIM} aux_dims={TOM_AUX_REWARD_DIM} "
            f"joint_head_only=True type={detected_type}",
            flush=True,
        )
        if abs(ensemble_weight - 1.0) > 1e-8:
            print(
                "  [Reward] NOTE: ensemble_weight is ignored for GRPO candidate "
                "ranking because z-only heads are prompt-constant within a group.",
                flush=True,
            )

    # ── Context encoding (image + text) ──
    def _encode_context_batch(self, samples: List[Dict]):
        texts = []
        images = []
        for s in samples:
            # Build processor-compatible chat text
            synth = {
                "context_text": format_reward_context(s),
                "image_path": resolve_image_with_warning(s, self.image_dir),
            }
            text = _build_context_text_for_processor(
                synth, self.model_type, self.processor,
            )
            texts.append(text)
            img = None
            if synth["image_path"]:
                img = load_and_resize_image(synth["image_path"])
            images.append(img)

        if self.model_type == "qwen2.5-vl":
            has_images = [img is not None for img in images]
            if any(has_images):
                if not all(has_images):
                    missing = sum(1 for has_image in has_images if not has_image)
                    raise ValueError(
                        "Mixed image availability while encoding reward contexts "
                        f"({missing}/{len(images)} missing)."
                    )
                # No truncation when images are in the text — it can sever
                # the inline <image> token block.
                enc = self.processor(
                    text=texts, images=images,
                    return_tensors="pt", padding=True,
                )
            else:
                enc = self.processor(
                    text=texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_ctx_len,
                )
        else:
            pad_id = self.tokenizer.pad_token_id or 0
            all_ids = []
            for t in texts:
                e = self.tokenizer(
                    t, return_tensors="pt", truncation=True, max_length=self.max_ctx_len,
                )
                all_ids.append(e.input_ids.squeeze(0))
            mx = max(x.shape[0] for x in all_ids)
            ids = torch.full((len(all_ids), mx), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, x in enumerate(all_ids):
                ids[i, :x.shape[0]] = x
                mask[i, :x.shape[0]] = 1
            enc = {"input_ids": ids, "attention_mask": mask}

        enc = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
               for k, v in enc.items()}

        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            mu1, mu2 = self.model.encode_context_deterministic(
                enc["input_ids"], enc["attention_mask"],
                pixel_values=enc.get("pixel_values"),
                image_grid_thw=enc.get("image_grid_thw"),
            )
        return mu1, mu2

    @torch.no_grad()
    def score(self, samples: List[Dict], completions: List[str]) -> List[float]:
        """
        Args:
            samples: len N, MMRole example dicts (one per candidate).
                     The same original example appears group_size times.
            completions: len N, candidate response strings.
        Returns:
            list of N scalar rewards.
        """
        assert len(samples) == len(completions)

        # Unique contexts by example_id (or object id) to amortize encoding cost
        seen = {}
        unique_samples = []
        for i, s in enumerate(samples):
            key = s.get("example_id") or id(s)
            if key not in seen:
                seen[key] = len(unique_samples)
                unique_samples.append(s)

        # Encode unique contexts in chunks
        unique_z1 = []
        unique_z2 = []
        chunk = 4
        for i in range(0, len(unique_samples), chunk):
            batch = unique_samples[i:i + chunk]
            mu1, mu2 = self._encode_context_batch(batch)
            for j in range(len(batch)):
                unique_z1.append(mu1[j])
                unique_z2.append(mu2[j])

        # Map per-candidate to the corresponding z1/z2
        z1_list = []
        z2_list = []
        for s in samples:
            key = s.get("example_id") or id(s)
            idx = seen[key]
            z1_list.append(unique_z1[idx])
            z2_list.append(unique_z2[idx])

        # Score candidates in chunks (multimodal response encoding)
        rewards: List[float] = []
        for i in range(0, len(completions), chunk):
            batch_completions = completions[i:i + chunk]
            batch_samples = samples[i:i + chunk]
            z1 = torch.stack(z1_list[i:i + chunk], dim=0).to(self.device)
            z2 = torch.stack(z2_list[i:i + chunk], dim=0).to(self.device)

            resp_texts, resp_images = [], []
            for sample, comp in zip(batch_samples, batch_completions):
                img_path = resolve_image_with_warning(sample, self.image_dir)
                resp_texts.append(
                    _build_response_text_for_processor(
                        comp, img_path, self.model_type, self.processor,
                    )
                )
                resp_images.append(
                    load_and_resize_image(img_path) if img_path else None
                )

            if self.model_type == "qwen2.5-vl":
                has_images = [img is not None for img in resp_images]
                if any(has_images):
                    if not all(has_images):
                        missing = sum(1 for has_image in has_images if not has_image)
                        raise ValueError(
                            "Mixed image availability while scoring reward responses "
                            f"({missing}/{len(resp_images)} missing)."
                        )
                    resp_enc = self.processor(
                        text=resp_texts, images=resp_images,
                        return_tensors="pt", padding=True,
                    )
                else:
                    resp_enc = self.processor(
                        text=resp_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_resp_len,
                    )
            else:
                pad_id = self.tokenizer.pad_token_id or 0
                all_ids = []
                for t in resp_texts:
                    enc = self.tokenizer(
                        t, return_tensors="pt",
                        truncation=True, max_length=self.max_resp_len,
                    )
                    all_ids.append(enc.input_ids.squeeze(0))
                mx = max(x.shape[0] for x in all_ids)
                ids_t = torch.full((len(all_ids), mx), pad_id, dtype=torch.long)
                mask_t = torch.zeros_like(ids_t)
                for j, x in enumerate(all_ids):
                    ids_t[j, :x.shape[0]] = x
                    mask_t[j, :x.shape[0]] = 1
                resp_enc = {"input_ids": ids_t, "attention_mask": mask_t}

            resp_enc = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in resp_enc.items()
            }
            resp_ids = resp_enc["input_ids"]
            resp_mask = resp_enc["attention_mask"]

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                joint, _, _ = self.model.forward_reward_with_z(
                    z1, z2, resp_ids, resp_mask,
                    pixel_values=resp_enc.get("pixel_values"),
                    image_grid_thw=resp_enc.get("image_grid_thw"),
                )

            joint = joint.float()
            if self.scoring_dim_indices is not None:
                scalar = joint[:, self.scoring_dim_indices].mean(dim=1)
            else:
                scalar = joint.mean(dim=1)
            rewards.extend(scalar.cpu().tolist())

        return rewards


# ──────────────────────────────────────────────────────────────────────────────
# GRPO dataset (same prompts as Stage 1 SFT, chain-of-belief format)
# ──────────────────────────────────────────────────────────────────────────────
class GRPOBeliefDataset(Dataset):
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

    def attach_mental_prefixes(self, prefix_map: Dict[str, str]) -> int:
        attached = 0
        for example in self.examples:
            example_id = example.get("example_id", "")
            prefix = prefix_map.get(example_id, "")
            if prefix:
                example["_mental_prefix"] = prefix
                attached += 1
        return attached

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_grpo_prompt(example: dict) -> str:
    speaker = example.get("speaker_name", "Speaker")
    partner = example.get("partner_name", "Partner")
    speaker_profile = example.get("speaker_profile", "")[:500]
    partner_profile = example.get("partner_profile", "")[:500]

    history_lines = []
    for t in example.get("dialogue_history", []):
        history_lines.append(f"[{t.get('speaker','?')}]: {t.get('utterance','')}")
    history_text = "\n".join(history_lines[-4:])

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
    mental_prefix = example.get("_mental_prefix", "").strip()
    if mental_prefix:
        prompt = (
            f"{prompt}\n\n"
            f"## Mental Prefix\n"
            f"{mental_prefix}\n\n"
            "Use this mental prefix as a compact guide to the scene's ToM demands, "
            "but still ground every section in the actual image and dialogue context."
        )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# GRPO Trainer (learned reward)
# ──────────────────────────────────────────────────────────────────────────────
class GRPOLearnedRewardTrainer:
    def __init__(self, policy_model, ref_model,
                 reward_model: FrozenVisualToMRewardModel,
                 processor, model_type: str, image_dir: str,
                 policy_device="cuda:0", ref_device="cuda:0",
                 group_size: int = 8, max_gen_len: int = 512,
                 clip_eps: float = 0.2, kl_coeff: float = 0.05,
                 temperature: float = 0.8, top_p: float = 0.95,
                 max_grad_norm: float = 1.0,
                 reward_clip: float = 0.0,
                 gen_chunk_size: int = 2):
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
        self.reward_clip = reward_clip
        self.gen_chunk_size = gen_chunk_size
        self.max_policy_seq_len = self._infer_model_max_seq_len(policy_model)
        self.max_ref_seq_len = self._infer_model_max_seq_len(ref_model)
        limits = [v for v in [self.max_policy_seq_len, self.max_ref_seq_len] if v is not None]
        self.max_logprob_seq_len = min(limits) if limits else None

    @staticmethod
    def _assert_valid_log_probs(log_probs_list, name: str, positive_tol: float = 1e-3):
        if not log_probs_list:
            return
        flat = torch.cat([lp.detach().float().flatten().cpu() for lp in log_probs_list])
        if not torch.isfinite(flat).all():
            n_nan = int(torch.isnan(flat).sum().item())
            n_inf = int(torch.isinf(flat).sum().item())
            raise RuntimeError(
                f"{name} log-probs are non-finite (n_nan={n_nan}, n_inf={n_inf}). "
                "This usually indicates a broken forward pass or device instability."
            )
        max_lp = float(flat.max().item())
        if max_lp > positive_tol:
            raise RuntimeError(
                f"{name} log-probs are invalid: max={max_lp:.6f} > {positive_tol}. "
                "Log-probabilities must be <= 0. This run should be treated as invalid "
                "(likely a corrupted forward pass or device-placement problem)."
            )

    def _ensure_nonempty_completion_ids(self, completion_ids: torch.Tensor) -> torch.Tensor:
        if completion_ids.numel() > 0:
            return completion_ids.detach().clone()

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise RuntimeError(
                "Policy generated an empty completion and tokenizer has no eos_token_id "
                "for a safe fallback token."
            )
        return torch.tensor(
            [eos_id],
            device=completion_ids.device,
            dtype=completion_ids.dtype if completion_ids.dtype != torch.bool else torch.long,
        )

    @staticmethod
    def _infer_model_vocab_size(model) -> Optional[int]:
        if model is None:
            return None
        try:
            embed = model.get_input_embeddings()
        except Exception:
            embed = None
        if embed is None or not hasattr(embed, "weight"):
            return None
        return int(embed.weight.shape[0])

    @staticmethod
    def _infer_model_max_seq_len(model) -> Optional[int]:
        if model is None:
            return None

        def _collect(cfg) -> List[int]:
            values = []
            if cfg is None:
                return values
            for key in ("max_position_embeddings", "n_positions", "seq_length"):
                value = getattr(cfg, key, None)
                if isinstance(value, int) and 0 < value < 1_000_000:
                    values.append(int(value))
            return values

        cfg = getattr(model, "config", None)
        candidates = _collect(cfg)
        candidates.extend(_collect(getattr(cfg, "text_config", None)))
        return min(candidates) if candidates else None

    @staticmethod
    def _validate_full_input_ids(full_ids: torch.Tensor, vocab_size: Optional[int],
                                 context: str) -> None:
        if vocab_size is None:
            return
        ids_cpu = full_ids.detach().to("cpu")
        min_id = int(ids_cpu.min().item())
        max_id = int(ids_cpu.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise RuntimeError(
                f"{context}: input_ids out of range for embedding vocab_size={vocab_size} "
                f"(min_id={min_id}, max_id={max_id}, seq_len={full_ids.shape[1]})."
            )

    def _truncate_completion_for_logprob(self, prompt_ids: torch.Tensor,
                                         completion_ids: torch.Tensor,
                                         example_id: str) -> torch.Tensor:
        completion_ids = completion_ids.detach().clone()
        if self.max_logprob_seq_len is None:
            return completion_ids

        prompt_len = int(prompt_ids.shape[0])
        if prompt_len >= self.max_logprob_seq_len:
            raise RuntimeError(
                f"Prompt too long for Stage 2 log-prob recomputation: "
                f"prompt_len={prompt_len}, max_logprob_seq_len={self.max_logprob_seq_len}, "
                f"example_id={example_id or '<unknown>'}."
            )

        max_completion_len = self.max_logprob_seq_len - prompt_len
        if completion_ids.shape[0] <= max_completion_len:
            return completion_ids

        key = (example_id or "<unknown>", prompt_len, completion_ids.shape[0], max_completion_len)
        if key not in _TRUNCATED_COMPLETION_WARNINGS:
            _TRUNCATED_COMPLETION_WARNINGS.add(key)
            print(
                "  [WARN] Truncating completion for Stage 2 log-prob recomputation: "
                f"example_id={example_id or '<unknown>'} "
                f"prompt_len={prompt_len} completion_len={completion_ids.shape[0]} "
                f"-> {max_completion_len} (limit={self.max_logprob_seq_len}).",
                flush=True,
            )
        return completion_ids[:max_completion_len]

    @staticmethod
    def _validate_gather_inputs(logits: torch.Tensor, completion_ids: torch.Tensor,
                                context: str) -> None:
        if completion_ids.numel() == 0:
            raise RuntimeError(f"{context}: completion_ids is unexpectedly empty.")
        vocab_size = int(logits.shape[-1])
        ids_cpu = completion_ids.detach().to("cpu")
        min_id = int(ids_cpu.min().item())
        max_id = int(ids_cpu.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise RuntimeError(
                f"{context}: completion token ids out of range for vocab_size={vocab_size} "
                f"(min_id={min_id}, max_id={max_id}, num_tokens={completion_ids.numel()})."
            )

    @torch.no_grad()
    def generate_candidates(
        self,
        examples: List[Dict],
        num_return_sequences: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Tuple:
        self.policy.eval()
        all_completions = []
        all_prompt_ids = []
        all_prompt_aux = []   # per-candidate {pixel_values, image_grid_thw}
        all_completion_ids = []
        all_samples = []  # one entry per candidate (same example repeated)
        total_sequences = num_return_sequences or self.group_size
        use_sampling = self.temperature if temperature is None else temperature
        use_top_p = self.top_p if top_p is None else top_p
        use_do_sample = True if do_sample is None else do_sample

        for example in examples:
            prompt_text = build_grpo_prompt(example)
            image_path = resolve_image_with_warning(example, self.image_dir)
            inputs, _ = prepare_generation_inputs(
                prompt_text, image_path, self.processor,
                self.model_type, device=str(self.device),
            )
            prompt_ids = inputs["input_ids"][0]
            prompt_ids_cache = prompt_ids.detach().clone().cpu()
            aux = {}
            if "pixel_values" in inputs:
                aux["pixel_values"] = inputs["pixel_values"].detach().clone().cpu()
            if "image_grid_thw" in inputs:
                aux["image_grid_thw"] = inputs["image_grid_thw"].detach().clone().cpu()

            # Generate in sub-chunks to cap peak KV-cache memory at
            # O(gen_chunk) instead of O(group_size).
            gen_chunk = max(1, min(total_sequences, self.gen_chunk_size))
            gathered = []
            remaining = total_sequences
            while remaining > 0:
                this_chunk = min(gen_chunk, remaining)
                gen_kwargs = dict(
                    **inputs,
                    min_new_tokens=1,
                    max_new_tokens=self.max_gen_len,
                    do_sample=use_do_sample,
                    num_return_sequences=this_chunk,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if use_do_sample:
                    gen_kwargs["temperature"] = use_sampling
                    gen_kwargs["top_p"] = use_top_p
                sub = self.policy.generate(**gen_kwargs)
                for s in sub:
                    gathered.append(s.detach().cpu())
                del sub
                torch.cuda.empty_cache()
                remaining -= this_chunk
            outputs = gathered

            for seq in outputs:
                seq = seq.to(prompt_ids.device)
                completion_ids = self._ensure_nonempty_completion_ids(
                    seq[prompt_ids.shape[0]:]
                )
                completion_ids = self._truncate_completion_for_logprob(
                    prompt_ids, completion_ids, example.get("example_id", "")
                )
                completion_ids_cache = completion_ids.detach().clone().cpu()
                completion_text = self.tokenizer.decode(
                    completion_ids_cache, skip_special_tokens=True,
                ).strip()

                all_completions.append(completion_text)
                all_prompt_ids.append(prompt_ids_cache)
                all_prompt_aux.append(aux)
                all_completion_ids.append(completion_ids_cache)
                all_samples.append(example)

        return (all_completions, all_prompt_ids, all_prompt_aux,
                all_completion_ids, all_samples)

    def score_completions(self, samples: List[Dict], completions: List[str]) -> torch.Tensor:
        scoring_texts = [completion_to_reward_text(c) for c in completions]
        rewards = self.reward_model.score(samples, scoring_texts)
        if torch.cuda.is_available() and str(self.reward_model.device).startswith("cuda"):
            torch.cuda.synchronize(self.reward_model.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        structure_bonus = torch.tensor(
            [chain_structure_bonus(c) for c in completions],
            dtype=torch.float32, device=self.device,
        )
        rewards_tensor = rewards_tensor + structure_bonus
        if self.reward_clip > 0:
            rewards_tensor = torch.tanh(rewards_tensor / self.reward_clip) * self.reward_clip
        if not torch.isfinite(rewards_tensor).all():
            raise RuntimeError("Non-finite reward detected in Stage 2 GRPO scoring.")
        return rewards_tensor

    @torch.no_grad()
    def evaluate_examples(self, examples: List[Dict]) -> Dict[str, float]:
        completions, _, _, _, samples = self.generate_candidates(
            examples,
            num_return_sequences=1,
            do_sample=False,
        )
        rewards_tensor = self.score_completions(samples, completions)
        return {
            "mean_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std().item() if rewards_tensor.numel() > 1 else 0.0,
            "format_rate": (
                sum(1 for c in completions if has_valid_format(c)) / max(len(completions), 1)
            ),
        }

    def compute_log_probs(self, model, prompt_ids_list, prompt_aux_list,
                          completion_ids_list, target_device=None,
                          eval_mode: bool = True):
        dev = target_device or self.device
        log_probs_list = []
        was_training = model.training if hasattr(model, "training") else False
        if eval_mode and hasattr(model, "eval"):
            model.eval()
        try:
            for prompt_ids, aux, completion_ids in zip(
                prompt_ids_list, prompt_aux_list, completion_ids_list
            ):
                completion_ids = completion_ids.to(prompt_ids.device)
                full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(dev)
                attention_mask = torch.ones_like(full_ids)
                self._validate_full_input_ids(
                    full_ids,
                    self._infer_model_vocab_size(model),
                    context="compute_log_probs",
                )

                fwd_kwargs = dict(input_ids=full_ids, attention_mask=attention_mask)
                if aux.get("pixel_values") is not None:
                    fwd_kwargs["pixel_values"] = aux["pixel_values"].to(dev)
                if aux.get("image_grid_thw") is not None:
                    fwd_kwargs["image_grid_thw"] = aux["image_grid_thw"].to(dev)

                with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**fwd_kwargs)
                prompt_len = prompt_ids.shape[0]
                raw_logits = outputs.logits[
                    0, prompt_len - 1: prompt_len - 1 + completion_ids.shape[0]
                ]
                comp_logits = raw_logits.float()
                if not torch.isfinite(comp_logits).all():
                    n_nan = int(torch.isnan(comp_logits).sum().item())
                    n_inf = int(torch.isinf(comp_logits).sum().item())
                    finite = comp_logits[torch.isfinite(comp_logits)]
                    finite_stats = (
                        f"finite_min={finite.min().item():.2f} "
                        f"finite_max={finite.max().item():.2f}"
                        if finite.numel() > 0 else "all non-finite"
                    )
                    raise RuntimeError(
                        f"compute_log_probs: non-finite logits from model "
                        f"(device={dev}, shape={tuple(comp_logits.shape)}, "
                        f"dtype_raw={raw_logits.dtype}, n_nan={n_nan}, "
                        f"n_inf={n_inf}, {finite_stats}). "
                        f"This indicates a broken forward pass in the model, "
                        f"not a downstream numerical issue."
                    )
                comp_logits = comp_logits.clamp_(-50.0, 50.0)
                del outputs
                if torch.cuda.is_available() and str(dev).startswith("cuda"):
                    torch.cuda.synchronize(dev)
                if comp_logits.shape[0] != completion_ids.shape[0]:
                    raise RuntimeError(
                        f"compute_log_probs: completion/logit length mismatch "
                        f"(prompt_len={prompt_len}, completion_len={completion_ids.shape[0]}, "
                        f"logit_rows={comp_logits.shape[0]})."
                    )
                self._validate_gather_inputs(
                    comp_logits, completion_ids, context="compute_log_probs"
                )
                comp_lp = F.log_softmax(comp_logits, dim=-1)
                token_lp = comp_lp.gather(
                    1, completion_ids.unsqueeze(0).to(dev).T,
                ).squeeze(-1)
                if not torch.isfinite(token_lp).all():
                    raise RuntimeError(
                        "compute_log_probs: non-finite token log-probs "
                        f"(device={dev}, seq_len={completion_ids.shape[0]}). "
                        "Likely an upstream CUDA error or OOM — rerun with "
                        "CUDA_LAUNCH_BLOCKING=1 to localize."
                    )
                log_probs_list.append(token_lp.to(self.device))
        finally:
            if eval_mode and was_training and hasattr(model, "train"):
                model.train()
        return log_probs_list

    def grpo_step(self, examples: List[Dict], optimizer, scheduler=None):
        t0 = time.time()
        num_examples = len(examples)

        # Keep policy in eval() throughout the step so dropout doesn't desync
        # old vs. new log-probs (LoRA params still get gradients in eval mode).
        self.policy.eval()

        print(f"    [1/5] Generating {self.group_size}x{num_examples} candidates...",
              flush=True)
        (completions, prompt_ids_list, prompt_aux_list,
         completion_ids_list, samples) = self.generate_candidates(examples)
        t1 = time.time()
        print(f"    [1/5] Done ({t1-t0:.0f}s)", flush=True)

        print(f"    [2/5] Scoring {len(completions)} candidates (learned reward)...",
              flush=True)
        rewards_tensor = self.score_completions(samples, completions)

        t2 = time.time()
        print(f"    [2/5] Done ({t2-t1:.0f}s) "
              f"mean={rewards_tensor.mean():.4f} std={rewards_tensor.std():.4f}",
              flush=True)

        # Group-normalized advantages
        advantages = torch.zeros_like(rewards_tensor)
        for i in range(num_examples):
            start = i * self.group_size
            end = start + self.group_size
            group = rewards_tensor[start:end]
            advantages[start:end] = (group - group.mean()) / (group.std() + 1e-8)
        advantages = advantages.clamp(-3.0, 3.0)

        print(f"    [3/5] Old log probs...", flush=True)
        with torch.no_grad():
            old_log_probs_list = self.compute_log_probs(
                self.policy, prompt_ids_list, prompt_aux_list, completion_ids_list,
            )
            self._assert_valid_log_probs(old_log_probs_list, "old")
            ref_log_probs_list = None
            if self.ref_model is not None:
                print(f"    [4/5] Ref log probs...", flush=True)
                ref_log_probs_list = self.compute_log_probs(
                    self.ref_model, prompt_ids_list, prompt_aux_list,
                    completion_ids_list, target_device=self.ref_device,
                )
                self._assert_valid_log_probs(ref_log_probs_list, "ref")
        t3 = time.time()
        print(f"    [4/5] Done ({t3-t2:.0f}s)", flush=True)

        # Diagnostic: log-prob statistics. At step 1 before any update, policy
        # and ref are the same weights, so old_lp and ref_lp should be bitwise
        # identical (or differ only by bf16 precision noise, well under 0.1).
        # Large differences point to a structural issue (wrong dtype path,
        # wrong device, wrong image injection, etc.).
        try:
            diag_old = torch.cat([lp.float().flatten() for lp in old_log_probs_list])
            if ref_log_probs_list is not None:
                diag_ref = torch.cat(
                    [lp.float().flatten() for lp in ref_log_probs_list]
                )
                diag_delta = (diag_ref - diag_old).abs()
                print(
                    f"    [diag] old_lp mean={diag_old.mean().item():.3f} "
                    f"min={diag_old.min().item():.3f} max={diag_old.max().item():.3f} | "
                    f"ref_lp mean={diag_ref.mean().item():.3f} "
                    f"min={diag_ref.min().item():.3f} max={diag_ref.max().item():.3f} | "
                    f"|ref-old| mean={diag_delta.mean().item():.4f} "
                    f"max={diag_delta.max().item():.4f}",
                    flush=True,
                )
            else:
                print(
                    f"    [diag] old_lp mean={diag_old.mean().item():.3f} "
                    f"min={diag_old.min().item():.3f} max={diag_old.max().item():.3f}",
                    flush=True,
                )
        except Exception as e:
            print(f"    [diag] log-prob stats failed: {e}", flush=True)

        # PPO-clip policy gradient
        print(f"    [5/5] Policy gradient...", flush=True)
        all_loss = torch.tensor(0.0, device=self.device)
        all_kl = torch.tensor(0.0, device=self.device)
        denom = len(prompt_ids_list)
        optimizer.zero_grad(set_to_none=True)
        policy_was_training = self.policy.training
        # Disable dropout during PPO/GRPO log-prob evaluation and gradient
        # computation. In train() mode, LoRA dropout can make the "current"
        # policy disagree wildly with the old/ref policies even before a real
        # parameter update, which shows up as exploding KL and loss.
        self.policy.eval()

        for j in range(len(prompt_ids_list)):
            full_ids = torch.cat([
                prompt_ids_list[j],
                completion_ids_list[j].to(prompt_ids_list[j].device),
            ], dim=0).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(full_ids)
            self._validate_full_input_ids(
                full_ids,
                self._infer_model_vocab_size(self.policy),
                context="policy_gradient",
            )

            fwd_kwargs = dict(input_ids=full_ids, attention_mask=attention_mask)
            aux_j = prompt_aux_list[j]
            if aux_j.get("pixel_values") is not None:
                fwd_kwargs["pixel_values"] = aux_j["pixel_values"].to(self.device)
            if aux_j.get("image_grid_thw") is not None:
                fwd_kwargs["image_grid_thw"] = aux_j["image_grid_thw"].to(self.device)

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                outputs = self.policy(**fwd_kwargs)
            prompt_len = prompt_ids_list[j].shape[0]
            comp_logits = outputs.logits[
                0, prompt_len - 1: prompt_len - 1 + completion_ids_list[j].shape[0]
            ].float()
            del outputs
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            if comp_logits.shape[0] != completion_ids_list[j].shape[0]:
                raise RuntimeError(
                    f"policy_gradient: completion/logit length mismatch "
                    f"(prompt_len={prompt_len}, completion_len={completion_ids_list[j].shape[0]}, "
                    f"logit_rows={comp_logits.shape[0]})."
                )
            self._validate_gather_inputs(
                comp_logits, completion_ids_list[j], context="policy_gradient"
            )
            comp_lp = F.log_softmax(comp_logits, dim=-1)
            token_lp = comp_lp.gather(
                1, completion_ids_list[j].unsqueeze(0).to(self.device).T,
            ).squeeze(-1)

            old_lp = old_log_probs_list[j].detach()
            if not torch.isfinite(old_lp).all():
                _warn_nonfinite_logprob(
                    "old", j, "using current token log-probs as the PPO anchor"
                )
                old_lp = token_lp.detach()
            log_ratio_pg = (token_lp - old_lp).clamp(-10.0, 10.0)
            ratio = torch.exp(log_ratio_pg)
            adv = advantages[j]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            kl = torch.tensor(0.0, device=self.device)
            if ref_log_probs_list is not None and self.kl_coeff > 0:
                ref_lp = ref_log_probs_list[j].detach()
                if not torch.isfinite(ref_lp).all():
                    _warn_nonfinite_logprob(
                        "ref", j, "dropping KL regularization for this candidate"
                    )
                else:
                    log_ratio = (token_lp - ref_lp).clamp(-10.0, 10.0)
                    kl = (torch.exp(log_ratio) - log_ratio - 1).mean()

            sample_loss = policy_loss + self.kl_coeff * kl
            scaled_loss = sample_loss / denom
            if not torch.isfinite(scaled_loss):
                raise RuntimeError(
                    "Non-finite GRPO sample loss detected before backward() "
                    f"at candidate {j}. "
                    f"policy_loss={float(policy_loss.detach().cpu())} "
                    f"kl={float(kl.detach().cpu())} "
                    f"ratio_max={float(ratio.detach().max().cpu())} "
                    f"ratio_min={float(ratio.detach().min().cpu())}"
                )
            scaled_loss.backward()
            all_loss = all_loss + sample_loss.detach()
            all_kl = all_kl + kl.detach()

            del full_ids, attention_mask, comp_logits, comp_lp, token_lp
            del old_lp, ratio, surr1, surr2, policy_loss, kl, sample_loss, scaled_loss
            if "outputs" in locals():
                del outputs

        all_loss = all_loss / denom
        if not torch.isfinite(all_loss):
            raise RuntimeError("Non-finite GRPO loss detected before backward().")

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            max_norm=self.max_grad_norm,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if policy_was_training:
            self.policy.train()

        t4 = time.time()
        print(f"    [5/5] Done ({t4-t3:.0f}s) | Total: {t4-t0:.0f}s", flush=True)

        return {
            "policy_loss": all_loss.item(),
            "mean_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std().item(),
            "mean_kl": all_kl.item() / len(prompt_ids_list),
            "mean_advantage": advantages.mean().item(),
            "format_rate": sum(1 for c in completions if has_valid_format(c)) / len(completions),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for flag, path in (
        ("--sft_checkpoint", args.sft_checkpoint),
        ("--reward_checkpoint_dir", args.reward_checkpoint_dir),
        ("--mental_prefix_checkpoint_dir", args.mental_prefix_checkpoint_dir),
    ):
        if path and not os.path.exists(path):
            hint = " Pass --sft_checkpoint '' to start from a fresh LoRA." if flag == "--sft_checkpoint" else ""
            raise FileNotFoundError(f"{flag} {path} does not exist.{hint}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        if args.policy_device:
            policy_device_name = args.policy_device
        elif n_gpus >= 1:
            policy_device_name = "cuda:0"
        else:
            policy_device_name = "cpu"

        if args.ref_device:
            ref_device = args.ref_device
        elif n_gpus >= 2:
            ref_device = "cuda:1"
        elif n_gpus >= 1:
            ref_device = "cuda:0"
        else:
            ref_device = "cpu"

        if args.reward_device:
            reward_device = args.reward_device
        elif n_gpus >= 3:
            reward_device = "cuda:2"
        elif n_gpus >= 2:
            reward_device = "cuda:1"
        elif n_gpus >= 1:
            reward_device = "cuda:0"
        else:
            reward_device = "cpu"
    else:
        policy_device_name = "cpu"
        ref_device = "cpu"
        reward_device = "cpu"

    policy_device = torch.device(policy_device_name)
    print(
        f"Devices: policy={policy_device} ref={ref_device} reward={reward_device} "
        f"(visible_gpus={n_gpus})",
        flush=True,
    )
    if ref_device == reward_device and str(ref_device).startswith("cuda"):
        print(
            "  [WARN] Reference model and reward model share the same GPU. "
            "If ref log-probs become invalid, split them across separate GPUs.",
            flush=True,
        )

    model_type = args.model_type or detect_model_type(args.base_model)
    reward_model_type = (
        args.reward_model_type
        or detect_model_type(args.reward_base_model or args.base_model)
    )
    print(f"Model type: {model_type}", flush=True)
    print(f"Reward model type: {reward_model_type}", flush=True)

    # Parse scoring dims subset
    scoring_dim_indices = None
    if args.reward_scoring_dims:
        names = [d.strip() for d in args.reward_scoring_dims.split(",") if d.strip()]
        bad = [d for d in names if d not in MMROLE_REWARD_DIMS]
        if bad:
            raise ValueError(f"Invalid scoring dims {bad}; valid={MMROLE_REWARD_DIMS}")
        scoring_dim_indices = [MMROLE_REWARD_DIMS.index(d) for d in names]
        print(f"  Reward scoring subset: {names} (idx {scoring_dim_indices})", flush=True)

    # Dataset + optional frozen mental-prefix precomputation
    dataset = GRPOBeliefDataset(
        args.train_path, args.image_dir, max_examples=args.max_examples,
    )
    val_examples = []
    val_dataset = None
    if args.val_path and os.path.exists(args.val_path):
        val_dataset = GRPOBeliefDataset(
            args.val_path, args.image_dir,
            max_examples=args.val_max_examples,
        )

    if args.mental_prefix_checkpoint_dir:
        mental_prefix_base = (
            args.mental_prefix_base_model
            or args.reward_base_model
            or args.base_model
        )
        mental_prefix_type = (
            args.mental_prefix_model_type
            or detect_model_type(mental_prefix_base)
        )
        print(">> Precomputing frozen mental prefixes...", flush=True)
        prefix_model = FrozenMentalPrefixModel(
            base_model_name=mental_prefix_base,
            checkpoint_dir=args.mental_prefix_checkpoint_dir,
            model_type=mental_prefix_type,
            z_dim=args.mental_prefix_z_dim,
            device=args.mental_prefix_device or str(reward_device),
            image_dir=args.image_dir,
            max_ctx_len=args.mental_prefix_max_ctx_len,
        )
        train_prefixes = prefix_model.build_prefix_map(
            dataset.examples, batch_size=args.mental_prefix_batch_size,
        )
        n_train_prefix = dataset.attach_mental_prefixes(train_prefixes)
        print(
            f"  Mental prefixes attached to train GRPO examples: "
            f"{n_train_prefix}/{len(dataset.examples)}",
            flush=True,
        )
        if val_dataset:
            val_prefixes = prefix_model.build_prefix_map(
                val_dataset.examples, batch_size=args.mental_prefix_batch_size,
            )
            n_val_prefix = val_dataset.attach_mental_prefixes(val_prefixes)
            print(
                f"  Mental prefixes attached to val GRPO examples: "
                f"{n_val_prefix}/{len(val_dataset.examples)}",
                flush=True,
            )
        del prefix_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if val_dataset:
        n_val = min(args.val_prompts, len(val_dataset))
        if n_val > 0:
            rng = random.Random(args.seed)
            val_indices = rng.sample(range(len(val_dataset)), n_val)
            val_examples = [val_dataset[i] for i in val_indices]
            print(f"  Fixed val prompts: {len(val_examples)}", flush=True)

    # Frozen reward model
    print(">> Loading frozen reward model...", flush=True)
    reward_model = FrozenVisualToMRewardModel(
        base_model_name=args.reward_base_model or args.base_model,
        checkpoint_dir=args.reward_checkpoint_dir,
        model_type=reward_model_type,
        z_dim=args.z_dim,
        device=str(reward_device),
        image_dir=args.image_dir,
        scoring_dim_indices=scoring_dim_indices,
        ensemble_weight=args.ensemble_weight,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
    )

    # Policy
    print(">> Loading policy model...", flush=True)
    base_policy, processor, _ = load_base_model(args.base_model, model_type)

    if args.sft_checkpoint:
        print(f"  Loading SFT LoRA from {args.sft_checkpoint}...", flush=True)
        policy_model = PeftModel.from_pretrained(
            base_policy, args.sft_checkpoint, is_trainable=True,
        )
    else:
        print("  --sft_checkpoint is empty: initializing a fresh LoRA", flush=True)
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        else:
            target_modules = default_lora_target_modules(model_type)
        print(f"  LoRA target modules: {target_modules}", flush=True)
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_policy, lora_config)

    policy_model = policy_model.to(policy_device)
    try:
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  Policy gradient checkpointing: enabled with use_reentrant=False",
              flush=True)
    except TypeError:
        policy_model.gradient_checkpointing_enable()
        print("  Policy gradient checkpointing: enabled with default settings",
              flush=True)
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
    for name, p in policy_model.named_parameters():
        p.requires_grad = "lora_" in name
    unexpected_trainable = [
        name for name, p in policy_model.named_parameters()
        if p.requires_grad and "lora_" not in name
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Stage 2 policy should only update LoRA weights, but found non-LoRA "
            f"trainable params: {unexpected_trainable[:10]}"
        )
    lora_trainable = [
        name for name, p in policy_model.named_parameters()
        if p.requires_grad
    ]
    if not lora_trainable:
        raise RuntimeError("Stage 2 policy has no trainable LoRA parameters.")
    print(
        f"  Verified trainable parameters: {len(lora_trainable)} tensors, LoRA-only",
        flush=True,
    )
    policy_model.print_trainable_parameters()
    if hasattr(policy_model, "config"):
        policy_model.config.use_cache = False

    # Reference model (frozen SFT or base)
    #
    # IMPORTANT: load the ref exactly like the policy (is_trainable=True) so
    # LoRA weights are kept in the same dtype and the forward path is
    # numerically identical. Then freeze all parameters. If we omit
    # is_trainable=True, PEFT takes a different code path that can produce
    # non-finite logits in Qwen2.5-VL's attention layers.
    print(">> Loading reference model...", flush=True)
    ref_base, _, _ = load_base_model(args.base_model, model_type)
    if args.sft_checkpoint:
        ref_model = PeftModel.from_pretrained(
            ref_base, args.sft_checkpoint, is_trainable=True,
        )
    else:
        ref_model = ref_base
    ref_model = ref_model.to(ref_device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    if hasattr(ref_model, "config"):
        ref_model.config.use_cache = False

    # Optimizer + schedule
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = args.num_iterations
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = GRPOLearnedRewardTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        processor=processor,
        model_type=model_type,
        image_dir=args.image_dir,
        policy_device=str(policy_device),
        ref_device=str(ref_device),
        group_size=args.group_size,
        max_gen_len=args.max_gen_len,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        temperature=args.temperature,
        top_p=args.top_p,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        gen_chunk_size=args.gen_chunk_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 2: GRPO with LEARNED ToM Reward", flush=True)
    print(f"  Base model:         {args.base_model}", flush=True)
    print(f"  SFT checkpoint:     {args.sft_checkpoint}", flush=True)
    print(f"  Reward checkpoint:  {args.reward_checkpoint_dir}", flush=True)
    print(f"  Reward head:        joint only", flush=True)
    print(f"  Iterations:         {args.num_iterations}", flush=True)
    print(f"  Group size:         {args.group_size}", flush=True)
    print(f"  Prompts/iter:       {args.prompts_per_iter}", flush=True)
    print(f"  Eval every:         {args.eval_every}", flush=True)
    print(
        f"  Mental prefixes:    {'enabled' if args.mental_prefix_checkpoint_dir else 'disabled'}",
        flush=True,
    )
    print(f"  LR / KL / clip_eps: {args.lr} / {args.kl_coeff} / {args.clip_eps}", flush=True)
    print(f"  Output:             {args.output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    best_reward = -float("inf")
    best_selection_reward = -float("inf")
    patience_counter = 0
    start_iteration = 1
    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    if args.resume:
        resume_dir = require_resume_dir(args.output_dir)
        state = load_resume_state(resume_dir)
        check_resume_args(state["args"], args, RESUME_INVARIANT_ARGS)
        load_lora_weights(policy_model, os.path.join(resume_dir, "policy"))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_reward, best_selection_reward = state["best_reward"], state["best_selection_reward"]
        patience_counter = state["patience_counter"]
        start_iteration = (
            args.num_iterations + 1 if state["stopped_early"] else state["next_iteration"]
        )
        restore_rng_state(state["rng"])
        print(f"  Resumed from {resume_dir} at iteration {start_iteration}", flush=True)
        log_file = open(log_path, "a")
    else:
        if os.path.exists(log_path):
            print(f"  Overwriting existing training log at {log_path}.", flush=True)
        log_file = open(log_path, "w")

    def save_resume(next_iteration, stopped_early):
        save_resume_dir(
            args.output_dir,
            lambda d: policy_model.save_pretrained(os.path.join(d, "policy")),
            {
                "args": vars(args), "next_iteration": next_iteration,
                "stopped_early": stopped_early, "best_reward": best_reward,
                "best_selection_reward": best_selection_reward,
                "patience_counter": patience_counter,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            },
        )

    start_time = time.time()
    for iteration in range(start_iteration, args.num_iterations + 1):
        batch_indices = random.sample(
            range(len(dataset)), min(args.prompts_per_iter, len(dataset)),
        )
        batch_examples = [dataset[i] for i in batch_indices]

        print(f"\n  Iteration {iteration}/{args.num_iterations}", flush=True)
        metrics = trainer.grpo_step(batch_examples, optimizer, scheduler)

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  reward={metrics['mean_reward']:.4f}±{metrics['std_reward']:.4f} "
            f"loss={metrics['policy_loss']:.4f} kl={metrics['mean_kl']:.4f} "
            f"format={metrics['format_rate']:.0%} lr={lr_now:.2e} time={elapsed:.0f}s",
            flush=True,
        )

        log_file.write(json.dumps({
            "iteration": iteration, **metrics, "lr": lr_now, "elapsed": elapsed,
        }) + "\n")
        log_file.flush()

        selection_reward = None if val_examples else metrics["mean_reward"]
        if val_examples and args.eval_every > 0 and (
            iteration % args.eval_every == 0 or iteration == 1
        ):
            val_metrics = trainer.evaluate_examples(val_examples)
            selection_reward = val_metrics["mean_reward"]
            print(
                f"  val_reward={val_metrics['mean_reward']:.4f}±{val_metrics['std_reward']:.4f} "
                f"val_format={val_metrics['format_rate']:.0%}",
                flush=True,
            )
            log_file.write(json.dumps({
                "iteration": iteration,
                "val_mean_reward": val_metrics["mean_reward"],
                "val_std_reward": val_metrics["std_reward"],
                "val_format_rate": val_metrics["format_rate"],
            }) + "\n")
            log_file.flush()

        if metrics["mean_reward"] > best_reward:
            best_reward = metrics["mean_reward"]

        if selection_reward is not None and selection_reward > best_selection_reward:
            best_selection_reward = selection_reward
            patience_counter = 0
            best_dir = os.path.join(args.output_dir, "best")
            policy_model.save_pretrained(best_dir)
            try:
                processor.save_pretrained(best_dir)
            except Exception:
                pass
            print(
                f"  New best selection reward ({selection_reward:.4f})! "
                f"Saved to {best_dir}",
                flush=True,
            )
        elif selection_reward is not None:
            patience_counter += 1

        if args.save_every > 0 and iteration % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"iter_{iteration}")
            policy_model.save_pretrained(ckpt_dir)

        stop = args.patience > 0 and patience_counter >= args.patience
        if args.resume_every > 0 and (
            iteration % args.resume_every == 0 or stop or iteration == args.num_iterations
        ):
            log_file.flush()
            save_resume(iteration + 1, stop)
        if stop:
            print(f"\n  Early stopping: no improvement for {args.patience} iters",
                  flush=True)
            break

    log_file.close()

    final_dir = os.path.join(args.output_dir, "final")
    policy_model.save_pretrained(final_dir)
    try:
        processor.save_pretrained(final_dir)
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 2 (learned reward) GRPO complete", flush=True)
    print(f"  Best train reward: {best_reward:.4f}", flush=True)
    print(f"  Best selection reward: {best_selection_reward:.4f}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nNext: Stage 3 DPO", flush=True)
    print(f"  python stage3_dpo_contrastive.py \\", flush=True)
    print(f"    --grpo_checkpoint {os.path.join(args.output_dir, 'best')} \\", flush=True)
    print(f"    --base_model {args.base_model}", flush=True)

    del policy_model, ref_model, reward_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: GRPO with Frozen Visual ToM Reward Model"
    )
    # Models
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat"])
    parser.add_argument("--sft_checkpoint", type=str,
                        default="projects/mmrole/checkpoints/stage1_sft/best",
                        help="Stage 1 SFT adapter to start from; pass '' for a fresh LoRA.")
    parser.add_argument("--reward_base_model", type=str, default="",
                        help="VLM used to train the reward model (defaults to --base_model)")
    parser.add_argument("--reward_model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat"],
                        help="Reward-model backbone type (auto-detected from --reward_base_model if empty)")
    parser.add_argument("--reward_checkpoint_dir", type=str, required=True,
                        help="Stage 0 checkpoint dir (contains lora_adapter/ + head .pth files)")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument(
        "--ensemble_weight", type=float, default=1.0,
        help="Deprecated backward-compatibility flag. GRPO candidate ranking now "
             "uses only the response-dependent joint head.",
    )
    parser.add_argument("--reward_scoring_dims", type=str, default="",
                        help="Comma-separated subset of MMROLE_REWARD_DIMS to score on "
                             "(empty = use all)")
    parser.add_argument("--reward_clip", type=float, default=1.0)

    # LoRA (used only if no SFT checkpoint)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="Comma-separated LoRA target module names; defaults depend on model_type")

    # Data
    parser.add_argument("--train_path", type=str,
                        default="projects/mmrole/training_data/train/belief_prediction.jsonl")
    parser.add_argument("--val_path", type=str,
                        default="projects/mmrole/training_data/val/belief_prediction.jsonl")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--val_max_examples", type=int, default=200)
    parser.add_argument("--val_prompts", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--mental_prefix_checkpoint_dir", type=str, default="",
                        help="Optional Stage 0 checkpoint dir used to build frozen mental prefixes.")
    parser.add_argument("--mental_prefix_base_model", type=str, default="",
                        help="Base model used by the Stage 0 mental-prefix checkpoint "
                             "(defaults to --reward_base_model or --base_model).")
    parser.add_argument("--mental_prefix_model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat"],
                        help="Mental-prefix backbone type (auto-detected from "
                             "--mental_prefix_base_model if empty).")
    parser.add_argument("--mental_prefix_device", type=str, default="",
                        help="Device for frozen mental-prefix precomputation "
                             "(defaults to --reward_device when available).")
    parser.add_argument("--mental_prefix_batch_size", type=int, default=8)
    parser.add_argument("--mental_prefix_z_dim", type=int, default=128)
    parser.add_argument("--mental_prefix_max_ctx_len", type=int, default=1024)

    # GRPO
    parser.add_argument("--num_iterations", type=int, default=600)
    parser.add_argument("--prompts_per_iter", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--gen_chunk_size", type=int, default=2,
                        help="Max num_return_sequences per generate() call. "
                             "Lower values reduce peak KV-cache memory.")
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training / output
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/checkpoints/stage2_grpo_learned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--policy_device", type=str, default="",
                        help="Explicit policy device (e.g. cuda:0). "
                             "Defaults to cuda:0 when CUDA is available.")
    parser.add_argument("--ref_device", type=str, default="",
                        help="Explicit reference-model device (e.g. cuda:1). "
                             "Defaults to cuda:1 when 2+ GPUs are visible.")
    parser.add_argument("--reward_device", type=str, default="",
                        help="Explicit reward-model device (e.g. cuda:2). "
                             "Defaults to cuda:2 when 3+ GPUs are visible.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <output_dir>/last after an interruption.")
    parser.add_argument("--resume_every", type=int, default=10,
                        help="Refresh <output_dir>/last every N iterations.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
