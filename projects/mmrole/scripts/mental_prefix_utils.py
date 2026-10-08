"""
Utilities for frozen mental-prefix conditioning from a Stage 0 checkpoint.

This module does not inject continuous prefix embeddings into the policy
backbone. Instead, it uses the frozen Stage 0 latent encoder plus z-only ToM
heads to produce a compact textual "mental prefix" that can be prepended to the
policy prompt during Stage 1/2 training. This is the lowest-risk way to add
direct mental conditioning without rewriting the multimodal token path.
"""

import os
from typing import Dict, List, Optional

import torch
from peft import PeftModel

from model_utils import (
    detect_model_type, load_base_model, get_tokenizer,
    resolve_image, load_and_resize_image,
)
from stage0_reward_model_visual_tom import (
    VisualRecursiveToMRewardModel,
    CUSTOM_HEAD_NAMES as REWARD_HEAD_NAMES,
    format_reward_context,
    _build_context_text_for_processor,
)


MENTAL_PREFIX_DIM_NAMES = [
    "first_order_tom",
    "second_order_tom",
    "belief_divergence",
]


def _bucket_score(score: float) -> str:
    score = float(max(0.0, min(1.0, score)))
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def format_mental_prefix_text(scores: List[float]) -> str:
    scores = [float(max(0.0, min(1.0, s))) for s in scores]
    first, second, divergence = scores
    return (
        "<mental_prefix>\n"
        f"- First-order ToM demand: {_bucket_score(first)} ({first:.2f})\n"
        f"- Second-order ToM demand: {_bucket_score(second)} ({second:.2f})\n"
        f"- Belief divergence: {_bucket_score(divergence)} ({divergence:.2f})\n"
        "</mental_prefix>"
    )


class FrozenMentalPrefixModel:
    """Loads Stage 0 and predicts compact ToM prefix text from context/image."""

    def __init__(self, base_model_name: str, checkpoint_dir: str,
                 model_type: Optional[str] = None,
                 z_dim: int = 128, device: str = "cuda:0",
                 image_dir: str = "", max_ctx_len: int = 1024):
        self.device = device
        self.image_dir = image_dir
        self.max_ctx_len = max_ctx_len
        required = ["lora_adapter"] + [
            f"{name}.pth" for name in ("z1_mu", "z2_mu", "z1_only_reward_head", "z_combined_reward_head")
        ]
        missing = [name for name in required if not os.path.exists(os.path.join(checkpoint_dir, name))]
        if missing:
            raise FileNotFoundError(f"Mental-prefix checkpoint {checkpoint_dir} is missing {missing}.")

        base_model, processor, detected_type = load_base_model(
            base_model_name, model_type,
        )
        self.model_type = detected_type
        self.processor = processor
        self.tokenizer = get_tokenizer(processor, detected_type)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_path = os.path.join(checkpoint_dir, "lora_adapter")
        if os.path.exists(lora_path):
            base_model = PeftModel.from_pretrained(
                base_model, lora_path, torch_dtype=torch.bfloat16,
            )
            base_model = base_model.merge_and_unload()

        base_model = base_model.to(device)
        self.model = VisualRecursiveToMRewardModel(
            base_model, detected_type, z_dim=z_dim,
        ).to(device)

        for head_name in REWARD_HEAD_NAMES:
            path = os.path.join(checkpoint_dir, f"{head_name}.pth")
            if os.path.exists(path):
                getattr(self.model, head_name).load_state_dict(
                    torch.load(path, map_location=device, weights_only=True)
                )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _encode_context_batch(self, samples: List[Dict]):
        texts = []
        images = []
        for sample in samples:
            synth = {
                "context_text": format_reward_context(sample),
                "image_path": resolve_image(sample, self.image_dir),
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
                        "Mixed image availability while encoding mental prefixes "
                        f"({missing}/{len(images)} missing)."
                    )
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
            for text in texts:
                enc = self.tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=self.max_ctx_len,
                )
                all_ids.append(enc.input_ids.squeeze(0))
            max_batch_len = max(ids.shape[0] for ids in all_ids)
            ids = torch.full((len(all_ids), max_batch_len), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, token_ids in enumerate(all_ids):
                ids[i, :token_ids.shape[0]] = token_ids
                mask[i, :token_ids.shape[0]] = 1
            enc = {"input_ids": ids, "attention_mask": mask}

        enc = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in enc.items()
        }
        with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            mu1, mu2 = self.model.encode_context_deterministic(
                enc["input_ids"], enc["attention_mask"],
                pixel_values=enc.get("pixel_values"),
                image_grid_thw=enc.get("image_grid_thw"),
            )
        return mu1, mu2

    @torch.no_grad()
    def build_prefix_map(self, examples: List[Dict], batch_size: int = 8) -> Dict[str, str]:
        prefix_map: Dict[str, str] = {}
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            mu1, mu2 = self._encode_context_batch(batch)
            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                z1_scores = self.model.z1_only_reward_head(mu1).float()
                zc_scores = self.model.z_combined_reward_head(
                    torch.cat([mu1, mu2], dim=1)
                ).float()
            combined = (0.5 * (z1_scores + zc_scores)).clamp_(0.0, 1.0)
            for example, scores in zip(batch, combined.cpu().tolist()):
                example_id = example.get("example_id", "")
                if example_id:
                    prefix_map[example_id] = format_mental_prefix_text(scores)
        return prefix_map
