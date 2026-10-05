"""
Shared model utilities for the Visual ToM training pipeline.
Supports:
  - Qwen/Qwen2.5-VL-7B-Instruct
  - Qwen/Qwen-VL-Chat
  - llava-hf/llava-v1.6-mistral-7b-hf

Model types:
  - "qwen2.5-vl": Qwen2VLForConditionalGeneration + AutoProcessor
  - "qwen-vl-chat": AutoModelForCausalLM + AutoTokenizer (trust_remote_code)
  - "llava-next": LlavaNextForConditionalGeneration + LlavaNextProcessor
"""

import os
import torch
from typing import Optional, List, Dict, Tuple, Union
from PIL import Image

# ── BeamSearchScorer patch (Qwen-VL-Chat needs it, removed in transformers>=4.45)
import transformers
try:
    from transformers import BeamSearchScorer  # noqa
except ImportError:
    class _BeamSearchScorerStub:
        pass
    transformers.BeamSearchScorer = _BeamSearchScorerStub
    if hasattr(transformers, "generation"):
        transformers.generation.BeamSearchScorer = _BeamSearchScorerStub


# Shared system prompt for chat-style backbones.
ROLEPLAY_SYSTEM_PROMPT = (
    "You are a dedicated role-playing assistant designed to immerse yourself "
    "fully in the character you are portraying."
)


# ──────────────────────────────────────────────────────────────────────────────
# Model type detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_model_type(model_name: str) -> str:
    """Auto-detect model type from HuggingFace model name."""
    name_lower = model_name.lower().replace("/", "-")
    if "llava" in name_lower and ("v1.6" in name_lower or "next" in name_lower or "mistral" in name_lower):
        return "llava-next"
    if "qwen2.5-vl" in name_lower or "qwen2_5" in name_lower:
        return "qwen2.5-vl"
    if "qwen2-vl" in name_lower:
        return "qwen2-vl"
    if "qwen-vl" in name_lower:
        return "qwen-vl-chat"
    # Default to qwen2.5-vl for unknown / generic qwen names
    return "qwen2.5-vl"


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────

def resolve_image(example: dict, image_dir: str) -> Optional[str]:
    """Resolve image path from an example dict."""
    for key in ["image_local", "image"]:
        img = example.get(key, "")
        if not img:
            continue
        full = os.path.join(image_dir, img)
        if os.path.exists(full):
            return full
        fname = os.path.basename(img)
        for sub in ["coco", "character", ""]:
            cand = os.path.join(image_dir, sub, fname)
            if os.path.exists(cand):
                return cand
    return None


def load_and_resize_image(image_path: str, max_side: int = 512) -> Optional[Image.Image]:
    """Load and resize image, returns None on failure."""
    try:
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_side:
            ratio = max_side / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Model + processor/tokenizer loading
# ──────────────────────────────────────────────────────────────────────────────

def load_base_model(model_name: str, model_type: Optional[str] = None,
                    torch_dtype=torch.bfloat16):
    """Load base model and processor/tokenizer. Returns (model, processor, model_type).

    Does NOT move to device — caller handles device placement / device_map.
    """
    if model_type is None:
        model_type = detect_model_type(model_name)

    print(f"  Loading model: {model_name} (type={model_type})", flush=True)

    if model_type == "qwen2.5-vl":
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLCls
        except ImportError:
            # Older transformers only had Qwen2VL; fall back (may fail for 2.5).
            from transformers import Qwen2VLForConditionalGeneration as _VLCls

        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = _VLCls.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        return model, processor, model_type

    elif model_type == "qwen2-vl":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        return model, processor, model_type

    elif model_type == "qwen-vl-chat":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        return model, tokenizer, model_type

    elif model_type == "llava-next":
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        processor = LlavaNextProcessor.from_pretrained(model_name)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        return model, processor, model_type

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_tokenizer(processor_or_tokenizer, model_type: str):
    """Extract the callable HuggingFace tokenizer from a processor or tokenizer.

    Qwen-VL-Chat's QWenTokenizer has a `.tokenizer` attribute pointing to a
    tiktoken.Encoding backend, which is NOT callable. We must guard against
    returning that internal object instead of the tokenizer itself.
    """
    if hasattr(processor_or_tokenizer, "tokenizer"):
        inner = processor_or_tokenizer.tokenizer
        if callable(inner):
            return inner
    return processor_or_tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Chat text construction
# ──────────────────────────────────────────────────────────────────────────────

def build_full_chat_text(prompt: str, response: str,
                         image_path: Optional[str],
                         processor_or_tokenizer, model_type: str) -> str:
    """Build complete chat text (prompt + response) for training."""
    if model_type == "qwen2.5-vl":
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": response},
        ]
        return processor_or_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    elif model_type == "llava-next":
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]
        return processor_or_tokenizer.apply_chat_template(
            messages, add_generation_prompt=False
        )

    elif model_type == "qwen-vl-chat":
        # Build query with image tags
        if image_path:
            query_parts = [{"image": image_path}, {"text": prompt}]
            query = processor_or_tokenizer.from_list_format(query_parts)
        else:
            query = prompt
        # ChatML format
        text = (
            f"<|im_start|>system\n{ROLEPLAY_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        return text


def build_generation_text(prompt: str, image_path: Optional[str],
                          processor_or_tokenizer, model_type: str) -> str:
    """Build prompt text for generation (no response)."""
    if model_type == "qwen2.5-vl":
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return processor_or_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    elif model_type == "llava-next":
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return processor_or_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    elif model_type == "qwen-vl-chat":
        if image_path:
            query_parts = [{"image": image_path}, {"text": prompt}]
            query = processor_or_tokenizer.from_list_format(query_parts)
        else:
            query = prompt
        text = (
            f"<|im_start|>system\n{ROLEPLAY_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return text


def build_prompt_only_text(prompt: str, processor_or_tokenizer,
                           model_type: str) -> str:
    """Build prompt-only text (no image) for computing prompt token length."""
    if model_type == "qwen2.5-vl":
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return processor_or_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    elif model_type == "llava-next":
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return processor_or_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    elif model_type == "qwen-vl-chat":
        text = (
            f"<|im_start|>system\n{ROLEPLAY_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization for training
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_batch(texts: List[str], images: List[Optional[Image.Image]],
                   processor_or_tokenizer, model_type: str,
                   max_len: int = 2048) -> dict:
    """Tokenize a batch of texts + images for training.

    Returns dict with input_ids, attention_mask (and pixel_values for qwen2.5-vl).
    """
    if model_type in {"qwen2.5-vl", "llava-next"}:
        has_images = [img is not None for img in images]
        if any(has_images):
            if not all(has_images):
                missing = sum(1 for has_image in has_images if not has_image)
                raise ValueError(
                    f"Mixed image availability in a {model_type} batch "
                    f"({missing}/{len(images)} missing). Resolve missing image paths "
                    "before multimodal tokenization."
                )
            # Do not truncate multimodal examples: truncation can cut through the
            # processor-expanded image token block and desynchronize text/image ids.
            inputs = processor_or_tokenizer(
                text=texts, images=images,
                return_tensors="pt", padding=True,
            )
        else:
            inputs = processor_or_tokenizer(
                text=texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
        return inputs

    elif model_type == "qwen-vl-chat":
        # Qwen-VL-Chat: images are embedded in text via <img> tags
        # Tokenize each example individually, then pad
        all_input_ids = []
        for text in texts:
            enc = processor_or_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_len,
            )
            all_input_ids.append(enc.input_ids.squeeze(0))

        # Pad to max length in batch
        max_batch_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = processor_or_tokenizer.pad_token_id or 0

        padded_ids = torch.full((len(texts), max_batch_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_batch_len), dtype=torch.long)

        for i, ids in enumerate(all_input_ids):
            padded_ids[i, :ids.shape[0]] = ids
            attention_mask[i, :ids.shape[0]] = 1

        return {"input_ids": padded_ids, "attention_mask": attention_mask}


def compute_prompt_length(prompt: str, processor_or_tokenizer,
                          model_type: str, max_len: int = 2048,
                          image: Optional[Image.Image] = None,
                          image_path: Optional[str] = None) -> int:
    """Compute the token length of just the prompt (for label masking)."""
    if model_type in {"qwen2.5-vl", "llava-next"}:
        prompt_text = build_generation_text(
            prompt, "__IMAGE__" if image is not None else None,
            processor_or_tokenizer, model_type,
        )
        if image is not None:
            enc = processor_or_tokenizer(
                text=[prompt_text], images=[image],
                return_tensors="pt", padding=True,
            )
        else:
            enc = processor_or_tokenizer(
                text=[prompt_text], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
        return int(enc["attention_mask"][0].sum().item())

    prompt_text = build_generation_text(
        prompt, image_path, processor_or_tokenizer, model_type
    )
    tokenizer = get_tokenizer(processor_or_tokenizer, model_type)
    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=max_len
    ).input_ids
    return len(prompt_ids)


def create_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  prompt_lengths: List[int]) -> torch.Tensor:
    """Create labels tensor by masking prompt tokens and padding."""
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lengths):
        labels[i, :min(plen, labels.shape[1])] = -100
    labels[attention_mask == 0] = -100
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Generation utilities
# ──────────────────────────────────────────────────────────────────────────────

def prepare_generation_inputs(prompt: str, image_path: Optional[str],
                              processor_or_tokenizer, model_type: str,
                              device: str = "cuda") -> Tuple[dict, int]:
    """Prepare inputs for model.generate(). Returns (inputs_dict, prompt_length)."""
    gen_text = build_generation_text(prompt, image_path, processor_or_tokenizer, model_type)

    if model_type in {"qwen2.5-vl", "llava-next"}:
        image = None
        if image_path:
            image = load_and_resize_image(image_path)
        if image is not None:
            inputs = processor_or_tokenizer(
                text=[gen_text], images=[image],
                return_tensors="pt", padding=True,
            ).to(device)
        else:
            inputs = processor_or_tokenizer(
                text=[gen_text], return_tensors="pt", padding=True,
            ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        return inputs, prompt_len

    elif model_type == "qwen-vl-chat":
        inputs = processor_or_tokenizer(
            gen_text, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        return inputs, prompt_len


# ──────────────────────────────────────────────────────────────────────────────
# LoRA configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def default_lora_target_modules(model_type: str) -> Union[str, List[str]]:
    """Return the default LoRA target module names for each supported model type.

    For qwen-vl-chat a regex string is returned instead of a list because the
    ViT MLP also has a layer named 'c_proj', so a plain suffix-match list would
    accidentally wrap ViT layers with LoRA adapters.  PEFT uses re.fullmatch
    on the full module path when target_modules is a string, which lets us
    restrict LoRA to the LM decoder (transformer.h.*) only.
    """
    if model_type in {"qwen2.5-vl", "qwen2-vl"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    if model_type == "qwen-vl-chat":
        # Regex: only LM decoder layers under transformer.h.*, not ViT layers.
        # The ViT's VisualAttentionBlock MLP has a 'c_proj' layer; using a list
        # of plain names would wrap those too, doubling ViT activation memory.
        return r"transformer\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(w1|w2))"
    if model_type == "llava-next":
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    # Fallback: standard decoder-only names
    return ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]
