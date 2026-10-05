#!/usr/bin/env python3
"""
Step 6a: Generate Responses from Baseline VLMs on MMRole Official Test
========================================================================
Runs open-source VLMs on the official test set to produce responses
for ToM evaluation (step6_evaluate_tom.py).

Supported models:
  - Qwen/Qwen-VL-Chat
  - llava-hf/llava-v1.6-mistral-7b-hf  (LLaVA-NeXT-Mistral-7B)
  - 01-ai/Yi-VL-6B

Usage:
    python step6a_generate_responses.py \
        --model qwen-vl-chat \
        --test_path projects/mmrole/mmrole_official_test_annotated_clean.jsonl \
        --output_path projects/mmrole/eval_responses_qwen_vl_chat.jsonl

    # Run all 3 baselines
    python step6a_generate_responses.py --model all
"""

import os
import json
import argparse
import time
import torch
from typing import Optional
from PIL import Image

from model_utils import load_base_model, prepare_generation_inputs

# ---------------------------------------------------------------------------
# Patch: Qwen-VL-Chat imports BeamSearchScorer which was removed in
# transformers>=4.45.  Inject a stub so the import doesn't crash.
# ---------------------------------------------------------------------------
import transformers
try:
    from transformers import BeamSearchScorer  # noqa
except ImportError:
    class _BeamSearchScorerStub:
        """Stub for removed BeamSearchScorer — Qwen-VL trust_remote_code needs it at import time."""
        pass
    transformers.BeamSearchScorer = _BeamSearchScorerStub
    transformers.generation.BeamSearchScorer = _BeamSearchScorerStub


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "qwen2.5-vl-local": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "short_name": "qwen2_5_vl_local",
        "type": "qwen2.5-vl",
    },
    "qwen-vl-chat": {
        "hf_id": "Qwen/Qwen-VL-Chat",   #Qwen/Qwen-VL-Chat
        "short_name": "qwen_vl_chat",
        "type": "qwen-vl",
    },
    "llava-next-mistral-7b": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "short_name": "llava_next_mistral_7b",
        "type": "llava-next",
    },
    "yi-vl-6b": {
        "hf_id": "01-ai/Yi-VL-6B",
        "short_name": "yi_vl_6b",
        "type": "yi-vl",
    },
}


# ---------------------------------------------------------------------------
# Prompt builder (consistent with MMRole test format)
# ---------------------------------------------------------------------------

def build_mmrole_prompt(annotation: dict) -> str:
    """Build the role-play prompt from our annotated test example."""
    agents = annotation["agents"]
    ctx = annotation["interaction_context"]
    speaker = agents["speaker"]
    partner = agents["partner"]

    history_lines = []
    for t in ctx.get("dialogue_history", []):
        history_lines.append(f"[{t['speaker']}]: {t['utterance']}")
    history_text = "\n".join(history_lines)

    dtype = ctx.get("dialogue_type", "inter_role")

    if dtype in ("inter-role", "inter_role"):
        prompt = (
            f"Please step into the shoes of {speaker['name']}. "
            f"Imagine you are talking with {partner['name']} about the given image. "
            f"This requires a deep understanding of the character's background, "
            f"including their personality, experiences, abilities, and relationships.\n\n"
            f"The description of {speaker['name']}:\n{speaker['profile'][:1500]}\n\n"
            f"The description of {partner['name']}:\n{partner['profile'][:1500]}\n\n"
        )
        if history_text:
            prompt += (
                f"The conversation history between {speaker['name']} and {partner['name']}:\n"
                f"{history_text}\n\n"
            )
        prompt += (
            f"Please respond to the following words of {partner['name']} about the image "
            f"using the distinctive tone, manner and vocabulary of {speaker['name']}:\n"
            f"{ctx.get('current_utterance', '')}"
        )
    elif dtype in ("human-role", "human_role"):
        prompt = (
            f"Please step into the shoes of {speaker['name']}. "
            f"Imagine you are talking with a curious human about the given image.\n\n"
            f"The description of {speaker['name']}:\n{speaker['profile'][:1500]}\n\n"
        )
        if history_text:
            prompt += f"Conversation history:\n{history_text}\n\n"
        prompt += (
            f"Please respond about the image using the distinctive tone, manner "
            f"and vocabulary of {speaker['name']}:\n"
            f"{ctx.get('current_utterance', '')}"
        )
    else:  # comment
        prompt = (
            f"Please step into the shoes of {speaker['name']}. "
            f"Comment on the given image as {speaker['name']} would.\n\n"
            f"The description of {speaker['name']}:\n{speaker['profile'][:1500]}\n\n"
            f"Please provide your comment using the distinctive tone, manner "
            f"and vocabulary of {speaker['name']}."
        )

    return prompt


def resolve_image(scene: dict, image_dir: str) -> Optional[str]:
    for key in ["image_local", "image"]:
        img = scene.get(key, "")
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


# ---------------------------------------------------------------------------
# Qwen-VL-Chat
# ---------------------------------------------------------------------------

def load_qwen_vl(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    return model, tokenizer


def generate_qwen_vl(model, tokenizer, prompt: str, image_path: Optional[str]) -> str:
    query_parts = []
    if image_path:
        query_parts.append({"image": image_path})
    query_parts.append({"text": prompt})

    query = tokenizer.from_list_format(query_parts)
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def load_qwen25_vl_local(model_id: str, device: str,
                         adapter_path: Optional[str] = None):
    model, processor, _ = load_base_model(
        model_id,
        model_type="qwen2.5-vl",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if adapter_path:
        from peft import PeftModel
        from transformers import AutoProcessor
        if os.path.exists(os.path.join(adapter_path, "preprocessor_config.json")):
            processor = AutoProcessor.from_pretrained(
                adapter_path,
                trust_remote_code=True,
                min_pixels=256 * 28 * 28,
                max_pixels=512 * 28 * 28,
            )
        model = PeftModel.from_pretrained(model, adapter_path)

    target_device = device
    if target_device == "auto":
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(target_device).eval()
    model.config.use_cache = True
    return model, processor


def generate_qwen25_vl_local(model, processor, prompt: str,
                             image_path: Optional[str],
                             max_new_tokens: int = 512) -> str:
    device = next(model.parameters()).device
    inputs, prompt_len = prepare_generation_inputs(
        prompt,
        image_path,
        processor,
        "qwen2.5-vl",
        device=str(device),
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response_ids = output[0][prompt_len:]
    return processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# LLaVA-NeXT (Mistral-7B)
# ---------------------------------------------------------------------------

def load_llava_next(model_id: str, device: str):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    ).eval()
    return model, processor


def generate_llava_next(model, processor, prompt: str, image_path: Optional[str]) -> str:
    conversation = [
        {
            "role": "user",
            "content": [],
        }
    ]
    if image_path:
        conversation[0]["content"].append({"type": "image"})
    conversation[0]["content"].append({"type": "text", "text": prompt})

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image = None
    if image_path:
        image = Image.open(image_path).convert("RGB")

    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    decoded = processor.decode(output[0], skip_special_tokens=True)
    # Extract assistant response after [/INST]
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return decoded


# ---------------------------------------------------------------------------
# Yi-VL-6B
# ---------------------------------------------------------------------------

def _build_yi_vl_projector(mm_hidden_size, hidden_size, dtype):
    """Build Yi-VL's mm_projector: Linear-LayerNorm-GELU-Linear-LayerNorm.

    The BUAADreamer/Yi-VL-6B-hf conversion saved these as linear_{1..4}
    but layers 2 and 4 are actually LayerNorm (1-D weights), which crashes
    when transformers tries to load them into nn.Linear (2-D weights).
    """
    import torch.nn as nn
    proj = nn.Sequential(
        nn.Linear(mm_hidden_size, hidden_size, dtype=dtype),   # linear_1
        nn.LayerNorm(hidden_size, dtype=dtype),                 # linear_2 (actually LN)
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size, dtype=dtype),       # linear_3
        nn.LayerNorm(hidden_size, dtype=dtype),                 # linear_4 (actually LN)
    )
    return proj


def load_yi_vl(model_id: str = "BUAADreamer/Yi-VL-6B-hf", device_map: str = "auto"):
    # BUAADreamer/Yi-VL-6B-hf's projector has LayerNorm layers saved under
    # "linear_2"/"linear_4" keys.  Transformers' LlavaMultiModalProjector only
    # has two nn.Linear layers, so loading crashes with a shape mismatch.
    # Fix: let from_pretrained handle all key remapping (it knows how), just
    # skip the mismatched projector layers, then reload them manually.
    import torch.nn as nn
    from transformers import (
        LlavaForConditionalGeneration, LlavaConfig,
        AutoTokenizer, CLIPImageProcessor,
    )
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    import json

    hf_id = "BUAADreamer/Yi-VL-6B-hf"

    # 1) Tokenizer / image processor
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    image_processor = CLIPImageProcessor.from_pretrained(hf_id)

    # 2) Load model on CPU first (no device_map) so we can swap the projector
    #    before accelerate wraps the modules.
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        ignore_mismatched_sizes=True,   # lets the 1D-vs-2D projector keys through
        low_cpu_mem_usage=True,
    )

    # 3) Build the correct 5-layer projector and load its weights *before*
    #    attaching it to the model (avoids transformers/__setattr__ hooks).
    config = model.config
    mm_hidden = config.vision_config.hidden_size   # 1280
    hidden    = config.text_config.hidden_size      # 4096
    projector = _build_yi_vl_projector(mm_hidden, hidden, torch.float16)

    # Load just the projector weights from the checkpoint
    index_path = hf_hub_download(hf_id, "model.safetensors.index.json")
    index = json.load(open(index_path))
    proj_shard = index["weight_map"]["multi_modal_projector.linear_1.weight"]
    shard_path = hf_hub_download(hf_id, proj_shard)

    # Map checkpoint names → Sequential indices (index 2 is GELU, no params)
    proj_key_map = {
        "multi_modal_projector.linear_1": 0,
        "multi_modal_projector.linear_2": 1,
        "multi_modal_projector.linear_3": 3,
        "multi_modal_projector.linear_4": 4,
    }
    proj_sd = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("multi_modal_projector."):
                continue
            parts = key.rsplit(".", 1)
            prefix, suffix = parts[0], parts[1]
            if prefix in proj_key_map:
                new_key = f"{proj_key_map[prefix]}.{suffix}"
                proj_sd[new_key] = f.get_tensor(key)

    projector.load_state_dict(proj_sd)

    # 4) Swap in the loaded projector and move to GPU
    #    multi_modal_projector is a property delegating to model.model
    model.model.multi_modal_projector = projector
    model = model.cuda().eval()
    return model, (tokenizer, image_processor)


def generate_yi_vl(model, processor, prompt: str, image_path: Optional[str]) -> str:
    tokenizer, image_processor = processor

    if image_path:
        image = Image.open(image_path).convert("RGB")
        # LlavaForConditionalGeneration expects <image> token in the prompt
        full_prompt = f"<image>\n{prompt}"
    else:
        image = None
        full_prompt = prompt

    # Build inputs via the processor pipeline
    conversation = [{"role": "user", "content": full_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
    else:
        text = full_prompt

    # Tokenize text
    text_inputs = tokenizer(text, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(model.device)

    gen_kwargs = dict(input_ids=input_ids, max_new_tokens=512, do_sample=False)

    if image is not None:
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(dtype=torch.float16, device=model.device)
        gen_kwargs["pixel_values"] = pixel_values

        # Expand the single <image> token into N placeholder tokens so
        # LlavaForConditionalGeneration can scatter visual features.
        image_token_id = model.config.image_token_index  # 64000
        ps = model.config.vision_config.patch_size
        img_sz = model.config.vision_config.image_size
        n_patches = (img_sz // ps) ** 2  # 1024
        ids = input_ids[0].tolist()
        expanded = []
        for tid in ids:
            if tid == image_token_id:
                expanded.extend([image_token_id] * n_patches)
            else:
                expanded.append(tid)
        input_ids = torch.tensor([expanded], device=model.device)
        gen_kwargs["input_ids"] = input_ids

    with torch.no_grad():
        output = model.generate(**gen_kwargs)

    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Unified generation
# ---------------------------------------------------------------------------

def load_model(model_key: str, device: str = "auto",
               base_model: Optional[str] = None,
               adapter_path: Optional[str] = None):
    info = MODEL_REGISTRY[model_key]
    hf_id = base_model or info["hf_id"]
    print(f"Loading {hf_id} ...")

    if info["type"] == "qwen2.5-vl":
        return load_qwen25_vl_local(hf_id, device, adapter_path) + ("qwen2.5-vl",)
    elif info["type"] == "qwen-vl":
        return load_qwen_vl(hf_id, device) + ("qwen-vl",)
    elif info["type"] == "llava-next":
        return load_llava_next(hf_id, device) + ("llava-next",)
    elif info["type"] == "yi-vl":
        return load_yi_vl(hf_id, device) + ("yi-vl",)
    else:
        raise ValueError(f"Unknown model type: {info['type']}")


def generate(model, processor_or_tokenizer, model_type: str,
             prompt: str, image_path: Optional[str],
             max_new_tokens: int = 512) -> str:
    if model_type == "qwen2.5-vl":
        return generate_qwen25_vl_local(
            model, processor_or_tokenizer, prompt, image_path, max_new_tokens
        )
    elif model_type == "qwen-vl":
        return generate_qwen_vl(model, processor_or_tokenizer, prompt, image_path)
    elif model_type == "llava-next":
        return generate_llava_next(model, processor_or_tokenizer, prompt, image_path)
    elif model_type == "yi-vl":
        return generate_yi_vl(model, processor_or_tokenizer, prompt, image_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_generation(model_key: str, test_path: str, output_path: str,
                   image_dir: str, device: str, max_examples: int,
                   base_model: Optional[str] = None,
                   adapter_path: Optional[str] = None,
                   max_new_tokens: int = 512):
    info = MODEL_REGISTRY[model_key]

    # Load test data
    annotations = []
    with open(test_path) as f:
        for line in f:
            annotations.append(json.loads(line))
    if max_examples > 0:
        annotations = annotations[:max_examples]
    print(f"Test examples: {len(annotations)}")

    # Load model
    model, processor, model_type = load_model(
        model_key, device, base_model=base_model, adapter_path=adapter_path
    )
    print(f"Model loaded: {base_model or info['hf_id']}")
    if adapter_path:
        print(f"Adapter loaded: {adapter_path}")

    # Check for resume
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    completed.add(d["example_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Already completed: {len(completed)}")

    remaining = [a for a in annotations if a["example_id"] not in completed]
    print(f"To generate: {len(remaining)}")

    start = time.time()
    done = 0
    errors = 0

    with open(output_path, "a") as out_f:
        for ann in remaining:
            try:
                prompt = build_mmrole_prompt(ann)
                image_path = resolve_image(ann.get("scene", {}), image_dir)

                response = generate(
                    model, processor, model_type, prompt, image_path,
                    max_new_tokens=max_new_tokens,
                )

                result = {
                    "example_id": ann["example_id"],
                    "dialogue_id": ann.get("dialogue_id", ""),
                    "model": base_model or info["hf_id"],
                    "model_key": model_key,
                    "adapter_path": adapter_path or "",
                    "response": response,
                    "speaker": ann["agents"]["speaker"]["name"],
                    "partner": ann["agents"]["partner"]["name"],
                    "dialogue_type": ann.get("test_metadata", {}).get("dialogue_type", ""),
                    "distribution": ann.get("test_metadata", {}).get("distribution", ""),
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1

                if done % 20 == 0:
                    elapsed = time.time() - start
                    print(f"  {done}/{len(remaining)} done "
                          f"({elapsed:.0f}s, {done/elapsed:.1f} ex/s)")

            except Exception as e:
                errors += 1
                print(f"  Error on {ann['example_id']}: {str(e)[:100]}")

    elapsed = time.time() - start
    print(f"\nDone: {done}, Errors: {errors}, Time: {elapsed:.0f}s")
    print(f"Output: {output_path}")

    # Free GPU memory
    del model, processor
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Generate VLM responses on MMRole test set")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to run (or 'all' for all baselines)")
    parser.add_argument("--test_path", type=str,
                        default="projects/mmrole/mmrole_official_test_annotated_clean.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/eval_responses")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--base_model", type=str, default="",
                        help="Override HF base model for local checkpoint runs")
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Optional LoRA adapter path for local checkpoint runs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "all":
        models_to_run = list(MODEL_REGISTRY.keys())
    else:
        models_to_run = [args.model]

    for model_key in models_to_run:
        info = MODEL_REGISTRY[model_key]
        output_path = os.path.join(args.output_dir, f"{info['short_name']}.jsonl")
        print(f"\n{'='*60}")
        print(f"  {model_key} → {output_path}")
        print(f"{'='*60}\n")
        run_generation(model_key, args.test_path, output_path,
                       args.image_dir, args.device, args.max_examples,
                       base_model=args.base_model or None,
                       adapter_path=args.adapter_path or None,
                       max_new_tokens=args.max_new_tokens)

    print(f"\n{'='*60}")
    print(f"All response generation complete!")
    print(f"{'='*60}")
    print(f"\nNext: evaluate with ToM dimensions:")
    for model_key in models_to_run:
        info = MODEL_REGISTRY[model_key]
        resp_path = os.path.join(args.output_dir, f"{info['short_name']}.jsonl")
        print(f"  python step6_evaluate_tom.py response --responses_path {resp_path}")


if __name__ == "__main__":
    main()
