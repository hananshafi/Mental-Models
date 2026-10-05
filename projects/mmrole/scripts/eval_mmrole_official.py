#!/usr/bin/env python3
"""
Official MMRole Evaluation Pipeline (Reproduce Paper Results)
==============================================================
Two-stage evaluation following the official MMRole methodology:

  Stage 1 (inference): Generate model responses for test prompts
  Stage 2 (scoring):   Score responses using the MMRole reward model
                        (QWen-VL-Chat fine-tuned) against GT responses

The reward model compares Model A (evaluated) vs Model B (ground truth)
on 8 dimensions, outputting score pairs. Final metric = ratio (A/B).
  - ratio > 1.0  → model beats GT
  - ratio = 1.0  → on par with GT
  - ratio < 1.0  → worse than GT

Supports:
  - Local HF models (QWen, LLaMA, Mistral, etc.)
  - API models (GPT-4o, Claude, etc.) via litellm
  - MMRole Reward Model (official QWen-VL-Chat fine-tuned)
  - GPT-4o as fallback judge (if RM not available)
  - Resume support (skips already-scored samples)

Usage:
    # Stage 1: Generate answers with a local model
    python eval_mmrole_official.py infer \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --test_dir projects/mmrole/raw_data \
        --image_dir projects/mmrole/images \
        --output_dir projects/mmrole/eval_answers/qwen7b

    # Stage 2: Score using official RM
    python eval_mmrole_official.py score \
        --rm_path YanqiDai/MMRole-Eval_RM \
        --answers_dir projects/mmrole/eval_answers/qwen7b \
        --image_dir projects/mmrole/images \
        --output_dir projects/mmrole/eval_reviews/qwen7b

    # Stage 2 alt: Score using GPT-4o as judge
    python eval_mmrole_official.py score \
        --use_gpt4_judge \
        --answers_dir projects/mmrole/eval_answers/qwen7b \
        --image_dir projects/mmrole/images \
        --output_dir projects/mmrole/eval_reviews/qwen7b_gpt4judge

    # Aggregate results
    python eval_mmrole_official.py results \
        --reviews_dir projects/mmrole/eval_reviews/qwen7b
"""

import os
import sys
import json
import csv
import base64
import argparse
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
from PIL import Image

import torch
torch.manual_seed(1234)

from model_utils import load_base_model, prepare_generation_inputs

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SCORE_KEYS = [
    "Instruction Adherence",
    "Fluency",
    "Coherency",
    "Image-Text Relevance",
    "Response Accuracy",
    "Personality Consistency",
    "Knowledge Consistency",
    "Tone Consistency",
]

SCORE_KEY_SHORT = {
    "Instruction Adherence": "IA",
    "Fluency": "Flu",
    "Coherency": "Coh",
    "Image-Text Relevance": "ITR",
    "Response Accuracy": "RA",
    "Personality Consistency": "PC",
    "Knowledge Consistency": "KC",
    "Tone Consistency": "TC",
}

DIMENSION_CATEGORIES = {
    "Fundamental Conversational Skills": ["Instruction Adherence", "Fluency", "Coherency"],
    "Multimodal Understanding": ["Image-Text Relevance", "Response Accuracy"],
    "Role-Playing Qualities": ["Personality Consistency", "Knowledge Consistency", "Tone Consistency"],
}

TEST_FILES = [
    "data_test_in-distribution_inter-role_test.jsonl",
    "data_test_in-distribution_human-role_test.jsonl",
    "data_test_in-distribution_comment_test.jsonl",
    "data_test_out-of-distribution_inter-role_test.jsonl",
    "data_test_out-of-distribution_human-role_test.jsonl",
    "data_test_out-of-distribution_comment_test.jsonl",
]

SYSTEM_ROLEPLAY = (
    "You are a dedicated role-playing assistant designed to immerse yourself "
    "fully in the character you are portraying."
)

SYSTEM_EVALUATOR = (
    "You are an objective and precise evaluator, specializing in rigorously "
    "assessing the role-playing and multimodal understanding abilities of "
    "various models."
)

MODEL_REGISTRY = {
    "qwen2.5-vl-local": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "short_name": "qwen2_5_vl_local",
        "type": "qwen2.5-vl",
    },
    "qwen-vl-chat": {
        "hf_id": "Qwen/Qwen-VL-Chat",
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


def resolve_image_path(image_field: str, image_dir: str) -> str:
    """Map dataset image paths to local filesystem.

    Official code uses ``os.path.join(image_dir, data['image'])`` directly.
    We try that first, then fall back to a remapped path for cases where
    COCO images were downloaded flat into ``images/coco/``.
    """
    # 1. Direct join (official behaviour)
    direct = os.path.join(image_dir, image_field)
    if os.path.exists(direct):
        return direct
    # 2. Remapped: COCO/train2017/xxx.jpg -> coco/xxx.jpg
    remapped = image_field.replace("COCO/train2017/", "coco/").replace("COCO/val2017/", "coco/")
    full = os.path.join(image_dir, remapped)
    if os.path.exists(full):
        return full
    return direct  # return best guess (will surface a clear error downstream)


def encode_image_b64(image_path: str) -> Optional[str]:
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_test_data(test_dir: str) -> Dict[str, List[dict]]:
    """Load all official MMRole test JSONL files.

    Returns a dict keyed by ``(dist_type, interaction_type)`` strings that
    mirror the official directory layout:
        ``in-test/comment_test``, ``out-test/inter-role_test``, etc.
    """
    data_by_split = {}
    for fname in TEST_FILES:
        path = os.path.join(test_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found, skipping.")
            continue
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        # e.g. "data_test_in-distribution_inter-role_test.jsonl"
        #   -> dist_prefix = "in-test", out_name = "inter-role_test"
        bare = fname.replace("data_test_", "").replace(".jsonl", "")
        # bare = "in-distribution_inter-role_test"
        if bare.startswith("in-distribution_"):
            dist_prefix = "in-test"
            out_name = bare.replace("in-distribution_", "")
        elif bare.startswith("out-of-distribution_"):
            dist_prefix = "out-test"
            out_name = bare.replace("out-of-distribution_", "")
        else:
            dist_prefix = "other"
            out_name = bare
        split_key = f"{dist_prefix}/{out_name}"
        data_by_split[split_key] = samples
        print(f"  Loaded {len(samples)} samples -> {split_key}")
    return data_by_split


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Inference (Generate Model Answers)
# ──────────────────────────────────────────────────────────────────────────────
def run_inference_local(model, tokenizer, data_by_split, image_dir, output_dir,
                        device="cuda", max_new_tokens=512, temperature=0.7,
                        vlm_type=None, processor=None):
    """Generate answers using a local HF model.

    Args:
        vlm_type: One of ``"qwen-vl"``, ``"llava-next"``, ``"yi-vl"``, or
            ``None`` for text-only models.
        processor: HF image processor (required for llava-next / yi-vl).
    """
    for split_name, samples in data_by_split.items():
        out_path = os.path.join(output_dir, f"{split_name}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Resume: load existing
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            done_ids = {d["id"] for d in existing if d.get("conversations", [{}])[1].get("answer")}
            print(f"  Resuming {split_name}: {len(done_ids)}/{len(samples)} done")
        else:
            existing = []
            done_ids = set()

        # Build id->existing map
        existing_map = {d["id"]: d for d in existing}

        results = []
        for sample in tqdm(samples, desc=split_name):
            sid = sample["id"]
            if sid in done_ids:
                results.append(existing_map[sid])
                continue

            image_path = resolve_image_path(sample["image"], image_dir)
            question = sample["conversations"][0]["value"]

            if vlm_type == "qwen2.5-vl":
                answer = _infer_qwen25_vl(model, processor, question, image_path,
                                          max_new_tokens, temperature)
            elif vlm_type == "qwen-vl":
                answer = _infer_qwen_vl(model, tokenizer, question, image_path)
            elif vlm_type == "llava-next":
                answer = _infer_llava_next(model, processor, question, image_path,
                                           device, max_new_tokens, temperature)
            elif vlm_type == "yi-vl":
                answer = _infer_yi_vl(model, tokenizer, question, image_path,
                                      )
            else:
                answer = _infer_text(model, tokenizer, question, device,
                                     max_new_tokens, temperature)

            out_sample = dict(sample)
            out_sample["conversations"] = [
                sample["conversations"][0],
                {"from": "assistant", "value": sample["conversations"][1]["value"],
                 "answer": answer},
            ]
            results.append(out_sample)

            # Save incrementally
            if len(results) % 10 == 0:
                with open(out_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(results)} answers saved.")


def _infer_text(model, tokenizer, question, device, max_new_tokens, temperature):
    """Generate with a text-only causal LM."""
    prompt = f"{SYSTEM_ROLEPLAY}\n\n{question}"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def _infer_qwen_vl(model, tokenizer, question, image_path):
    """Generate with Qwen-VL-Chat using its native .chat() API."""
    try:
        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])
        response, _ = model.chat(tokenizer, query=query, history=None,
                                 system=SYSTEM_ROLEPLAY)
        return response
    except Exception as e:
        print(f"  Qwen-VL inference error: {e}")
        return ""


def _infer_llava_next(model, processor, question, image_path,
                      device, max_new_tokens, temperature):
    """Generate with LLaVA-NeXT (llava-hf) using the HF processor pipeline."""
    from PIL import Image
    try:
        image = Image.open(image_path).convert("RGB")
        # LLaVA-NeXT uses <image> token in the prompt
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_ROLEPLAY}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        # Decode only newly generated tokens
        response_ids = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(response_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  LLaVA-NeXT inference error: {e}")
        return ""


def _infer_qwen25_vl(model, processor, question, image_path, max_new_tokens, temperature):
    """Generate with Qwen2.5-VL using the shared model_utils pipeline."""
    try:
        device = str(next(model.parameters()).device)
        inputs, prompt_len = prepare_generation_inputs(
            question, image_path, processor, "qwen2.5-vl", device=device,
        )
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
            )
        response_ids = output[0][prompt_len:]
        return processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  Qwen2.5-VL inference error: {e}")
        return ""


def _infer_yi_vl(model, processor, prompt: str, image_path: Optional[str]) -> str:
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


def run_inference_api(model_name: str, data_by_split, image_dir, output_dir,
                      max_tokens=512, temperature=0.7):
    """Generate answers using an API model (GPT-4o, Claude, etc.)."""
    from openai import OpenAI
    client = OpenAI()

    for split_name, samples in data_by_split.items():
        out_path = os.path.join(output_dir, f"{split_name}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            done_ids = {d["id"] for d in existing if d.get("conversations", [{}])[1].get("answer")}
        else:
            existing = []
            done_ids = set()

        existing_map = {d["id"]: d for d in existing}
        results = []

        for sample in tqdm(samples, desc=split_name):
            sid = sample["id"]
            if sid in done_ids:
                results.append(existing_map[sid])
                continue

            image_path = resolve_image_path(sample["image"], image_dir)
            question = sample["conversations"][0]["value"]
            img_b64 = encode_image_b64(image_path)

            messages = [{"role": "system", "content": SYSTEM_ROLEPLAY}]
            content = []
            if img_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            content.append({"type": "text", "text": question})
            messages.append({"role": "user", "content": content})

            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    answer = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"  API error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
                    answer = ""

            out_sample = dict(sample)
            out_sample["conversations"] = [
                sample["conversations"][0],
                {"from": "assistant", "value": sample["conversations"][1]["value"],
                 "answer": answer},
            ]
            results.append(out_sample)

            if len(results) % 10 == 0:
                with open(out_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(results)} answers saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Scoring (Reward Model or GPT-4o Judge)
# ──────────────────────────────────────────────────────────────────────────────
def build_eval_prompt(question, evaluated_answer, groundtruth_answer,
                      role, script, other_role, score_key):
    """Build the official MMRole evaluation prompt."""
    if script == "Hypothetical Characters":
        role_script = f"{role}, a person who is one of the Hypothetical Characters,"
        if other_role == "a curious human":
            other_role_script = other_role
        else:
            other_role_script = f"{other_role}, a person who is one of the Hypothetical Characters,"
    else:
        role_script = f"{role} from {script}"
        if other_role == "a curious human":
            other_role_script = other_role
        else:
            other_role_script = f"{other_role} from {script}"

    the_other_role = other_role if other_role != "a curious human" else "the curious human"

    task = (
        f"The task instruction of the two models is to directly role-play as "
        f"{role_script} and talk with {other_role_script} about the given image "
        f"using the distinctive tone, manner and vocabulary of {role}."
    )

    aspect_desc = {
        "Instruction Adherence": f"Instruction Adherence: Do the responses accurately adhere to the task instruction, directly role-playing as {role} and only including words that {role} should say, without any additional explanatory prefixes or suffixes?",
        "Fluency": "Fluency: Are the responses grammatically correct and smoothly articulated?",
        "Coherency": "Coherency: Do the responses maintain a coherent thread of dialogue without contradicting earlier parts of the conversation or previously established facts?",
        "Image-Text Relevance": "Image-Text Relevance: Are the responses closely related to the visual content of the image?",
        "Response Accuracy": f"Response Accuracy: Do the responses accurately answer {the_other_role}'s words or appropriately initiate a conversation based on the image?",
        "Personality Consistency": f"Personality Consistency: Do the responses accurately and sufficiently reflect the personality of {role}?",
        "Knowledge Consistency": f"Knowledge Consistency: Are the responses consistent with the factual knowledge that {role} should possess, including experiences, abilities, and relationships?",
        "Tone Consistency": f"Tone Consistency: Do the responses maintain a consistent tone that aligns with {role}'s typical manner of speaking and catchphrases, rather than resembling the style of AI assistants?",
    }

    text = (
        f"## **[Question Start]**\n\n{question}\n\n## **[Question End]**\n\n\n"
        f"## **[Model A's Response Start]**\n\n{evaluated_answer}\n\n## **[Model A's Response End]**\n\n\n"
        f"## **[Model B's Response Start]**\n\n{groundtruth_answer}\n\n## **[Model B's Response End]**\n\n\n"
        f"## **[Instruction]**\n\n"
        f"{task}\n\n"
        f"Please evaluate the following aspect of each model's response:\n"
        f"{aspect_desc[score_key]}\n\n"
        f"Please provide a brief qualitative evaluation for the relative performance "
        f"of the two models, followed by paired quantitative scores from 1 to 10, "
        f"where 1 indicates poor performance and 10 indicates excellent performance.\n\n"
        f"The output should be in the following format:\n"
        f"{{Qualitative Evaluation}}, [Scores]: ({{the score of Model A}}, {{the score of Model B}})\n\n"
        f"Please ensure that your evaluations are unbiased and that the order in which "
        f"the responses were presented does not affect your judgment."
    )
    return text


def parse_score_pair(review: str):
    """Extract [score_A, score_B] from '[Scores]: (X, Y)' format.

    Returns a list of two floats on success, or an empty dict ``{}`` on
    failure — matching the official ``get_review_score()`` behaviour.
    """
    try:
        parts = review.split("[Scores]:")
        assert len(parts) == 2, "Missing [Scores]: delimiter"
        score_text = parts[1].strip().split("\n")[0].strip()
        score_text = score_text.replace("(", "").replace(")", "")
        score_pair = score_text.split(",")
        if len(score_pair) != 2:
            raise ValueError("Invalid score pair.")
        result = []
        for s in score_pair:
            s = s.strip()
            if ":" in s:
                s = s.split(":")[1].strip()
            result.append(float(s))
        return result
    except Exception as e:
        print(f"{e} You must manually fix the score pair.")
        return {}


def recover_score_pair(rm_entry):
    """Return a [score_a, score_b] pair from a review entry when possible."""
    if not isinstance(rm_entry, dict):
        return []
    score = rm_entry.get("score", [])
    if isinstance(score, list) and len(score) == 2:
        return score
    review = rm_entry.get("review", "")
    if review:
        parsed = parse_score_pair(review)
        if isinstance(parsed, list) and len(parsed) == 2:
            return parsed
    return []


def score_with_rm(rm_model, rm_tokenizer, answers_dir, image_dir, output_dir, device="cuda"):
    """Score using the official MMRole reward model (QWen-VL-Chat fine-tuned)."""
    # Collect answer JSON files (may be in subdirectories like in-test/, out-test/)
    answer_files = []
    for root, _dirs, files in os.walk(answers_dir):
        for f in files:
            if f.endswith(".json"):
                answer_files.append(os.path.relpath(os.path.join(root, f), answers_dir))
    if not answer_files:
        print("No answer files found!")
        return

    for rel_path in sorted(answer_files):
        print(f"\nScoring {rel_path}...")
        in_path = os.path.join(answers_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(in_path) as f:
            data_list = json.load(f)

        # Resume: load existing reviews
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            existing_map = {d["id"]: d for d in existing}
        else:
            existing_map = {}

        for data in tqdm(data_list, desc=rel_path):
            sid = data["id"]
            if sid in existing_map and "rm_review" in existing_map[sid]:
                # Check if all 8 dims are scored
                rm = existing_map[sid]["rm_review"]
                if all(k in rm and isinstance(rm[k].get("score"), list)
                       and len(rm[k]["score"]) == 2 for k in SCORE_KEYS):
                    data["rm_review"] = rm
                    continue

            evaluated_answer = data["conversations"][1].get("answer", "")
            groundtruth_answer = data["conversations"][1]["value"]

            if not evaluated_answer:
                print(f"  WARNING: No answer for {sid}, skipping.")
                continue

            image_path = resolve_image_path(data["image"], image_dir)
            question = data["conversations"][0]["value"]
            role = data["role"]
            script = data["script"]
            other_role = data["other_role"]

            rm_review_dict = {}
            for score_key in SCORE_KEYS:
                eval_prompt = build_eval_prompt(
                    question, evaluated_answer, groundtruth_answer,
                    role, script, other_role, score_key,
                )

                try:
                    query = rm_tokenizer.from_list_format([
                        {"image": image_path},
                        {"text": eval_prompt},
                    ])
                    response, _ = rm_model.chat(
                        rm_tokenizer, query=query, history=None,
                        system=SYSTEM_EVALUATOR,
                    )
                    score = parse_score_pair(response)
                    rm_review_dict[score_key] = {
                        "review": response,
                        "score": score,
                    }
                except Exception as e:
                    print(f"  RM error for {sid}/{score_key}: {e}")
                    rm_review_dict[score_key] = {"review": "", "score": {}}

            data["rm_review"] = rm_review_dict

            # Save after every sample (matches official behaviour — crash-safe)
            with open(out_path, "w") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)

        with open(out_path, "w") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(data_list)} reviewed samples to {out_path}")


def score_with_gpt4(answers_dir, image_dir, output_dir, judge_model="gpt-4o"):
    """Score using GPT-4o as judge (fallback when RM not available)."""
    from openai import OpenAI
    client = OpenAI()

    # Collect answer JSON files (may be in subdirectories)
    answer_files = []
    for root, _dirs, files in os.walk(answers_dir):
        for f in files:
            if f.endswith(".json"):
                answer_files.append(os.path.relpath(os.path.join(root, f), answers_dir))
    if not answer_files:
        print("No answer files found!")
        return

    for rel_path in sorted(answer_files):
        print(f"\nScoring {rel_path} with {judge_model}...")
        in_path = os.path.join(answers_dir, rel_path)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(in_path) as f:
            data_list = json.load(f)

        # Resume
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            existing_map = {d["id"]: d for d in existing}
        else:
            existing_map = {}

        for data in tqdm(data_list, desc=rel_path):
            sid = data["id"]
            if sid in existing_map and "rm_review" in existing_map[sid]:
                rm = existing_map[sid]["rm_review"]
                if all(k in rm and isinstance(rm[k].get("score"), list)
                       and len(rm[k]["score"]) == 2 for k in SCORE_KEYS):
                    data["rm_review"] = rm
                    continue

            evaluated_answer = data["conversations"][1].get("answer", "")
            groundtruth_answer = data["conversations"][1]["value"]

            if not evaluated_answer:
                continue

            image_path = resolve_image_path(data["image"], image_dir)
            question = data["conversations"][0]["value"]
            role = data["role"]
            script = data["script"]
            other_role = data["other_role"]

            img_b64 = encode_image_b64(image_path)

            rm_review_dict = {}
            for score_key in SCORE_KEYS:
                eval_prompt = build_eval_prompt(
                    question, evaluated_answer, groundtruth_answer,
                    role, script, other_role, score_key,
                )

                messages = [{"role": "system", "content": SYSTEM_EVALUATOR}]
                content = []
                if img_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                content.append({"type": "text", "text": eval_prompt})
                messages.append({"role": "user", "content": content})

                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=judge_model,
                            messages=messages,
                            max_tokens=512,
                            temperature=0.0,
                        )
                        review = resp.choices[0].message.content.strip()
                        score = parse_score_pair(review)
                        rm_review_dict[score_key] = {
                            "review": review,
                            "score": score,
                        }
                        break
                    except Exception as e:
                        print(f"  API error ({attempt+1}): {e}")
                        time.sleep(2 ** attempt)
                        rm_review_dict[score_key] = {"review": "", "score": {}}

            data["rm_review"] = rm_review_dict

            # Save after every sample (crash-safe)
            with open(out_path, "w") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)

        with open(out_path, "w") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(data_list)} reviewed samples to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Aggregate Results
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_results(reviews_dir, output_path=None, official_format=False):
    """Compute ratio scores per dimension, per split, and overall."""
    # Collect review JSON files (may be in subdirectories like in-test/, out-test/)
    review_files = []
    for root, _dirs, files in os.walk(reviews_dir):
        for f in files:
            if f.endswith(".json"):
                rel = os.path.relpath(os.path.join(root, f), reviews_dir)
                review_files.append(rel)
    if not review_files:
        print("No review files found!")
        return

    if official_format:
        if output_path is None:
            output_path = os.path.join(reviews_dir, "results.csv")
        os.makedirs(output_path, exist_ok=True)
        for rel_path in sorted(review_files):
            split_name = rel_path.replace(".json", "")
            path = os.path.join(reviews_dir, rel_path)
            result_file = os.path.join(output_path, rel_path.replace(".json", ".csv"))
            os.makedirs(os.path.dirname(result_file), exist_ok=True)

            with open(path) as f:
                data_list = json.load(f)

            sum_scores = {key: 0.0 for key in SCORE_KEYS}
            count = 0

            for data in data_list:
                rm_review = data.get("rm_review", {})
                if rm_review:
                    valid = True
                    for score_key in SCORE_KEYS:
                        score_pair = recover_score_pair(rm_review.get(score_key, {}))
                        if len(score_pair) != 2:
                            valid = False
                            print(
                                f"Warning: Invalid score pair for "
                                f"{data.get('id', data.get('example_id', 'unknown'))} "
                                f"/ {score_key} in {path}"
                            )
                            break
                        score = score_pair[0] / score_pair[1]
                        sum_scores[score_key] += score
                    if valid:
                        count += 1
                else:
                    print(f"Warning: No RM review for {data.get('id', data.get('example_id', 'unknown'))}")

            if count == 0:
                print(f"Warning: 0 valid reviews for {path}")
                avg_scores = {key: 0.0 for key in sum_scores}
            else:
                avg_scores = {key: sum_scores[key] / count for key in sum_scores}

            if count != 72 and count != 26:
                print(f"Warning: {count} reviews for {path}")

            with open(result_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["count"] + SCORE_KEYS)
                writer.writerow([count] + [avg_scores[key] for key in SCORE_KEYS])
            print(f"Saved official-format results to {result_file}")
        return

    # Per-split and overall
    all_ratios = {k: [] for k in SCORE_KEYS}
    split_ratios = {}

    for rel_path in sorted(review_files):
        split_name = rel_path.replace(".json", "")
        path = os.path.join(reviews_dir, rel_path)

        with open(path) as f:
            data_list = json.load(f)

        split_ratios[split_name] = {k: [] for k in SCORE_KEYS}
        n_scored = 0
        n_failed = 0

        for data in data_list:
            rm = data.get("rm_review", {})
            if not rm:
                n_failed += 1
                continue

            scored = True
            for sk in SCORE_KEYS:
                score = rm.get(sk, {}).get("score", [])
                if isinstance(score, list) and len(score) == 2 and score[1] > 0:
                    ratio = score[0] / score[1]
                    split_ratios[split_name][sk].append(ratio)
                    all_ratios[sk].append(ratio)
                else:
                    scored = False

            if scored:
                n_scored += 1
            else:
                n_failed += 1

        print(f"\n{split_name}: {n_scored} scored, {n_failed} failed/missing")

    # Print results
    print("\n" + "=" * 90)
    print("MMROLE EVALUATION RESULTS (ratio: model_score / gt_score)")
    print("=" * 90)

    # Per-split table
    for split_name, ratios in sorted(split_ratios.items()):
        n = min(len(v) for v in ratios.values()) if ratios else 0
        if n == 0:
            continue
        print(f"\n--- {split_name} (n={n}) ---")
        header = "  " + "  ".join(f"{SCORE_KEY_SHORT[k]:>6s}" for k in SCORE_KEYS) + "   Overall"
        print(header)
        vals = []
        row = "  "
        for sk in SCORE_KEYS:
            avg = sum(ratios[sk]) / len(ratios[sk]) if ratios[sk] else 0
            vals.append(avg)
            row += f"{avg:6.3f}  "
        overall = sum(vals) / len(vals) if vals else 0
        row += f"  {overall:.3f}"
        print(row)

    # Overall table
    print(f"\n{'=' * 90}")
    print(f"OVERALL (n={min(len(v) for v in all_ratios.values())})")
    print("=" * 90)
    header = "  " + "  ".join(f"{SCORE_KEY_SHORT[k]:>6s}" for k in SCORE_KEYS) + "   Overall"
    print(header)
    vals = []
    row = "  "
    for sk in SCORE_KEYS:
        avg = sum(all_ratios[sk]) / len(all_ratios[sk]) if all_ratios[sk] else 0
        vals.append(avg)
        row += f"{avg:6.3f}  "
    overall = sum(vals) / len(vals) if vals else 0
    row += f"  {overall:.3f}"
    print(row)

    # By category
    print(f"\nBy Category:")
    for cat_name, cat_dims in DIMENSION_CATEGORIES.items():
        cat_vals = [sum(all_ratios[d]) / len(all_ratios[d]) for d in cat_dims if all_ratios[d]]
        cat_avg = sum(cat_vals) / len(cat_vals) if cat_vals else 0
        print(f"  {cat_name}: {cat_avg:.3f}")

    # In-dist vs out-dist
    for dist_type, dist_label in [("in-test", "in-distribution"), ("out-test", "out-of-distribution")]:
        dist_ratios = {k: [] for k in SCORE_KEYS}
        for split_name, ratios in split_ratios.items():
            if split_name.startswith(dist_type):
                for sk in SCORE_KEYS:
                    dist_ratios[sk].extend(ratios[sk])
        n = min(len(v) for v in dist_ratios.values()) if dist_ratios else 0
        if n > 0:
            vals = [sum(dist_ratios[sk]) / len(dist_ratios[sk]) for sk in SCORE_KEYS]
            avg = sum(vals) / len(vals)
            print(f"\n  {dist_label} (n={n}): Overall = {avg:.3f}")

    # Save CSV
    if output_path is None:
        output_path = os.path.join(reviews_dir, "results.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n"] + [SCORE_KEY_SHORT[k] for k in SCORE_KEYS] + ["Overall"])
        for split_name, ratios in sorted(split_ratios.items()):
            n = min(len(v) for v in ratios.values()) if ratios else 0
            vals = [sum(ratios[sk]) / len(ratios[sk]) if ratios[sk] else 0 for sk in SCORE_KEYS]
            overall = sum(vals) / len(vals) if vals else 0
            writer.writerow([split_name, n] + [f"{v:.4f}" for v in vals] + [f"{overall:.4f}"])
        # Overall row
        n = min(len(v) for v in all_ratios.values())
        vals = [sum(all_ratios[sk]) / len(all_ratios[sk]) if all_ratios[sk] else 0 for sk in SCORE_KEYS]
        overall = sum(vals) / len(vals) if vals else 0
        writer.writerow(["OVERALL", n] + [f"{v:.4f}" for v in vals] + [f"{overall:.4f}"])
    print(f"\nResults saved to {output_path}")


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
    return model, (tokenizer,image_processor)

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
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Official MMRole Evaluation Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Stage to run")

    # ── infer ──
    p_infer = subparsers.add_parser("infer", help="Stage 1: Generate model answers")
    p_infer.add_argument("--model_name", type=str, required=True,
                         help="HF model name or API model name (e.g. gpt-4o)")
    p_infer.add_argument("--adapter_path", type=str, default=None,
                         help="LoRA adapter path (optional)")
    p_infer.add_argument("--test_dir", type=str,
                         default="projects/mmrole/raw_data")
    p_infer.add_argument("--image_dir", type=str,
                         default="projects/mmrole/images")
    p_infer.add_argument("--output_dir", type=str, required=True)
    p_infer.add_argument("--api", action="store_true", default=False,
                         help="Use API model instead of local HF model")
    p_infer.add_argument("--vlm", type=str, default=None,
                         help="VLM type: qwen-vl, llava-next, yi-vl (auto-detected if model_name is in MODEL_REGISTRY)")
    p_infer.add_argument("--max_new_tokens", type=int, default=512)
    p_infer.add_argument("--temperature", type=float, default=0.7)
    p_infer.add_argument("--gpu", type=str, default="0")

    # ── score ──
    p_score = subparsers.add_parser("score", help="Stage 2: Score answers with RM or GPT-4o")
    p_score.add_argument("--rm_path", type=str, default="YanqiDai/MMRole-Eval_RM",
                         help="Path to MMRole reward model")
    p_score.add_argument("--answers_dir", type=str, required=True)
    p_score.add_argument("--image_dir", type=str,
                         default="projects/mmrole/images")
    p_score.add_argument("--output_dir", type=str, required=True)
    p_score.add_argument("--use_gpt4_judge", action="store_true", default=False,
                         help="Use GPT-4o as judge instead of RM")
    p_score.add_argument("--judge_model", type=str, default="gpt-4o")
    p_score.add_argument("--use_lora", action="store_true", default=False,
                         help="RM weights are LoRA adapters (use AutoPeftModelForCausalLM)")
    p_score.add_argument("--no_random", action="store_true", default=False,
                         help="Disable sampling for deterministic RM output")
    p_score.add_argument("--gpu", type=str, default="0")

    # ── results ──
    p_results = subparsers.add_parser("results", help="Stage 3: Aggregate and display results")
    p_results.add_argument("--reviews_dir", type=str, required=True)
    p_results.add_argument("--output_csv", type=str, default=None)
    p_results.add_argument("--official_format", action="store_true", default=False,
                           help="Match released MMRole RM_result.py output layout "
                                "(per-review-file CSVs under output_csv/results_dir)")

    args = parser.parse_args()

    if args.command == "infer":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"=== Stage 1: Inference ({args.model_name}) ===")

        data_by_split = load_test_data(args.test_dir)

        if args.api:
            run_inference_api(
                args.model_name, data_by_split, args.image_dir, args.output_dir,
                max_tokens=args.max_new_tokens, temperature=args.temperature,
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Resolve model name via registry (allows short names like "qwen-vl-chat")
            reg_entry = MODEL_REGISTRY.get(args.model_name)
            hf_id = reg_entry["hf_id"] if reg_entry else args.model_name
            vlm_type = reg_entry["type"] if reg_entry else (args.vlm if args.vlm else None)
            print(f"  Loading model: {hf_id}  (type={vlm_type})")

            processor = None  # Only used by llava-next

            if vlm_type == "qwen2.5-vl":
                from transformers import AutoProcessor
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as _VLCls
                except ImportError:
                    from transformers import Qwen2VLForConditionalGeneration as _VLCls
                processor = AutoProcessor.from_pretrained(
                    hf_id, trust_remote_code=True,
                    min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
                )
                if processor.tokenizer.pad_token is None:
                    processor.tokenizer.pad_token = processor.tokenizer.eos_token
                model = _VLCls.from_pretrained(
                    hf_id, torch_dtype=torch.bfloat16, device_map="auto",
                ).eval()
                tokenizer = processor.tokenizer

            elif vlm_type == "qwen-vl":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, device_map="cuda:0", trust_remote_code=True,
                ).eval()

            elif vlm_type == "llava-next":
                from transformers import LlavaNextForConditionalGeneration, AutoProcessor
                processor = AutoProcessor.from_pretrained(hf_id)
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    hf_id, torch_dtype=torch.float16, device_map="cuda:0",
                ).eval()
                tokenizer = processor.tokenizer

            elif vlm_type == "yi-vl":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model, tokenizer = load_yi_vl(hf_id, device) #+ ("yi-vl",)
                model.eval()
                # tokenizer = AutoTokenizer.from_pretrained(
                #     hf_id, trust_remote_code=True, use_fast=False,
                # )
                # model = AutoModelForCausalLM.from_pretrained(
                #     hf_id, device_map="cuda:0", trust_remote_code=True,
                #     torch_dtype=torch.float16,
                # ).eval()

            else:
                # Text-only model
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id, torch_dtype=torch.bfloat16,
                ).to(device)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

            # Optional LoRA adapter (works for any model type)
            if args.adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model, args.adapter_path, torch_dtype=torch.bfloat16,
                )
                # Keep the adapter unmerged during inference. For Qwen2.5-VL we
                # observed measurable drift after merge_and_unload(), while the
                # live PEFT path matched training-time behavior more closely.
                print(f"  LoRA adapter loaded (unmerged) from {args.adapter_path}")
                model.eval()

            run_inference_local(
                model, tokenizer, data_by_split, args.image_dir, args.output_dir,
                device=str(device), max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, vlm_type=vlm_type, processor=processor,
            )

    elif args.command == "score":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if args.use_gpt4_judge:
            print(f"=== Stage 2: Scoring with {args.judge_model} ===")
            score_with_gpt4(
                args.answers_dir, args.image_dir, args.output_dir,
                judge_model=args.judge_model,
            )
        else:
            print(f"=== Stage 2: Scoring with RM ({args.rm_path}) ===")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if args.no_random:
                torch.manual_seed(1234)
                torch.cuda.manual_seed(1234)
                torch.cuda.manual_seed_all(1234)
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            print(f"  Loading RM: {args.rm_path}")
            rm_tokenizer = AutoTokenizer.from_pretrained(
                args.rm_path, trust_remote_code=True,
            )
            if args.use_lora:
                from peft import AutoPeftModelForCausalLM
                rm_model = AutoPeftModelForCausalLM.from_pretrained(
                    args.rm_path, device_map="cuda:0", trust_remote_code=True,
                ).eval()
            else:
                rm_model = AutoModelForCausalLM.from_pretrained(
                    args.rm_path, device_map="cuda:0", trust_remote_code=True,
                ).eval()
            if args.no_random:
                rm_model.generation_config.do_sample = False

            score_with_rm(
                rm_model, rm_tokenizer,
                args.answers_dir, args.image_dir, args.output_dir,
            )

    elif args.command == "results":
        print(f"=== Stage 3: Aggregate Results ===")
        aggregate_results(
            args.reviews_dir,
            output_path=args.output_csv,
            official_format=args.official_format,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
