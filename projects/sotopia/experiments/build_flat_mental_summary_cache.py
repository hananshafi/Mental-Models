#!/usr/bin/env python3
"""Build LLM-fused flat mental summaries for the BIT ablation.

Output JSONL schema consumed by stage1_train_mental_supervision_variants.py:
  {
    "key": sha1([source_mental1_text, source_mental2_text]),
    "source_mental1_text": "...",
    "source_mental2_text": "...",
    "flat_mental1_text": "one fused narrative sentence",
    "flat_mental2_text": "one fused narrative sentence"
  }

Use --provider openai for parallel standard OpenAI API generation, --provider hf
for a local Hugging Face causal LM, or --write_prompts_only to dump prompts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))

from stage1_train_coupled_mental_reward_v3 import RecursiveToMDataset  # noqa: E402


def mental_pair_key(mental1: str, mental2: str) -> str:
    payload = json.dumps([mental1 or "", mental2 or ""], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_unique_pairs(data_path: str, limit: int | None = None) -> OrderedDict[str, tuple[str, str]]:
    dataset = RecursiveToMDataset(data_path, tokenizer=None)
    pairs: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for sample in dataset.samples:
        mental1 = sample.get("mental1_text", "")
        mental2 = sample.get("mental2_text", "")
        if not mental1.strip() and not mental2.strip():
            continue
        key = mental_pair_key(mental1, mental2)
        if key not in pairs:
            pairs[key] = (mental1, mental2)
        if limit and len(pairs) >= limit:
            break
    return pairs


def build_prompt(mental1: str, mental2: str) -> str:
    return f"""Rewrite these structured mental-state annotations into flat narrative supervision.

Rules:
- Preserve the meaning and all concrete content.
- Do not use headings, bullet points, or explicit segment labels.
- Do not copy the labels Partner Belief, Strategic Intent, Thought Process, Second-Order Belief, Second-Order Intent, or Second-Order Thought.
- Each output should be one fluent sentence.
- Return only valid JSON with keys flat_mental1_text and flat_mental2_text.

First annotation:
{mental1 if mental1.strip() else "N/A"}

Second annotation:
{mental2 if mental2.strip() else "N/A"}
"""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")
    return json.loads(match.group(0))


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["key"])
    return done


def render_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a careful data-rewriting assistant."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"


def generate_one(model, tokenizer, prompt: str, device: str, max_new_tokens: int,
                 temperature: float, top_p: float) -> dict[str, str]:
    rendered = render_chat_prompt(tokenizer, prompt)
    enc = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=2048).to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    obj = extract_json(decoded)
    return {
        "flat_mental1_text": str(obj["flat_mental1_text"]).strip(),
        "flat_mental2_text": str(obj["flat_mental2_text"]).strip(),
    }


def openai_chat_completion(prompt: str, args: argparse.Namespace, api_key: str) -> dict[str, str]:
    payload = {
        "model": args.openai_model,
        "messages": [
            {"role": "system", "content": "You are a careful data-rewriting assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        args.openai_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    obj = extract_json(content)
    return {
        "flat_mental1_text": str(obj["flat_mental1_text"]).strip(),
        "flat_mental2_text": str(obj["flat_mental2_text"]).strip(),
    }


def openai_chat_completion_with_retries(prompt: str, args: argparse.Namespace, api_key: str) -> dict[str, str]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            return openai_chat_completion(prompt, args, api_key)
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt >= args.max_retries:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt >= args.max_retries:
                raise
        time.sleep(min(args.retry_max_sleep, args.retry_base_sleep * (2 ** attempt)))
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def run_openai_parallel(
    pending_items: list[tuple[int, str, str, str]],
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    api_key = os.environ.get(args.openai_api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.openai_api_key_env} is not set.")

    completed = 0
    failed = 0

    def worker(item: tuple[int, str, str, str]) -> dict[str, Any]:
        idx, key, mental1, mental2 = item
        prompt = build_prompt(mental1, mental2)
        rec: dict[str, Any] = {
            "key": key,
            "source_mental1_text": mental1,
            "source_mental2_text": mental2,
            "provider": "openai",
            "model": args.openai_model,
            "_idx": idx,
        }
        rec.update(openai_chat_completion_with_retries(prompt, args, api_key))
        return rec

    with out_path.open("a") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            future_to_item = {executor.submit(worker, item): item for item in pending_items}
            for future in concurrent.futures.as_completed(future_to_item):
                idx, key, _, _ = future_to_item[future]
                try:
                    rec = future.result()
                    rec.pop("_idx", None)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                except Exception as exc:
                    failed += 1
                    print(f"[FAILED] idx={idx} key={key}: {exc}", flush=True)
                done = completed + failed
                if done % args.progress_every == 0 or done == len(pending_items):
                    print(
                        f"Progress: {done}/{len(pending_items)} "
                        f"completed={completed} failed={failed}",
                        flush=True,
                    )
    if failed:
        raise RuntimeError(f"{failed} OpenAI requests failed. Re-run to resume missing keys.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--provider", choices=["hf", "openai"], default="hf")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--openai_model", default="gpt-4o-mini")
    parser.add_argument("--openai_base_url", default="https://api.openai.com/v1")
    parser.add_argument("--openai_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_base_sleep", type=float, default=1.0)
    parser.add_argument("--retry_max_sleep", type=float, default=30.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--write_prompts_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = load_unique_pairs(args.data_path, limit=args.limit)
    done = load_done_keys(out_path)
    print(f"Loaded {len(pairs)} unique mental pairs; {len(done)} already done.", flush=True)

    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.provider == "openai" and not args.write_prompts_only:
        pending_items = [
            (idx, key, mental1, mental2)
            for idx, (key, (mental1, mental2)) in enumerate(pairs.items(), start=1)
            if key not in done
        ]
        print(
            f"Running OpenAI standard API in parallel: model={args.openai_model} "
            f"parallelism={args.parallelism} pending={len(pending_items)}",
            flush=True,
        )
        run_openai_parallel(pending_items, out_path, args)
        print(f"Done: {out_path}", flush=True)
        return

    if not args.write_prompts_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    with out_path.open("a") as f:
        for idx, (key, (mental1, mental2)) in enumerate(pairs.items(), start=1):
            if key in done:
                continue
            prompt = build_prompt(mental1, mental2)
            rec: dict[str, Any] = {
                "key": key,
                "source_mental1_text": mental1,
                "source_mental2_text": mental2,
            }
            if args.write_prompts_only:
                rec["prompt"] = prompt
            else:
                rec["provider"] = "hf"
                rec["model"] = args.model_name
                rec.update(generate_one(
                    model, tokenizer, prompt, device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if idx % 25 == 0:
                print(f"Wrote {idx}/{len(pairs)}", flush=True)

    print(f"Done: {out_path}", flush=True)


if __name__ == "__main__":
    main()
