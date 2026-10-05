#!/usr/bin/env python3
"""
Step 5c: Annotate MMRole Training Data with Official 8-Dim Reward Scores
=========================================================================
Uses the OpenAI API to score MMRole training examples with the official
8 response-level MMRole dimensions.

This script is designed for current MMRole training data and supports:

1. belief_prediction.jsonl
   - scores `current_utterance`
   - writes `mmrole_reward_scores`

2. preference_pairs.jsonl
   - scores `preferred_response` and `rejected_response`
   - writes `preferred_mmrole_reward_scores`
   - writes `rejected_mmrole_reward_scores`

Outputs are written beside the inputs by default:
  belief_prediction_mmrole_reward_openai.jsonl
  preference_pairs_mmrole_reward_openai.jsonl

The script is resumable: if an output record already contains all required
score fields for an example_id, that example is skipped automatically.

Example:
    export OPENAI_API_KEY='...'

    # Annotate belief examples only
    python step5c_annotate_mmrole_reward_dims_openai.py \
        --jobs belief \
        --model gpt-4o-mini

    # Annotate both belief + preference data
    python step5c_annotate_mmrole_reward_dims_openai.py \
        --jobs belief,preference \
        --model gpt-4o-mini

    # Use GPT-5.4-mini instead
    python step5c_annotate_mmrole_reward_dims_openai.py \
        --jobs belief,preference \
        --model gpt-5.4-mini
"""

import os
import json
import time
import base64
import argparse
import threading
import hashlib
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


MMROLE_DIMENSIONS = {
    "instruction_adherence": (
        "Instruction Adherence: Does the response accurately adhere to the task "
        "instruction, directly role-playing as {speaker_name} and only including "
        "words that {speaker_name} should say, without any additional explanatory "
        "prefixes or suffixes?"
    ),
    "fluency": "Fluency: Is the response grammatically correct and smoothly articulated?",
    "coherency": (
        "Coherency: Does the response maintain a coherent thread of dialogue "
        "without contradicting earlier parts of the conversation or previously "
        "established facts?"
    ),
    "image_text_relevance": (
        "Image-Text Relevance: Is the response closely related to the visual "
        "content of the image?"
    ),
    "response_accuracy": (
        "Response Accuracy: Does the response accurately answer the partner's "
        "words or appropriately initiate a conversation based on the image?"
    ),
    "personality_consistency": (
        "Personality Consistency: Does the response accurately and sufficiently "
        "reflect the personality of {speaker_name}?"
    ),
    "knowledge_consistency": (
        "Knowledge Consistency: Is the response consistent with the factual "
        "knowledge that {speaker_name} should possess, including experiences, "
        "abilities, and relationships?"
    ),
    "tone_consistency": (
        "Tone Consistency: Does the response maintain a consistent tone that "
        "aligns with {speaker_name}'s typical manner of speaking and catchphrases, "
        "rather than resembling the style of AI assistants?"
    ),
}


JOB_SPECS = {
    "belief": {
        "input_name": "belief_prediction.jsonl",
        "output_name": "belief_prediction_mmrole_reward_openai.jsonl",
        "response_specs": [
            {
                "response_field": "current_utterance",
                "score_key": "mmrole_reward_scores",
                "raw_key": "mmrole_reward_judge_raw",
            },
        ],
    },
    "preference": {
        "input_name": "preference_pairs.jsonl",
        "output_name": "preference_pairs_mmrole_reward_openai.jsonl",
        "response_specs": [
            {
                "response_field": "preferred_response",
                "score_key": "preferred_mmrole_reward_scores",
                "raw_key": "preferred_mmrole_reward_judge_raw",
            },
            {
                "response_field": "rejected_response",
                "score_key": "rejected_mmrole_reward_scores",
                "raw_key": "rejected_mmrole_reward_judge_raw",
            },
        ],
    },
}


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SPLITS = ("train", "val", "test_in")

SYSTEM_PROMPT = (
    "You are an objective and precise evaluator specializing in MMRole-style "
    "multimodal role-playing assessment. Score only the provided response. "
    "Be strict, calibrated, and consistent across examples."
)


def parse_dimensions(dimensions_arg: str) -> List[str]:
    if not dimensions_arg or dimensions_arg.strip().lower() == "all":
        return list(MMROLE_DIMENSIONS.keys())

    dims = [d.strip() for d in dimensions_arg.split(",") if d.strip()]
    invalid = [d for d in dims if d not in MMROLE_DIMENSIONS]
    if invalid:
        raise ValueError(
            f"Invalid dimensions {invalid}. Valid choices: {list(MMROLE_DIMENSIONS.keys())}"
        )
    return dims


def parse_jobs(jobs_arg: str) -> List[str]:
    jobs = [job.strip() for job in (jobs_arg or "").split(",") if job.strip()]
    if not jobs:
        raise ValueError("No jobs specified.")
    invalid = [job for job in jobs if job not in JOB_SPECS]
    if invalid:
        raise ValueError(f"Invalid jobs {invalid}. Valid choices: {list(JOB_SPECS.keys())}")
    return jobs


def encode_image_b64(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image(example: Dict[str, Any], image_dir: str) -> Optional[str]:
    for key in ("image_local", "image"):
        image = example.get(key, "")
        if not image:
            continue

        full = os.path.join(image_dir, image)
        if os.path.exists(full):
            return full

        fname = os.path.basename(image)
        for subdir in ("coco", "character", ""):
            candidate = os.path.join(image_dir, subdir, fname)
            if os.path.exists(candidate):
                return candidate
    return None


def format_history(dialogue_history: List[Dict[str, Any]], max_turns: int = 8) -> str:
    turns = dialogue_history[-max_turns:]
    if not turns:
        return "(start of conversation)"
    return "\n".join(
        f"[{turn.get('speaker', '?')}]: {turn.get('utterance', '')}"
        for turn in turns
    )


def build_prompt(example: Dict[str, Any], dimensions: List[str], response_text: str) -> str:
    speaker_name = example.get("speaker_name", "Speaker")
    partner_name = example.get("partner_name", "Partner")
    speaker_profile = str(example.get("speaker_profile", ""))[:1800]
    partner_profile = str(example.get("partner_profile", ""))[:900]
    history = format_history(example.get("dialogue_history", []))
    response = str(response_text).strip()

    dimension_lines = []
    for dim in dimensions:
        criterion = MMROLE_DIMENSIONS[dim].format(speaker_name=speaker_name)
        dimension_lines.append(f'- "{dim}": {criterion}')
    dimension_text = "\n".join(dimension_lines)

    return f"""Evaluate one role-play response using the official MMRole reward dimensions.

## Context
- Character being role-played: {speaker_name}
- Character profile: {speaker_profile}
- Talking with: {partner_name}
- Partner profile: {partner_profile}
- Image: (attached if available)
- Dialogue history:
{history}

## Response to evaluate
{response}

## Dimensions to score
{dimension_text}

## Instructions
- Score each dimension from 1 to 10.
- Use the image when available.
- Judge only the provided response, not an ideal answer you wish had been given.
- Be especially careful with image_text_relevance, response_accuracy, knowledge_consistency, personality_consistency, and tone_consistency.
- Return ONLY valid JSON.
- Use exactly these keys and numeric values only.

Expected JSON format:
{json.dumps({dim: 0 for dim in dimensions}, indent=2)}
"""


def extract_json_object(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {candidate[:200]}")
    return candidate[start:end + 1]


def parse_scores(text: str, dimensions: List[str]) -> Dict[str, float]:
    obj_text = extract_json_object(text)
    data = json.loads(obj_text)
    scores = {}

    for dim in dimensions:
        if dim not in data:
            raise ValueError(f"Missing dimension '{dim}' in response: {obj_text[:200]}")
        value = data[dim]
        if isinstance(value, dict):
            value = value.get("score")
        score = float(value)
        # GPT-4o-mini occasionally emits 0.0 for a dimension even when the
        # rubric asks for 1..10. Treat that as the minimum valid score instead
        # of failing the whole record. Still reject clearly broken outputs.
        if 0.0 <= score <= 10.0:
            score = max(1.0, score)
        else:
            raise ValueError(f"Out-of-range score for {dim}: {score}")
        scores[dim] = score

    return scores


def write_jsonl_line(path: str, record: Dict[str, Any], lock: threading.Lock):
    line = json.dumps(record) + "\n"
    with lock:
        with open(path, "a") as f:
            f.write(line)


def record_has_all_scores(record: Dict[str, Any], response_specs: List[Dict[str, str]]) -> bool:
    return all(spec["score_key"] in record for spec in response_specs)


def build_resume_key(record: Dict[str, Any], response_specs: List[Dict[str, str]]) -> str:
    """Build a row-level identity key that survives annotation output.

    We cannot key resumability only on example_id because preference_pairs.jsonl
    contains many rows for the same example_id. Instead, hash the original
    record content after dropping generated annotation fields.
    """
    drop_keys = set()
    for spec in response_specs:
        drop_keys.add(spec["score_key"])
        drop_keys.add(spec["raw_key"])

    payload = {
        key: value
        for key, value in record.items()
        if key not in drop_keys
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def load_completed_ids(output_path: str, response_specs: List[Dict[str, str]]) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record_has_all_scores(record, response_specs):
                done.add(build_resume_key(record, response_specs))
    return done


def save_metadata(meta_path: str, payload: Dict[str, Any]):
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)


class MMRoleRewardAnnotator:
    def __init__(self, api_key: str, model: str,
                 image_dir: str, base_url: str = "",
                 rate_limit_delay: float = 0.15,
                 max_retries: int = 4,
                 max_completion_tokens: int = 256):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.image_dir = image_dir
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.stats = {"calls": 0, "errors": 0, "parse_errors": 0}
        self._lock = threading.Lock()

    def _call_model(self, prompt: str, image_path: Optional[str]) -> str:
        content = []
        if image_path:
            b64 = encode_image_b64(image_path)
            if b64:
                ext = os.path.splitext(image_path)[1].lower()
                mime = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(ext, "image/jpeg")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "low",
                    },
                })
        content.append({"type": "text", "text": prompt})

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.1,
                )
                with self._lock:
                    self.stats["calls"] += 1
                return resp.choices[0].message.content or ""
            except Exception as e:
                error_text = str(e)
                if "rate limit" in error_text.lower():
                    time.sleep(min(60, 2 ** (attempt + 2)))
                else:
                    with self._lock:
                        self.stats["errors"] += 1
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
        raise RuntimeError("Exhausted retries without a response.")

    def annotate_record(self, record: Dict[str, Any], dimensions: List[str],
                        response_specs: List[Dict[str, str]],
                        store_raw_response: bool) -> Dict[str, Any]:
        out = dict(record)
        image_path = resolve_image(record, self.image_dir)

        for spec in response_specs:
            response_field = spec["response_field"]
            response_text = str(record.get(response_field, "")).strip()
            if not response_text:
                raise ValueError(f"Empty response field '{response_field}'")

            prompt = build_prompt(record, dimensions, response_text)
            raw_response = self._call_model(prompt, image_path)
            try:
                scores = parse_scores(raw_response, dimensions)
            except Exception:
                with self._lock:
                    self.stats["parse_errors"] += 1
                raise

            out[spec["score_key"]] = scores
            if store_raw_response:
                out[spec["raw_key"]] = raw_response
        return out


def process_split(input_path: str, output_path: str, annotator: MMRoleRewardAnnotator,
                  dimensions: List[str], response_specs: List[Dict[str, str]],
                  max_workers: int, max_examples: int, store_raw_response: bool):
    print(f"\nReading: {input_path}", flush=True)
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_examples > 0 and len(records) >= max_examples:
                break

    done_ids = load_completed_ids(output_path, response_specs)
    pending = [
        record for record in records
        if build_resume_key(record, response_specs) not in done_ids
    ]
    failed_path = output_path + ".failed"

    print(
        f"  Loaded={len(records)} done={len(done_ids)} pending={len(pending)} "
        f"output={output_path}",
        flush=True,
    )

    if not pending:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                annotator.annotate_record,
                record,
                dimensions,
                response_specs,
                store_raw_response,
            ): record
            for record in pending
        }

        for future in as_completed(futures):
            record = futures[future]
            example_id = record.get("example_id", "?")
            try:
                annotated = future.result()
                write_jsonl_line(output_path, annotated, write_lock)
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"  Progress: {completed}/{len(pending)} "
                        f"(last example_id={example_id})",
                        flush=True,
                    )
            except Exception as e:
                failure = {
                    "example_id": example_id,
                    "resume_key": build_resume_key(record, response_specs),
                    "error": str(e),
                }
                write_jsonl_line(failed_path, failure, write_lock)
                print(f"  Failed: {example_id} -> {str(e)[:200]}", flush=True)


def build_job_paths(training_root: str, splits: List[str], job_name: str) -> List[tuple]:
    spec = JOB_SPECS[job_name]
    pairs = []
    for split in splits:
        split_dir = os.path.join(training_root, split)
        pairs.append((
            os.path.join(split_dir, spec["input_name"]),
            os.path.join(split_dir, spec["output_name"]),
        ))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Annotate MMRole training data with official 8-dim reward scores using OpenAI models."
    )
    parser.add_argument("--training_root", type=str,
                        default="projects/mmrole/training_data")
    parser.add_argument("--splits", type=str,
                        default=",".join(DEFAULT_SPLITS),
                        help="Comma-separated splits under --training_root.")
    parser.add_argument("--jobs", type=str, default="belief,preference",
                        help="Comma-separated subset of {belief,preference}.")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--dimensions", type=str, default="all",
                        help="Comma-separated subset of official MMRole dimensions, or 'all'.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base_url", type=str, default="",
                        help="Optional OpenAI-compatible base URL. Leave empty for OpenAI.")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--rate_limit_delay", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--max_completion_tokens", type=int, default=256)
    parser.add_argument("--store_raw_response", action="store_true",
                        help="Keep the raw judge output in each annotated record.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Set {args.api_key_env} in the environment before running."
        )

    dimensions = parse_dimensions(args.dimensions)
    jobs = parse_jobs(args.jobs)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    annotator = MMRoleRewardAnnotator(
        api_key=api_key,
        model=args.model,
        image_dir=args.image_dir,
        base_url=args.base_url,
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
    )

    print("MMRole OpenAI reward annotation", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Base URL: {args.base_url or '(OpenAI default)'}", flush=True)
    print(f"  Dimensions: {dimensions}", flush=True)
    print(f"  Jobs: {jobs}", flush=True)
    print(f"  Splits: {splits}", flush=True)
    print(f"  Max workers: {args.max_workers}", flush=True)

    for job_name in jobs:
        spec = JOB_SPECS[job_name]
        print(f"\n=== Job: {job_name} ===", flush=True)
        for input_path, output_path in build_job_paths(args.training_root, splits, job_name):
            if not os.path.exists(input_path):
                print(f"Skipping missing input: {input_path}", flush=True)
                continue

            meta_path = output_path + ".meta.json"
            save_metadata(meta_path, {
                "job": job_name,
                "input_path": input_path,
                "output_path": output_path,
                "image_dir": args.image_dir,
                "dimensions": dimensions,
                "response_specs": spec["response_specs"],
                "model": args.model,
                "base_url": args.base_url,
                "api_key_env": args.api_key_env,
                "max_workers": args.max_workers,
                "rate_limit_delay": args.rate_limit_delay,
                "max_retries": args.max_retries,
                "max_completion_tokens": args.max_completion_tokens,
                "store_raw_response": args.store_raw_response,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            process_split(
                input_path=input_path,
                output_path=output_path,
                annotator=annotator,
                dimensions=dimensions,
                response_specs=spec["response_specs"],
                max_workers=args.max_workers,
                max_examples=args.max_examples,
                store_raw_response=args.store_raw_response,
            )

    print("\nDone.", flush=True)
    print(f"  Calls: {annotator.stats['calls']}", flush=True)
    print(f"  Errors: {annotator.stats['errors']}", flush=True)
    print(f"  Parse errors: {annotator.stats['parse_errors']}", flush=True)


if __name__ == "__main__":
    main()
