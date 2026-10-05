#!/usr/bin/env python3
"""Generate SOTOPIA-balanced counterfactual candidates with an OpenAI model.

The script samples real SOTOPIA episode-heldout turns and asks an LLM to create
matched hidden-state A/B branches plus response candidates. It does not require
the openai Python package; it uses the HTTPS API directly.

Output schema matches run_counterfactual_flip_experiment.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You create rigorous SOTOPIA counterfactual evaluation items.

Each item must keep the observable dialogue context fixed while changing only a
hidden partner mental state. The two candidate replies must be similar in
length, politeness, fluency, and general social quality. Without the hidden
state, a reward model should be close to indifferent. With hidden state A,
candidate A should be better. With hidden state B, candidate B should be better.

Do not write candidates that reveal the hidden-state label. Do not make one
candidate obviously rude, unsafe, incoherent, or much longer than the other.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def slug(text: str, max_len: int = 96) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not clean:
        clean = hashlib.sha1(text.encode()).hexdigest()[:12]
    return clean[:max_len].strip("_")


def key_for_record(rec: dict[str, Any]) -> str:
    return f"{rec.get('episode_id')}::t{rec.get('turn_num')}::{rec.get('speaker')}"


def insert_state_context(context: str, mental_state: str) -> str:
    lines = context.rstrip().splitlines()
    if lines and re.match(r"^Turn\s+\d+\s+\|\s+.*:\s*$", lines[-1]):
        return "\n".join(lines[:-1] + [f"Counterfactual partner mental state: {mental_state}", lines[-1]])
    return f"{context.rstrip()}\nCounterfactual partner mental state: {mental_state}"


def clean_context(context: str, max_chars: int) -> str:
    context = context.strip()
    if len(context) <= max_chars:
        return context
    # Preserve the current turn and as much recent dialogue as possible.
    return context[-max_chars:]


def eligible_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        context = rec.get("context_text", "")
        if not context or not rec.get("response_text"):
            continue
        if int(rec.get("turn_num", 0)) < 1:
            continue
        if len(context) < 350:
            continue
        out.append(rec)
    return out


def sample_records(records: list[dict[str, Any]], max_records: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_scenario.setdefault(str(rec.get("scenario", "")), []).append(rec)
    for group in by_scenario.values():
        rng.shuffle(group)
    scenarios = list(by_scenario)
    rng.shuffle(scenarios)

    selected = []
    while scenarios and len(selected) < max_records:
        next_scenarios = []
        for scenario in scenarios:
            group = by_scenario[scenario]
            if group and len(selected) < max_records:
                selected.append(group.pop())
            if group:
                next_scenarios.append(scenario)
        scenarios = next_scenarios
    return selected


def make_user_prompt(rec: dict[str, Any], pairs_per_record: int, max_context_chars: int) -> str:
    payload = {
        "episode_id": rec.get("episode_id"),
        "turn_num": rec.get("turn_num"),
        "speaker": rec.get("speaker"),
        "scenario": rec.get("scenario"),
        "observable_context": clean_context(rec.get("context_text", ""), max_context_chars),
        "original_response": rec.get("response_text"),
        "original_first_order_mental_model": rec.get("mental1_text"),
        "original_second_order_mental_model": rec.get("mental2_text"),
    }
    return f"""Create {pairs_per_record} counterfactual pair(s) for this SOTOPIA turn.

Return only valid JSON with this exact top-level shape:
{{
  "pairs": [
    {{
      "axis": "short_name_for_the_hidden_mental_state_axis",
      "state_a": {{
        "name": "short_state_a_name",
        "mental_state": "Partner Belief: ... Strategic Intent: ... Thought Process: ..."
      }},
      "state_b": {{
        "name": "short_state_b_name",
        "mental_state": "Partner Belief: ... Strategic Intent: ... Thought Process: ..."
      }},
      "candidate_a": "speaker reply that is best under state_a",
      "candidate_b": "speaker reply that is best under state_b",
      "why_balanced": "one sentence explaining why observable-only scoring should be near indifferent"
    }}
  ]
}}

Hard constraints:
- Use the same speaker and situation as the observable context.
- Candidate A and B must both be socially plausible and similar in length.
- Candidate A should be correct only because state_a is true.
- Candidate B should be correct only because state_b is true.
- Do not make either candidate mention hidden states, labels, or the words candidate/state.
- Avoid obvious good/bad pairs; both replies should look reasonable without hidden mental-state information.

SOTOPIA turn:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def read_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        return Path(args.api_key_file).read_text().strip()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY or pass --api-key-file.")


def call_openai(args: argparse.Namespace, api_key: str, rec: dict[str, Any]) -> dict[str, Any]:
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(rec, args.pairs_per_record, args.max_context_chars)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
    last_error = None
    for attempt in range(args.max_retries):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"]
            return {
                "record_key": key_for_record(rec),
                "source": {
                    "episode_id": rec.get("episode_id"),
                    "turn_num": rec.get("turn_num"),
                    "speaker": rec.get("speaker"),
                    "scenario": rec.get("scenario"),
                    "response_text": rec.get("response_text"),
                    "mental1_text": rec.get("mental1_text"),
                    "mental2_text": rec.get("mental2_text"),
                    "context_text": rec.get("context_text"),
                },
                "model": args.model,
                "response": json.loads(content),
                "usage": parsed.get("usage", {}),
            }
        except urllib.error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {err_text[:500]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(min(30, (2 ** attempt) + random.random()))
    raise RuntimeError(f"OpenAI call failed for {key_for_record(rec)}: {last_error}")


def valid_pair(pair: dict[str, Any]) -> bool:
    required = ["axis", "state_a", "state_b", "candidate_a", "candidate_b"]
    if any(key not in pair for key in required):
        return False
    if not isinstance(pair.get("state_a"), dict) or not isinstance(pair.get("state_b"), dict):
        return False
    cand_a = str(pair.get("candidate_a", "")).strip()
    cand_b = str(pair.get("candidate_b", "")).strip()
    if len(cand_a) < 35 or len(cand_b) < 35:
        return False
    if cand_a == cand_b:
        return False
    if max(len(cand_a), len(cand_b)) / max(1, min(len(cand_a), len(cand_b))) > 1.8:
        return False
    for state_key in ["state_a", "state_b"]:
        mental = str(pair[state_key].get("mental_state", "")).strip()
        if "Partner Belief:" not in mental or "Strategic Intent:" not in mental:
            return False
    return True


def materialize_records(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for row in cache_rows:
        source = row["source"]
        context = source["context_text"].rstrip()
        response = row["response"]
        for idx, pair in enumerate(response.get("pairs", [])):
            if not valid_pair(pair):
                continue
            pair_key = f"{row['record_key']}::{pair.get('axis')}::{idx}"
            pair_id = f"sotopia_llm_cf_{slug(pair_key)}"
            if pair_id in seen:
                continue
            seen.add(pair_id)
            state_a_mental = pair["state_a"]["mental_state"].strip()
            state_b_mental = pair["state_b"]["mental_state"].strip()
            out.append({
                "pair_id": pair_id,
                "observable_context": context,
                "candidate_a": " ".join(str(pair["candidate_a"]).split()),
                "candidate_b": " ".join(str(pair["candidate_b"]).split()),
                "state_a": {
                    "name": str(pair["state_a"].get("name", "state_a")),
                    "z_context": insert_state_context(context, state_a_mental),
                    "correct": "a",
                },
                "state_b": {
                    "name": str(pair["state_b"].get("name", "state_b")),
                    "z_context": insert_state_context(context, state_b_mental),
                    "correct": "b",
                },
                "source": {
                    "dataset": "sotopia_episode_heldout_val",
                    "generation_model": row["model"],
                    "episode_id": source.get("episode_id"),
                    "turn_num": source.get("turn_num"),
                    "speaker": source.get("speaker"),
                    "scenario": source.get("scenario"),
                    "original_response_text": source.get("response_text"),
                    "original_mental1_text": source.get("mental1_text"),
                    "original_mental2_text": source.get("mental2_text"),
                    "why_balanced": pair.get("why_balanced"),
                },
                "axis": str(pair.get("axis", "")),
                "notes": "LLM-generated SOTOPIA-heldout counterfactual candidate; filter and review before final use.",
            })
    return out


def load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def main() -> None:
    args = parse_args()
    api_key = read_key(args)
    source_records = sample_records(
        eligible_records(read_jsonl(Path(args.source_jsonl))),
        max_records=args.max_records,
        seed=args.seed,
    )

    cache_path = Path(args.cache_jsonl)
    cached = load_cache(cache_path)
    done = {row["record_key"] for row in cached}
    todo = [rec for rec in source_records if key_for_record(rec) not in done]
    print(json.dumps({
        "source_records": len(source_records),
        "cached_records": len(cached),
        "todo_records": len(todo),
        "model": args.model,
        "cache_jsonl": args.cache_jsonl,
    }, indent=2), flush=True)

    if todo and not args.materialize_only:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_rec = {executor.submit(call_openai, args, api_key, rec): rec for rec in todo}
            for future in futures.as_completed(future_to_rec):
                rec = future_to_rec[future]
                try:
                    row = future.result()
                    append_jsonl(cache_path, row)
                    print(f"[ok] {row['record_key']}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] {key_for_record(rec)}: {exc}", flush=True)
                    if args.fail_fast:
                        raise

    cache_rows = load_cache(cache_path)
    records = materialize_records(cache_rows)
    write_jsonl(Path(args.output_jsonl), records)
    summary = {
        "source_jsonl": args.source_jsonl,
        "cache_jsonl": args.cache_jsonl,
        "output_jsonl": args.output_jsonl,
        "model": args.model,
        "seed": args.seed,
        "cache_rows": len(cache_rows),
        "materialized_pairs": len(records),
        "pairs_per_record": args.pairs_per_record,
        "curation_status": "llm_generated_sotopia_heldout_candidates_needs_filtering_and_review",
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--cache-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=80)
    parser.add_argument("--pairs-per-record", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
