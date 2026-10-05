#!/usr/bin/env python3
"""Generate role-targeted SOTOPIA counterfactual ranking items.

Each generated item keeps the observable SOTOPIA context fixed while flipping
exactly one hidden mental role: belief, intent, or thought.  Candidate A should
be better under state A; candidate B should be better under state B.  The script
stores both structured BIT contexts and fused flat-summary contexts so the same
benchmark can compare BIT and flat-supervision reward models fairly.
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
ROLES = ["belief", "intent", "thought"]

SYSTEM_PROMPT = """You create rigorous role-targeted SOTOPIA counterfactual ranking items.

The item must keep the observable dialogue context fixed.  Change exactly one
hidden mental role: belief, intent, or thought.  The other two hidden roles must
remain matched and semantically constant across state A and state B.

Candidate A must be better only under state A.  Candidate B must be better only
under state B.  Both candidates must be socially plausible, similar in length,
similar in politeness, and close to indistinguishable without the hidden role.
Do not make one response obviously rude, unsafe, incoherent, much shorter, or
much longer.  Do not reveal state labels or mention hidden-state wording.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def slug(text: str, max_len: int = 96) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (clean or hashlib.sha1(text.encode()).hexdigest()[:12])[:max_len].strip("_")


def record_key(rec: dict[str, Any]) -> str:
    return f"{rec.get('episode_id')}::t{rec.get('turn_num')}::{rec.get('speaker')}"


def clean_context(context: str, max_chars: int) -> str:
    context = context.strip()
    if len(context) <= max_chars:
        return context
    return context[-max_chars:]


def insert_state_context(context: str, state_text: str, prefix: str) -> str:
    state_line = f"{prefix}: {state_text.strip()}"
    lines = context.rstrip().splitlines()
    if lines and re.match(r"^Turn\s+\d+\s+\|\s+.*:\s*$", lines[-1]):
        return "\n".join(lines[:-1] + [state_line, lines[-1]])
    return f"{context.rstrip()}\n{state_line}"


def structured_state_text(roles: dict[str, str]) -> str:
    return (
        f"Partner Belief: {roles['belief'].strip()} | "
        f"Strategic Intent: {roles['intent'].strip()} | "
        f"Thought Process: {roles['thought'].strip()}"
    )


def flat_state_text(fused_summary: str) -> str:
    return re.sub(r"\s+", " ", fused_summary).strip()


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
        mental = f"{rec.get('mental1_text', '')} {rec.get('mental2_text', '')}"
        if not all(token in mental for token in ["Partner Belief:", "Strategic Intent:", "Thought Process:"]):
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


def make_prompt(rec: dict[str, Any], role: str, max_context_chars: int) -> str:
    payload = {
        "episode_id": rec.get("episode_id"),
        "turn_num": rec.get("turn_num"),
        "speaker": rec.get("speaker"),
        "scenario": rec.get("scenario"),
        "observable_context": clean_context(rec.get("context_text", ""), max_context_chars),
        "original_response": rec.get("response_text"),
        "original_first_order_mental_model": rec.get("mental1_text"),
        "target_role_to_flip": role,
    }
    return f"""Create one role-targeted counterfactual ranking item.

Return only valid JSON with this exact top-level shape:
{{
  "role": "{role}",
  "state_a": {{
    "name": "short_state_a_name",
    "roles": {{
      "belief": "...",
      "intent": "...",
      "thought": "..."
    }},
    "flat_summary": "one fluent sentence fusing the same belief/intent/thought content without labels"
  }},
  "state_b": {{
    "name": "short_state_b_name",
    "roles": {{
      "belief": "...",
      "intent": "...",
      "thought": "..."
    }},
    "flat_summary": "one fluent sentence fusing the same belief/intent/thought content without labels"
  }},
  "candidate_a": "speaker reply that is better under state_a",
  "candidate_b": "speaker reply that is better under state_b",
  "why_role_targeted": "one sentence explaining how only {role} changed",
  "why_observed_balanced": "one sentence explaining why observable-only scoring should be near indifferent"
}}

Hard constraints:
- Flip exactly the target role: {role}.
- Keep the other two roles as semantically matched as possible across state A and state B.
- Candidate A must be correct under state A; candidate B must be correct under state B.
- Both candidates must be natural continuations from the same speaker.
- Both candidates must be similar in length, politeness, and generic social quality.
- Avoid obvious good/bad pairs.

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


def call_openai(args: argparse.Namespace, api_key: str, rec: dict[str, Any], role: str) -> dict[str, Any]:
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_prompt(rec, role, args.max_context_chars)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(args.max_retries):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"]
            return {
                "record_key": record_key(rec),
                "role": role,
                "model": args.model,
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
    raise RuntimeError(f"OpenAI call failed for {record_key(rec)} role={role}: {last_error}")


def valid_response(resp: dict[str, Any], expected_role: str) -> bool:
    if resp.get("role") != expected_role:
        return False
    for state_key in ["state_a", "state_b"]:
        state = resp.get(state_key)
        if not isinstance(state, dict):
            return False
        roles = state.get("roles")
        if not isinstance(roles, dict):
            return False
        if any(not str(roles.get(role, "")).strip() for role in ROLES):
            return False
        if not str(state.get("flat_summary", "")).strip():
            return False
    cand_a = str(resp.get("candidate_a", "")).strip()
    cand_b = str(resp.get("candidate_b", "")).strip()
    if len(cand_a) < 35 or len(cand_b) < 35 or cand_a == cand_b:
        return False
    if max(len(cand_a), len(cand_b)) / max(1, min(len(cand_a), len(cand_b))) > 1.75:
        return False
    return True


def materialize(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for row in cache_rows:
        resp = row.get("response", {})
        expected_role = row.get("role")
        if expected_role not in ROLES or not valid_response(resp, expected_role):
            continue
        source = row["source"]
        context = source["context_text"].rstrip()
        pair_key = f"{row['record_key']}::{expected_role}"
        pair_id = f"sotopia_role_cf_{slug(pair_key)}"
        if pair_id in seen:
            continue
        seen.add(pair_id)

        state_a_roles = {role: str(resp["state_a"]["roles"][role]).strip() for role in ROLES}
        state_b_roles = {role: str(resp["state_b"]["roles"][role]).strip() for role in ROLES}
        state_a_structured = structured_state_text(state_a_roles)
        state_b_structured = structured_state_text(state_b_roles)
        state_a_flat = flat_state_text(str(resp["state_a"]["flat_summary"]))
        state_b_flat = flat_state_text(str(resp["state_b"]["flat_summary"]))
        out.append({
            "pair_id": pair_id,
            "role": expected_role,
            "observable_context": context,
            "candidate_a": " ".join(str(resp["candidate_a"]).split()),
            "candidate_b": " ".join(str(resp["candidate_b"]).split()),
            "state_a": {
                "name": str(resp["state_a"].get("name", f"{expected_role}_state_a")),
                "roles": state_a_roles,
                "z_context": insert_state_context(context, state_a_structured, "Counterfactual partner mental state"),
                "flat_context": insert_state_context(context, state_a_flat, "Counterfactual partner mental summary"),
                "correct": "a",
            },
            "state_b": {
                "name": str(resp["state_b"].get("name", f"{expected_role}_state_b")),
                "roles": state_b_roles,
                "z_context": insert_state_context(context, state_b_structured, "Counterfactual partner mental state"),
                "flat_context": insert_state_context(context, state_b_flat, "Counterfactual partner mental summary"),
                "correct": "b",
            },
            "source": {
                "dataset": "sotopia_stage1_val",
                "generation_model": row["model"],
                "episode_id": source.get("episode_id"),
                "turn_num": source.get("turn_num"),
                "speaker": source.get("speaker"),
                "scenario": source.get("scenario"),
                "original_response_text": source.get("response_text"),
                "original_mental1_text": source.get("mental1_text"),
                "original_mental2_text": source.get("mental2_text"),
                "why_role_targeted": resp.get("why_role_targeted"),
                "why_observed_balanced": resp.get("why_observed_balanced"),
            },
            "notes": "LLM-generated role-targeted SOTOPIA held-out counterfactual; review before final publication.",
        })
    return out


def load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", default="projects/sotopia/runs/analysis/mental_latent_qwen_v3_epoch4_stage1val1508/analysis_records_subset.jsonl")
    parser.add_argument("--cache-jsonl", default="projects/sotopia/experiments/runs/stage1/role_targeted_counterfactual/cache_gpt4o_seed42.jsonl")
    parser.add_argument("--output-jsonl", default="projects/sotopia/experiments/runs/stage1/role_targeted_counterfactual/role_targeted_counterfactual_pairs_gpt4o_seed42.jsonl")
    parser.add_argument("--summary-json", default="projects/sotopia/experiments/runs/stage1/role_targeted_counterfactual/generation_summary.json")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--roles", nargs="+", default=ROLES, choices=ROLES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=1300)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_records = sample_records(
        eligible_records(read_jsonl(Path(args.source_jsonl))),
        max_records=args.max_records,
        seed=args.seed,
    )
    cache_path = Path(args.cache_jsonl)
    cached = load_cache(cache_path)
    done = {(row.get("record_key"), row.get("role")) for row in cached}
    tasks = [
        (rec, role)
        for rec in source_records
        for role in args.roles
        if (record_key(rec), role) not in done
    ]
    print(json.dumps({
        "source_records": len(source_records),
        "cached_items": len(cached),
        "todo_items": len(tasks),
        "roles": args.roles,
        "model": args.model,
    }, indent=2), flush=True)

    if tasks and not args.materialize_only:
        api_key = read_key(args)
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(call_openai, args, api_key, rec, role): (rec, role) for rec, role in tasks}
            for future in futures.as_completed(future_to_task):
                rec, role = future_to_task[future]
                try:
                    row = future.result()
                    append_jsonl(cache_path, row)
                    print(f"[ok] {row['record_key']} role={row['role']}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] {record_key(rec)} role={role}: {exc}", flush=True)
                    if args.fail_fast:
                        raise

    records = materialize(load_cache(cache_path))
    write_jsonl(Path(args.output_jsonl), records)
    role_counts = {role: sum(1 for rec in records if rec.get("role") == role) for role in ROLES}
    summary = {
        "source_jsonl": args.source_jsonl,
        "cache_jsonl": args.cache_jsonl,
        "output_jsonl": args.output_jsonl,
        "model": args.model,
        "seed": args.seed,
        "max_records": args.max_records,
        "materialized_pairs": len(records),
        "role_counts": role_counts,
        "curation_status": "llm_generated_role_targeted_candidates_needs_review_or_observed_only_filtering",
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
