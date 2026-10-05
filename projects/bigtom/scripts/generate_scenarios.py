"""
Expand BigToM's 200 seed scenarios into a larger training set using Qwen.

Approach:
  1. Load the 200 seed rows (19 fields, ';' delimited) from bigtom.csv.
  2. Sample k=4 seeds per request as few-shot exemplars.
  3. Ask Qwen for a new scenario in strict JSON (19 fields).
  4. Validate schema + reject malformed rows.
  5. Append to an output CSV in the same 19-field format so BigToM's
     generate_conditions.py can process it unchanged.

Usage:
  python generate_scenarios.py --n 2000 --out ../data/bigtom_qwen.csv
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from qwen_client import QwenClient, parallel_chat


FIELDS = [
    "story",                # context + desire + percept + belief + causal event
    "aware_event",          # "<Agent> perceives the event"
    "not_aware_event",      # "<Agent> does not perceive the event"
    "action_new_state",     # action assuming new state
    "action_init_state",    # action assuming initial state
    "belief_question",
    "desire_question",
    "action_question",
    "belief_aware",
    "desire_aware",
    "action_aware",
    "belief_not_aware",
    "desire_not_aware",
    "action_not_aware",
    "random_event",
    "aware_random",
    "not_aware_random",
    "source",               # "auto" | "qwen"
    "init_belief_idx",      # 0 or 1
]


SYSTEM = (
    "You generate BigToM-style theory-of-mind scenarios. "
    "Each scenario is a single JSON object following the schema exactly. "
    "No markdown, no commentary — output ONLY the JSON object."
)


PROMPT_TEMPLATE = """You will generate ONE new BigToM scenario following the schema below.

## Schema (strict)

A JSON object with these keys, all strings unless noted:
- story: exactly 5 sentences separated by '. ':
    (1) Context: an agent in a situation/location.
    (2) Desire: the agent's goal.
    (3) Perception cue: object is observed in a specific state (do NOT say the agent knows it).
    (4) Belief: "<Agent> believes that <object> is <state>." matching sentence 3 exactly.
    (5) Causal event: external event changes the object to another extreme state. Do NOT mention the agent.
- aware_event: "<Agent> perceives/notices/sees the event." (no object state).
- not_aware_event: "<Agent> does not perceive/notice/see the event." (no object state).
- action_new_state: action the agent takes assuming the new (post-event) state.
- action_init_state: action the agent takes assuming the original state.
- belief_question: "Does <Agent> believe the <object> is <stateA> or <stateB>?"
- desire_question: "What does <Agent> want to do ...?"
- action_question: "What will <Agent> do?"
- belief_aware: what the agent believes after perceiving the event.
- desire_aware: the agent's desire after perceiving the event.
- action_aware: the action the agent takes after perceiving the event.
- belief_not_aware: what the agent believes if they did NOT perceive the event.
- desire_not_aware: the agent's desire if they did NOT perceive the event.
- action_not_aware: the action the agent takes if they did NOT perceive the event.
- random_event: a SEPARATE unrelated event that fits the context but cannot change the object's state.
- aware_random: "<Agent> perceives the random event." (no object state).
- not_aware_random: "<Agent> does not perceive the random event." (no object state).
- init_belief_idx: integer 0 or 1 (pick randomly; indicates the polarity of the initial belief).

Be creative. Use UNCOMMON names. Do NOT reuse contexts from the examples below.

## Examples (already valid scenarios)

{examples}

## Your turn

Output a new scenario as a single JSON object only. No prose, no markdown fences.
"""


def load_seeds(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 19:
                continue
            rows.append(row[:19])
    return rows


def seed_to_json(row):
    return {k: v for k, v in zip(FIELDS, row)}


def few_shot_block(seeds, k, rng):
    picks = rng.sample(seeds, k)
    blocks = []
    for i, row in enumerate(picks):
        blocks.append(f"### Example {i+1}\n```json\n{json.dumps(seed_to_json(row), ensure_ascii=False, indent=2)}\n```")
    return "\n\n".join(blocks)


def build_prompt(seeds, k, rng):
    return PROMPT_TEMPLATE.format(examples=few_shot_block(seeds, k, rng))


def parse_response(text: str):
    """Extract the first JSON object from a Qwen response."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _norm(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def split_story_sentences(story: str):
    return [s.strip() for s in story.split(".") if s.strip()]


def scenario_signature(obj) -> str:
    key_fields = [
        obj.get("story", ""),
        obj.get("aware_event", ""),
        obj.get("not_aware_event", ""),
        obj.get("belief_aware", ""),
        obj.get("belief_not_aware", ""),
        obj.get("action_aware", ""),
        obj.get("action_not_aware", ""),
    ]
    payload = "||".join(_norm(v) for v in key_fields)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_existing_signatures(path: Path):
    sigs = set()
    if not path.exists():
        return sigs
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 19:
                continue
            obj = {k: v for k, v in zip(FIELDS, row[:19])}
            sigs.add(scenario_signature(obj))
    return sigs


def validate(obj):
    if not isinstance(obj, dict):
        return False
    for f in FIELDS:
        if f == "init_belief_idx":
            continue
        if f == "source":
            continue
        if f not in obj or not isinstance(obj[f], str) or not obj[f].strip():
            return False
    story_sents = split_story_sentences(obj["story"].strip())
    if len(story_sents) != 5:
        return False
    if "believ" not in story_sents[3].lower():
        return False
    if _norm(obj["belief_aware"]) == _norm(obj["belief_not_aware"]):
        return False
    if _norm(obj["action_aware"]) == _norm(obj["action_not_aware"]):
        return False
    if _norm(obj["aware_event"]) == _norm(obj["not_aware_event"]):
        return False
    if _norm(story_sents[4]) == _norm(obj["random_event"]):
        return False
    return True


def to_row(obj):
    out = []
    for f in FIELDS:
        if f == "source":
            out.append("qwen")
        elif f == "init_belief_idx":
            out.append(str(obj.get("init_belief_idx", 0)))
        else:
            out.append(obj.get(f, "").replace(";", ",").replace("\n", " ").strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_csv", type=Path,
                    default=Path("third_party/src/bigtom/data/bigtom/bigtom.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("projects/bigtom/data/bigtom_qwen.csv"))
    ap.add_argument("--n", type=int, default=2000, help="number of new scenarios to generate")
    ap.add_argument("--k_shot", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64, help="parallel calls per flush")
    ap.add_argument("--max_workers", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = load_seeds(args.seed_csv)
    print(f"Loaded {len(seeds)} seed scenarios from {args.seed_csv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    existing_sigs = load_existing_signatures(args.out)
    if args.out.exists():
        with open(args.out) as f:
            existing = sum(1 for _ in f)
        print(f"Found existing {existing} rows in {args.out} — appending")

    client = QwenClient()
    fout = open(args.out, "a", newline="")
    writer = csv.writer(fout, delimiter=";")

    total_written = existing
    total_rejected = 0
    target = args.n

    while total_written - existing < target:
        batch_needed = min(args.batch, target - (total_written - existing))
        prompts = [
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(seeds, args.k_shot, rng)},
            ]
            for _ in range(batch_needed)
        ]
        outs = parallel_chat(
            client, prompts,
            max_workers=args.max_workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        kept = 0
        for text in outs:
            obj = parse_response(text)
            if not obj or not validate(obj):
                total_rejected += 1
                continue
            if "init_belief_idx" not in obj:
                obj["init_belief_idx"] = rng.randint(0, 1)
            sig = scenario_signature(obj)
            if sig in existing_sigs:
                total_rejected += 1
                continue
            writer.writerow(to_row(obj))
            existing_sigs.add(sig)
            kept += 1
            total_written += 1
        fout.flush()
        new_count = total_written - existing
        print(f"  batch: kept {kept}/{len(outs)} | total new {new_count}/{target} "
              f"| rejected overall {total_rejected}")

    fout.close()
    print(f"Done. Wrote {total_written - existing} new scenarios to {args.out} "
          f"(total rows in file: {total_written}). Rejected {total_rejected}.")


if __name__ == "__main__":
    main()
