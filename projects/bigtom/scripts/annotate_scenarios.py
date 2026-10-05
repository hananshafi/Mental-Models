"""
Annotate BigToM scenarios with first-order + second-order mental state targets.

For every scenario in the input CSV we emit TWO training examples (one for the
aware condition, one for the not-aware condition). For each example, Qwen-235B
with reasoning enabled produces:

    first_order_belief      : what the agent believes about the object state
    second_order_belief     : what the agent believes an observer believes
                              about the object state
    second_order_observer   : name of the observer (Qwen introduces one)
    rationale               : short CoT explaining why the beliefs hold,
                              used as the SFT target for the policy
    paraphrase              : story rewritten with different wording
                              (once per scenario, shared by both conditions)

Output: JSONL, one row per training example.

Usage:
    python annotate_scenarios.py --in ../data/bigtom_qwen.csv \\
                                 --out ../data/bigtom_qwen_annotated.jsonl
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from qwen_client import QwenClient, parallel_chat

# Schema of the upstream scenario CSV (19 fields per row).
SCENARIO_FIELDS = [
    "story", "aware_event", "not_aware_event",
    "action_new_state", "action_init_state",
    "belief_question", "desire_question", "action_question",
    "belief_aware", "desire_aware", "action_aware",
    "belief_not_aware", "desire_not_aware", "action_not_aware",
    "random_event", "aware_random", "not_aware_random",
    "source", "init_belief_idx",
]


SYSTEM = (
    "You are a theory-of-mind annotator. Given a BigToM scenario, you produce "
    "a strict JSON object with first-order and second-order belief annotations, "
    "plus a short chain-of-thought rationale and a paraphrase. "
    "Return ONLY the JSON object, no markdown fencing, no commentary."
)


PROMPT_TEMPLATE = """You will annotate the following theory-of-mind scenario.

### Scenario

Story: {story}

Causal branch (the event DID happen). Two alternative continuations follow:
 - If the agent perceives the event: "{aware_event}"
 - If the agent does NOT perceive the event: "{not_aware_event}"

Gold belief when aware:      {belief_aware}
Gold belief when not aware:  {belief_not_aware}
Gold action when aware:      {action_aware}
Gold action when not aware:  {action_not_aware}

Belief question:  {belief_question}
Action question:  {action_question}

### Your task

Invent ONE plausible second character (observer) who is physically present near
the object at the time of the external event — for example, a bystander,
coworker, friend, or passerby that naturally fits the scene. This observer
directly witnesses the external event. The main agent may or may not be aware
that the observer saw it.

Then produce the following JSON object (no extra fields):

{{
  "observer_name": "<one short name you invented for the observer>",
  "observer_role": "<one-phrase role that fits the scene, e.g. 'coworker at the cafe'>",
  "paraphrase": "<rewrite the original story in different words, same 5 beats, same facts>",

  "aware": {{
    "first_order_belief": "<what the AGENT believes about the object state when the AGENT perceived the event>",
    "second_order_belief": "<what the AGENT believes the OBSERVER believes about the object state, in this aware branch>",
    "rationale": "<2-4 sentence chain-of-thought that walks from the story facts to the belief and the action the agent will take. End with the final action.>"
  }},

  "not_aware": {{
    "first_order_belief": "<what the AGENT believes about the object state when the AGENT did NOT perceive the event>",
    "second_order_belief": "<what the AGENT believes the OBSERVER believes about the object state, in this not-aware branch. Be careful: the agent may not even know the observer saw anything.>",
    "rationale": "<2-4 sentence chain-of-thought ending with the final action.>"
  }}
}}

Rules:
- first_order_belief in the `aware` branch must match the world state AFTER the event.
- first_order_belief in the `not_aware` branch must match the world state BEFORE the event (the agent is mistaken).
- second_order_belief must be the AGENT's model of the OBSERVER's belief, not the observer's actual belief.
- Never mention markdown. Output only the JSON object.
"""


def load_scenarios(path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 19:
                continue
            rows.append({k: v for k, v in zip(SCENARIO_FIELDS, row[:19])})
    return rows


def build_prompt(sc: Dict[str, str]) -> str:
    return PROMPT_TEMPLATE.format(**sc)


def parse_response(text: str):
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


def scenario_uid(sc: Dict[str, str]) -> str:
    key_fields = [
        sc["story"],
        sc["aware_event"],
        sc["not_aware_event"],
        sc["belief_aware"],
        sc["belief_not_aware"],
        sc["action_aware"],
        sc["action_not_aware"],
    ]
    payload = "||".join(_norm(v) for v in key_fields)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def validate(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    for k in ("observer_name", "observer_role", "paraphrase", "aware", "not_aware"):
        if k not in obj:
            return False
    for branch in ("aware", "not_aware"):
        b = obj[branch]
        if not isinstance(b, dict):
            return False
        for k in ("first_order_belief", "second_order_belief", "rationale"):
            if k not in b or not isinstance(b[k], str) or not b[k].strip():
                return False
    if _norm(obj["aware"]["first_order_belief"]) == _norm(obj["not_aware"]["first_order_belief"]):
        return False
    return True


def emit(obj: Dict, sc: Dict[str, str], sc_idx: int) -> List[Dict]:
    """Turn one annotated scenario into two flat training examples."""
    base = {
        "scenario_id": sc_idx,
        "scenario_uid": scenario_uid(sc),
        "story": sc["story"],
        "paraphrase": obj["paraphrase"],
        "observer_name": obj["observer_name"],
        "observer_role": obj["observer_role"],
        "belief_question": sc["belief_question"],
        "desire_question": sc["desire_question"],
        "action_question": sc["action_question"],
        "init_belief_idx": int(sc.get("init_belief_idx", 0) or 0),
    }
    out = []
    for cond, percept_key, gold_belief, gold_action in (
        ("aware", "aware_event", sc["belief_aware"], sc["action_aware"]),
        ("not_aware", "not_aware_event", sc["belief_not_aware"], sc["action_not_aware"]),
    ):
        row = dict(base)
        row["condition"] = cond                       # "aware" | "not_aware"
        row["percept"] = sc[percept_key]              # perception sentence appended to story
        row["gold_belief"] = gold_belief              # BigToM gold string (z1 label)
        row["gold_action"] = gold_action              # BigToM gold action string
        row["first_order_belief"] = obj[cond]["first_order_belief"]
        row["second_order_belief"] = obj[cond]["second_order_belief"]
        row["rationale"] = obj[cond]["rationale"]
        out.append(row)
    return out


def load_existing_ids(path: Path):
    done = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                key = r.get("scenario_uid", str(r["scenario_id"]))
                done.setdefault(key, set()).add(r["condition"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("projects/bigtom/data/bigtom_qwen.csv"),
                    help="scenario CSV to annotate")
    ap.add_argument("--out", type=Path,
                    default=Path("projects/bigtom/data/bigtom_qwen_annotated.jsonl"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=6000,
                    help="generous budget since reasoning eats tokens")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=None,
                    help="only annotate the first N scenarios (debug)")
    args = ap.parse_args()

    scenarios = load_scenarios(args.inp)
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenarios from {args.inp}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_existing_ids(args.out)
    todo = []
    already_done = 0
    for i, s in enumerate(scenarios):
        key = scenario_uid(s)
        if done_ids.get(key) == {"aware", "not_aware"}:
            already_done += 1
            continue
        todo.append((i, s))
    print(f"Already annotated: {already_done}. Remaining: {len(todo)}.")

    client = QwenClient()

    fout = open(args.out, "a")
    kept = 0
    rejected = 0

    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        prompts = [
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(s)},
            ]
            for _, s in chunk
        ]
        outs = parallel_chat(
            client, prompts,
            max_workers=args.max_workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            # Qwen3 reasoning ON — higher-quality belief annotations.
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        for (idx, sc), text in zip(chunk, outs):
            obj = parse_response(text)
            if not obj or not validate(obj):
                rejected += 1
                continue
            for row in emit(obj, sc, idx):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
        fout.flush()
        print(f"  [{start+len(chunk)}/{len(todo)}]  kept={kept}  rejected={rejected}")

    fout.close()
    print(f"Done. Kept {kept} / {len(todo)} scenarios ({kept*2} training rows). Rejected {rejected}.")


if __name__ == "__main__":
    main()
