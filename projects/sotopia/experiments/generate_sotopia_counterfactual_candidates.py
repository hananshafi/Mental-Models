#!/usr/bin/env python3
"""Generate SOTOPIA-heldout counterfactual candidate pairs.

The output uses the same schema as run_counterfactual_flip_experiment.py:
the observable context is a real SOTOPIA held-out turn, while state_a/state_b
insert different partner mental states before the speaker's response slot.

This is intentionally deterministic. It creates candidate stimuli for later
observed-only balancing and human/LLM curation, not a final human-curated
benchmark by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CounterfactualAxis:
    name: str
    keywords: tuple[str, ...]
    state_a: str
    state_b: str
    candidates_a: tuple[str, ...]
    candidates_b: tuple[str, ...]


AXES = [
    CounterfactualAxis(
        name="direct_ask_vs_constraint_check",
        keywords=("donate", "fund", "money", "afford", "budget", "buy", "sell", "price", "pay", "loan"),
        state_a=(
            "Partner Belief: The partner is receptive and has enough flexibility to agree to a stronger request. "
            "Strategic Intent: They want the speaker to make a clear, concrete ask. "
            "Thought Process: A direct proposal will help them decide and move the interaction forward."
        ),
        state_b=(
            "Partner Belief: The partner feels constrained and may resist if pushed too quickly. "
            "Strategic Intent: They want their limits acknowledged before making a commitment. "
            "Thought Process: Checking constraints first will preserve trust and make agreement more likely."
        ),
        candidates_a=(
            "I want to be direct about what I am hoping for here. Could we commit to the stronger option now and make it concrete?",
            "Given what we both want, I think the best next step is to agree to the more ambitious plan now.",
            "I think this is the moment to make a clear commitment. Would you be willing to take the bigger step with me?",
        ),
        candidates_b=(
            "Before I ask for anything more, I want to understand what would actually be workable for you.",
            "I do not want to pressure you past your limits. What version of this would feel realistic on your side?",
            "Let us first figure out the constraint you are working with, then choose a step that still helps.",
        ),
    ),
    CounterfactualAxis(
        name="firm_boundary_vs_empathy_first",
        keywords=("argument", "upset", "noise", "late", "deadline", "mistake", "conflict", "hurt", "angry", "apologize"),
        state_a=(
            "Partner Belief: The partner understands the situation but may avoid changing unless the boundary is explicit. "
            "Strategic Intent: They need a clear expectation from the speaker. "
            "Thought Process: A calm firm boundary will make the next step unambiguous."
        ),
        state_b=(
            "Partner Belief: The partner is emotionally strained and fears being judged. "
            "Strategic Intent: They need the speaker to understand their side before setting terms. "
            "Thought Process: Empathy first will reduce defensiveness and make later agreement possible."
        ),
        candidates_a=(
            "I want to be clear about what needs to change from here, because this has started to affect me.",
            "I understand there may be reasons, but we still need a clear agreement about how to handle this going forward.",
            "I need to set a boundary here so that we both know what is acceptable next time.",
        ),
        candidates_b=(
            "Before we decide what should happen next, I want to understand what was going on for you.",
            "I can tell this may be more complicated than it looks, so I want to hear your side first.",
            "I do not want to jump straight to rules if something difficult was happening for you.",
        ),
    ),
    CounterfactualAxis(
        name="urgent_action_vs_patient_followup",
        keywords=("urgent", "today", "now", "soon", "late", "delay", "deadline", "emergency", "risk", "problem"),
        state_a=(
            "Partner Belief: The partner thinks delay will create a real cost or missed opportunity. "
            "Strategic Intent: They want decisive action now. "
            "Thought Process: A concrete immediate plan is more helpful than more discussion."
        ),
        state_b=(
            "Partner Belief: The partner has time and is wary of being rushed. "
            "Strategic Intent: They want a careful follow-up rather than immediate pressure. "
            "Thought Process: Patience will help them stay engaged and make a better decision."
        ),
        candidates_a=(
            "Let us act on this now: I can take the first step today and we can confirm the details right away.",
            "I think waiting will make this harder, so I would like us to decide on the next concrete step now.",
            "This seems time-sensitive enough that I want to move quickly and make a firm plan today.",
        ),
        candidates_b=(
            "There is no need to force a decision this minute. I can follow up after you have had time to think.",
            "Let us slow down and make sure the plan fits before we commit to anything today.",
            "I would rather give this a careful follow-up than rush you into a decision now.",
        ),
    ),
    CounterfactualAxis(
        name="disclosure_vs_discretion",
        keywords=("secret", "private", "tell", "know", "truth", "honest", "information", "credential", "confidential"),
        state_a=(
            "Partner Belief: The partner already knows enough context and wants the speaker to be direct. "
            "Strategic Intent: They want transparent acknowledgment. "
            "Thought Process: Naming the sensitive issue will reduce confusion and build trust."
        ),
        state_b=(
            "Partner Belief: The partner values privacy and would feel harmed by unnecessary disclosure. "
            "Strategic Intent: They want the speaker to protect sensitive information. "
            "Thought Process: Discretion will preserve trust better than saying too much."
        ),
        candidates_a=(
            "I think it is better to be direct about the sensitive part rather than talk around it.",
            "Since this affects the decision, I should name what is really going on as clearly as I can.",
            "I do not want to hide the important context from you, so I will be straightforward.",
        ),
        candidates_b=(
            "I want to respect the private parts of this, so I will keep the explanation focused on what affects us now.",
            "Some details are not mine to share, but I can still explain the practical next step.",
            "I do not want to expose anything sensitive unnecessarily, so let me keep this careful and limited.",
        ),
    ),
    CounterfactualAxis(
        name="encourage_commitment_vs_preserve_autonomy",
        keywords=("convince", "persuade", "support", "help", "agree", "decision", "choose", "plan", "friend"),
        state_a=(
            "Partner Belief: The partner wants encouragement and is close to agreeing. "
            "Strategic Intent: They want the speaker to confidently guide them toward commitment. "
            "Thought Process: Positive momentum will help them act on an intention they already have."
        ),
        state_b=(
            "Partner Belief: The partner worries that the speaker is trying to control their choice. "
            "Strategic Intent: They need autonomy and room to decide. "
            "Thought Process: A low-pressure response will make them more receptive."
        ),
        candidates_a=(
            "I believe this is worth committing to, and I would really like us to take that step together.",
            "You already seem open to this, so I want to encourage us to turn that into a real decision.",
            "I think we can make this work if we both commit to the plan now.",
        ),
        candidates_b=(
            "I do not want you to feel pushed. I can explain my view, but the choice should still feel like yours.",
            "Take the time you need; I would rather you decide freely than feel talked into it.",
            "I can share why it matters to me, but I want to leave you room to choose what feels right.",
        ),
    ),
    CounterfactualAxis(
        name="repair_accountability_vs_explain_context",
        keywords=("sorry", "apology", "apologize", "trust", "relationship", "hurt", "forgot", "missed", "wrong"),
        state_a=(
            "Partner Belief: The partner mainly needs accountability and a repair attempt. "
            "Strategic Intent: They want the speaker to own the impact before explaining. "
            "Thought Process: A direct apology will make the relationship feel repairable."
        ),
        state_b=(
            "Partner Belief: The partner believes there may be missing context and wants explanation before judgment. "
            "Strategic Intent: They want to understand what happened. "
            "Thought Process: Explaining the circumstances will make the response feel fair."
        ),
        candidates_a=(
            "You are right to be upset. I should own my part in this, and I want to make it right.",
            "I can see that this affected you. I am sorry, and I want to repair the trust I damaged.",
            "Before I explain anything, I want to acknowledge that my part in this was not okay.",
        ),
        candidates_b=(
            "I hear why it looks bad, but I think the context matters, and I would like to explain what happened.",
            "Can I walk you through what was going on from my side before we decide what this means?",
            "There is more behind this than I was able to say at first, and I think explaining it will help.",
        ),
    ),
]

NEUTRAL_AXES = ("encourage_commitment_vs_preserve_autonomy", "firm_boundary_vs_empathy_first", "urgent_action_vs_patient_followup")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def slug(text: str, max_len: int = 48) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return clean[:max_len].strip("_") or hashlib.md5(text.encode()).hexdigest()[:10]


def short_goal(context: str) -> str:
    match = re.search(r"Goal:\s*(.*?)(?:\n|$)", context)
    if not match:
        return "the current goal"
    goal = re.sub(r"\s+", " ", match.group(1)).strip()
    return goal[:180]


def insert_state_context(context: str, mental_state: str) -> str:
    lines = context.rstrip().splitlines()
    if lines and re.match(r"^Turn\s+\d+\s+\|\s+.*:\s*$", lines[-1]):
        return "\n".join(lines[:-1] + [f"Counterfactual partner mental state: {mental_state}", lines[-1]])
    return f"{context.rstrip()}\nCounterfactual partner mental state: {mental_state}"


def axis_score(axis: CounterfactualAxis, text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in axis.keywords if kw in lower)


def pick_axes(record: dict[str, Any], max_axes: int) -> list[CounterfactualAxis]:
    text = " ".join([
        str(record.get("scenario", "")),
        str(record.get("context_text", "")),
        str(record.get("mental1_text", "")),
    ])
    scored = [(axis_score(axis, text), axis) for axis in AXES]
    scored.sort(key=lambda item: (-item[0], item[1].name))
    chosen = [axis for score, axis in scored if score > 0][:max_axes]
    if len(chosen) < max_axes:
        by_name = {axis.name: axis for axis in AXES}
        for name in NEUTRAL_AXES:
            axis = by_name[name]
            if axis not in chosen:
                chosen.append(axis)
            if len(chosen) >= max_axes:
                break
    return chosen[:max_axes]


def eligible_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for rec in records:
        context = rec.get("context_text", "")
        if not context or not rec.get("response_text"):
            continue
        if len(context) < 250 or len(context) > 6000:
            continue
        if int(rec.get("turn_num", 0)) < 1:
            continue
        filtered.append(rec)
    return filtered


def make_pair(record: dict[str, Any], axis: CounterfactualAxis, variant_idx: int) -> dict[str, Any]:
    cand_a = axis.candidates_a[variant_idx % len(axis.candidates_a)]
    cand_b = axis.candidates_b[variant_idx % len(axis.candidates_b)]
    context = record["context_text"].rstrip()
    pair_key = f"{record.get('episode_id')}_t{record.get('turn_num')}_{axis.name}_{variant_idx}"
    pair_id = f"sotopia_cf_{slug(pair_key, 90)}"
    goal = short_goal(context)

    state_a = (
        f"{axis.state_a} Local speaker goal: {goal}. "
        "Correct response preference: candidate A is better than candidate B under this hidden state."
    )
    state_b = (
        f"{axis.state_b} Local speaker goal: {goal}. "
        "Correct response preference: candidate B is better than candidate A under this hidden state."
    )

    return {
        "pair_id": pair_id,
        "observable_context": context,
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "state_a": {
            "name": f"{axis.name}_state_a",
            "z_context": insert_state_context(context, state_a),
            "correct": "a",
        },
        "state_b": {
            "name": f"{axis.name}_state_b",
            "z_context": insert_state_context(context, state_b),
            "correct": "b",
        },
        "source": {
            "dataset": "sotopia_episode_heldout_val",
            "episode_id": record.get("episode_id"),
            "turn_num": record.get("turn_num"),
            "speaker": record.get("speaker"),
            "scenario": record.get("scenario"),
            "original_response_text": record.get("response_text"),
            "original_mental1_text": record.get("mental1_text"),
            "original_mental2_text": record.get("mental2_text"),
        },
        "axis": axis.name,
        "notes": (
            "Automatically generated SOTOPIA-heldout counterfactual candidate. "
            "Use observed-only reward filtering and human/LLM review before treating as a final benchmark."
        ),
    }


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    records = eligible_records(read_jsonl(Path(args.source_jsonl)))
    rng.shuffle(records)

    pairs: list[dict[str, Any]] = []
    used_episode_turn_axes: set[tuple[str, int, str]] = set()
    for record in records:
        if len(pairs) >= args.max_pairs:
            break
        axes = pick_axes(record, args.axes_per_record)
        for axis in axes:
            key = (str(record.get("episode_id")), int(record.get("turn_num", 0)), axis.name)
            if key in used_episode_turn_axes:
                continue
            used_episode_turn_axes.add(key)
            for variant_idx in range(args.variants_per_axis):
                pairs.append(make_pair(record, axis, variant_idx))
                if len(pairs) >= args.max_pairs:
                    break
            if len(pairs) >= args.max_pairs:
                break

    write_jsonl(Path(args.output_jsonl), pairs)
    meta = {
        "source_jsonl": args.source_jsonl,
        "output_jsonl": args.output_jsonl,
        "seed": args.seed,
        "eligible_records": len(records),
        "generated_pairs": len(pairs),
        "axes_per_record": args.axes_per_record,
        "variants_per_axis": args.variants_per_axis,
        "curation_status": "automatic_sotopia_heldout_candidate_generation",
        "warning": "Not human curated; filter by observed-only balance and review before final benchmark use.",
    }
    if args.metadata_json:
        Path(args.metadata_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metadata_json, "w") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metadata-json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=160)
    parser.add_argument("--axes-per-record", type=int, default=2)
    parser.add_argument("--variants-per-axis", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
