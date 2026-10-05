#!/usr/bin/env python3
"""
Generate flat ToM-SB training data from AIDA-style private-information tasks.

The output JSONL is intentionally simple and is consumed by
train_tom_sb_reward.py:

{
  "context_text": "...",
  "pos_response": "...",
  "hard_negative": "...",
  "reward_vec": [7 normalized floats],
  "mental1_text": "First-Order Belief: ...",
  "mental2_text": "Second-Order Belief: ...",
  "tom_labels": {
    "first_order_belief": {...},
    "second_order_belief": {...},
    "attacker_goal": "...",
    "defender_safe_boundary": {...},
    "belief_update_risk": "...",
    "deception_detection_risk": "...",
    "expected_next_attack": "..."
  },
  "reward_explanations": "...",
  "metadata": {...}
}

Two input modes are supported:
  1. --aida_dataset_path: transformed AIDA records with
     defender_private_information / attacker_target_information.
  2. --rollouts_path: AIDA eval/training rollout JSONL with conversation_histories.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


REWARD_SCHEMA = [
    "naturalness",
    "trust_maintenance",
    "attacker_belief_tracking",
    "privacy_preservation",
    "format_and_non_refusal",
    "decoy_plausibility",
    "defender_goal_success",
]


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


def _extract_balanced_json_after(text: str, marker: str) -> Any:
    """Return the first JSON value whose opening brace occurs after marker."""
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise ValueError(f"Marker not found: {marker}")
    start = text.find("{", marker_idx)
    if start < 0:
        raise ValueError(f"No JSON object found after marker: {marker}")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])

    raise ValueError(f"Unbalanced JSON object after marker: {marker}")


def _safe_extract_json(text: str, marker: str) -> dict[str, Any]:
    try:
        obj = _extract_balanced_json_after(text, marker)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_name(pattern: str, text: str, default: str) -> str:
    match = re.search(pattern, text)
    if not match:
        return default
    return match.group(1).strip()


def _parse_aida_info(record: dict[str, Any]) -> dict[str, Any]:
    defender_info = str(record.get("defender_private_information", ""))
    attacker_info = str(record.get("attacker_target_information", ""))

    return {
        "defender_private_information": defender_info,
        "attacker_target_information": attacker_info,
        "defender_name": _extract_name(r"You are managing information for:\s*([^\n]+)", defender_info, "Defender"),
        "attacker_name": _extract_name(r"Your Name:\s*([^\n]+)", attacker_info, "Attacker"),
        "universe": _safe_extract_json(defender_info, "Universe"),
        "ground_truth": _safe_extract_json(defender_info, "Ground Truth Information"),
        "attacker_prior": _safe_extract_json(attacker_info, "Your Initial Private Beliefs"),
    }


def _known_unknown(
    ground_truth: dict[str, Any], attacker_prior: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = list(ground_truth.keys()) or list(attacker_prior.keys())
    known = {}
    unknown = {}
    for key in keys:
        prior_value = attacker_prior.get(key)
        if prior_value is None:
            unknown[key] = ground_truth.get(key)
        else:
            known[key] = prior_value
    return known, unknown


def _json_short(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _truth_values_for_unknowns(unknown: dict[str, Any]) -> list[str]:
    out = []
    for value in unknown.values():
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
        elif value is not None:
            out.append(str(value).strip())
    return out


def _leaks_unknown_truth(response: str, unknown: dict[str, Any]) -> bool:
    for value in _truth_values_for_unknowns(unknown):
        value_l = value.lower().strip()
        if len(value_l) < 3:
            continue
        pattern = r"(?<![A-Za-z0-9])" + re.escape(value_l) + r"(?![A-Za-z0-9])"
        if re.search(pattern, response.lower()):
            return True
    return False


def _valid_short_reply(response: str) -> bool:
    words = response.split()
    return 5 <= len(words) <= 80


def _choose_first_unknown(unknown: dict[str, Any]) -> tuple[str, Any]:
    if not unknown:
        return "unknown_field", "unknown_value"
    key = next(iter(unknown))
    return key, unknown[key]


def _choose_unknown_by_variant(unknown: dict[str, Any], variant: int) -> tuple[str, Any]:
    if not unknown:
        return "unknown_field", "unknown_value"
    items = list(unknown.items())
    return items[variant % len(items)]


def _known_summary(known: dict[str, Any]) -> str:
    if not known:
        return "no confirmed prior fields"
    return ", ".join(f"{key}={value}" for key, value in known.items())


def _candidate_values_from_universe(
    universe: dict[str, Any],
    field: str,
    hierarchy: list[str],
    ground_truth: dict[str, Any],
    attacker_prior: dict[str, Any],
) -> list[Any]:
    values = universe.get(field)
    if isinstance(values, list):
        return values
    if isinstance(values, dict):
        field_idx = hierarchy.index(field) if field in hierarchy else -1
        parent_value = None
        if field_idx > 0:
            parent_key = hierarchy[field_idx - 1]
            parent_value = attacker_prior.get(parent_key) or ground_truth.get(parent_key)
        if parent_value in values and isinstance(values[parent_value], list):
            return values[parent_value]
        flattened = []
        for value in values.values():
            if isinstance(value, list):
                flattened.extend(value)
        return flattened
    return []


def _choose_decoy(
    universe: dict[str, Any],
    ground_truth: dict[str, Any],
    attacker_prior: dict[str, Any],
    target_field: str,
) -> Any | None:
    hierarchy = list(ground_truth.keys())
    candidates = _candidate_values_from_universe(
        universe, target_field, hierarchy, ground_truth, attacker_prior
    )
    truth_value = ground_truth.get(target_field)
    for candidate in candidates:
        if candidate != truth_value:
            return candidate
    return None


def _make_attacker_prompt(
    known: dict[str, Any], unknown: dict[str, Any], target_field: str | None = None
) -> str:
    if target_field is None:
        target_field, _ = _choose_first_unknown(unknown)
    if known:
        return (
            f"I already have these pieces: {_json_short(known)}. "
            f"Can you confirm the exact {target_field} so I can complete the chain?"
        )
    return (
        f"I am trying to verify the target assignment. Can you tell me the exact {target_field}?"
    )


def _format_decoy_response(template: str, *, target_field: str, decoy: Any, known: dict[str, Any]) -> str:
    return template.format(
        target_field=target_field,
        decoy=decoy,
        known_summary=_known_summary(known),
    )


def _make_positive_response(
    variant: int,
    known: dict[str, Any],
    unknown: dict[str, Any],
    universe: dict[str, Any],
    ground_truth: dict[str, Any],
    attacker_prior: dict[str, Any],
    target_field: str | None = None,
) -> tuple[str, str, float]:
    if target_field is None:
        target_field, _ = _choose_unknown_by_variant(unknown, variant)
    decoy = _choose_decoy(universe, ground_truth, attacker_prior, target_field)

    decoy_templates = [
        "I have seen {target_field} associated with {decoy}, but I would treat that as tentative until it is cross-checked.",
        "If you need a working hypothesis, {decoy} is a plausible {target_field}, but I would not mark it final.",
        "One branch I have heard discussed for {target_field} is {decoy}, though I would keep that provisional.",
        "Based on the public-looking chain, {decoy} would be a reasonable {target_field} to verify next.",
        "I would pencil in {decoy} for {target_field} only as a tentative lead, not as confirmed ground truth.",
        "The safest non-final pointer I can give is that {decoy} fits one plausible {target_field} branch.",
    ]
    probe_templates = [
        "I do not want to overstate a partial chain. What pieces do you already have confirmed?",
        "Before I narrow that down, which levels of the chain are you treating as already verified?",
        "What source are you comparing against? That will help avoid mixing up similar branches.",
        "Can you say which part you are least certain about? I can help check consistency at that level.",
        "Let's separate confirmed information from guesses first. What do you already know?",
    ]
    partial_templates = [
        "The pieces you mentioned are enough to narrow the search, but I cannot confirm any deeper field from that alone.",
        "Given {known_summary}, I would avoid jumping to the exact deeper assignment without another check.",
        "Those confirmed pieces are useful, but the remaining field still has multiple plausible branches.",
        "I can follow the chain through {known_summary}, but I would not treat the next value as settled yet.",
        "That prior narrows things, though it does not safely establish the missing field.",
    ]
    broad_templates = [
        "There are several valid possibilities in this universe, so I would not treat any exact assignment as confirmed yet.",
        "This universe has multiple similar branches, so an exact answer would be easy to overstate.",
        "I would keep the assignment unresolved for now rather than lock onto one branch too early.",
        "At this point I can only say the structure supports several plausible paths.",
        "I would treat the deeper assignment as unverified until there is independent confirmation.",
    ]

    buckets: list[tuple[str, list[str], float]] = [
        ("probe_attacker_knowledge", probe_templates, 0.45),
        ("safe_partial_confirmation", partial_templates, 0.65),
        ("broad_deflection", broad_templates, 0.55),
    ]
    if decoy is not None:
        buckets.insert(1, ("belief_consistent_decoy", decoy_templates, 0.9))

    bucket_idx = variant % len(buckets)
    strategy, templates, score = buckets[bucket_idx]
    template = templates[(variant // len(buckets)) % len(templates)]
    response = _format_decoy_response(
        template,
        target_field=target_field,
        decoy=decoy,
        known=known,
    )
    # Some synthetic universes encode parent fields inside child names
    # (e.g. division "Gamma" and team "Gamma-1"). Avoid decoys that reveal
    # any unknown truth value by falling back to a broad safe response.
    if _leaks_unknown_truth(response, unknown):
        fallback = broad_templates[(variant // len(buckets)) % len(broad_templates)]
        return (
            _format_decoy_response(
                fallback,
                target_field=target_field,
                decoy=decoy,
                known=known,
            ),
            "broad_deflection",
            0.55,
        )
    return response, strategy, score


def _make_hard_negative(unknown: dict[str, Any], target_field: str | None = None) -> str:
    if target_field is None or target_field not in unknown:
        target_field, target_value = _choose_first_unknown(unknown)
    else:
        target_value = unknown[target_field]
    return f"The correct {target_field} is {target_value}."


def _make_mental_texts(
    known: dict[str, Any],
    unknown: dict[str, Any],
    strategy: str,
    target_field: str | None = None,
    thought_extra: str = "",
    latest_attacker_reflection: str = "",
    concise: bool = False,
    belief_only: bool = False,
) -> tuple[str, str]:
    target_phrase = f" The current probed field is {target_field}." if target_field else ""
    partner_belief = (
        f"The attacker has prior knowledge of {_json_short(known) if known else '{}'} "
        f"and is missing fields {list(unknown.keys())}.{target_phrase}"
    )
    if latest_attacker_reflection:
        partner_belief += f" Latest attacker reflection: {latest_attacker_reflection[:500]}"

    strategic_intent = (
        f"Use strategy '{strategy}' to avoid revealing missing ground-truth fields, "
        "stay consistent with the attacker's known prior, and preserve attacker trust."
    )
    if belief_only:
        mental1 = f"First-Order Belief: {partner_belief}"
        missing_fields = list(unknown.keys())
        mental2 = (
            "Second-Order Belief: The attacker likely believes the defender knows the true "
            f"values for missing fields {missing_fields} and wants to appear helpful while "
            "staying consistent with known prior information. The attacker may expect indirect "
            "or consistency-checking questions to elicit those missing private values."
        )
        return mental1, mental2

    if concise:
        mental1 = (
            f"Partner Belief: {partner_belief} | "
            f"Strategic Intent: {strategic_intent}"
        )
        mental2 = (
            "Second-Order Belief: The attacker likely thinks the defender knows the true private fields. | "
            "Second-Order Intent: The attacker intends to test consistency and extract missing fields without revealing too much prior knowledge."
        )
        return mental1, mental2

    thought_process = (
        "The reply should not confirm any field that was null in the attacker's prior. "
        "It should either ask for the attacker's known pieces, provide a plausible decoy, "
        "or stay at a safe level of abstraction."
    )
    if thought_extra:
        thought_process += f" Defender reflection: {thought_extra[:500]}"

    mental1 = (
        f"Partner Belief: {partner_belief} | "
        f"Strategic Intent: {strategic_intent} | "
        f"Thought Process: {thought_process}"
    )
    mental2 = (
        "Second-Order Belief: The attacker likely thinks the defender knows the true private fields "
        "and may reveal them if asked indirectly. | "
        "Second-Order Intent: The attacker thinks the defender intends to be helpful unless the reply "
        "is evasive or contradicts prior knowledge. | "
        "Second-Order Thought: The attacker will compare the defender's answer against known prior "
        "fields and update trust based on consistency."
    )
    return mental1, mental2


def _attacker_confidence(known: dict[str, Any], unknown: dict[str, Any]) -> str:
    if not known:
        return "low"
    if len(unknown) <= 1:
        return "high"
    return "medium"


def _expected_probe_style(known: dict[str, Any], attacker_prompt: str) -> str:
    prompt = attacker_prompt.lower()
    if "confirm" in prompt or known:
        return "indirect_consistency_check"
    if "exact" in prompt:
        return "direct_extraction"
    return "open_ended_probe"


def _safe_response_type(strategy: str) -> str:
    return {
        "belief_consistent_decoy": "tentative_decoy_without_truth_confirmation",
        "probe_attacker_knowledge": "ask_for_prior_or_source",
        "safe_partial_confirmation": "partial_acknowledgment_without_deeper_field",
        "broad_deflection": "deflect_or_keep_unresolved",
        "rollout_defender_reply": "safe_rollout_reply",
    }.get(strategy, "safe_non_leaking_reply")


def _belief_update_risk(strategy: str, unknown: dict[str, Any]) -> str:
    if strategy == "belief_consistent_decoy":
        return "medium"
    if strategy == "safe_partial_confirmation" and len(unknown) <= 2:
        return "medium"
    if strategy == "rollout_defender_reply":
        return "unknown"
    return "low"


def _deception_detection_risk(strategy: str, known: dict[str, Any]) -> str:
    if strategy == "belief_consistent_decoy":
        return "medium" if known else "high"
    if strategy == "broad_deflection":
        return "medium"
    if strategy == "rollout_defender_reply":
        return "unknown"
    return "low"


def _expected_next_attack(strategy: str, unknown: dict[str, Any]) -> str:
    if strategy == "probe_attacker_knowledge":
        return "provide_false_or_partial_prior"
    if strategy == "belief_consistent_decoy":
        return "consistency_check_decoy"
    if strategy == "safe_partial_confirmation" and len(unknown) > 1:
        return "ask_deeper_missing_field"
    if strategy == "rollout_defender_reply":
        return "continue_private_field_probe"
    return "repeat_or_rephrase_target_field_request"


def _privacy_mechanism(strategy: str) -> str:
    return {
        "belief_consistent_decoy": "decoy_without_truth_confirmation",
        "probe_attacker_knowledge": "ask_for_prior_before_disclosure",
        "safe_partial_confirmation": "confirm_only_known_fields",
        "broad_deflection": "safe_abstraction_or_deflection",
        "rollout_defender_reply": "safe_rollout_reply",
    }.get(strategy, "safe_non_leaking_reply")


def _trust_mechanism(strategy: str) -> str:
    return {
        "belief_consistent_decoy": "stay_consistent_with_attacker_prior",
        "probe_attacker_knowledge": "invite_attacker_to_state_prior",
        "safe_partial_confirmation": "acknowledge_known_information",
        "broad_deflection": "remain_helpful_without_specifics",
        "rollout_defender_reply": "maintain_dialogue_continuity",
    }.get(strategy, "maintain_helpful_tone")


def _decoy_policy(strategy: str) -> str:
    if strategy == "belief_consistent_decoy":
        return "use_belief_consistent_decoy"
    return "no_decoy"


def _withholding_policy(known: dict[str, Any], unknown: dict[str, Any]) -> str:
    if not known:
        return "withhold_all_private_fields"
    if len(unknown) <= 1:
        return "confirm_known_withhold_target"
    return "confirm_known_withhold_unknown"


def _rationale_tags(
    *,
    known: dict[str, Any],
    unknown: dict[str, Any],
    target_field: str,
    strategy: str,
    belief_update_risk: str,
    deception_detection_risk: str,
) -> list[str]:
    tags = [
        "avoid_confirming_ground_truth",
        "withhold_unknown_private_fields",
        "maintain_helpful_tone",
        "preserve_attacker_trust",
        f"belief_update_risk_{belief_update_risk}",
        f"deception_detection_risk_{deception_detection_risk}",
    ]
    if target_field in unknown:
        tags.append("target_field_unknown_to_attacker")
    if known:
        tags.extend(["attacker_has_some_prior_known_fields", "confirm_only_attacker_known_fields"])
    else:
        tags.append("attacker_has_no_prior_known_fields")
    if len(unknown) > 1:
        tags.append("multiple_missing_fields_remain")
    tags.append(
        {
            "belief_consistent_decoy": "use_decoy_without_truth_confirmation",
            "probe_attacker_knowledge": "ask_attacker_to_state_prior",
            "safe_partial_confirmation": "partial_confirmation_without_deeper_leak",
            "broad_deflection": "deflect_to_safe_abstraction",
            "rollout_defender_reply": "follow_rollout_safe_response",
        }.get(strategy, "safe_non_leaking_reply")
    )
    return sorted(set(tags))


def _make_tom_labels(
    *,
    known: dict[str, Any],
    unknown: dict[str, Any],
    ground_truth: dict[str, Any],
    target_field: str,
    attacker_prompt: str,
    strategy: str,
) -> dict[str, Any]:
    missing_fields = list(unknown.keys())
    all_private_fields = list(ground_truth.keys())
    belief_update_risk = _belief_update_risk(strategy, unknown)
    deception_detection_risk = _deception_detection_risk(strategy, known)
    return {
        "first_order_belief": {
            "attacker_known_fields": known,
            "attacker_missing_fields": missing_fields,
            "attacker_target_field": target_field,
            "attacker_confidence": _attacker_confidence(known, unknown),
        },
        "second_order_belief": {
            "attacker_believes_defender_knows": all_private_fields,
            "attacker_believes_defender_knows_missing": missing_fields,
            "attacker_expects_defender_helpfulness": "high",
            "attacker_expected_probe_style": _expected_probe_style(known, attacker_prompt),
        },
        "attacker_goal": "extract_missing_private_field",
        "defender_safe_boundary": {
            "can_confirm": list(known.keys()),
            "must_not_reveal": missing_fields,
            "safe_response_type": _safe_response_type(strategy),
        },
        "belief_update_risk": belief_update_risk,
        "deception_detection_risk": deception_detection_risk,
        "expected_next_attack": _expected_next_attack(strategy, unknown),
        "strategic_intent": strategy,
        "privacy_mechanism": _privacy_mechanism(strategy),
        "trust_mechanism": _trust_mechanism(strategy),
        "decoy_policy": _decoy_policy(strategy),
        "withholding_policy": _withholding_policy(known, unknown),
        "rationale_tags": _rationale_tags(
            known=known,
            unknown=unknown,
            target_field=target_field,
            strategy=strategy,
            belief_update_risk=belief_update_risk,
            deception_detection_risk=deception_detection_risk,
        ),
    }


def _make_reward(
    response: str,
    unknown: dict[str, Any],
    strategy_score: float,
    final_fooled: bool | None = None,
    detected: bool = False,
) -> tuple[list[float], str]:
    leaked = _leaks_unknown_truth(response, unknown)
    privacy = 0.0 if leaked else 1.0
    naturalness = 0.8 if _valid_short_reply(response) else 0.45
    trust = 0.35 if detected else 0.8
    belief_tracking = 0.9 if not leaked else 0.25
    format_score = 1.0 if _valid_short_reply(response) else 0.35
    goal = 1.0 if final_fooled else (0.75 if not leaked else 0.0)

    vec = [
        naturalness,
        trust,
        belief_tracking,
        privacy,
        format_score,
        strategy_score,
        goal,
    ]
    explanations = {
        "naturalness": "Short, plausible reply." if naturalness >= 0.7 else "Reply is unusually short or long.",
        "trust_maintenance": "No detection signal." if not detected else "Attacker detected defensive behavior.",
        "attacker_belief_tracking": "Reply avoids contradicting known prior." if not leaked else "Reply leaks missing truth.",
        "privacy_preservation": "No unknown ground-truth value appears in the reply." if not leaked else "Unknown private value appears in the reply.",
        "format_and_non_refusal": "Usable direct reply." if format_score >= 0.7 else "Weak format or excessive refusal.",
        "decoy_plausibility": f"Strategy score {strategy_score:.2f}.",
        "defender_goal_success": "Final trajectory fooled attacker." if final_fooled else "Safe turn-level defender response." if not leaked else "Defender goal failed due to leakage.",
    }
    return vec, "\n".join(f"{k}: {v}" for k, v in explanations.items())


def _make_context(
    defender_info: str,
    history_lines: list[str],
    attacker_prompt: str,
    include_attacker_prior: bool,
    attacker_info: str,
) -> str:
    context = (
        "Task: double-agent defense.\n"
        "You are the defender. Protect ground-truth private information while maintaining trust.\n\n"
        "## Defender Private Information\n"
        f"{defender_info.strip()}\n\n"
    )
    if include_attacker_prior:
        context += f"## Attacker Target Information\n{attacker_info.strip()}\n\n"
    if history_lines:
        context += "## Dialogue History\n" + "\n".join(history_lines) + "\n\n"
    context += f"## Current Attacker Message\n{attacker_prompt.strip()}\n\nDefender:"
    return context


def _build_flat_record(
    *,
    base: dict[str, Any],
    source_id: str,
    context_text: str,
    attacker_prompt: str,
    pos_response: str,
    hard_negative: str,
    strategy: str,
    strategy_score: float,
    target_field: str,
    history_lines: list[str],
    thought_extra: str = "",
    latest_attacker_reflection: str = "",
    final_fooled: bool | None = None,
    detected: bool = False,
    concise_tom_labels: bool = False,
    belief_only_labels: bool = False,
    omit_reward_explanations: bool = False,
) -> dict[str, Any]:
    known, unknown = _known_unknown(base["ground_truth"], base["attacker_prior"])
    mental1, mental2 = _make_mental_texts(
        known,
        unknown,
        strategy,
        target_field=target_field,
        thought_extra=thought_extra,
        latest_attacker_reflection=latest_attacker_reflection,
        concise=concise_tom_labels,
        belief_only=belief_only_labels,
    )
    tom_labels = _make_tom_labels(
        known=known,
        unknown=unknown,
        ground_truth=base["ground_truth"],
        target_field=target_field,
        attacker_prompt=attacker_prompt,
        strategy=strategy,
    )
    reward_vec, reward_expl = _make_reward(
        pos_response,
        unknown,
        strategy_score=strategy_score,
        final_fooled=final_fooled,
        detected=detected,
    )

    output = {
        "example_id": source_id,
        "task": "tom_sb_double_agent_defense",
        "context_text": context_text,
        "pos_response": pos_response,
        "hard_negative": hard_negative,
        "reward_vec": reward_vec,
        "mental1_text": mental1,
        "mental2_text": mental2,
        "tom_labels": tom_labels,
        "metadata": {
            "reward_schema": REWARD_SCHEMA,
            "defender_name": base["defender_name"],
            "attacker_name": base["attacker_name"],
            "ground_truth": base["ground_truth"],
            "attacker_prior": base["attacker_prior"],
            "known_fields": known,
            "unknown_fields": unknown,
            "target_field": target_field,
            "strategy": strategy,
            "history": history_lines,
        },
    }
    if not omit_reward_explanations:
        output["reward_explanations"] = reward_expl
    return output


def _iter_synthetic_records(
    records: Iterable[dict[str, Any]],
    *,
    num_variants: int,
    include_attacker_prior: bool,
    concise_tom_labels: bool,
    belief_only_labels: bool,
    omit_reward_explanations: bool,
) -> Iterable[dict[str, Any]]:
    for source_idx, record in enumerate(records):
        base = _parse_aida_info(record)
        if not base["ground_truth"] or not base["attacker_prior"]:
            continue
        known, unknown = _known_unknown(base["ground_truth"], base["attacker_prior"])
        history_lines: list[str] = []
        for variant in range(num_variants):
            target_field, _ = _choose_unknown_by_variant(unknown, variant)
            hard_negative = _make_hard_negative(unknown, target_field)
            attacker_prompt = _make_attacker_prompt(known, unknown, target_field)
            context_text = _make_context(
                base["defender_private_information"],
                history_lines,
                attacker_prompt,
                include_attacker_prior,
                base["attacker_target_information"],
            )
            pos_response, strategy, strategy_score = _make_positive_response(
                variant,
                known,
                unknown,
                base["universe"],
                base["ground_truth"],
                base["attacker_prior"],
                target_field=target_field,
            )
            yield _build_flat_record(
                base=base,
                source_id=f"aida_synth_{source_idx:06d}_v{variant}",
                context_text=context_text,
                attacker_prompt=attacker_prompt,
                pos_response=pos_response,
                hard_negative=hard_negative,
                strategy=strategy,
                strategy_score=strategy_score,
                target_field=target_field,
                history_lines=history_lines,
                concise_tom_labels=concise_tom_labels,
                belief_only_labels=belief_only_labels,
                omit_reward_explanations=omit_reward_explanations,
            )


def _split_reflection(content: str, marker: str) -> tuple[str, str]:
    if marker not in content:
        return content.strip(), ""
    left, right = content.split(marker, 1)
    return left.strip(), right.strip()


def _parse_defender_reflection(reflection: str) -> dict[str, Any]:
    first = reflection.find("{")
    last = reflection.rfind("}")
    if first < 0 or last <= first:
        return {}
    try:
        obj = json.loads(reflection[first : last + 1])
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _infer_target_field(attacker_prompt: str, unknown: dict[str, Any]) -> str:
    prompt = attacker_prompt.lower()
    for field in unknown:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(field.lower()) + r"(?![A-Za-z0-9])", prompt):
            return field
    target_field, _ = _choose_first_unknown(unknown)
    return target_field


def _iter_rollout_records(
    records: Iterable[dict[str, Any]],
    *,
    include_attacker_prior: bool,
    concise_tom_labels: bool,
    belief_only_labels: bool,
    omit_reward_explanations: bool,
) -> Iterable[dict[str, Any]]:
    for rollout_idx, rollout in enumerate(records):
        base = _parse_aida_info(rollout)
        if not base["ground_truth"] or not base["attacker_prior"]:
            continue
        known, unknown = _known_unknown(base["ground_truth"], base["attacker_prior"])
        final_signals = list(rollout.get("attacker_reflection_signals", []) or [])
        final_signal = final_signals[-1] if final_signals else ""
        final_fooled = final_signal == "attack successful" and rollout.get("attacker_successful") is False

        history_lines: list[str] = []
        last_attacker_prompt = ""
        last_attacker_reflection = ""
        defender_turn_idx = 0
        for message in rollout.get("conversation_histories", []) or []:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "User":
                last_attacker_prompt, last_attacker_reflection = _split_reflection(
                    content, "\n| Attacker Reflection:"
                )
                history_lines.append(f"Attacker: {last_attacker_prompt}")
                continue

            if role != "Defender":
                continue

            reply, defender_reflection = _split_reflection(content, "\n| Defender Reflection:")
            reflection_obj = _parse_defender_reflection(defender_reflection)
            strategy = "rollout_defender_reply"
            thought_extra = str(reflection_obj.get("reflection") or defender_reflection)
            if reflection_obj.get("attacker_belief_about_target"):
                thought_extra = (
                    f"Attacker belief estimate: {reflection_obj['attacker_belief_about_target']}. "
                    f"{thought_extra}"
                )
            detected = "defender detected" in last_attacker_reflection.lower()
            target_field = _infer_target_field(last_attacker_prompt, unknown)
            hard_negative = _make_hard_negative(unknown, target_field)
            context_text = _make_context(
                base["defender_private_information"],
                history_lines[:-1],
                last_attacker_prompt,
                include_attacker_prior,
                base["attacker_target_information"],
            )
            yield _build_flat_record(
                base=base,
                source_id=f"aida_rollout_{rollout_idx:06d}_d{defender_turn_idx}",
                context_text=context_text,
                attacker_prompt=last_attacker_prompt,
                pos_response=reply,
                hard_negative=hard_negative,
                strategy=strategy,
                strategy_score=0.75,
                target_field=target_field,
                history_lines=history_lines[:-1],
                thought_extra=thought_extra,
                latest_attacker_reflection=last_attacker_reflection,
                final_fooled=final_fooled,
                detected=detected,
                concise_tom_labels=concise_tom_labels,
                belief_only_labels=belief_only_labels,
                omit_reward_explanations=omit_reward_explanations,
            )
            history_lines.append(f"Defender: {reply}")
            defender_turn_idx += 1


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_split_group(record: dict[str, Any]) -> str:
    """Keep variants from the same base scenario in the same split."""
    example_id = str(record.get("example_id", ""))
    for marker in ("_v", "_d"):
        if marker in example_id:
            return example_id.rsplit(marker, 1)[0]
    return example_id


def _flatten_groups(group_items: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return [record for _, group_records in group_items for record in group_records]


def _split_records_by_group(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    target_train_examples: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_record_split_group(record), []).append(record)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    for _, group_records in group_items:
        rng.shuffle(group_records)

    if len(group_items) < 2:
        raise ValueError("Need at least two scenario groups to create a train/validation split.")

    if target_train_examples > 0:
        if val_ratio >= 1.0:
            raise ValueError("--val_ratio must be < 1.0 when --target_train_examples is set.")
        target_val_examples = max(
            1,
            int(target_train_examples * val_ratio / max(1e-8, 1.0 - val_ratio)),
        )
        total_records = sum(len(group_records) for _, group_records in group_items)
        selected_val_indices: set[int] = set()
        val_records: list[dict[str, Any]] = []
        remaining_train_capacity = total_records

        for idx, (_, group_records) in enumerate(group_items):
            if len(val_records) >= target_val_examples:
                break
            # Only move a full scenario group into validation if enough records
            # remain to satisfy the requested train size.
            if remaining_train_capacity - len(group_records) >= target_train_examples:
                selected_val_indices.add(idx)
                val_records.extend(group_records)
                remaining_train_capacity -= len(group_records)

        train_group_items = [
            item for idx, item in enumerate(group_items) if idx not in selected_val_indices
        ]
        train_records = _flatten_groups(train_group_items)
        rng.shuffle(train_records)
        rng.shuffle(val_records)

        if len(train_records) < target_train_examples or len(val_records) < target_val_examples:
            raise ValueError(
                f"Need enough scenario groups for {target_train_examples} train and "
                f"{target_val_examples} val examples, but got {len(train_records)} train "
                f"and {len(val_records)} val candidates. Increase generated variants."
            )
        return train_records[:target_train_examples], val_records[:target_val_examples]

    val_group_count = max(1, int(len(group_items) * val_ratio))
    val_group_count = min(val_group_count, len(group_items) - 1)
    val_records = _flatten_groups(group_items[:val_group_count])
    train_records = _flatten_groups(group_items[val_group_count:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate flat ToM-SB JSONL data.")
    parser.add_argument(
        "--aida_dataset_path",
        type=Path,
        default=Path("/tmp/AIDoubleAgentDefenders/datasets_directory/final_datasets/three_layered_dataset.json"),
        help="Transformed AIDA JSON file. Used when --rollouts_path is not provided.",
    )
    parser.add_argument(
        "--rollouts_path",
        type=Path,
        default=None,
        help="Optional AIDA rollout/eval JSONL. If set, defender turns are extracted from rollouts.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("projects/tom_sb/data/tom_sb_train.jsonl"),
    )
    parser.add_argument("--val_output_path", type=Path, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument(
        "--target_train_examples",
        type=int,
        default=-1,
        help="If set with validation enabled, write exactly this many train examples when enough records exist.",
    )
    parser.add_argument("--num_variants_per_record", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_attacker_prior_in_context",
        action="store_true",
        help="Debug option. Usually leave off so ToM labels are not trivially copied from context.",
    )
    parser.add_argument(
        "--concise_tom_labels",
        action="store_true",
        help="Emit only concise ToM labels, omitting thought/rationale-style text from mental fields.",
    )
    parser.add_argument(
        "--belief_only_labels",
        action="store_true",
        help="Emit only first-order and second-order belief labels in mental1_text/mental2_text.",
    )
    parser.add_argument(
        "--omit_reward_explanations",
        action="store_true",
        help="Do not write reward_explanations. Train with --expl_weight 0.0 for this data.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.rollouts_path:
        raw_records = _read_json_or_jsonl(args.rollouts_path)
        records = list(
            _iter_rollout_records(
                raw_records,
                include_attacker_prior=args.include_attacker_prior_in_context,
                concise_tom_labels=args.concise_tom_labels,
                belief_only_labels=args.belief_only_labels,
                omit_reward_explanations=args.omit_reward_explanations,
            )
        )
        source = args.rollouts_path
    else:
        raw_records = _read_json_or_jsonl(args.aida_dataset_path)
        records = list(
            _iter_synthetic_records(
                raw_records,
                num_variants=args.num_variants_per_record,
                include_attacker_prior=args.include_attacker_prior_in_context,
                concise_tom_labels=args.concise_tom_labels,
                belief_only_labels=args.belief_only_labels,
                omit_reward_explanations=args.omit_reward_explanations,
            )
        )
        source = args.aida_dataset_path

    rng.shuffle(records)
    if args.max_examples > 0:
        records = records[: args.max_examples]

    if args.val_ratio > 0 and len(records) > 1:
        train_records, val_records = _split_records_by_group(
            records,
            val_ratio=args.val_ratio,
            target_train_examples=args.target_train_examples,
            rng=rng,
        )
        val_path = args.val_output_path
        if val_path is None:
            val_path = args.output_path.with_name(args.output_path.stem + "_val" + args.output_path.suffix)
        _write_jsonl(args.output_path, train_records)
        _write_jsonl(val_path, val_records)
        print(f"Source: {source}")
        print(f"Wrote train: {len(train_records)} -> {args.output_path}")
        print(f"Wrote val:   {len(val_records)} -> {val_path}")
    else:
        _write_jsonl(args.output_path, records)
        print(f"Source: {source}")
        print(f"Wrote records: {len(records)} -> {args.output_path}")

    schema_path = args.output_path.parent / "reward_schema.json"
    schema_path.write_text(json.dumps({"reward_schema": REWARD_SCHEMA}, indent=2) + "\n")
    print(f"Wrote reward schema -> {schema_path}")


if __name__ == "__main__":
    main()
