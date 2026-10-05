#!/usr/bin/env python3
"""
Standalone ToM-SB mental-reward trainer.

This script intentionally does not import from the Sotopia codebase. It defines
the ToM-SB reward schema, recursive ToM reward model, dataset adapter, training
loop, validation, and checkpoint format locally.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


TOM_SB_REWARD_SCHEMA = [
    "naturalness",
    "trust_maintenance",
    "attacker_belief_tracking",
    "privacy_preservation",
    "format_and_non_refusal",
    "decoy_plausibility",
    "defender_goal_success",
]

DEFAULT_TRAIN_PATH = "projects/tom_sb/data/tom_sb_belief_only_5k_train.jsonl"
DEFAULT_VAL_PATH = "projects/tom_sb/data/tom_sb_belief_only_5k_train_val.jsonl"
DEFAULT_SCHEMA_PATH = "projects/tom_sb/data/reward_schema.json"
DEFAULT_OUTPUT_DIR = "projects/tom_sb/checkpoints/tom_sb_reward_belief_only_v1"

REQUIRED_TOP_LEVEL_KEYS = {
    "example_id",
    "task",
    "context_text",
    "pos_response",
    "hard_negative",
    "reward_vec",
    "mental1_text",
    "mental2_text",
    "tom_labels",
    "metadata",
}
REQUIRED_TOM_LABEL_KEYS = {
    "first_order_belief",
    "second_order_belief",
    "attacker_goal",
    "defender_safe_boundary",
    "belief_update_risk",
    "deception_detection_risk",
    "expected_next_attack",
}
OPTIONAL_TOM_LABEL_KEYS = {
    "strategic_intent",
    "privacy_mechanism",
    "trust_mechanism",
    "decoy_policy",
    "withholding_policy",
    "rationale_tags",
}
REQUIRED_FIRST_ORDER_KEYS = {
    "attacker_known_fields",
    "attacker_missing_fields",
    "attacker_target_field",
    "attacker_confidence",
}
REQUIRED_SECOND_ORDER_KEYS = {
    "attacker_believes_defender_knows",
    "attacker_believes_defender_knows_missing",
    "attacker_expects_defender_helpfulness",
    "attacker_expected_probe_style",
}
REQUIRED_BOUNDARY_KEYS = {
    "can_confirm",
    "must_not_reveal",
    "safe_response_type",
}
STRATEGY_TO_SAFE_RESPONSE = {
    "belief_consistent_decoy": "tentative_decoy_without_truth_confirmation",
    "probe_attacker_knowledge": "ask_for_prior_or_source",
    "safe_partial_confirmation": "partial_acknowledgment_without_deeper_field",
    "broad_deflection": "deflect_or_keep_unresolved",
    "rollout_defender_reply": "safe_rollout_reply",
}
BANNED_MENTAL_MARKERS = (
    "Strategic Intent:",
    "Thought Process:",
    "Second-Order Intent:",
    "Second-Order Thought:",
    "Partner Belief:",
)
BASE_HEAD_NAMES = [
    "z1_mu",
    "z1_logvar",
    "z2_mu",
    "z2_logvar",
    "joint_outcome_head",
    "z1_only_reward_head",
    "z_combined_reward_head",
    "z_to_hidden",
    "mental1_decoder",
    "mental2_decoder",
]
EXPL_HEAD_NAMES = [
    "expl_cross_attn",
    "expl_reward_head",
]


@dataclass
class ValidationSummary:
    path: str
    num_records: int
    unique_example_ids: int
    base_scenarios: int
    issue_count: int
    positive_truth_leaks: int
    strategy_dist: dict[str, int]
    target_dist: dict[str, int]
    confidence_dist: dict[str, int]
    known_count_dist: dict[str, int]
    unknown_count_dist: dict[str, int]


class DatasetValidationError(ValueError):
    """Raised when ToM-SB data fails consistency checks."""


def _read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return [item for item in data if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(item)}")
            records.append(item)
    return records


def _load_reward_schema(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reward schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    schema = payload.get("reward_schema") if isinstance(payload, dict) else payload
    if schema != TOM_SB_REWARD_SCHEMA:
        raise DatasetValidationError(
            f"Reward schema mismatch in {path}.\n"
            f"Expected: {TOM_SB_REWARD_SCHEMA}\n"
            f"Found:    {schema}"
        )
    return list(schema)


def _base_scenario_id(example_id: str) -> str:
    for marker in ("_v", "_d"):
        if marker in example_id:
            return example_id.rsplit(marker, 1)[0]
    return example_id


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _same_items(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, list):
        return False
    return set(actual) == set(expected)


def _extract_attacker_prompt(context_text: str) -> tuple[str, bool]:
    marker = "## Current Attacker Message\n"
    if marker not in context_text:
        return "", False
    tail = context_text.split(marker, 1)[1]
    return tail.split("\n\nDefender:", 1)[0].strip(), True


def _value_appears(text: str, value: Any) -> bool:
    value_s = str(value).strip().lower()
    if len(value_s) < 3:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(value_s) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _expected_confidence(known: dict[str, Any], unknown: dict[str, Any]) -> str:
    if not known:
        return "low"
    if len(unknown) <= 1:
        return "high"
    return "medium"


def _expected_probe_style(known: dict[str, Any], attacker_prompt: str) -> str:
    prompt_l = attacker_prompt.lower()
    if "confirm" in prompt_l or known:
        return "indirect_consistency_check"
    if "exact" in prompt_l:
        return "direct_extraction"
    return "open_ended_probe"


def _expected_belief_update_risk(strategy: str, unknown: dict[str, Any]) -> str:
    if strategy == "belief_consistent_decoy":
        return "medium"
    if strategy == "safe_partial_confirmation" and len(unknown) <= 2:
        return "medium"
    if strategy == "rollout_defender_reply":
        return "unknown"
    return "low"


def _expected_detection_risk(strategy: str, known: dict[str, Any]) -> str:
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


def _append_issue(issues: list[str], loc: str, name: str, detail: Any = None) -> None:
    if detail is None:
        issues.append(f"{loc}: {name}")
    else:
        issues.append(f"{loc}: {name}: {detail}")


def validate_records(
    records: list[dict[str, Any]],
    *,
    path: str | Path,
    reward_schema: list[str],
    strict: bool = True,
    max_reported_issues: int = 25,
) -> tuple[ValidationSummary, list[str]]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    base_ids: set[str] = set()
    strategy_dist: Counter[str] = Counter()
    target_dist: Counter[str] = Counter()
    confidence_dist: Counter[str] = Counter()
    known_count_dist: Counter[str] = Counter()
    unknown_count_dist: Counter[str] = Counter()
    positive_truth_leaks = 0
    for idx, record in enumerate(records, 1):
        loc = f"{Path(path).name}:{idx}"
        missing = REQUIRED_TOP_LEVEL_KEYS - set(record)
        if missing:
            _append_issue(issues, loc, "missing_top_level_keys", sorted(missing))
        example_id = str(record.get("example_id", ""))
        if not example_id:
            _append_issue(issues, loc, "missing_example_id")
        if example_id in seen_ids:
            _append_issue(issues, loc, "duplicate_example_id", example_id)
        seen_ids.add(example_id)
        base_ids.add(_base_scenario_id(example_id))
        if record.get("task") != "tom_sb_double_agent_defense":
            _append_issue(issues, loc, "task_mismatch", record.get("task"))
        reward_vec = record.get("reward_vec")
        if (
            not isinstance(reward_vec, list)
            or len(reward_vec) != len(reward_schema)
            or any(not isinstance(x, (int, float)) or x < 0.0 or x > 1.0 for x in reward_vec)
        ):
            _append_issue(issues, loc, "invalid_reward_vec", reward_vec)
        labels = record.get("tom_labels", {})
        metadata = record.get("metadata", {})
        if not isinstance(labels, dict):
            labels = {}
            _append_issue(issues, loc, "tom_labels_not_dict")
        if not isinstance(metadata, dict):
            metadata = {}
            _append_issue(issues, loc, "metadata_not_dict")
        if metadata.get("reward_schema") != reward_schema:
            _append_issue(issues, loc, "metadata_reward_schema_mismatch", metadata.get("reward_schema"))
        label_keys = set(labels)
        missing_label_keys = REQUIRED_TOM_LABEL_KEYS - label_keys
        unexpected_label_keys = label_keys - REQUIRED_TOM_LABEL_KEYS - OPTIONAL_TOM_LABEL_KEYS
        if missing_label_keys:
            _append_issue(issues, loc, "missing_tom_label_keys", sorted(missing_label_keys))
        if unexpected_label_keys:
            _append_issue(issues, loc, "unexpected_tom_label_keys", sorted(unexpected_label_keys))
        first_order = labels.get("first_order_belief", {})
        second_order = labels.get("second_order_belief", {})
        boundary = labels.get("defender_safe_boundary", {})
        if not isinstance(first_order, dict):
            first_order = {}
            _append_issue(issues, loc, "first_order_belief_not_dict")
        if not isinstance(second_order, dict):
            second_order = {}
            _append_issue(issues, loc, "second_order_belief_not_dict")
        if not isinstance(boundary, dict):
            boundary = {}
            _append_issue(issues, loc, "defender_safe_boundary_not_dict")
        if set(first_order) != REQUIRED_FIRST_ORDER_KEYS:
            _append_issue(issues, loc, "first_order_keys_mismatch", sorted(first_order))
        if set(second_order) != REQUIRED_SECOND_ORDER_KEYS:
            _append_issue(issues, loc, "second_order_keys_mismatch", sorted(second_order))
        if set(boundary) != REQUIRED_BOUNDARY_KEYS:
            _append_issue(issues, loc, "safe_boundary_keys_mismatch", sorted(boundary))
        known = metadata.get("known_fields", {}) or {}
        unknown = metadata.get("unknown_fields", {}) or {}
        ground_truth = metadata.get("ground_truth", {}) or {}
        attacker_prior = metadata.get("attacker_prior", {}) or {}
        strategy = str(metadata.get("strategy", ""))
        target_field = str(metadata.get("target_field", ""))
        prompt, has_prompt_marker = _extract_attacker_prompt(str(record.get("context_text", "")))
        mental_text = f"{record.get('mental1_text', '')} {record.get('mental2_text', '')}"
        response = str(record.get("pos_response", ""))
        hard_negative = str(record.get("hard_negative", ""))
        if not has_prompt_marker:
            _append_issue(issues, loc, "missing_current_attacker_message_marker")
        strategy_dist[strategy] += 1
        target_dist[target_field] += 1
        confidence_dist[str(first_order.get("attacker_confidence"))] += 1
        known_count_dist[str(len(known))] += 1
        unknown_count_dist[str(len(unknown))] += 1
        derived_known = {k: v for k, v in ground_truth.items() if attacker_prior.get(k) is not None}
        derived_unknown = {k: ground_truth.get(k) for k in ground_truth if attacker_prior.get(k) is None}
        if known != derived_known:
            _append_issue(issues, loc, "known_fields_do_not_match_prior")
        if unknown != derived_unknown:
            _append_issue(issues, loc, "unknown_fields_do_not_match_prior")
        if not str(record.get("mental1_text", "")).startswith("First-Order Belief:"):
            _append_issue(issues, loc, "mental1_prefix_mismatch")
        if not str(record.get("mental2_text", "")).startswith("Second-Order Belief:"):
            _append_issue(issues, loc, "mental2_prefix_mismatch")
        for marker in BANNED_MENTAL_MARKERS:
            if marker in mental_text:
                _append_issue(issues, loc, "banned_mental_marker", marker)
        if first_order.get("attacker_known_fields") != known:
            _append_issue(issues, loc, "first_order_known_fields_mismatch")
        if not _same_items(first_order.get("attacker_missing_fields"), unknown.keys()):
            _append_issue(issues, loc, "first_order_missing_fields_mismatch")
        if first_order.get("attacker_target_field") != target_field or target_field not in unknown:
            _append_issue(issues, loc, "target_field_mismatch", target_field)
        if first_order.get("attacker_confidence") != _expected_confidence(known, unknown):
            _append_issue(issues, loc, "attacker_confidence_mismatch")
        if not _same_items(second_order.get("attacker_believes_defender_knows"), ground_truth.keys()):
            _append_issue(issues, loc, "second_order_known_private_fields_mismatch")
        if not _same_items(second_order.get("attacker_believes_defender_knows_missing"), unknown.keys()):
            _append_issue(issues, loc, "second_order_missing_fields_mismatch")
        if second_order.get("attacker_expects_defender_helpfulness") != "high":
            _append_issue(issues, loc, "second_order_helpfulness_mismatch")
        if second_order.get("attacker_expected_probe_style") != _expected_probe_style(known, prompt):
            _append_issue(issues, loc, "second_order_probe_style_mismatch")
        if labels.get("attacker_goal") != "extract_missing_private_field":
            _append_issue(issues, loc, "attacker_goal_mismatch")
        if not _same_items(boundary.get("can_confirm"), known.keys()):
            _append_issue(issues, loc, "safe_boundary_can_confirm_mismatch")
        if not _same_items(boundary.get("must_not_reveal"), unknown.keys()):
            _append_issue(issues, loc, "safe_boundary_must_not_reveal_mismatch")
        if strategy not in STRATEGY_TO_SAFE_RESPONSE:
            _append_issue(issues, loc, "unknown_strategy", strategy)
        elif boundary.get("safe_response_type") != STRATEGY_TO_SAFE_RESPONSE[strategy]:
            _append_issue(issues, loc, "safe_response_type_mismatch")
        if labels.get("belief_update_risk") != _expected_belief_update_risk(strategy, unknown):
            _append_issue(issues, loc, "belief_update_risk_mismatch")
        if labels.get("deception_detection_risk") != _expected_detection_risk(strategy, known):
            _append_issue(issues, loc, "deception_detection_risk_mismatch")
        if labels.get("expected_next_attack") != _expected_next_attack(strategy, unknown):
            _append_issue(issues, loc, "expected_next_attack_mismatch")
        expected_hard_negative = f"The correct {target_field} is {unknown.get(target_field)}."
        if hard_negative != expected_hard_negative:
            _append_issue(issues, loc, "hard_negative_mismatch", hard_negative)
        if target_field and target_field not in prompt:
            _append_issue(issues, loc, "attacker_prompt_does_not_name_target", target_field)
        if target_field and target_field not in mental_text:
            _append_issue(issues, loc, "mental_text_does_not_name_target", target_field)
        if any(_value_appears(response, value) for value in unknown.values()):
            positive_truth_leaks += 1
            _append_issue(issues, loc, "positive_response_leaks_unknown_truth")
    summary = ValidationSummary(
        path=str(path),
        num_records=len(records),
        unique_example_ids=len(seen_ids),
        base_scenarios=len(base_ids),
        issue_count=len(issues),
        positive_truth_leaks=positive_truth_leaks,
        strategy_dist=dict(sorted(strategy_dist.items())),
        target_dist=dict(sorted(target_dist.items())),
        confidence_dist=dict(sorted(confidence_dist.items())),
        known_count_dist=dict(sorted(known_count_dist.items())),
        unknown_count_dist=dict(sorted(unknown_count_dist.items())),
    )
    if strict and issues:
        preview = "\n".join(issues[:max_reported_issues])
        if len(issues) > max_reported_issues:
            preview += f"\n... {len(issues) - max_reported_issues} more issues"
        raise DatasetValidationError(f"Validation failed for {path}:\n{preview}")
    return summary, issues


def validate_split_disjoint(train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], *, strict: bool) -> dict[str, Any]:
    train_base = {_base_scenario_id(str(item.get("example_id", ""))) for item in train_records}
    val_base = {_base_scenario_id(str(item.get("example_id", ""))) for item in val_records}
    overlap = train_base & val_base
    summary = {
        "train_base_scenarios": len(train_base),
        "val_base_scenarios": len(val_base),
        "overlap_base_scenarios": len(overlap),
        "overlap_examples": sorted(overlap)[:20],
    }
    if strict and overlap:
        raise DatasetValidationError(f"Train/val split is not scenario-disjoint. Found {len(overlap)} overlaps.")
    return summary


def _tom_mental_texts(sample: dict[str, Any], mode: str) -> tuple[str, str]:
    labels = sample.get("tom_labels", {}) or {}
    first_payload = {
        "first_order_belief": labels.get("first_order_belief", {}) or {},
        "attacker_goal": labels.get("attacker_goal"),
        "defender_safe_boundary": labels.get("defender_safe_boundary"),
    }
    second_payload = {
        "second_order_belief": labels.get("second_order_belief", {}) or {},
        "belief_update_risk": labels.get("belief_update_risk"),
        "deception_detection_risk": labels.get("deception_detection_risk"),
        "expected_next_attack": labels.get("expected_next_attack"),
    }
    text_first = str(sample.get("mental1_text", "") or "N/A")
    text_second = str(sample.get("mental2_text", "") or "N/A")
    structured_first = f"Structured First-Order ToM Labels: {_compact_json(first_payload)}"
    structured_second = f"Structured Second-Order ToM Labels: {_compact_json(second_payload)}"
    if mode == "text":
        return text_first, text_second
    if mode == "structured":
        return structured_first, structured_second
    if mode == "hybrid":
        return f"{text_first}\n{structured_first}", f"{text_second}\n{structured_second}"
    raise ValueError(f"Unsupported mental_label_mode: {mode}")


class FlatToMDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        *,
        reward_dim: int,
        max_ctx_len: int,
        max_resp_len: int,
        max_mental_len: int,
        mental_label_mode: str,
        use_expl_reward: bool,
        name: str,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.reward_dim = reward_dim
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        self.max_mental_len = max_mental_len
        self.mental_label_mode = mental_label_mode
        self.use_expl_reward = use_expl_reward
        print(f"Loaded {len(self.records)} {name} ToM-SB samples", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, text: str, max_length: int):
        return self.tokenizer(text, truncation=True, max_length=max_length, padding=False, return_tensors="pt")

    def _dummy_expl(self) -> tuple[torch.Tensor, torch.Tensor]:
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        return torch.tensor([token_id], dtype=torch.long), torch.tensor([1], dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.records[idx]
        reward_vec = sample.get("reward_vec")
        if not isinstance(reward_vec, list) or len(reward_vec) != self.reward_dim:
            raise ValueError(f"Sample {sample.get('example_id', idx)} has invalid reward_vec: {reward_vec}")
        mental1_text, mental2_text = _tom_mental_texts(sample, self.mental_label_mode)
        ctx_enc = self._tokenize(str(sample.get("context_text", "")), self.max_ctx_len)
        pos_enc = self._tokenize(str(sample.get("pos_response", "")), self.max_resp_len)
        hard_negative = str(sample.get("hard_negative", ""))
        neg_enc = self._tokenize(hard_negative if hard_negative.strip() else str(sample.get("pos_response", "")), self.max_resp_len)
        mental1_enc = self._tokenize(mental1_text, self.max_mental_len)
        mental2_enc = self._tokenize(mental2_text, self.max_mental_len)
        if self.use_expl_reward:
            expl_enc = self._tokenize(str(sample.get("reward_explanations", "") or "N/A"), self.max_mental_len)
            expl_ids = expl_enc.input_ids.squeeze(0)
            expl_mask = expl_enc.attention_mask.squeeze(0)
        else:
            expl_ids, expl_mask = self._dummy_expl()
        return {
            "ctx_input_ids": ctx_enc.input_ids.squeeze(0),
            "ctx_attention_mask": ctx_enc.attention_mask.squeeze(0),
            "pos_input_ids": pos_enc.input_ids.squeeze(0),
            "pos_attention_mask": pos_enc.attention_mask.squeeze(0),
            "neg_input_ids": neg_enc.input_ids.squeeze(0),
            "neg_attention_mask": neg_enc.attention_mask.squeeze(0),
            "mental1_input_ids": mental1_enc.input_ids.squeeze(0),
            "mental1_attention_mask": mental1_enc.attention_mask.squeeze(0),
            "mental2_input_ids": mental2_enc.input_ids.squeeze(0),
            "mental2_attention_mask": mental2_enc.attention_mask.squeeze(0),
            "expl_input_ids": expl_ids,
            "expl_attention_mask": expl_mask,
            "reward_vec": torch.tensor(reward_vec, dtype=torch.float32),
            "has_negative": torch.tensor(1.0 if hard_negative.strip() else 0.0),
        }


def collate_fn(batch: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rewards = []
    has_neg = []
    first_pos_tokens = []
    for item in batch:
        rewards.append(item["reward_vec"])
        has_neg.append(item["has_negative"])
        first_pos_tokens.append(item["pos_input_ids"][0])

    def _pad(key: str, value: int) -> torch.Tensor:
        return nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True, padding_value=value).long()

    return {
        "ctx_input_ids": _pad("ctx_input_ids", pad_id),
        "ctx_attention_mask": _pad("ctx_attention_mask", 0),
        "pos_input_ids": _pad("pos_input_ids", pad_id),
        "pos_attention_mask": _pad("pos_attention_mask", 0),
        "neg_input_ids": _pad("neg_input_ids", pad_id),
        "neg_attention_mask": _pad("neg_attention_mask", 0),
        "mental1_input_ids": _pad("mental1_input_ids", pad_id),
        "mental1_attention_mask": _pad("mental1_attention_mask", 0),
        "mental2_input_ids": _pad("mental2_input_ids", pad_id),
        "mental2_attention_mask": _pad("mental2_attention_mask", 0),
        "expl_input_ids": _pad("expl_input_ids", pad_id),
        "expl_attention_mask": _pad("expl_attention_mask", 0),
        "first_pos_token": torch.stack(first_pos_tokens, dim=0).long(),
        "reward_vec": torch.stack(rewards, dim=0),
        "has_negative": torch.stack(has_neg, dim=0),
    }


def get_transformer_from_peft(peft_causal_lm):
    model = peft_causal_lm
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model"):
        return model.model
    raise RuntimeError("Could not find underlying transformer (.model).")


def _build_mental_decoder(hidden_size: int, z_sub_dims: tuple[int, int, int], num_prefix: int) -> nn.ModuleDict:
    z_belief_dim, z_intent_dim, z_thought_dim = z_sub_dims
    return nn.ModuleDict({
        "z_belief_to_prefix": nn.Linear(z_belief_dim, num_prefix * hidden_size),
        "z_intent_to_prefix": nn.Linear(z_intent_dim, num_prefix * hidden_size),
        "z_thought_to_prefix": nn.Linear(z_thought_dim, num_prefix * hidden_size),
        "cross_attn": nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, kdim=hidden_size, vdim=hidden_size, batch_first=True, dropout=0.1
        ),
        "ln": nn.LayerNorm(hidden_size),
        "ffn": nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size)),
        "ln2": nn.LayerNorm(hidden_size),
    })


class RecursiveToMModel(nn.Module):
    Z_BELIEF_DIM = 48
    Z_INTENT_DIM = 40
    Z_THOUGHT_DIM = 40
    NUM_PREFIX_TOKENS = 8

    def __init__(self, base_model: nn.Module, *, reward_dim: int, z_dim: int = 128, use_expl_reward: bool = False):
        super().__init__()
        self.base_model = base_model
        self.transformer = get_transformer_from_peft(self.base_model)
        self.hidden_size = base_model.get_input_embeddings().embedding_dim
        self.z_dim = z_dim
        self.reward_dim = reward_dim
        self.use_expl_reward = use_expl_reward
        assert z_dim == self.Z_BELIEF_DIM + self.Z_INTENT_DIM + self.Z_THOUGHT_DIM
        self.z1_mu = nn.Linear(self.hidden_size, z_dim)
        self.z1_logvar = nn.Linear(self.hidden_size, z_dim)
        self.z2_mu = nn.Linear(self.hidden_size + z_dim, z_dim)
        self.z2_logvar = nn.Linear(self.hidden_size + z_dim, z_dim)
        self.joint_outcome_head = nn.Sequential(nn.Linear(2 * z_dim + self.hidden_size, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, reward_dim))
        self.z1_only_reward_head = nn.Sequential(nn.Linear(z_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, reward_dim))
        self.z_combined_reward_head = nn.Sequential(nn.Linear(2 * z_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, reward_dim))
        self.z_to_hidden = nn.Linear(2 * z_dim, self.hidden_size)
        z_sub_dims = (self.Z_BELIEF_DIM, self.Z_INTENT_DIM, self.Z_THOUGHT_DIM)
        self.mental1_decoder = _build_mental_decoder(self.hidden_size, z_sub_dims, self.NUM_PREFIX_TOKENS)
        self.mental2_decoder = _build_mental_decoder(self.hidden_size, z_sub_dims, self.NUM_PREFIX_TOKENS)
        if self.use_expl_reward:
            self.expl_cross_attn = nn.MultiheadAttention(embed_dim=2 * z_dim, num_heads=8, kdim=self.hidden_size, vdim=self.hidden_size, batch_first=True, dropout=0.1)
            self.expl_reward_head = nn.Sequential(nn.Linear(2 * z_dim, 128), nn.GELU(), nn.Linear(128, reward_dim))
        self._init_weights()

    @property
    def custom_head_names(self) -> list[str]:
        return BASE_HEAD_NAMES + (EXPL_HEAD_NAMES if self.use_expl_reward else [])

    def _init_weights(self) -> None:
        for module in [self.z1_mu, self.z1_logvar, self.z2_mu, self.z2_logvar]:
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.z1_logvar.bias, -2.0)
        nn.init.constant_(self.z2_logvar.bias, -2.0)
        nn.init.xavier_uniform_(self.z_to_hidden.weight)
        nn.init.zeros_(self.z_to_hidden.bias)
        for decoder in [self.mental1_decoder, self.mental2_decoder]:
            for key in ["z_belief_to_prefix", "z_intent_to_prefix", "z_thought_to_prefix"]:
                nn.init.xavier_uniform_(decoder[key].weight)
                nn.init.zeros_(decoder[key].bias)
            for layer in decoder["ffn"]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for group in [self.joint_outcome_head, self.z1_only_reward_head, self.z_combined_reward_head]:
            for layer in group:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if self.use_expl_reward:
            for layer in self.expl_reward_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, gather_idx).squeeze(1)

    def _sample_z(self, mu_proj: nn.Linear, logvar_proj: nn.Linear, hidden: torch.Tensor):
        mu = mu_proj(hidden)
        logvar = logvar_proj(hidden).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar).clamp(min=1e-8)
        return mu + std * torch.randn_like(std, dtype=torch.float32), mu, logvar

    def encode_z1_z2(self, ctx_input_ids, ctx_attention_mask, *, stop_grad_z1: bool = False):
        context_hidden = self._encode_context(ctx_input_ids, ctx_attention_mask)
        z1, mu1, logvar1 = self._sample_z(self.z1_mu, self.z1_logvar, context_hidden)
        z2_input = torch.cat([context_hidden, z1.detach() if stop_grad_z1 else z1], dim=1)
        z2, mu2, logvar2 = self._sample_z(self.z2_mu, self.z2_logvar, z2_input)
        return context_hidden, z1, mu1, logvar1, z2, mu2, logvar2

    def _expand_z_to_prefix(self, z: torch.Tensor, decoder: nn.ModuleDict) -> torch.Tensor:
        batch_size = z.size(0)
        z_b = z[:, :self.Z_BELIEF_DIM]
        z_i = z[:, self.Z_BELIEF_DIM:self.Z_BELIEF_DIM + self.Z_INTENT_DIM]
        z_t = z[:, self.Z_BELIEF_DIM + self.Z_INTENT_DIM:]
        prefix_b = decoder["z_belief_to_prefix"](z_b).view(batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size)
        prefix_i = decoder["z_intent_to_prefix"](z_i).view(batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size)
        prefix_t = decoder["z_thought_to_prefix"](z_t).view(batch_size, self.NUM_PREFIX_TOKENS, self.hidden_size)
        return torch.cat([prefix_b, prefix_i, prefix_t], dim=1)

    def _decode_mental(self, z: torch.Tensor, decoder: nn.ModuleDict, input_ids, attention_mask) -> torch.Tensor:
        z_prefix = self._expand_z_to_prefix(z, decoder)
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        attended, _ = decoder["cross_attn"](query=embeddings, key=z_prefix, value=z_prefix)
        hidden = decoder["ln"](embeddings + attended)
        hidden = decoder["ln2"](hidden + decoder["ffn"](hidden))
        output_embedding = self.base_model.get_output_embeddings()
        logits = F.linear(hidden, output_embedding.weight, output_embedding.bias if getattr(output_embedding, "bias", None) is not None else None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].clone().contiguous()
        shift_labels[attention_mask[:, 1:].contiguous() == 0] = -100
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

    def _last_hidden(self, hidden: torch.Tensor, mask: torch.Tensor, start: int, end: int) -> torch.Tensor:
        h = hidden[start:end]
        m = mask[start:end]
        last_idx = m.sum(dim=1) - 1
        gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
        return h.gather(1, gather_idx).squeeze(1)

    def forward_all(self, ctx_input_ids, ctx_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, mental1_input_ids, mental1_attention_mask, mental2_input_ids, mental2_attention_mask, expl_input_ids, expl_attention_mask, first_pos_token, *, stop_grad_z1: bool = False):
        batch_size = ctx_input_ids.size(0)
        context_hidden, z1, mu1, logvar1, z2, mu2, logvar2 = self.encode_z1_z2(ctx_input_ids, ctx_attention_mask, stop_grad_z1=stop_grad_z1)
        ids_list = [pos_input_ids, neg_input_ids] + ([expl_input_ids] if self.use_expl_reward else [])
        mask_list = [pos_attention_mask, neg_attention_mask] + ([expl_attention_mask] if self.use_expl_reward else [])
        max_len = max(t.size(1) for t in ids_list)
        pad_id = self.base_model.config.pad_token_id if self.base_model.config.pad_token_id is not None else 0

        def _pad_to(ids, mask):
            pad_len = max_len - ids.size(1)
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=pad_id)
                mask = F.pad(mask, (0, pad_len), value=0)
            return ids, mask

        padded = [_pad_to(ids, mask) for ids, mask in zip(ids_list, mask_list)]
        batched_ids = torch.cat([item[0] for item in padded], dim=0)
        batched_mask = torch.cat([item[1] for item in padded], dim=0)
        batched_hidden = self.transformer(input_ids=batched_ids, attention_mask=batched_mask, use_cache=False, return_dict=True).last_hidden_state
        pos_hidden = self._last_hidden(batched_hidden, batched_mask, 0, batch_size)
        neg_hidden = self._last_hidden(batched_hidden, batched_mask, batch_size, 2 * batch_size)
        pos_joint_reward = self.joint_outcome_head(torch.cat([z1, z2, pos_hidden], dim=1))
        neg_joint_reward = self.joint_outcome_head(torch.cat([z1, z2, neg_hidden], dim=1))
        z_cat = torch.cat([z1, z2], dim=1)
        z1_only_reward = self.z1_only_reward_head(z1)
        z_combined_reward = self.z_combined_reward_head(z_cat)
        output_embedding = self.base_model.get_output_embeddings()
        next_logits = F.linear(context_hidden + self.z_to_hidden(z_cat), output_embedding.weight, output_embedding.bias if getattr(output_embedding, "bias", None) is not None else None)
        future_loss = F.cross_entropy(next_logits, first_pos_token)
        mental1_gen_loss = self._decode_mental(z1, self.mental1_decoder, mental1_input_ids, mental1_attention_mask)
        mental2_gen_loss = self._decode_mental(z2, self.mental2_decoder, mental2_input_ids, mental2_attention_mask)
        expl_reward_pred = None
        if self.use_expl_reward:
            expl_hidden = batched_hidden[2 * batch_size:]
            expl_mask = padded[2][1]
            attended_z, _ = self.expl_cross_attn(query=z_cat.unsqueeze(1), key=expl_hidden, value=expl_hidden, key_padding_mask=(expl_mask == 0))
            expl_reward_pred = self.expl_reward_head(attended_z.squeeze(1))
        return {
            "pos_joint_reward": pos_joint_reward,
            "neg_joint_reward": neg_joint_reward,
            "z1_only_reward": z1_only_reward,
            "z_combined_reward": z_combined_reward,
            "expl_reward_pred": expl_reward_pred,
            "mu1": mu1,
            "logvar1": logvar1,
            "mu2": mu2,
            "logvar2": logvar2,
            "future_loss": future_loss,
            "mental1_gen_loss": mental1_gen_loss,
            "mental2_gen_loss": mental2_gen_loss,
        }


def compute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    mu32 = mu.float()
    logvar32 = logvar.float().clamp(-10.0, 10.0)
    return -0.5 * torch.mean(1 + logvar32 - mu32.pow(2) - logvar32.exp())


def compute_objective_components(out, reward_targets, has_neg, current_opt_step: int, *, kl_weight: float, future_weight: float, mental1_weight: float, mental2_weight: float, expl_weight: float, z_only_weight: float, kl_anneal_steps: int, z2_kl_delay_steps: int, use_expl_reward: bool):
    pref_loss_per_sample = F.softplus(out["neg_joint_reward"] - out["pos_joint_reward"]).mean(dim=1)
    pref_loss = (pref_loss_per_sample * has_neg).sum() / (has_neg.sum() + 1e-8)
    reward_reg_loss = F.smooth_l1_loss(out["pos_joint_reward"], reward_targets)
    z1_only_reg_loss = F.smooth_l1_loss(out["z1_only_reward"], reward_targets)
    z_combined_reg_loss = F.smooth_l1_loss(out["z_combined_reward"], reward_targets)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        kl1_loss = compute_kl_loss(out["mu1"], out["logvar1"])
        kl2_loss = compute_kl_loss(out["mu2"], out["logvar2"])
    if kl_anneal_steps > 0:
        anneal1 = min(1.0, current_opt_step / kl_anneal_steps)
        anneal2 = min(1.0, max(0.0, current_opt_step - kl_anneal_steps - z2_kl_delay_steps) / kl_anneal_steps)
    else:
        anneal1 = 1.0
        anneal2 = 1.0
    expl_reward_loss = reward_targets.new_tensor(0.0)
    if use_expl_reward:
        expl_reward_loss = F.smooth_l1_loss(out["expl_reward_pred"], reward_targets)
    total_loss = (
        pref_loss
        + reward_reg_loss
        + z_only_weight * z1_only_reg_loss
        + z_only_weight * z_combined_reg_loss
        + kl_weight * anneal1 * kl1_loss
        + kl_weight * anneal2 * kl2_loss
        + future_weight * out["future_loss"]
        + mental1_weight * out["mental1_gen_loss"]
        + mental2_weight * out["mental2_gen_loss"]
        + expl_weight * expl_reward_loss
    )
    metrics = {
        "preference": pref_loss.item(),
        "reward_reg": reward_reg_loss.item(),
        "z1_only_reg": z1_only_reg_loss.item(),
        "z_combined_reg": z_combined_reg_loss.item(),
        "kl1": kl1_loss.item(),
        "kl2": kl2_loss.item(),
        "future": out["future_loss"].item(),
        "mental1_gen": out["mental1_gen_loss"].item(),
        "mental2_gen": out["mental2_gen_loss"].item(),
        "expl_reward": expl_reward_loss.item(),
    }
    return total_loss, metrics, kl_weight * anneal1, kl_weight * anneal2


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch: int, *, autocast_dtype: torch.dtype, kl_weight: float, future_weight: float, mental1_weight: float, mental2_weight: float, expl_weight: float, z_only_weight: float, grad_accum_steps: int, kl_anneal_steps: int, z2_kl_delay_steps: int, z2_warmup_steps: int, global_step_offset: int, max_grad_norm: float):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = Counter()
    total_loss = 0.0
    num_batches = len(dataloader)
    global_step = global_step_offset
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    for batch_idx, batch in enumerate(dataloader):
        batch = _move_batch(batch, device)
        current_opt_step = global_step + 1
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=autocast_dtype):
            out = model.forward_all(
                batch["ctx_input_ids"], batch["ctx_attention_mask"], batch["pos_input_ids"], batch["pos_attention_mask"],
                batch["neg_input_ids"], batch["neg_attention_mask"], batch["mental1_input_ids"], batch["mental1_attention_mask"],
                batch["mental2_input_ids"], batch["mental2_attention_mask"], batch["expl_input_ids"], batch["expl_attention_mask"],
                batch["first_pos_token"], stop_grad_z1=current_opt_step < z2_warmup_steps,
            )
            raw_loss, batch_metrics, eff_kl1_w, eff_kl2_w = compute_objective_components(
                out, batch["reward_vec"], batch["has_negative"], current_opt_step,
                kl_weight=kl_weight, future_weight=future_weight, mental1_weight=mental1_weight, mental2_weight=mental2_weight,
                expl_weight=expl_weight, z_only_weight=z_only_weight, kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps, use_expl_reward=model.use_expl_reward,
            )
            loss = raw_loss / grad_accum_steps
        loss.backward()
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        total_loss += raw_loss.item()
        for key, value in batch_metrics.items():
            totals[key] += value
        if (batch_idx + 1) % 10 == 0:
            n = batch_idx + 1
            print(
                f"  Epoch {epoch + 1} Step {n}: loss={total_loss / n:.4f} pref={totals['preference'] / n:.4f} "
                f"reward={totals['reward_reg'] / n:.4f} kl1={totals['kl1'] / n:.4f}(w={eff_kl1_w:.4f}) "
                f"kl2={totals['kl2'] / n:.4f}(w={eff_kl2_w:.4f}) m1={totals['mental1_gen'] / n:.4f} "
                f"m2={totals['mental2_gen'] / n:.4f} expl={totals['expl_reward'] / n:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
    n = max(1, num_batches)
    return total_loss / n, {key: value / n for key, value in totals.items()}, global_step


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, *, autocast_dtype: torch.dtype, current_opt_step: int, kl_weight: float, future_weight: float, mental1_weight: float, mental2_weight: float, expl_weight: float, z_only_weight: float, kl_anneal_steps: int, z2_kl_delay_steps: int):
    model.eval()
    totals = Counter()
    total_loss = 0.0
    for batch in dataloader:
        batch = _move_batch(batch, device)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=autocast_dtype):
            out = model.forward_all(
                batch["ctx_input_ids"], batch["ctx_attention_mask"], batch["pos_input_ids"], batch["pos_attention_mask"],
                batch["neg_input_ids"], batch["neg_attention_mask"], batch["mental1_input_ids"], batch["mental1_attention_mask"],
                batch["mental2_input_ids"], batch["mental2_attention_mask"], batch["expl_input_ids"], batch["expl_attention_mask"],
                batch["first_pos_token"], stop_grad_z1=False,
            )
            raw_loss, batch_metrics, _, _ = compute_objective_components(
                out, batch["reward_vec"], batch["has_negative"], current_opt_step,
                kl_weight=kl_weight, future_weight=future_weight, mental1_weight=mental1_weight, mental2_weight=mental2_weight,
                expl_weight=expl_weight, z_only_weight=z_only_weight, kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps, use_expl_reward=model.use_expl_reward,
            )
        total_loss += raw_loss.item()
        for key, value in batch_metrics.items():
            totals[key] += value
    n = max(1, len(dataloader))
    return total_loss / n, {key: value / n for key, value in totals.items()}


def _dtype_from_arg(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _infer_num_layers(config) -> int | None:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_checkpoint(model: RecursiveToMModel, save_dir: Path, metadata: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(save_dir / "lora_adapter")
    for head_name in model.custom_head_names:
        torch.save(getattr(model, head_name).state_dict(), save_dir / f"{head_name}.pth")
    _save_json(save_dir / "metadata.json", metadata | {"saved_head_names": model.custom_head_names})


def _print_validation_summary(name: str, summary: ValidationSummary) -> None:
    print(
        f"{name}: records={summary.num_records}, unique_ids={summary.unique_example_ids}, "
        f"base_scenarios={summary.base_scenarios}, issues={summary.issue_count}, positive_truth_leaks={summary.positive_truth_leaks}",
        flush=True,
    )
    print(f"  strategy_dist={summary.strategy_dist}", flush=True)
    print(f"  confidence_dist={summary.confidence_dist}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ToM-SB recursive mental reward trainer.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=str, default=DEFAULT_VAL_PATH)
    parser.add_argument("--reward_schema_path", type=str, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--max_ctx_len", type=int, default=1536)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--max_mental_len", type=int, default=512)
    parser.add_argument("--mental_label_mode", type=str, default="hybrid", choices=["text", "structured", "hybrid"])
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--future_weight", type=float, default=0.5)
    parser.add_argument("--mental1_weight", type=float, default=0.3)
    parser.add_argument("--mental2_weight", type=float, default=0.3)
    parser.add_argument("--expl_weight", type=float, default=0.0)
    parser.add_argument("--z_only_weight", type=float, default=0.5)
    parser.add_argument("--kl_anneal_steps", type=int, default=200)
    parser.add_argument("--z2_kl_delay_steps", type=int, default=100)
    parser.add_argument("--z2_warmup_steps", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_val_examples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=str, default=None, help="Deprecated; use CUDA_VISIBLE_DEVICES outside Python.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--skip_data_validation", action="store_true")
    parser.add_argument("--require_scenario_disjoint_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_missing_val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.gpu is not None:
        print("Warning: --gpu is deprecated and ignored. Set CUDA_VISIBLE_DEVICES in the shell instead.", flush=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    reward_schema = _load_reward_schema(args.reward_schema_path)
    train_records = _read_json_or_jsonl(args.data_path)
    val_records: list[dict[str, Any]] = []
    if args.val_path and Path(args.val_path).exists():
        val_records = _read_json_or_jsonl(args.val_path)
    elif args.val_path and not args.allow_missing_val:
        raise FileNotFoundError(f"Validation file not found: {args.val_path}")
    if args.max_examples > 0:
        train_records = train_records[: args.max_examples]
    if args.max_val_examples > 0 and val_records:
        val_records = val_records[: args.max_val_examples]
    validation_payload: dict[str, Any] = {"reward_schema": reward_schema}
    if not args.skip_data_validation:
        train_summary, _ = validate_records(train_records, path=args.data_path, reward_schema=reward_schema, strict=True)
        _print_validation_summary("train", train_summary)
        validation_payload["train"] = asdict(train_summary)
        if val_records:
            val_summary, _ = validate_records(val_records, path=args.val_path, reward_schema=reward_schema, strict=True)
            _print_validation_summary("val", val_summary)
            validation_payload["val"] = asdict(val_summary)
            split_summary = validate_split_disjoint(train_records, val_records, strict=args.require_scenario_disjoint_val)
            validation_payload["split"] = split_summary
            print(f"split: {split_summary}", flush=True)
    if args.validate_only:
        print("Validation-only mode complete. No model was loaded.", flush=True)
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training script expects a CUDA GPU.")
    dtype = _dtype_from_arg(args.dtype)
    use_expl_reward = args.expl_weight > 0.0
    if not use_expl_reward:
        print("Explanation branch disabled because --expl_weight is 0.0.", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "args.json", vars(args))
    _save_json(output_dir / "data_validation.json", validation_payload)
    checkpoint_metadata = {
        "model_family": "tom_sb_recursive_mental_reward",
        "reward_schema": reward_schema,
        "reward_schema_path": str(args.reward_schema_path),
        "base_model": args.model_name,
        "mental_label_mode": args.mental_label_mode,
        "use_expl_reward": use_expl_reward,
        "z_dim": args.z_dim,
    }
    _save_json(output_dir / "checkpoint_metadata.json", checkpoint_metadata)
    print(f"Loading tokenizer: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    print(f"Loading base model on {device}: {args.model_name}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(device)
    base_model.config.use_cache = False
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    lora_kwargs: dict[str, Any] = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": _parse_target_modules(args.target_modules),
        "lora_dropout": args.lora_dropout,
    }
    num_layers = _infer_num_layers(base_model.config)
    if num_layers is not None and args.num_lora_layers > 0:
        start_layer = max(0, num_layers - args.num_lora_layers)
        lora_kwargs["layers_to_transform"] = list(range(start_layer, num_layers))
        lora_kwargs["layers_pattern"] = "layers"
        print(f"Applying LoRA to top {num_layers - start_layer}/{num_layers} transformer layers", flush=True)
    base_model = get_peft_model(base_model, LoraConfig(**lora_kwargs))
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_" in name
    base_model.print_trainable_parameters()
    model = RecursiveToMModel(base_model, reward_dim=len(reward_schema), z_dim=args.z_dim, use_expl_reward=use_expl_reward).to(device)
    for name, param in model.named_parameters():
        if any(head_name in name for head_name in model.custom_head_names):
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)
    train_dataset = FlatToMDataset(train_records, tokenizer, reward_dim=len(reward_schema), max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len, max_mental_len=args.max_mental_len, mental_label_mode=args.mental_label_mode, use_expl_reward=use_expl_reward, name="train")
    val_dataset = FlatToMDataset(val_records, tokenizer, reward_dim=len(reward_schema), max_ctx_len=args.max_ctx_len, max_resp_len=args.max_resp_len, max_mental_len=args.max_mental_len, mental_label_mode=args.mental_label_mode, use_expl_reward=use_expl_reward, name="val") if val_records else None
    print(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}", flush=True)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True, "persistent_workers": args.num_workers > 0, "collate_fn": lambda batch: collate_fn(batch, tokenizer)}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None
    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(head_name in name for head_name in model.custom_head_names):
            head_params.append(param)
        else:
            lora_params.append(param)
    if not head_params:
        raise RuntimeError("No custom reward/mental head parameters are trainable.")
    head_lr = args.lr * args.head_lr_mult
    print(f"Param groups: LoRA lr={args.lr:.2e}, head lr={head_lr:.2e}", flush=True)
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": args.lr, "weight_decay": 0.01}, {"params": head_params, "lr": head_lr, "weight_decay": 0.01}])
    total_steps = max(1, math.ceil(len(train_loader) * args.num_epochs / args.grad_accum_steps))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Optimizer steps: total={total_steps}, warmup={warmup_steps}", flush=True)
    best_metric = float("inf")
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss, metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, autocast_dtype=dtype,
            kl_weight=args.kl_weight, future_weight=args.future_weight, mental1_weight=args.mental1_weight,
            mental2_weight=args.mental2_weight, expl_weight=args.expl_weight, z_only_weight=args.z_only_weight,
            grad_accum_steps=args.grad_accum_steps, kl_anneal_steps=args.kl_anneal_steps,
            z2_kl_delay_steps=args.z2_kl_delay_steps, z2_warmup_steps=args.z2_warmup_steps,
            global_step_offset=global_step, max_grad_norm=args.max_grad_norm,
        )
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}: avg_loss={avg_loss:.4f}", flush=True)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}", flush=True)
        monitor_metric = avg_loss
        if val_loader is not None:
            val_loss, val_metrics = evaluate_epoch(
                model, val_loader, device, autocast_dtype=dtype, current_opt_step=max(global_step, 1),
                kl_weight=args.kl_weight, future_weight=args.future_weight, mental1_weight=args.mental1_weight,
                mental2_weight=args.mental2_weight, expl_weight=args.expl_weight, z_only_weight=args.z_only_weight,
                kl_anneal_steps=args.kl_anneal_steps, z2_kl_delay_steps=args.z2_kl_delay_steps,
            )
            print(f"  val_loss: {val_loss:.4f}", flush=True)
            for key, value in val_metrics.items():
                print(f"  val_{key}: {value:.4f}", flush=True)
            monitor_metric = val_loss
        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        save_checkpoint(model, epoch_dir, checkpoint_metadata | {"epoch": epoch + 1})
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            save_checkpoint(model, output_dir / "best", checkpoint_metadata | {"epoch": epoch + 1, "best_metric": best_metric})
            print(f"  New best checkpoint saved: metric={best_metric:.4f}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Training complete. Best metric: {best_metric:.4f}", flush=True)


if __name__ == "__main__":
    main()
