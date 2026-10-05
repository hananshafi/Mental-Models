#!/usr/bin/env python3
"""Compatibility wrapper for the standalone ToM-SB mental-reward trainer."""

from __future__ import annotations

if __name__ == "__main__":
    from train_tom_sb_mental_reward_standalone import main as _standalone_main

    _standalone_main()
    raise SystemExit

'''
Legacy adapter code below is intentionally inert. Use
train_tom_sb_mental_reward_standalone.py for all ToM-SB training.

import argparse
import gc
import importlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_PATH = "projects/tom_sb/data/tom_sb_belief_only_5k_train.jsonl"
DEFAULT_VAL_PATH = "projects/tom_sb/data/tom_sb_belief_only_5k_train_val.jsonl"
DEFAULT_OUTPUT_DIR = "projects/tom_sb/checkpoints/tom_sb_reward_belief_only_v1"
DEFAULT_STAGE1_DIR = "projects/sotopia"

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


def _base_scenario_id(example_id: str) -> str:
    for marker in ("_v", "_d"):
        if marker in example_id:
            return example_id.rsplit(marker, 1)[0]
    return example_id


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_attacker_prompt(context_text: str) -> str:
    marker = "## Current Attacker Message\n"
    if marker not in context_text:
        return ""
    tail = context_text.split(marker, 1)[1]
    return tail.split("\n\nDefender:", 1)[0].strip()


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
    reward_dim: int,
    strict: bool = True,
    max_reported_issues: int = 25,
) -> tuple[ValidationSummary, list[str]]:
    """Validate ToM-SB records for schema, label, and leakage consistency."""
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
            or len(reward_vec) != reward_dim
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

        if set(labels) != REQUIRED_TOM_LABEL_KEYS:
            _append_issue(issues, loc, "tom_label_keys_mismatch", sorted(labels))

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
        prompt = _extract_attacker_prompt(str(record.get("context_text", "")))
        mental_text = f"{record.get('mental1_text', '')} {record.get('mental2_text', '')}"
        response = str(record.get("pos_response", ""))
        hard_negative = str(record.get("hard_negative", ""))

        strategy_dist[strategy] += 1
        target_dist[target_field] += 1
        confidence_dist[str(first_order.get("attacker_confidence"))] += 1
        known_count_dist[str(len(known))] += 1
        unknown_count_dist[str(len(unknown))] += 1

        derived_known = {
            key: value for key, value in ground_truth.items() if attacker_prior.get(key) is not None
        }
        derived_unknown = {
            key: ground_truth.get(key) for key in ground_truth if attacker_prior.get(key) is None
        }
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
        if first_order.get("attacker_missing_fields") != list(unknown.keys()):
            _append_issue(issues, loc, "first_order_missing_fields_mismatch")
        if first_order.get("attacker_target_field") != target_field or target_field not in unknown:
            _append_issue(issues, loc, "target_field_mismatch", target_field)
        if first_order.get("attacker_confidence") != _expected_confidence(known, unknown):
            _append_issue(issues, loc, "attacker_confidence_mismatch")

        if second_order.get("attacker_believes_defender_knows") != list(ground_truth.keys()):
            _append_issue(issues, loc, "second_order_known_private_fields_mismatch")
        if second_order.get("attacker_believes_defender_knows_missing") != list(unknown.keys()):
            _append_issue(issues, loc, "second_order_missing_fields_mismatch")
        if second_order.get("attacker_expects_defender_helpfulness") != "high":
            _append_issue(issues, loc, "second_order_helpfulness_mismatch")
        if second_order.get("attacker_expected_probe_style") != _expected_probe_style(known, prompt):
            _append_issue(issues, loc, "second_order_probe_style_mismatch")

        if labels.get("attacker_goal") != "extract_missing_private_field":
            _append_issue(issues, loc, "attacker_goal_mismatch")
        if boundary.get("can_confirm") != list(known.keys()):
            _append_issue(issues, loc, "safe_boundary_can_confirm_mismatch")
        if boundary.get("must_not_reveal") != list(unknown.keys()):
            _append_issue(issues, loc, "safe_boundary_must_not_reveal_mismatch")
        expected_safe_response = STRATEGY_TO_SAFE_RESPONSE.get(strategy, "safe_non_leaking_reply")
        if boundary.get("safe_response_type") != expected_safe_response:
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


def validate_split_disjoint(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    *,
    strict: bool,
) -> dict[str, Any]:
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
        raise DatasetValidationError(
            "Train/val split is not scenario-disjoint. "
            f"Found {len(overlap)} overlapping base scenario ids, e.g. {sorted(overlap)[:10]}"
        )
    return summary


def _tom_mental_texts(sample: dict[str, Any], mode: str) -> tuple[str, str]:
    """Return first-order and second-order mental supervision strings."""
    labels = sample.get("tom_labels", {}) or {}
    first_order = labels.get("first_order_belief", {}) or {}
    second_order = labels.get("second_order_belief", {}) or {}

    first_payload = {
        "first_order_belief": first_order,
        "attacker_goal": labels.get("attacker_goal"),
        "defender_safe_boundary": labels.get("defender_safe_boundary"),
    }
    second_payload = {
        "second_order_belief": second_order,
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
    """Flat ToM-SB JSONL dataset for recursive ToM reward training."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        *,
        torch_module,
        reward_dim: int,
        max_ctx_len: int,
        max_resp_len: int,
        max_mental_len: int,
        mental_label_mode: str,
        name: str,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.reward_dim = reward_dim
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        self.max_mental_len = max_mental_len
        self.mental_label_mode = mental_label_mode
        self.name = name
        print(f"Loaded {len(self.records)} {name} ToM-SB samples", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, text: str, max_length: int):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.records[idx]
        reward_vec = sample.get("reward_vec")
        if not isinstance(reward_vec, list) or len(reward_vec) != self.reward_dim:
            raise ValueError(
                f"Sample {sample.get('example_id', idx)} has invalid reward_vec: {reward_vec}"
            )

        context_text = str(sample.get("context_text", ""))
        pos_response = str(sample.get("pos_response", ""))
        hard_negative = str(sample.get("hard_negative", ""))
        mental1_text, mental2_text = _tom_mental_texts(sample, self.mental_label_mode)
        reward_explanations = str(sample.get("reward_explanations", "") or "N/A")

        ctx_enc = self._tokenize(context_text, self.max_ctx_len)
        pos_enc = self._tokenize(pos_response, self.max_resp_len)
        neg_enc = self._tokenize(
            hard_negative if hard_negative.strip() else pos_response,
            self.max_resp_len,
        )
        mental1_enc = self._tokenize(mental1_text, self.max_mental_len)
        mental2_enc = self._tokenize(mental2_text, self.max_mental_len)
        expl_enc = self._tokenize(reward_explanations, self.max_mental_len)

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
            "expl_input_ids": expl_enc.input_ids.squeeze(0),
            "expl_attention_mask": expl_enc.attention_mask.squeeze(0),
            "reward_vec": self.torch.tensor(reward_vec, dtype=self.torch.float32),
            "has_negative": self.torch.tensor(1.0 if hard_negative.strip() else 0.0),
        }


def _import_stage1(stage1_module_dir: str | Path):
    module_dir = str(Path(stage1_module_dir).resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    return importlib.import_module("stage1_train_coupled_mental_reward_v3")


def _infer_num_layers(config) -> int | None:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _dtype_from_arg(torch_module, dtype_name: str):
    if dtype_name == "bf16":
        return torch_module.bfloat16
    if dtype_name == "fp16":
        return torch_module.float16
    if dtype_name == "fp32":
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _print_validation_summary(name: str, summary: ValidationSummary) -> None:
    print(
        f"{name}: records={summary.num_records}, unique_ids={summary.unique_example_ids}, "
        f"base_scenarios={summary.base_scenarios}, issues={summary.issue_count}, "
        f"positive_truth_leaks={summary.positive_truth_leaks}",
        flush=True,
    )
    print(f"  strategy_dist={summary.strategy_dist}", flush=True)
    print(f"  confidence_dist={summary.confidence_dist}", flush=True)


def _parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ToM-SB recursive mental reward model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=str, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1_module_dir", type=str, default=DEFAULT_STAGE1_DIR)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--max_ctx_len", type=int, default=1536)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--max_mental_len", type=int, default=384)
    parser.add_argument(
        "--mental_label_mode",
        type=str,
        default="hybrid",
        choices=["text", "structured", "hybrid"],
        help="How to supervise z1/z2 mental decoders from mental*_text and tom_labels.",
    )

    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_lora_layers", type=int, default=16)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--future_weight", type=float, default=0.5)
    parser.add_argument("--mental1_weight", type=float, default=0.3)
    parser.add_argument("--mental2_weight", type=float, default=0.3)
    parser.add_argument(
        "--expl_weight",
        type=float,
        default=0.0,
        help="Keep 0.0 for the belief-only data because reward_explanations are omitted.",
    )
    parser.add_argument("--z_only_weight", type=float, default=0.5)
    parser.add_argument("--kl_anneal_steps", type=int, default=200)
    parser.add_argument("--z2_kl_delay_steps", type=int, default=100)
    parser.add_argument("--z2_warmup_steps", type=int, default=100)

    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_val_examples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--skip_data_validation", action="store_true")
    parser.add_argument(
        "--require_scenario_disjoint_val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require train and validation splits to have no base scenario overlap.",
    )
    parser.add_argument("--allow_missing_val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    validation_payload: dict[str, Any] = {}
    if not args.skip_data_validation:
        train_summary, _ = validate_records(
            train_records,
            path=args.data_path,
            reward_dim=7,
            strict=True,
        )
        _print_validation_summary("train", train_summary)
        validation_payload["train"] = asdict(train_summary)

        if val_records:
            val_summary, _ = validate_records(
                val_records,
                path=args.val_path,
                reward_dim=7,
                strict=True,
            )
            _print_validation_summary("val", val_summary)
            validation_payload["val"] = asdict(val_summary)
            split_summary = validate_split_disjoint(
                train_records,
                val_records,
                strict=args.require_scenario_disjoint_val,
            )
            validation_payload["split"] = split_summary
            print(f"split: {split_summary}", flush=True)

    if args.validate_only:
        print("Validation-only mode complete. No model was loaded.", flush=True)
        return

    import torch
    from torch.utils.data import DataLoader

    stage1 = _import_stage1(args.stage1_module_dir)

    if stage1.REWARD_DIM != 7:
        raise RuntimeError(f"Expected Stage 1 reward dim 7, got {stage1.REWARD_DIM}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training script expects a CUDA GPU.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "args.json", vars(args))
    if validation_payload:
        _save_json(output_dir / "data_validation.json", validation_payload)

    print(f"Loading tokenizer: {args.model_name}", flush=True)
    tokenizer = stage1.AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = _dtype_from_arg(torch, args.dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading base model: {args.model_name}", flush=True)
    base_model = stage1.AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
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
        print(
            f"Applying LoRA to top {num_layers - start_layer}/{num_layers} transformer layers",
            flush=True,
        )

    lora_config = stage1.LoraConfig(**lora_kwargs)
    base_model = stage1.get_peft_model(base_model, lora_config)
    for name, param in base_model.named_parameters():
        param.requires_grad = "lora_" in name
    base_model.print_trainable_parameters()

    model = stage1.RecursiveToMModel(
        base_model,
        reward_dim=stage1.REWARD_DIM,
        z_dim=args.z_dim,
    ).to(device)

    for name, param in model.named_parameters():
        if any(head_name in name for head_name in stage1.CUSTOM_HEAD_NAMES):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    train_dataset = FlatToMDataset(
        train_records,
        tokenizer,
        torch_module=torch,
        reward_dim=stage1.REWARD_DIM,
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        max_mental_len=args.max_mental_len,
        mental_label_mode=args.mental_label_mode,
        name="train",
    )
    val_dataset = None
    if val_records:
        val_dataset = FlatToMDataset(
            val_records,
            tokenizer,
            torch_module=torch,
            reward_dim=stage1.REWARD_DIM,
            max_ctx_len=args.max_ctx_len,
            max_resp_len=args.max_resp_len,
            max_mental_len=args.max_mental_len,
            mental_label_mode=args.mental_label_mode,
            name="val",
        )

    print(
        f"Dataset split: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_dataset is not None else 0}",
        flush=True,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": lambda batch: stage1.collate_fn(batch, tokenizer),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(head_name in name for head_name in stage1.CUSTOM_HEAD_NAMES):
            head_params.append(param)
        else:
            lora_params.append(param)

    if not lora_params:
        print("Warning: no LoRA parameters are trainable.", flush=True)
    if not head_params:
        raise RuntimeError("No custom reward/mental head parameters are trainable.")

    head_lr = args.lr * args.head_lr_mult
    print(f"Param groups: LoRA lr={args.lr:.2e}, head lr={head_lr:.2e}", flush=True)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
        ]
    )
    total_steps = max(1, math.ceil(len(train_loader) * args.num_epochs / args.grad_accum_steps))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = stage1.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Optimizer steps: total={total_steps}, warmup={warmup_steps}", flush=True)

    best_metric = float("inf")
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss, metrics, global_step = stage1.train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            kl_weight=args.kl_weight,
            future_weight=args.future_weight,
            mental1_weight=args.mental1_weight,
            mental2_weight=args.mental2_weight,
            expl_weight=args.expl_weight,
            z_only_weight=args.z_only_weight,
            grad_accum_steps=args.grad_accum_steps,
            kl_anneal_steps=args.kl_anneal_steps,
            z2_kl_delay_steps=args.z2_kl_delay_steps,
            z2_warmup_steps=args.z2_warmup_steps,
            global_step_offset=global_step,
            max_grad_norm=args.max_grad_norm,
        )
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}: avg_loss={avg_loss:.4f}", flush=True)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}", flush=True)

        monitor_metric = avg_loss
        if val_loader is not None:
            val_loss, val_metrics = stage1.evaluate_epoch(
                model,
                val_loader,
                device,
                current_opt_step=max(global_step, 1),
                kl_weight=args.kl_weight,
                future_weight=args.future_weight,
                mental1_weight=args.mental1_weight,
                mental2_weight=args.mental2_weight,
                expl_weight=args.expl_weight,
                z_only_weight=args.z_only_weight,
                kl_anneal_steps=args.kl_anneal_steps,
                z2_kl_delay_steps=args.z2_kl_delay_steps,
            )
            print(f"  val_loss: {val_loss:.4f}", flush=True)
            for key, value in val_metrics.items():
                print(f"  val_{key}: {value:.4f}", flush=True)
            monitor_metric = val_loss

        epoch_dir = output_dir / f"epoch_{epoch}"
        stage1._save_checkpoint(model, str(epoch_dir))
        if monitor_metric < best_metric:
            best_metric = monitor_metric
            stage1._save_checkpoint(model, str(output_dir / "best"))
            print(f"  New best checkpoint saved: metric={best_metric:.4f}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training complete. Best metric: {best_metric:.4f}", flush=True)


if __name__ == "__main__":
    main()
'''
