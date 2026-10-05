#!/usr/bin/env python3
"""
ToM-SB mental-reward trainer with structured ToM prediction losses.

This script intentionally does not import from the Sotopia codebase. It reuses
the local ToM-SB validation/tokenization helpers from
train_tom_sb_mental_reward_standalone.py, but replaces free-form mental text
generation with direct first-order and second-order ToM prediction heads.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from train_tom_sb_mental_reward_standalone import (
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    ValidationSummary,
    _dtype_from_arg,
    _infer_num_layers,
    _load_reward_schema,
    _parse_target_modules,
    _read_json_or_jsonl,
    _save_json,
    compute_kl_loss,
    get_transformer_from_peft,
    validate_records,
    validate_split_disjoint,
)


DEFAULT_OUTPUT_DIR = "projects/tom_sb/checkpoints/tom_sb_reward_structured_tom_v1"

BASE_HEAD_NAMES = [
    "z1_mu",
    "z1_logvar",
    "z2_mu",
    "z2_logvar",
    "joint_outcome_head",
    "z1_only_reward_head",
    "z_combined_reward_head",
    "z_to_hidden",
    "m1_heads",
    "m2_heads",
]
EXPL_HEAD_NAMES = [
    "expl_cross_attn",
    "expl_reward_head",
]

DEFAULT_STRATEGIC_INTENT_VOCAB = [
    "belief_consistent_decoy",
    "probe_attacker_knowledge",
    "safe_partial_confirmation",
    "broad_deflection",
    "rollout_defender_reply",
    "unknown",
]
DEFAULT_PRIVACY_MECHANISM_VOCAB = [
    "decoy_without_truth_confirmation",
    "ask_for_prior_before_disclosure",
    "confirm_only_known_fields",
    "safe_abstraction_or_deflection",
    "safe_rollout_reply",
    "safe_non_leaking_reply",
]
DEFAULT_TRUST_MECHANISM_VOCAB = [
    "stay_consistent_with_attacker_prior",
    "invite_attacker_to_state_prior",
    "acknowledge_known_information",
    "remain_helpful_without_specifics",
    "maintain_dialogue_continuity",
    "maintain_helpful_tone",
]
DEFAULT_DECOY_POLICY_VOCAB = [
    "no_decoy",
    "use_belief_consistent_decoy",
]
DEFAULT_WITHHOLDING_POLICY_VOCAB = [
    "withhold_all_private_fields",
    "confirm_known_withhold_target",
    "confirm_known_withhold_unknown",
]
DEFAULT_RATIONALE_TAG_VOCAB = [
    "avoid_confirming_ground_truth",
    "withhold_unknown_private_fields",
    "maintain_helpful_tone",
    "preserve_attacker_trust",
    "target_field_unknown_to_attacker",
    "attacker_has_no_prior_known_fields",
    "attacker_has_some_prior_known_fields",
    "confirm_only_attacker_known_fields",
    "multiple_missing_fields_remain",
    "use_decoy_without_truth_confirmation",
    "ask_attacker_to_state_prior",
    "partial_confirmation_without_deeper_leak",
    "deflect_to_safe_abstraction",
    "follow_rollout_safe_response",
    "safe_non_leaking_reply",
    "belief_update_risk_low",
    "belief_update_risk_medium",
    "belief_update_risk_high",
    "belief_update_risk_unknown",
    "deception_detection_risk_low",
    "deception_detection_risk_medium",
    "deception_detection_risk_high",
    "deception_detection_risk_unknown",
]


def _ordered_add(vocab: list[str], value: Any) -> None:
    if value is None:
        return
    item = str(value)
    if item and item not in vocab:
        vocab.append(item)


def _ordered_add_many(vocab: list[str], values: Any) -> None:
    if isinstance(values, dict):
        values = values.keys()
    if isinstance(values, (str, bytes)):
        return
    if not isinstance(values, (list, tuple, set)):
        try:
            values = list(values)
        except TypeError:
            return
    for value in values:
        _ordered_add(vocab, value)


def _derive_privacy_mechanism(strategy: str) -> str:
    return {
        "belief_consistent_decoy": "decoy_without_truth_confirmation",
        "probe_attacker_knowledge": "ask_for_prior_before_disclosure",
        "safe_partial_confirmation": "confirm_only_known_fields",
        "broad_deflection": "safe_abstraction_or_deflection",
        "rollout_defender_reply": "safe_rollout_reply",
    }.get(strategy, "safe_non_leaking_reply")


def _derive_trust_mechanism(strategy: str) -> str:
    return {
        "belief_consistent_decoy": "stay_consistent_with_attacker_prior",
        "probe_attacker_knowledge": "invite_attacker_to_state_prior",
        "safe_partial_confirmation": "acknowledge_known_information",
        "broad_deflection": "remain_helpful_without_specifics",
        "rollout_defender_reply": "maintain_dialogue_continuity",
    }.get(strategy, "maintain_helpful_tone")


def _derive_decoy_policy(strategy: str) -> str:
    if strategy == "belief_consistent_decoy":
        return "use_belief_consistent_decoy"
    return "no_decoy"


def _derive_withholding_policy(known: dict[str, Any], unknown: dict[str, Any]) -> str:
    if not known:
        return "withhold_all_private_fields"
    if len(unknown) <= 1:
        return "confirm_known_withhold_target"
    return "confirm_known_withhold_unknown"


def _derive_rationale_tags(
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


def _strategy_reasoning_labels(sample: dict[str, Any]) -> dict[str, Any]:
    labels = sample.get("tom_labels", {}) or {}
    metadata = sample.get("metadata", {}) or {}
    known = metadata.get("known_fields", {}) or {}
    unknown = metadata.get("unknown_fields", {}) or {}
    strategy = str(labels.get("strategic_intent") or metadata.get("strategy") or "unknown")
    target_field = str(metadata.get("target_field") or (labels.get("first_order_belief", {}) or {}).get("attacker_target_field") or "")
    belief_update_risk = str(labels.get("belief_update_risk") or "unknown")
    deception_detection_risk = str(labels.get("deception_detection_risk") or "unknown")
    rationale_tags = labels.get("rationale_tags")
    if not isinstance(rationale_tags, list):
        rationale_tags = _derive_rationale_tags(
            known=known,
            unknown=unknown,
            target_field=target_field,
            strategy=strategy,
            belief_update_risk=belief_update_risk,
            deception_detection_risk=deception_detection_risk,
        )
    return {
        "strategic_intent": strategy,
        "privacy_mechanism": labels.get("privacy_mechanism") or _derive_privacy_mechanism(strategy),
        "trust_mechanism": labels.get("trust_mechanism") or _derive_trust_mechanism(strategy),
        "decoy_policy": labels.get("decoy_policy") or _derive_decoy_policy(strategy),
        "withholding_policy": labels.get("withholding_policy") or _derive_withholding_policy(known, unknown),
        "rationale_tags": rationale_tags,
    }


@dataclass
class StructuredToMSpec:
    field_vocab: list[str]
    attacker_confidence_vocab: list[str]
    attacker_goal_vocab: list[str]
    safe_response_type_vocab: list[str]
    helpfulness_vocab: list[str]
    probe_style_vocab: list[str]
    belief_update_risk_vocab: list[str]
    deception_detection_risk_vocab: list[str]
    expected_next_attack_vocab: list[str]
    strategic_intent_vocab: list[str]
    privacy_mechanism_vocab: list[str]
    trust_mechanism_vocab: list[str]
    decoy_policy_vocab: list[str]
    withholding_policy_vocab: list[str]
    rationale_tag_vocab: list[str]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "StructuredToMSpec":
        field_vocab: list[str] = []
        attacker_confidence_vocab = ["low", "medium", "high", "unknown"]
        attacker_goal_vocab = ["extract_missing_private_field"]
        safe_response_type_vocab = [
            "ask_for_prior_or_source",
            "tentative_decoy_without_truth_confirmation",
            "partial_acknowledgment_without_deeper_field",
            "deflect_or_keep_unresolved",
            "safe_rollout_reply",
        ]
        helpfulness_vocab = ["low", "medium", "high", "unknown"]
        probe_style_vocab = ["direct_extraction", "indirect_consistency_check", "open_ended_probe", "unknown"]
        belief_update_risk_vocab = ["low", "medium", "high", "unknown"]
        deception_detection_risk_vocab = ["low", "medium", "high", "unknown"]
        expected_next_attack_vocab = [
            "provide_false_or_partial_prior",
            "consistency_check_decoy",
            "ask_deeper_missing_field",
            "continue_private_field_probe",
            "repeat_or_rephrase_target_field_request",
            "unknown",
        ]
        strategic_intent_vocab = list(DEFAULT_STRATEGIC_INTENT_VOCAB)
        privacy_mechanism_vocab = list(DEFAULT_PRIVACY_MECHANISM_VOCAB)
        trust_mechanism_vocab = list(DEFAULT_TRUST_MECHANISM_VOCAB)
        decoy_policy_vocab = list(DEFAULT_DECOY_POLICY_VOCAB)
        withholding_policy_vocab = list(DEFAULT_WITHHOLDING_POLICY_VOCAB)
        rationale_tag_vocab = list(DEFAULT_RATIONALE_TAG_VOCAB)

        for record in records:
            labels = record.get("tom_labels", {}) or {}
            metadata = record.get("metadata", {}) or {}
            first = labels.get("first_order_belief", {}) or {}
            second = labels.get("second_order_belief", {}) or {}
            boundary = labels.get("defender_safe_boundary", {}) or {}
            strategy_reasoning = _strategy_reasoning_labels(record)

            for key in ("ground_truth", "known_fields", "unknown_fields"):
                _ordered_add_many(field_vocab, (metadata.get(key, {}) or {}).keys())
            _ordered_add_many(field_vocab, first.get("attacker_known_fields", {}) or {})
            _ordered_add_many(field_vocab, first.get("attacker_missing_fields", []) or [])
            _ordered_add(field_vocab, first.get("attacker_target_field"))
            _ordered_add_many(field_vocab, second.get("attacker_believes_defender_knows", []) or [])
            _ordered_add_many(field_vocab, second.get("attacker_believes_defender_knows_missing", []) or [])
            _ordered_add_many(field_vocab, boundary.get("can_confirm", []) or [])
            _ordered_add_many(field_vocab, boundary.get("must_not_reveal", []) or [])

            _ordered_add(attacker_confidence_vocab, first.get("attacker_confidence"))
            _ordered_add(attacker_goal_vocab, labels.get("attacker_goal"))
            _ordered_add(safe_response_type_vocab, boundary.get("safe_response_type"))
            _ordered_add(helpfulness_vocab, second.get("attacker_expects_defender_helpfulness"))
            _ordered_add(probe_style_vocab, second.get("attacker_expected_probe_style"))
            _ordered_add(belief_update_risk_vocab, labels.get("belief_update_risk"))
            _ordered_add(deception_detection_risk_vocab, labels.get("deception_detection_risk"))
            _ordered_add(expected_next_attack_vocab, labels.get("expected_next_attack"))
            _ordered_add(strategic_intent_vocab, strategy_reasoning.get("strategic_intent"))
            _ordered_add(privacy_mechanism_vocab, strategy_reasoning.get("privacy_mechanism"))
            _ordered_add(trust_mechanism_vocab, strategy_reasoning.get("trust_mechanism"))
            _ordered_add(decoy_policy_vocab, strategy_reasoning.get("decoy_policy"))
            _ordered_add(withholding_policy_vocab, strategy_reasoning.get("withholding_policy"))
            _ordered_add_many(rationale_tag_vocab, strategy_reasoning.get("rationale_tags", []) or [])

        if not field_vocab:
            raise ValueError("Could not infer any private-field labels from the ToM-SB records.")
        return cls(
            field_vocab=field_vocab,
            attacker_confidence_vocab=attacker_confidence_vocab,
            attacker_goal_vocab=attacker_goal_vocab,
            safe_response_type_vocab=safe_response_type_vocab,
            helpfulness_vocab=helpfulness_vocab,
            probe_style_vocab=probe_style_vocab,
            belief_update_risk_vocab=belief_update_risk_vocab,
            deception_detection_risk_vocab=deception_detection_risk_vocab,
            expected_next_attack_vocab=expected_next_attack_vocab,
            strategic_intent_vocab=strategic_intent_vocab,
            privacy_mechanism_vocab=privacy_mechanism_vocab,
            trust_mechanism_vocab=trust_mechanism_vocab,
            decoy_policy_vocab=decoy_policy_vocab,
            withholding_policy_vocab=withholding_policy_vocab,
            rationale_tag_vocab=rationale_tag_vocab,
        )

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)

    @property
    def n_fields(self) -> int:
        return len(self.field_vocab)

    def label_dims(self) -> dict[str, int]:
        return {
            "field_vocab": len(self.field_vocab),
            "attacker_confidence": len(self.attacker_confidence_vocab),
            "attacker_goal": len(self.attacker_goal_vocab),
            "safe_response_type": len(self.safe_response_type_vocab),
            "helpfulness": len(self.helpfulness_vocab),
            "probe_style": len(self.probe_style_vocab),
            "belief_update_risk": len(self.belief_update_risk_vocab),
            "deception_detection_risk": len(self.deception_detection_risk_vocab),
            "expected_next_attack": len(self.expected_next_attack_vocab),
            "strategic_intent": len(self.strategic_intent_vocab),
            "privacy_mechanism": len(self.privacy_mechanism_vocab),
            "trust_mechanism": len(self.trust_mechanism_vocab),
            "decoy_policy": len(self.decoy_policy_vocab),
            "withholding_policy": len(self.withholding_policy_vocab),
            "rationale_tags": len(self.rationale_tag_vocab),
        }

    def multi_hot(self, values: Any) -> torch.Tensor:
        return self._multi_hot_for_vocab(self.field_vocab, values, "field")

    @staticmethod
    def _multi_hot_for_vocab(vocab: list[str], values: Any, label_name: str) -> torch.Tensor:
        if isinstance(values, dict):
            values = values.keys()
        if values is None:
            values = []
        if isinstance(values, (str, bytes)):
            raise ValueError(f"Expected an iterable of {label_name} labels, got a string.")
        if not isinstance(values, (list, tuple, set)):
            try:
                values = list(values)
            except TypeError as exc:
                raise ValueError(f"Expected {label_name} list/dict for multi-hot label, got {type(values)}") from exc
        vec = torch.zeros(len(vocab), dtype=torch.float32)
        for value in values:
            item = str(value)
            if item not in vocab:
                raise ValueError(f"Unknown {label_name} label {item!r}; known labels={vocab}")
            vec[vocab.index(item)] = 1.0
        return vec

    def tag_multi_hot(self, values: Any) -> torch.Tensor:
        return self._multi_hot_for_vocab(self.rationale_tag_vocab, values, "rationale_tag")

    def field_index(self, value: Any) -> torch.Tensor:
        field = str(value)
        if field not in self.field_vocab:
            raise ValueError(f"Unknown target field {field!r}; known fields={self.field_vocab}")
        return torch.tensor(self.field_vocab.index(field), dtype=torch.long)

    @staticmethod
    def _cat_index(vocab: list[str], value: Any, label_name: str) -> torch.Tensor:
        item = str(value)
        if item not in vocab:
            raise ValueError(f"Unknown {label_name} label {item!r}; known labels={vocab}")
        return torch.tensor(vocab.index(item), dtype=torch.long)

    def cat_index(self, label_name: str, value: Any) -> torch.Tensor:
        vocab = {
            "attacker_confidence": self.attacker_confidence_vocab,
            "attacker_goal": self.attacker_goal_vocab,
            "safe_response_type": self.safe_response_type_vocab,
            "helpfulness": self.helpfulness_vocab,
            "probe_style": self.probe_style_vocab,
            "belief_update_risk": self.belief_update_risk_vocab,
            "deception_detection_risk": self.deception_detection_risk_vocab,
            "expected_next_attack": self.expected_next_attack_vocab,
            "strategic_intent": self.strategic_intent_vocab,
            "privacy_mechanism": self.privacy_mechanism_vocab,
            "trust_mechanism": self.trust_mechanism_vocab,
            "decoy_policy": self.decoy_policy_vocab,
            "withholding_policy": self.withholding_policy_vocab,
        }[label_name]
        return self._cat_index(vocab, value, label_name)


def _encode_structured_tom(sample: dict[str, Any], spec: StructuredToMSpec) -> dict[str, torch.Tensor]:
    labels = sample.get("tom_labels", {}) or {}
    first = labels.get("first_order_belief", {}) or {}
    second = labels.get("second_order_belief", {}) or {}
    boundary = labels.get("defender_safe_boundary", {}) or {}
    strategy_reasoning = _strategy_reasoning_labels(sample)
    return {
        "fo_known_fields": spec.multi_hot(first.get("attacker_known_fields", {}) or {}),
        "fo_missing_fields": spec.multi_hot(first.get("attacker_missing_fields", []) or []),
        "fo_target_field": spec.field_index(first.get("attacker_target_field")),
        "fo_attacker_confidence": spec.cat_index("attacker_confidence", first.get("attacker_confidence")),
        "fo_attacker_goal": spec.cat_index("attacker_goal", labels.get("attacker_goal")),
        "fo_can_confirm": spec.multi_hot(boundary.get("can_confirm", []) or []),
        "fo_must_not_reveal": spec.multi_hot(boundary.get("must_not_reveal", []) or []),
        "fo_safe_response_type": spec.cat_index("safe_response_type", boundary.get("safe_response_type")),
        "so_defender_knows": spec.multi_hot(second.get("attacker_believes_defender_knows", []) or []),
        "so_defender_knows_missing": spec.multi_hot(second.get("attacker_believes_defender_knows_missing", []) or []),
        "so_helpfulness": spec.cat_index("helpfulness", second.get("attacker_expects_defender_helpfulness")),
        "so_probe_style": spec.cat_index("probe_style", second.get("attacker_expected_probe_style")),
        "so_belief_update_risk": spec.cat_index("belief_update_risk", labels.get("belief_update_risk")),
        "so_deception_detection_risk": spec.cat_index("deception_detection_risk", labels.get("deception_detection_risk")),
        "so_expected_next_attack": spec.cat_index("expected_next_attack", labels.get("expected_next_attack")),
        "so_strategic_intent": spec.cat_index("strategic_intent", strategy_reasoning.get("strategic_intent")),
        "so_privacy_mechanism": spec.cat_index("privacy_mechanism", strategy_reasoning.get("privacy_mechanism")),
        "so_trust_mechanism": spec.cat_index("trust_mechanism", strategy_reasoning.get("trust_mechanism")),
        "so_decoy_policy": spec.cat_index("decoy_policy", strategy_reasoning.get("decoy_policy")),
        "so_withholding_policy": spec.cat_index("withholding_policy", strategy_reasoning.get("withholding_policy")),
        "so_rationale_tags": spec.tag_multi_hot(strategy_reasoning.get("rationale_tags", []) or []),
    }


class StructuredToMDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        *,
        spec: StructuredToMSpec,
        reward_dim: int,
        max_ctx_len: int,
        max_resp_len: int,
        use_expl_reward: bool,
        name: str,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.spec = spec
        self.reward_dim = reward_dim
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        self.use_expl_reward = use_expl_reward
        print(f"Loaded {len(self.records)} {name} structured ToM-SB samples", flush=True)

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
        ctx_enc = self._tokenize(str(sample.get("context_text", "")), self.max_ctx_len)
        pos_enc = self._tokenize(str(sample.get("pos_response", "")), self.max_resp_len)
        hard_negative = str(sample.get("hard_negative", ""))
        neg_enc = self._tokenize(hard_negative if hard_negative.strip() else str(sample.get("pos_response", "")), self.max_resp_len)
        if self.use_expl_reward:
            expl_enc = self._tokenize(str(sample.get("reward_explanations", "") or "N/A"), self.max_resp_len)
            expl_ids = expl_enc.input_ids.squeeze(0)
            expl_mask = expl_enc.attention_mask.squeeze(0)
        else:
            expl_ids, expl_mask = self._dummy_expl()
        item = {
            "ctx_input_ids": ctx_enc.input_ids.squeeze(0),
            "ctx_attention_mask": ctx_enc.attention_mask.squeeze(0),
            "pos_input_ids": pos_enc.input_ids.squeeze(0),
            "pos_attention_mask": pos_enc.attention_mask.squeeze(0),
            "neg_input_ids": neg_enc.input_ids.squeeze(0),
            "neg_attention_mask": neg_enc.attention_mask.squeeze(0),
            "expl_input_ids": expl_ids,
            "expl_attention_mask": expl_mask,
            "reward_vec": torch.tensor(reward_vec, dtype=torch.float32),
            "has_negative": torch.tensor(1.0 if hard_negative.strip() else 0.0),
        }
        item.update(_encode_structured_tom(sample, self.spec))
        return item


def collate_structured_fn(batch: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rewards = []
    has_neg = []
    first_pos_tokens = []
    structured_keys = [
        "fo_known_fields",
        "fo_missing_fields",
        "fo_target_field",
        "fo_attacker_confidence",
        "fo_attacker_goal",
        "fo_can_confirm",
        "fo_must_not_reveal",
        "fo_safe_response_type",
        "so_defender_knows",
        "so_defender_knows_missing",
        "so_helpfulness",
        "so_probe_style",
        "so_belief_update_risk",
        "so_deception_detection_risk",
        "so_expected_next_attack",
        "so_strategic_intent",
        "so_privacy_mechanism",
        "so_trust_mechanism",
        "so_decoy_policy",
        "so_withholding_policy",
        "so_rationale_tags",
    ]
    for item in batch:
        rewards.append(item["reward_vec"])
        has_neg.append(item["has_negative"])
        first_pos_tokens.append(item["pos_input_ids"][0])

    def _pad(key: str, value: int) -> torch.Tensor:
        return nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True, padding_value=value).long()

    collated = {
        "ctx_input_ids": _pad("ctx_input_ids", pad_id),
        "ctx_attention_mask": _pad("ctx_attention_mask", 0),
        "pos_input_ids": _pad("pos_input_ids", pad_id),
        "pos_attention_mask": _pad("pos_attention_mask", 0),
        "neg_input_ids": _pad("neg_input_ids", pad_id),
        "neg_attention_mask": _pad("neg_attention_mask", 0),
        "expl_input_ids": _pad("expl_input_ids", pad_id),
        "expl_attention_mask": _pad("expl_attention_mask", 0),
        "first_pos_token": torch.stack(first_pos_tokens, dim=0).long(),
        "reward_vec": torch.stack(rewards, dim=0),
        "has_negative": torch.stack(has_neg, dim=0),
    }
    for key in structured_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


def _make_prediction_head(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, output_dim),
    )


class StructuredRecursiveToMModel(nn.Module):
    Z_BELIEF_DIM = 48
    Z_INTENT_DIM = 40
    Z_THOUGHT_DIM = 40

    def __init__(
        self,
        base_model: nn.Module,
        *,
        reward_dim: int,
        spec: StructuredToMSpec,
        z_dim: int = 128,
        use_expl_reward: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.transformer = get_transformer_from_peft(self.base_model)
        self.hidden_size = base_model.get_input_embeddings().embedding_dim
        self.z_dim = z_dim
        self.reward_dim = reward_dim
        self.spec = spec
        self.use_expl_reward = use_expl_reward
        assert z_dim == self.Z_BELIEF_DIM + self.Z_INTENT_DIM + self.Z_THOUGHT_DIM

        self.z1_mu = nn.Linear(self.hidden_size, z_dim)
        self.z1_logvar = nn.Linear(self.hidden_size, z_dim)
        self.z2_mu = nn.Linear(self.hidden_size + z_dim, z_dim)
        self.z2_logvar = nn.Linear(self.hidden_size + z_dim, z_dim)

        self.joint_outcome_head = nn.Sequential(
            nn.Linear(2 * z_dim + self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, reward_dim),
        )
        self.z1_only_reward_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, reward_dim),
        )
        self.z_combined_reward_head = nn.Sequential(
            nn.Linear(2 * z_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, reward_dim),
        )
        self.z_to_hidden = nn.Linear(2 * z_dim, self.hidden_size)

        n_fields = spec.n_fields
        self.m1_heads = nn.ModuleDict(
            {
                "known_fields": _make_prediction_head(z_dim, n_fields),
                "missing_fields": _make_prediction_head(z_dim, n_fields),
                "target_field": _make_prediction_head(z_dim, n_fields),
                "attacker_confidence": _make_prediction_head(z_dim, len(spec.attacker_confidence_vocab)),
                "attacker_goal": _make_prediction_head(z_dim, len(spec.attacker_goal_vocab)),
                "can_confirm": _make_prediction_head(z_dim, n_fields),
                "must_not_reveal": _make_prediction_head(z_dim, n_fields),
                "safe_response_type": _make_prediction_head(z_dim, len(spec.safe_response_type_vocab)),
            }
        )
        self.m2_heads = nn.ModuleDict(
            {
                "defender_knows": _make_prediction_head(z_dim, n_fields),
                "defender_knows_missing": _make_prediction_head(z_dim, n_fields),
                "helpfulness": _make_prediction_head(z_dim, len(spec.helpfulness_vocab)),
                "probe_style": _make_prediction_head(z_dim, len(spec.probe_style_vocab)),
                "belief_update_risk": _make_prediction_head(z_dim, len(spec.belief_update_risk_vocab)),
                "deception_detection_risk": _make_prediction_head(z_dim, len(spec.deception_detection_risk_vocab)),
                "expected_next_attack": _make_prediction_head(z_dim, len(spec.expected_next_attack_vocab)),
                "strategic_intent": _make_prediction_head(z_dim, len(spec.strategic_intent_vocab)),
                "privacy_mechanism": _make_prediction_head(z_dim, len(spec.privacy_mechanism_vocab)),
                "trust_mechanism": _make_prediction_head(z_dim, len(spec.trust_mechanism_vocab)),
                "decoy_policy": _make_prediction_head(z_dim, len(spec.decoy_policy_vocab)),
                "withholding_policy": _make_prediction_head(z_dim, len(spec.withholding_policy_vocab)),
                "rationale_tags": _make_prediction_head(z_dim, len(spec.rationale_tag_vocab)),
            }
        )
        if self.use_expl_reward:
            self.expl_cross_attn = nn.MultiheadAttention(
                embed_dim=2 * z_dim,
                num_heads=8,
                kdim=self.hidden_size,
                vdim=self.hidden_size,
                batch_first=True,
                dropout=0.1,
            )
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
        for group in [self.joint_outcome_head, self.z1_only_reward_head, self.z_combined_reward_head]:
            for layer in group:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for heads in [self.m1_heads, self.m2_heads]:
            for head in heads.values():
                for layer in head:
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

    def _last_hidden(self, hidden: torch.Tensor, mask: torch.Tensor, start: int, end: int) -> torch.Tensor:
        h = hidden[start:end]
        m = mask[start:end]
        last_idx = m.sum(dim=1) - 1
        gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
        return h.gather(1, gather_idx).squeeze(1)

    def forward_all(
        self,
        ctx_input_ids,
        ctx_attention_mask,
        pos_input_ids,
        pos_attention_mask,
        neg_input_ids,
        neg_attention_mask,
        expl_input_ids,
        expl_attention_mask,
        first_pos_token,
        *,
        stop_grad_z1: bool = False,
    ) -> dict[str, Any]:
        batch_size = ctx_input_ids.size(0)
        context_hidden, z1, mu1, logvar1, z2, mu2, logvar2 = self.encode_z1_z2(
            ctx_input_ids, ctx_attention_mask, stop_grad_z1=stop_grad_z1
        )
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
        batched_hidden = self.transformer(
            input_ids=batched_ids,
            attention_mask=batched_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        pos_hidden = self._last_hidden(batched_hidden, batched_mask, 0, batch_size)
        neg_hidden = self._last_hidden(batched_hidden, batched_mask, batch_size, 2 * batch_size)
        pos_joint_reward = self.joint_outcome_head(torch.cat([z1, z2, pos_hidden], dim=1))
        neg_joint_reward = self.joint_outcome_head(torch.cat([z1, z2, neg_hidden], dim=1))
        z_cat = torch.cat([z1, z2], dim=1)
        z1_only_reward = self.z1_only_reward_head(z1)
        z_combined_reward = self.z_combined_reward_head(z_cat)
        output_embedding = self.base_model.get_output_embeddings()
        next_logits = F.linear(
            context_hidden + self.z_to_hidden(z_cat),
            output_embedding.weight,
            output_embedding.bias if getattr(output_embedding, "bias", None) is not None else None,
        )
        future_loss = F.cross_entropy(next_logits, first_pos_token)
        expl_reward_pred = None
        if self.use_expl_reward:
            expl_hidden = batched_hidden[2 * batch_size :]
            expl_mask = padded[2][1]
            attended_z, _ = self.expl_cross_attn(
                query=z_cat.unsqueeze(1),
                key=expl_hidden,
                value=expl_hidden,
                key_padding_mask=(expl_mask == 0),
            )
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
            "m1_logits": {name: head(z1) for name, head in self.m1_heads.items()},
            "m2_logits": {name: head(z2) for name, head in self.m2_heads.items()},
        }


def _bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float())


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels.long())


def _cat_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def _multi_exact(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits.float()) >= 0.5).float()
    return (pred == labels.float()).all(dim=1).float().mean().item()


def compute_structured_mental_losses(out: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    m1_logits = out["m1_logits"]
    m2_logits = out["m2_logits"]
    m1_components = {
        "m1_known_fields": _bce_loss(m1_logits["known_fields"], batch["fo_known_fields"]),
        "m1_missing_fields": _bce_loss(m1_logits["missing_fields"], batch["fo_missing_fields"]),
        "m1_target_field": _ce_loss(m1_logits["target_field"], batch["fo_target_field"]),
        "m1_attacker_confidence": _ce_loss(m1_logits["attacker_confidence"], batch["fo_attacker_confidence"]),
        "m1_attacker_goal": _ce_loss(m1_logits["attacker_goal"], batch["fo_attacker_goal"]),
        "m1_can_confirm": _bce_loss(m1_logits["can_confirm"], batch["fo_can_confirm"]),
        "m1_must_not_reveal": _bce_loss(m1_logits["must_not_reveal"], batch["fo_must_not_reveal"]),
        "m1_safe_response_type": _ce_loss(m1_logits["safe_response_type"], batch["fo_safe_response_type"]),
    }
    m2_components = {
        "m2_defender_knows": _bce_loss(m2_logits["defender_knows"], batch["so_defender_knows"]),
        "m2_defender_knows_missing": _bce_loss(m2_logits["defender_knows_missing"], batch["so_defender_knows_missing"]),
        "m2_helpfulness": _ce_loss(m2_logits["helpfulness"], batch["so_helpfulness"]),
        "m2_probe_style": _ce_loss(m2_logits["probe_style"], batch["so_probe_style"]),
        "m2_belief_update_risk": _ce_loss(m2_logits["belief_update_risk"], batch["so_belief_update_risk"]),
        "m2_deception_detection_risk": _ce_loss(m2_logits["deception_detection_risk"], batch["so_deception_detection_risk"]),
        "m2_expected_next_attack": _ce_loss(m2_logits["expected_next_attack"], batch["so_expected_next_attack"]),
        "m2_strategic_intent": _ce_loss(m2_logits["strategic_intent"], batch["so_strategic_intent"]),
        "m2_privacy_mechanism": _ce_loss(m2_logits["privacy_mechanism"], batch["so_privacy_mechanism"]),
        "m2_trust_mechanism": _ce_loss(m2_logits["trust_mechanism"], batch["so_trust_mechanism"]),
        "m2_decoy_policy": _ce_loss(m2_logits["decoy_policy"], batch["so_decoy_policy"]),
        "m2_withholding_policy": _ce_loss(m2_logits["withholding_policy"], batch["so_withholding_policy"]),
        "m2_rationale_tags": _bce_loss(m2_logits["rationale_tags"], batch["so_rationale_tags"]),
    }
    m1_loss = sum(m1_components.values()) / len(m1_components)
    m2_loss = sum(m2_components.values()) / len(m2_components)
    metrics = {key: value.item() for key, value in (m1_components | m2_components).items()}
    metrics.update(
        {
            "m1_known_exact": _multi_exact(m1_logits["known_fields"], batch["fo_known_fields"]),
            "m1_missing_exact": _multi_exact(m1_logits["missing_fields"], batch["fo_missing_fields"]),
            "m1_target_acc": _cat_accuracy(m1_logits["target_field"], batch["fo_target_field"]),
            "m1_safe_type_acc": _cat_accuracy(m1_logits["safe_response_type"], batch["fo_safe_response_type"]),
            "m2_missing_exact": _multi_exact(m2_logits["defender_knows_missing"], batch["so_defender_knows_missing"]),
            "m2_probe_style_acc": _cat_accuracy(m2_logits["probe_style"], batch["so_probe_style"]),
            "m2_next_attack_acc": _cat_accuracy(m2_logits["expected_next_attack"], batch["so_expected_next_attack"]),
            "m2_strategy_acc": _cat_accuracy(m2_logits["strategic_intent"], batch["so_strategic_intent"]),
            "m2_privacy_mechanism_acc": _cat_accuracy(m2_logits["privacy_mechanism"], batch["so_privacy_mechanism"]),
            "m2_rationale_exact": _multi_exact(m2_logits["rationale_tags"], batch["so_rationale_tags"]),
        }
    )
    return m1_loss, m2_loss, metrics


def compute_objective_components(
    out: dict[str, Any],
    batch: dict[str, torch.Tensor],
    current_opt_step: int,
    *,
    kl_weight: float,
    future_weight: float,
    mental1_weight: float,
    mental2_weight: float,
    expl_weight: float,
    z_only_weight: float,
    kl_anneal_steps: int,
    z2_kl_delay_steps: int,
    use_expl_reward: bool,
) -> tuple[torch.Tensor, dict[str, float], float, float]:
    reward_targets = batch["reward_vec"]
    has_neg = batch["has_negative"]
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
    mental1_loss, mental2_loss, structured_metrics = compute_structured_mental_losses(out, batch)
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
        + mental1_weight * mental1_loss
        + mental2_weight * mental2_loss
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
        "mental1_struct": mental1_loss.item(),
        "mental2_struct": mental2_loss.item(),
        "expl_reward": expl_reward_loss.item(),
    }
    metrics.update(structured_metrics)
    return total_loss, metrics, kl_weight * anneal1, kl_weight * anneal2


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    *,
    autocast_dtype: torch.dtype,
    kl_weight: float,
    future_weight: float,
    mental1_weight: float,
    mental2_weight: float,
    expl_weight: float,
    z_only_weight: float,
    grad_accum_steps: int,
    kl_anneal_steps: int,
    z2_kl_delay_steps: int,
    z2_warmup_steps: int,
    global_step_offset: int,
    max_grad_norm: float,
):
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
                batch["ctx_input_ids"],
                batch["ctx_attention_mask"],
                batch["pos_input_ids"],
                batch["pos_attention_mask"],
                batch["neg_input_ids"],
                batch["neg_attention_mask"],
                batch["expl_input_ids"],
                batch["expl_attention_mask"],
                batch["first_pos_token"],
                stop_grad_z1=current_opt_step < z2_warmup_steps,
            )
            raw_loss, batch_metrics, eff_kl1_w, eff_kl2_w = compute_objective_components(
                out,
                batch,
                current_opt_step,
                kl_weight=kl_weight,
                future_weight=future_weight,
                mental1_weight=mental1_weight,
                mental2_weight=mental2_weight,
                expl_weight=expl_weight,
                z_only_weight=z_only_weight,
                kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps,
                use_expl_reward=model.use_expl_reward,
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
                f"kl2={totals['kl2'] / n:.4f}(w={eff_kl2_w:.4f}) m1={totals['mental1_struct'] / n:.4f} "
                f"m2={totals['mental2_struct'] / n:.4f} m1_target_acc={totals['m1_target_acc'] / n:.3f} "
                f"m2_strategy_acc={totals['m2_strategy_acc'] / n:.3f} "
                f"m2_next_acc={totals['m2_next_attack_acc'] / n:.3f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
    n = max(1, num_batches)
    return total_loss / n, {key: value / n for key, value in totals.items()}, global_step


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    device,
    *,
    autocast_dtype: torch.dtype,
    current_opt_step: int,
    kl_weight: float,
    future_weight: float,
    mental1_weight: float,
    mental2_weight: float,
    expl_weight: float,
    z_only_weight: float,
    kl_anneal_steps: int,
    z2_kl_delay_steps: int,
):
    model.eval()
    totals = Counter()
    total_loss = 0.0
    for batch in dataloader:
        batch = _move_batch(batch, device)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda", dtype=autocast_dtype):
            out = model.forward_all(
                batch["ctx_input_ids"],
                batch["ctx_attention_mask"],
                batch["pos_input_ids"],
                batch["pos_attention_mask"],
                batch["neg_input_ids"],
                batch["neg_attention_mask"],
                batch["expl_input_ids"],
                batch["expl_attention_mask"],
                batch["first_pos_token"],
                stop_grad_z1=False,
            )
            raw_loss, batch_metrics, _, _ = compute_objective_components(
                out,
                batch,
                current_opt_step,
                kl_weight=kl_weight,
                future_weight=future_weight,
                mental1_weight=mental1_weight,
                mental2_weight=mental2_weight,
                expl_weight=expl_weight,
                z_only_weight=z_only_weight,
                kl_anneal_steps=kl_anneal_steps,
                z2_kl_delay_steps=z2_kl_delay_steps,
                use_expl_reward=model.use_expl_reward,
            )
        total_loss += raw_loss.item()
        for key, value in batch_metrics.items():
            totals[key] += value
    n = max(1, len(dataloader))
    return total_loss / n, {key: value / n for key, value in totals.items()}


def save_checkpoint(model: StructuredRecursiveToMModel, save_dir: Path, metadata: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(save_dir / "lora_adapter")
    for head_name in model.custom_head_names:
        torch.save(getattr(model, head_name).state_dict(), save_dir / f"{head_name}.pth")
    _save_json(save_dir / "metadata.json", metadata | {"saved_head_names": model.custom_head_names})


def _resolve_checkpoint_dir(path: str | Path) -> Path:
    ckpt_dir = Path(path)
    if (ckpt_dir / "best").is_dir() and not (ckpt_dir / "lora_adapter").is_dir():
        ckpt_dir = ckpt_dir / "best"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {ckpt_dir}")
    if not (ckpt_dir / "lora_adapter").is_dir():
        raise FileNotFoundError(f"Resume checkpoint is missing lora_adapter/: {ckpt_dir}")
    return ckpt_dir


def _load_checkpoint_metadata(ckpt_dir: Path) -> dict[str, Any]:
    metadata_path = ckpt_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Resume checkpoint is missing metadata.json: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected checkpoint metadata object in {metadata_path}")
    return metadata


def _validate_resume_metadata(
    *,
    ckpt_dir: Path,
    metadata: dict[str, Any],
    reward_schema: list[str],
    spec: StructuredToMSpec,
    use_expl_reward: bool,
    z_dim: int,
) -> None:
    if metadata.get("model_family") != "tom_sb_recursive_structured_mental_reward":
        raise ValueError(f"Checkpoint is not a structured ToM-SB reward checkpoint: {ckpt_dir}")
    if metadata.get("reward_schema") != reward_schema:
        raise ValueError(
            f"Resume checkpoint reward schema mismatch.\n"
            f"Checkpoint: {metadata.get('reward_schema')}\nCurrent:    {reward_schema}"
        )
    if metadata.get("structured_tom_spec") != spec.to_dict():
        raise ValueError("Resume checkpoint structured_tom_spec does not match the current train data/spec.")
    if bool(metadata.get("use_expl_reward", False)) != use_expl_reward:
        raise ValueError("Resume checkpoint use_expl_reward does not match current --expl_weight setting.")
    if int(metadata.get("z_dim", z_dim)) != z_dim:
        raise ValueError(f"Resume checkpoint z_dim={metadata.get('z_dim')} does not match current z_dim={z_dim}.")


def load_custom_heads(model: StructuredRecursiveToMModel, ckpt_dir: Path, device: torch.device) -> None:
    for head_name in model.custom_head_names:
        state_path = ckpt_dir / f"{head_name}.pth"
        if not state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint missing {state_path.name}: {state_path}")
        state = torch.load(state_path, map_location=device)
        getattr(model, head_name).load_state_dict(state)


def _fully_annealed_eval_step(kl_anneal_steps: int, z2_kl_delay_steps: int) -> int:
    if kl_anneal_steps <= 0:
        return 1
    return max(1, (2 * kl_anneal_steps) + z2_kl_delay_steps + 1)


def _print_validation_summary(name: str, summary: ValidationSummary) -> None:
    print(
        f"{name}: records={summary.num_records}, unique_ids={summary.unique_example_ids}, "
        f"base_scenarios={summary.base_scenarios}, issues={summary.issue_count}, "
        f"positive_truth_leaks={summary.positive_truth_leaks}",
        flush=True,
    )
    print(f"  strategy_dist={summary.strategy_dist}", flush=True)
    print(f"  confidence_dist={summary.confidence_dist}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ToM-SB recursive reward trainer with structured ToM losses.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=str, default=DEFAULT_VAL_PATH)
    parser.add_argument("--reward_schema_path", type=str, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a structured checkpoint directory, e.g. .../best. If a run root is passed, best/ is used.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--max_ctx_len", type=int, default=1536)
    parser.add_argument("--max_resp_len", type=int, default=256)
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
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Load --resume_from_checkpoint and run validation metrics only; no optimizer, training, or checkpoints.",
    )
    parser.add_argument("--skip_data_validation", action="store_true")
    parser.add_argument("--require_scenario_disjoint_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_missing_val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.gpu is not None:
        print("Warning: --gpu is deprecated and ignored. Set CUDA_VISIBLE_DEVICES in the shell instead.", flush=True)
    if args.eval_only and not args.resume_from_checkpoint:
        raise ValueError("--eval_only requires --resume_from_checkpoint.")
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

    resume_ckpt_dir: Path | None = None
    resume_metadata: dict[str, Any] | None = None
    if args.resume_from_checkpoint:
        resume_ckpt_dir = _resolve_checkpoint_dir(args.resume_from_checkpoint)
        resume_metadata = _load_checkpoint_metadata(resume_ckpt_dir)
        ckpt_spec = resume_metadata.get("structured_tom_spec")
        if not isinstance(ckpt_spec, dict):
            raise ValueError(f"Resume checkpoint is missing structured_tom_spec: {resume_ckpt_dir}")
        spec = StructuredToMSpec(**ckpt_spec)
    else:
        spec = StructuredToMSpec.from_records(train_records)
    for record in train_records:
        _encode_structured_tom(record, spec)
    for record in val_records:
        _encode_structured_tom(record, spec)
    print(f"Structured ToM spec: {json.dumps(spec.label_dims(), sort_keys=True)}", flush=True)

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
    if resume_ckpt_dir is not None and resume_metadata is not None:
        _validate_resume_metadata(
            ckpt_dir=resume_ckpt_dir,
            metadata=resume_metadata,
            reward_schema=reward_schema,
            spec=spec,
            use_expl_reward=use_expl_reward,
            z_dim=args.z_dim,
        )
        print(f"Resuming model weights from: {resume_ckpt_dir}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "args.json", vars(args))
    _save_json(output_dir / "data_validation.json", validation_payload)
    _save_json(output_dir / "structured_tom_spec.json", spec.to_dict())
    checkpoint_metadata = {
        "model_family": "tom_sb_recursive_structured_mental_reward",
        "reward_schema": reward_schema,
        "reward_schema_path": str(args.reward_schema_path),
        "base_model": args.model_name,
        "mental_loss_type": "structured_prediction",
        "structured_tom_spec": spec.to_dict(),
        "structured_tom_dims": spec.label_dims(),
        "use_expl_reward": use_expl_reward,
        "z_dim": args.z_dim,
        "resumed_from_checkpoint": str(resume_ckpt_dir) if resume_ckpt_dir else None,
        "resume_source_epoch": resume_metadata.get("epoch") if resume_metadata else None,
        "resume_source_best_metric": resume_metadata.get("best_metric") if resume_metadata else None,
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

    if resume_ckpt_dir is not None:
        print(f"Loading LoRA adapter from {resume_ckpt_dir / 'lora_adapter'}", flush=True)
        base_model = PeftModel.from_pretrained(base_model, resume_ckpt_dir / "lora_adapter", is_trainable=True)
    else:
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

    model = StructuredRecursiveToMModel(
        base_model,
        reward_dim=len(reward_schema),
        spec=spec,
        z_dim=args.z_dim,
        use_expl_reward=use_expl_reward,
    ).to(device)
    if resume_ckpt_dir is not None:
        load_custom_heads(model, resume_ckpt_dir, device)
        print("Loaded structured reward/ToM heads from checkpoint.", flush=True)
    for name, param in model.named_parameters():
        if any(head_name in name for head_name in model.custom_head_names):
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    train_dataset = StructuredToMDataset(
        train_records,
        tokenizer,
        spec=spec,
        reward_dim=len(reward_schema),
        max_ctx_len=args.max_ctx_len,
        max_resp_len=args.max_resp_len,
        use_expl_reward=use_expl_reward,
        name="train",
    )
    val_dataset = (
        StructuredToMDataset(
            val_records,
            tokenizer,
            spec=spec,
            reward_dim=len(reward_schema),
            max_ctx_len=args.max_ctx_len,
            max_resp_len=args.max_resp_len,
            use_expl_reward=use_expl_reward,
            name="val",
        )
        if val_records
        else None
    )
    print(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}", flush=True)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": lambda batch: collate_structured_fn(batch, tokenizer),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None
    if args.eval_only:
        if val_loader is None:
            raise ValueError("--eval_only requires a validation set. Provide --val_path.")
        eval_step = _fully_annealed_eval_step(args.kl_anneal_steps, args.z2_kl_delay_steps)
        print(f"Eval-only mode: running validation at fully annealed opt_step={eval_step}.", flush=True)
        val_loss, val_metrics = evaluate_epoch(
            model,
            val_loader,
            device,
            autocast_dtype=dtype,
            current_opt_step=eval_step,
            kl_weight=args.kl_weight,
            future_weight=args.future_weight,
            mental1_weight=args.mental1_weight,
            mental2_weight=args.mental2_weight,
            expl_weight=args.expl_weight,
            z_only_weight=args.z_only_weight,
            kl_anneal_steps=args.kl_anneal_steps,
            z2_kl_delay_steps=args.z2_kl_delay_steps,
        )
        print(f"eval_val_loss: {val_loss:.4f}", flush=True)
        for key, value in val_metrics.items():
            print(f"eval_val_{key}: {value:.4f}", flush=True)
        _save_json(
            output_dir / "eval_metrics.json",
            {
                "checkpoint": str(resume_ckpt_dir),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "eval_opt_step": eval_step,
            },
        )
        print(f"Eval-only complete. Metrics saved to {output_dir / 'eval_metrics.json'}", flush=True)
        return

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
        raise RuntimeError("No custom reward/structured ToM head parameters are trainable.")
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Optimizer steps: total={total_steps}, warmup={warmup_steps}", flush=True)

    best_metric = float("inf")
    global_step = 0
    for epoch in range(args.num_epochs):
        avg_loss, metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            autocast_dtype=dtype,
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
            val_loss, val_metrics = evaluate_epoch(
                model,
                val_loader,
                device,
                autocast_dtype=dtype,
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
