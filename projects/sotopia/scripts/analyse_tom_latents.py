#!/usr/bin/env python3
"""
Analyze the recursive ToM latent space used by the v3 Sotopia reward model.

This script builds three paper-facing artifacts from the same cached latent set:

1. Figure 1: UMAP/t-SNE of z colored by a fine-grained interaction label
2. Table 1: linear probe performance for intent / knowledge / strategy labels
3. Figure 2: latent traversal along a concept direction and how generated
   outputs / reward properties change

By default, probe labels are derived from observed response text only,
not from the mental-state annotation targets that trained the latent. You can
also provide an external JSONL of independently annotated labels via
``--label_source external --label_jsonl ...``.

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HOME="artifacts/huggingface/" python \
    projects/sotopia/analyze_recursive_tom_latents.py \
      --base_model_name Qwen/Qwen2.5-7B-Instruct \
      --checkpoint_dir projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/best \
      --data_path projects/sotopia/data/sotopia_turn_rewards_v3.jsonl \
      --output_dir projects/sotopia/runs/analysis/mental_latent_qwen_v3 \
      --label_source response_heuristic \
      --concept_field strategy \
      --concept_value negotiate_offer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

try:
    import umap  # type: ignore
except Exception:
    umap = None

from stage2_grpo_agent_training_v3 import (  # noqa: E402
    FrozenRewardModel,
    SOTOPIA_DIMENSIONS,
)


PLOT_BLUE = "#2F6FAE"
PLOT_GREEN = "#2CA25F"
PLOT_GREY = "#7A7A7A"
PLOT_LIGHT_GREY = "#D6D8DC"
PLOT_DARK_GREY = "#4B4B4B"
PLOT_PURPLE = "#7B61B5"
PLOT_PALETTE = [PLOT_BLUE, PLOT_GREEN, PLOT_PURPLE, PLOT_GREY]
PLOT_CMAP = LinearSegmentedColormap.from_list(
    "tom_blue_green_purple",
    [PLOT_LIGHT_GREY, PLOT_BLUE, PLOT_GREEN, PLOT_PURPLE],
)
PLOT_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "tom_grey_purple",
    [PLOT_LIGHT_GREY, PLOT_PURPLE],
)


def style_plot_axes(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#C7CBD1", linewidth=0.9, alpha=0.9)
    ax.grid(True, which="minor", color="#E8EAED", linewidth=0.5, alpha=0.9)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PLOT_DARK_GREY)
        spine.set_linewidth(1.0)


def style_plot_legend(legend: Any) -> None:
    if legend is None:
        return
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(PLOT_LIGHT_GREY)
    legend.get_frame().set_alpha(0.94)
    for text in legend.get_texts():
        text.set_fontsize(11)


def style_plot_colorbar(colorbar: Any, label: str | None = None) -> None:
    colorbar.ax.tick_params(labelsize=11, width=1.1)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    if label:
        colorbar.set_label(label, fontsize=12, fontweight="bold")


def label_color_map(labels: list[str]) -> dict[str, Any]:
    colors: dict[str, Any] = {}
    color_idx = 0
    for label in labels:
        if label == "other":
            colors[label] = (0.68, 0.68, 0.68, 0.60)
        else:
            colors[label] = PLOT_PALETTE[color_idx % len(PLOT_PALETTE)]
            color_idx += 1
    return colors


def annotate_panel(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=PLOT_DARK_GREY,
        bbox=dict(facecolor="white", edgecolor=PLOT_LIGHT_GREY, alpha=0.88, pad=3),
    )


REWARD_DIM = len(SOTOPIA_DIMENSIONS)
RESPONSE_HEURISTIC_VERSION = "response_only_v2"

INTENT_LABELS = [
    "negotiate",
    "persuade",
    "support",
    "coordinate",
    "decline",
    "socialize",
    "other",
]

KNOWLEDGE_LABELS = [
    "seek_information",
    "share_information",
    "speculate_uncertain",
    "state_confident_belief",
    "conceal_evasive",
    "other",
]

STRATEGY_LABELS = [
    "questioning",
    "offer_proposal",
    "validation",
    "commitment",
    "hedging",
    "boundary_setting",
    "explanation",
    "other",
]

INTERACTION_LABELS = [
    "ask_clarify",
    "make_offer",
    "validate_support",
    "commit_accept",
    "boundary_refusal",
    "coordinate_plan",
    "explain_disclose",
    "rapport_social",
    "evasive_redirect",
    "other",
]

HEDGE_WORDS = {
    "maybe", "perhaps", "might", "could", "possibly", "probably", "somewhat",
    "seems", "likely", "appears", "suggests",
}
UNCERTAINTY_WORDS = {
    "uncertain", "unsure", "guess", "wonder", "doubt", "unknown",
    "maybe", "perhaps", "might", "seems", "appears", "likely",
}
UNCERTAINTY_PHRASES = {
    "not sure",
    "don't know",
    "do not know",
    "hard to tell",
    "unclear",
    "i guess",
    "i think",
}
ASSERTIVE_WORDS = {
    "must", "need", "definitely", "absolutely", "certainly", "will", "should",
    "require", "insist", "important",
}
NEGOTIATION_WORDS = {
    "price", "offer", "deal", "compromise", "counter", "flexible", "willing",
    "accept", "agreement", "middle", "ground",
}
DISCLOSURE_WORDS = {
    "share", "explain", "tell", "reveal", "clarify", "describe", "inform",
}
SUPPORT_WORDS = {
    "support", "help", "care", "understand", "appreciate", "sorry",
    "reassure", "comfort", "glad", "thank",
}
CONCEAL_WORDS = {
    "private", "confidential", "secret", "hidden", "conceal", "withhold",
    "avoid", "deflect", "rather", "prefer", "comfortable",
}
COORDINATION_WORDS = {
    "schedule", "plan", "meeting", "meet", "timeline", "arrange",
    "coordinate", "together", "next step",
}
REFUSAL_WORDS = {
    "can't", "cannot", "won't", "wouldn't", "decline", "refuse",
    "no thanks", "prefer not", "not comfortable",
}
PERSUASION_WORDS = {
    "should", "need to", "important", "worth", "benefit", "consider",
    "encourage", "convince", "persuade",
}
QUESTION_CUES = {
    "what", "how", "why", "when", "where", "who", "could you",
    "can you", "would you", "do you", "are you", "tell me",
}


@dataclass
class TurnRecord:
    episode_id: str
    turn_num: int
    speaker: str
    scenario: str
    context_text: str
    response_text: str
    mental1_text: str
    mental2_text: str
    reward_vec: list[float]
    intent_label: str
    knowledge_label: str
    strategy_label: str
    interaction_label: str
    label_source: str

    @property
    def observed_text(self) -> str:
        return (
            f"{self.context_text}\nResponse: {self.response_text}"
        )

    @property
    def probe_text(self) -> str:
        if self.label_source == "response_heuristic":
            return self.response_text
        return self.observed_text


def normalize_score(dim: str, score: float) -> float:
    dim_ranges = {
        "believability": (0, 10),
        "relationship": (-5, 5),
        "knowledge": (0, 10),
        # Sotopia anchors these as negative-cost dimensions: 0 is best, -10 worst.
        "secret": (-10, 0),
        "social_rules": (-10, 0),
        "financial_and_material_benefits": (-5, 5),
        "goal": (0, 10),
    }
    lo, hi = dim_ranges[dim]
    return float((score - lo) / (hi - lo + 1e-8))


def _match_score(text: str, rules: OrderedDict[str, list[str]]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for label, phrases in rules.items():
        score = 0
        for phrase in phrases:
            if phrase in text:
                score += 1
        scores[label] = score
    return scores


def _pick_unique_best_label(scores: dict[str, int], fallback: str = "other") -> str:
    positive = {label: score for label, score in scores.items() if score > 0}
    if not positive:
        return fallback
    best_score = max(positive.values())
    winners = [label for label, score in positive.items() if score == best_score]
    if len(winners) != 1:
        return fallback
    return winners[0]


def derive_intent_label(response_text: str) -> str:
    text = response_text.lower()
    rules = OrderedDict(
        [
            (
                "negotiate",
                [
                    "compromise", "price", "deal", "negotiat", "offer",
                    "middle ground", "agreement", "counter", "budget",
                    "payment", "sell", "buy",
                ],
            ),
            (
                "persuade",
                [
                    "convince", "persuade", "encourage", "benefit",
                    "important", "need to", "should", "worth",
                    "consider", "make the case",
                ],
            ),
            (
                "support",
                [
                    "reassure", "support", "help", "assist", "comfort",
                    "sorry", "here for you", "care", "glad to help",
                ],
            ),
            (
                "coordinate",
                [
                    "let's", "schedule", "plan", "arrange", "coordinate",
                    "timeline", "next step", "meet", "together",
                    "follow up",
                ],
            ),
            (
                "decline",
                [
                    "can't", "cannot", "won't", "wouldn't", "decline",
                    "refuse", "prefer not", "not comfortable", "no thanks",
                    "rather not",
                ],
            ),
            (
                "socialize",
                [
                    "friend", "coffee", "dinner", "fun", "joke",
                    "playful", "flirt", "laugh", "hang out",
                    "rapport",
                ],
            ),
        ]
    )
    return _pick_unique_best_label(_match_score(text, rules))


def derive_knowledge_label(response_text: str) -> str:
    text = response_text.lower()
    rules = OrderedDict(
        [
            (
                "seek_information",
                [
                    "?", "what", "how", "why", "when", "where", "who",
                    "could you", "can you", "would you", "tell me",
                    "help me understand",
                ],
            ),
            (
                "share_information",
                [
                    "because", "for example", "let me explain", "the reason",
                    "specifically", "details", "here's", "here is",
                    "i have", "it means",
                ],
            ),
            (
                "speculate_uncertain",
                [
                    "maybe", "perhaps", "might", "could be", "not sure",
                    "i think", "i guess", "likely", "seems", "appears",
                ],
            ),
            (
                "state_confident_belief",
                [
                    "i know", "definitely", "certainly", "clearly",
                    "i'm sure", "must", "will", "it is",
                ],
            ),
            (
                "conceal_evasive",
                [
                    "prefer not", "rather not", "private", "confidential",
                    "can't share", "cannot share", "don't want to say",
                    "not disclose", "keep that private",
                ],
            ),
        ]
    )
    return _pick_unique_best_label(_match_score(text, rules))


def derive_strategy_label(response_text: str) -> str:
    text = response_text.lower()
    rules = OrderedDict(
        [
            (
                "questioning",
                [
                    "?", "what", "how", "why", "when", "where",
                    "could you", "can you", "would you", "tell me",
                ],
            ),
            (
                "offer_proposal",
                [
                    "offer", "how about", "i propose", "can we", "could we",
                    "let's", "compromise", "middle ground", "counter",
                    "would you take",
                ],
            ),
            (
                "validation",
                [
                    "i understand", "that makes sense", "appreciate", "thank",
                    "i hear you", "sorry", "glad", "valid",
                ],
            ),
            (
                "commitment",
                [
                    "i will", "i'll", "agreed", "sounds good", "done",
                    "i accept", "i can commit", "i'm in",
                ],
            ),
            (
                "hedging",
                [
                    "maybe", "perhaps", "might", "could", "possibly",
                    "likely", "somewhat", "a bit", "kind of", "i think",
                ],
            ),
            (
                "boundary_setting",
                [
                    "can't", "cannot", "won't", "wouldn't", "prefer not",
                    "not comfortable", "decline", "refuse",
                ],
            ),
            (
                "explanation",
                [
                    "because", "the reason", "for example", "specifically",
                    "to explain", "details", "clarify", "describe",
                ],
            ),
        ]
    )
    return _pick_unique_best_label(_match_score(text, rules))


def derive_interaction_label(response_text: str) -> str:
    text = response_text.lower()
    rules = OrderedDict(
        [
            ("make_offer", ["offer", "price", "deal", "compromise", "counter", "how about", "would you take"]),
            ("ask_clarify", ["?", "what", "how", "why", "when", "where", "could you", "can you", "tell me"]),
            ("validate_support", ["i understand", "appreciate", "thank", "sorry", "glad", "support"]),
            ("commit_accept", ["agreed", "sounds good", "done", "i accept", "i will", "i'll"]),
            ("boundary_refusal", ["can't", "cannot", "won't", "decline", "refuse", "prefer not", "not comfortable"]),
            ("coordinate_plan", ["schedule", "plan", "arrange", "timeline", "next step", "meet", "coordinate"]),
            ("explain_disclose", ["because", "the reason", "for example", "specifically", "let me explain", "details"]),
            ("rapport_social", ["friend", "fun", "coffee", "dinner", "joke", "playful", "laugh"]),
            ("evasive_redirect", ["rather not", "private", "confidential", "can't share", "don't want to say", "keep that private"]),
        ]
    )
    return _pick_unique_best_label(_match_score(text, rules))


def derive_labels(response_text: str) -> tuple[str, str, str, str]:
    intent = derive_intent_label(response_text)
    knowledge = derive_knowledge_label(response_text)
    strategy = derive_strategy_label(response_text)
    interaction = derive_interaction_label(response_text)
    return intent, knowledge, strategy, interaction


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_records_cache_meta(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cache_type": "records",
        "data_path": os.path.abspath(args.data_path),
        "label_source": args.label_source,
        "label_jsonl": os.path.abspath(args.label_jsonl) if args.label_jsonl else None,
        "response_heuristic_version": RESPONSE_HEURISTIC_VERSION,
    }


def build_latent_cache_meta(
    args: argparse.Namespace,
    records_path: Path,
    records: list[TurnRecord],
) -> dict[str, Any]:
    return {
        "cache_type": "latents",
        "base_model_name": args.base_model_name,
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "data_path": os.path.abspath(args.data_path),
        "label_source": args.label_source,
        "label_jsonl": os.path.abspath(args.label_jsonl) if args.label_jsonl else None,
        "z_dim": args.z_dim,
        "max_ctx_len": args.max_ctx_len,
        "num_records": len(records),
        "records_path": os.path.abspath(str(records_path)),
        "response_heuristic_version": RESPONSE_HEURISTIC_VERSION,
        "balanced_subset_enabled": not args.use_full_records,
        "balanced_subset_field": args.balanced_subset_field,
        "balanced_subset_per_label": args.balanced_subset_per_label,
        "balanced_subset_min_label_count": args.balanced_subset_min_label_count,
        "balanced_subset_keep_other": args.balanced_subset_keep_other,
        "records_signature": make_records_signature(records),
    }


def make_records_signature(records: list[TurnRecord]) -> str:
    digest = hashlib.sha1()
    for record in records:
        digest.update(
            f"{record.episode_id}\t{record.turn_num}\t{record.speaker}\n".encode("utf-8")
        )
    return digest.hexdigest()


def load_external_labels(path: str) -> dict[tuple[str, int, str], dict[str, str]]:
    labels: dict[tuple[str, int, str], dict[str, str]] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = (str(obj["episode_id"]), int(obj["turn_num"]), str(obj["speaker"]))
            labels[key] = {
                "intent_label": obj["intent_label"],
                "knowledge_label": obj["knowledge_label"],
                "strategy_label": obj["strategy_label"],
                "interaction_label": obj.get("interaction_label", obj["strategy_label"]),
            }
    return labels


def load_turn_records(
    data_path: str,
    label_source: str,
    external_labels: dict[tuple[str, int, str], dict[str, str]] | None = None,
) -> list[TurnRecord]:
    records: list[TurnRecord] = []
    with open(data_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
                _append_episode_records(records, episode, label_source, external_labels)
            except json.JSONDecodeError:
                parts = line.split("}{")
                for i, part in enumerate(parts):
                    obj_str = part
                    if i > 0:
                        obj_str = "{" + obj_str
                    if i < len(parts) - 1:
                        obj_str = obj_str + "}"
                    try:
                        episode = json.loads(obj_str)
                    except json.JSONDecodeError:
                        continue
                    _append_episode_records(records, episode, label_source, external_labels)
    return records


def _append_episode_records(
    records: list[TurnRecord],
    episode: dict[str, Any],
    label_source: str,
    external_labels: dict[tuple[str, int, str], dict[str, str]] | None,
) -> None:
    pe = episode["parsed_episode"]
    turns = pe.get("turns", [])
    turn_rewards = episode.get("turn_rewards", [])
    if not turns or not turn_rewards:
        return

    scenario = pe.get("scenario", "")
    episode_id = pe.get("pk", episode.get("episode_id", "unknown"))

    agent_1_name = pe.get("agent_1_name", "Agent 1")
    agent_2_name = pe.get("agent_2_name", "Agent 2")
    agent_1_bg = pe.get("agent_1_background", "")
    agent_2_bg = pe.get("agent_2_background", "")
    agent_1_goal = pe.get("agent_1_goal", "")
    agent_2_goal = pe.get("agent_2_goal", "")

    for tr in turn_rewards:
        turn_num = tr["turn"]
        speaker = tr["agent"]
        if speaker == agent_1_name:
            rewards_key = "agent_1_rewards"
            bg = agent_1_bg
            goal = agent_1_goal
            secret = pe.get("agent_1_secret", "")
        elif speaker == agent_2_name:
            rewards_key = "agent_2_rewards"
            bg = agent_2_bg
            goal = agent_2_goal
            secret = pe.get("agent_2_secret", "")
        else:
            continue

        agent_rewards = tr.get(rewards_key, {})
        if not agent_rewards:
            continue

        if turn_num >= len(turns):
            continue
        response_text = turns[turn_num].get("content", "").strip()
        if not response_text:
            continue

        reward_vec: list[float] = []
        for dim in SOTOPIA_DIMENSIONS:
            dim_data = agent_rewards.get(dim, {})
            score = dim_data.get("score", 0) if isinstance(dim_data, dict) else 0
            reward_vec.append(normalize_score(dim, score))

        mental_state = agent_rewards.get("mental_state", {})
        partner_belief = mental_state.get("partner_belief", "")
        strategic_intent = mental_state.get("strategic_intent", "")
        thought_process = mental_state.get("thought_process", "")
        second_order_belief = mental_state.get("second_order_belief", "")
        second_order_intent = mental_state.get("second_order_intent", "")
        second_order_thought = mental_state.get("second_order_thought", "")

        history_lines = []
        for prev_t in turns[:turn_num]:
            spk = prev_t.get("agent", "Unknown")
            act = prev_t.get("action", "said")
            content = prev_t.get("content", "")
            history_lines.append(f"Turn {prev_t['turn']+1} | {spk} {act}: {content}")

        mental1_parts = []
        if partner_belief:
            mental1_parts.append(f"Partner Belief: {partner_belief}")
        if strategic_intent:
            mental1_parts.append(f"Strategic Intent: {strategic_intent}")
        if thought_process:
            mental1_parts.append(f"Thought Process: {thought_process}")
        mental1_text = " | ".join(mental1_parts)

        mental2_parts = []
        if second_order_belief:
            mental2_parts.append(f"Second-Order Belief: {second_order_belief}")
        if second_order_intent:
            mental2_parts.append(f"Second-Order Intent: {second_order_intent}")
        if second_order_thought:
            mental2_parts.append(f"Second-Order Thought: {second_order_thought}")
        mental2_text = " | ".join(mental2_parts)

        context_text = (
            f"Scenario: {scenario}\n"
            f"Background: {bg}\n"
            f"Goal: {goal}\n"
            f"Secret: {secret if secret else 'None'}\n"
            f"Dialogue History:\n{os.linesep.join(history_lines)}\n"
            f"Turn {turn_num+1} | {speaker}:"
        )

        if label_source == "external":
            key = (str(episode_id), int(turn_num), str(speaker))
            if external_labels is None or key not in external_labels:
                raise KeyError(f"Missing external labels for {key}")
            label_bundle = external_labels[key]
            intent = label_bundle["intent_label"]
            knowledge = label_bundle["knowledge_label"]
            strategy = label_bundle["strategy_label"]
            interaction = label_bundle["interaction_label"]
            record_label_source = "external"
        else:
            intent, knowledge, strategy, interaction = derive_labels(response_text=response_text)
            record_label_source = "response_heuristic"
        records.append(
            TurnRecord(
                episode_id=episode_id,
                turn_num=turn_num,
                speaker=speaker,
                scenario=scenario,
                context_text=context_text,
                response_text=response_text,
                mental1_text=mental1_text,
                mental2_text=mental2_text,
                reward_vec=reward_vec,
                intent_label=intent,
                knowledge_label=knowledge,
                strategy_label=strategy,
                interaction_label=interaction,
                label_source=record_label_source,
            )
        )


def save_records_jsonl(records: list[TurnRecord], path: Path) -> None:
    with path.open("w") as f:
        for record in records:
            json.dump(asdict(record), f)
            f.write("\n")


def load_records_jsonl(path: Path) -> list[TurnRecord]:
    records: list[TurnRecord] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload.setdefault("label_source", "legacy_unknown")
            records.append(TurnRecord(**payload))
    return records


def load_or_extract_latents(
    args: argparse.Namespace,
    records: list[TurnRecord],
    cache_path: Path,
    cache_meta_path: Path,
    expected_cache_meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    current_cache_meta = load_json(cache_meta_path)
    if (
        cache_path.exists()
        and not args.recompute_latents
        and current_cache_meta == expected_cache_meta
    ):
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    bundle = FrozenRewardModel(
        base_model_name=args.base_model_name,
        checkpoint_dir=args.checkpoint_dir,
        z_dim=args.z_dim,
        device=args.device,
        scoring_dim_indices=None,
        ensemble_weight=args.ensemble_weight,
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = args.device

    z1s: list[np.ndarray] = []
    z2s: list[np.ndarray] = []
    ctx_hiddens: list[np.ndarray] = []
    reward_vecs = np.asarray([r.reward_vec for r in records], dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            contexts = [r.context_text for r in batch_records]
            enc = tokenizer(
                contexts,
                truncation=True,
                max_length=args.max_ctx_len,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

            with (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.startswith("cuda")
                else nullcontext()
            ):
                hidden = model._encode_sequence(input_ids, attention_mask)
                pooled = model._pool(hidden, attention_mask)
                z1, z2 = model.encode_context_to_z(input_ids, attention_mask)

            ctx_hiddens.append(pooled.float().cpu().numpy())
            z1s.append(z1.float().cpu().numpy())
            z2s.append(z2.float().cpu().numpy())

    arrays = {
        "z1": np.concatenate(z1s, axis=0).astype(np.float32),
        "z2": np.concatenate(z2s, axis=0).astype(np.float32),
        "z_concat": np.concatenate([np.concatenate(z1s, axis=0), np.concatenate(z2s, axis=0)], axis=1).astype(np.float32),
        "context_hidden": np.concatenate(ctx_hiddens, axis=0).astype(np.float32),
        "reward_vec": reward_vecs.astype(np.float32),
    }
    np.savez_compressed(cache_path, **arrays)
    save_json(cache_meta_path, expected_cache_meta)
    return arrays


def summarize_labels(records: list[TurnRecord], output_dir: Path) -> None:
    stats = {
        "intent": Counter(r.intent_label for r in records),
        "knowledge": Counter(r.knowledge_label for r in records),
        "strategy": Counter(r.strategy_label for r in records),
        "interaction": Counter(r.interaction_label for r in records),
    }
    serializable = {k: dict(v.most_common()) for k, v in stats.items()}
    with (output_dir / "label_stats.json").open("w") as f:
        json.dump(serializable, f, indent=2)


def select_balanced_subset(
    records: list[TurnRecord],
    field: str,
    per_label: int,
    min_label_count: int,
    keep_other: bool,
    seed: int,
) -> tuple[list[TurnRecord], dict[str, Any]]:
    if per_label <= 0:
        return records, {
            "subset_enabled": False,
            "reason": "per_label<=0",
            "original_num_records": len(records),
            "selected_num_records": len(records),
            "field": field,
        }

    label_attr = f"{field}_label"
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        label = getattr(record, label_attr)
        grouped_indices[label].append(idx)

    counts = {label: len(indices) for label, indices in grouped_indices.items()}
    eligible_labels = [
        label
        for label, count in sorted(counts.items())
        if count >= per_label and (keep_other or label != "other")
    ]

    selection_mode = "exact_per_label"
    target_per_label = per_label
    if not eligible_labels:
        fallback_labels = [
            label
            for label, count in sorted(counts.items())
            if count >= min_label_count and (keep_other or label != "other")
        ]
        if len(fallback_labels) < 2:
            return records, {
                "subset_enabled": False,
                "reason": "insufficient_labels_for_balanced_subset",
                "original_num_records": len(records),
                "selected_num_records": len(records),
                "field": field,
                "label_counts": counts,
            }
        eligible_labels = fallback_labels
        target_per_label = min(per_label, min(counts[label] for label in eligible_labels))
        selection_mode = "fallback_min_count"

    rng = random.Random(seed)
    selected_indices: list[int] = []
    selected_counts: dict[str, int] = {}
    for label in eligible_labels:
        candidates = list(grouped_indices[label])
        rng.shuffle(candidates)
        chosen = sorted(candidates[:target_per_label])
        selected_indices.extend(chosen)
        selected_counts[label] = len(chosen)

    selected_indices.sort()
    selected_records = [records[idx] for idx in selected_indices]
    stats = {
        "subset_enabled": True,
        "field": field,
        "selection_mode": selection_mode,
        "per_label_requested": per_label,
        "per_label_selected": target_per_label,
        "keep_other": keep_other,
        "min_label_count": min_label_count,
        "original_num_records": len(records),
        "selected_num_records": len(selected_records),
        "num_selected_labels": len(eligible_labels),
        "selected_labels": eligible_labels,
        "selected_label_counts": selected_counts,
        "full_label_counts": counts,
    }
    return selected_records, stats


def get_manifold_embeddings(
    features: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    features = StandardScaler().fit_transform(features)
    if method == "umap":
        if umap is None:
            raise RuntimeError("Requested UMAP, but umap-learn is not installed.")
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(features)
    if method == "tsne":
        perplexity = min(30, max(5, features.shape[0] // 25))
        reducer = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
        )
        return reducer.fit_transform(features)
    raise ValueError(f"Unknown manifold method: {method}")


def make_figure1(
    records: list[TurnRecord],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    method: str,
    feature_key: str,
    max_labels: int,
    min_label_count: int,
    seed: int,
) -> np.ndarray:
    if not records:
        raise ValueError("No records were loaded; Figure 1 cannot be created.")

    interaction_counts = Counter(r.interaction_label for r in records)
    selected_labels = [
        label for label, count in interaction_counts.most_common(max_labels)
        if count >= min_label_count and label != "other"
    ]
    labels = np.asarray(
        [r.interaction_label if r.interaction_label in selected_labels else "other" for r in records]
    )
    coords = get_manifold_embeddings(arrays[feature_key], method=method, seed=seed)

    point_rows = []
    for i, record in enumerate(records):
        point_rows.append({
            "episode_id": record.episode_id,
            "turn_num": record.turn_num,
            "speaker": record.speaker,
            "interaction_label": labels[i],
            "intent_label": record.intent_label,
            "knowledge_label": record.knowledge_label,
            "strategy_label": record.strategy_label,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        })

    with (output_dir / f"figure1_{feature_key}_{method}_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
        writer.writeheader()
        writer.writerows(point_rows)

    unique_labels = list(dict.fromkeys(labels.tolist()))
    color_map = label_color_map(unique_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18 if label != "other" else 10,
            alpha=0.85 if label != "other" else 0.35,
            color=color_map[label],
            label=f"{label} (n={int(mask.sum())})",
            rasterized=True,
        )
    style_plot_axes(ax, xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")
    style_plot_legend(ax.legend(loc="best", fontsize=11, frameon=True, ncol=2))
    fig.tight_layout()
    fig.savefig(output_dir / f"figure1_{feature_key}_{method}.png", dpi=220)
    plt.close(fig)
    return coords


def get_label_array(records: list[TurnRecord], field: str) -> np.ndarray:
    attr = f"{field}_label"
    return np.asarray([getattr(record, attr) for record in records])


def select_plot_labels(
    labels: np.ndarray,
    max_labels: int,
    min_label_count: int,
    include_other: bool = False,
) -> list[str]:
    selected = []
    for label, count in Counter(labels.tolist()).most_common():
        if label == "other" and not include_other:
            continue
        if count < min_label_count:
            continue
        selected.append(label)
        if len(selected) >= max_labels:
            break
    return selected


def _plot_density_contours(
    ax: plt.Axes,
    coords: np.ndarray,
    mask: np.ndarray,
    color: str,
    bins: int = 70,
) -> None:
    if int(mask.sum()) < 10:
        return
    x = coords[:, 0]
    y = coords[:, 1]
    x_pad = 0.03 * (float(x.max() - x.min()) + 1e-8)
    y_pad = 0.03 * (float(y.max() - y.min()) + 1e-8)
    x_range = (float(x.min() - x_pad), float(x.max() + x_pad))
    y_range = (float(y.min() - y_pad), float(y.max() + y_pad))
    hist, x_edges, y_edges = np.histogram2d(
        coords[mask, 0],
        coords[mask, 1],
        bins=bins,
        range=[x_range, y_range],
    )
    positive = hist[hist > 0]
    if len(positive) < 4:
        return
    levels = np.unique(np.percentile(positive, [55, 75, 90]))
    if len(levels) == 0:
        return
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers,
        y_centers,
        hist.T,
        levels=levels,
        colors=[color],
        linewidths=1.2,
        alpha=0.9,
    )


def make_figure1_density_enrichment(
    records: list[TurnRecord],
    coords: np.ndarray,
    output_dir: Path,
    method: str,
    feature_key: str,
    field: str,
    target_label: str,
) -> None:
    labels = get_label_array(records, field)
    if target_label not in set(labels.tolist()):
        return
    mask = labels == target_label

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(
        coords[~mask, 0],
        coords[~mask, 1],
        s=5,
        c=PLOT_LIGHT_GREY,
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        label=f"not {target_label} (n={int((~mask).sum())})",
    )
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=11,
        c=PLOT_GREEN,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        label=f"{target_label} (n={int(mask.sum())})",
    )
    _plot_density_contours(ax, coords, mask, color=PLOT_PURPLE)
    style_plot_axes(ax, xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")
    style_plot_legend(ax.legend(frameon=True, fontsize=11, loc="best"))
    fig.tight_layout()
    fig.savefig(
        output_dir / f"figure1_{feature_key}_{method}_density_{field}_{target_label}.png",
        dpi=220,
    )
    plt.close(fig)


def make_figure1_small_multiples(
    records: list[TurnRecord],
    coords: np.ndarray,
    output_dir: Path,
    method: str,
    feature_key: str,
    max_labels: int,
    min_label_count: int,
) -> None:
    labels = get_label_array(records, "interaction")
    selected_labels = select_plot_labels(
        labels,
        max_labels=max_labels,
        min_label_count=min_label_count,
        include_other=False,
    )
    if not selected_labels:
        return

    ncols = 3
    nrows = int(np.ceil(len(selected_labels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for idx, label in enumerate(selected_labels):
        ax = axes[idx // ncols][idx % ncols]
        mask = labels == label
        color = PLOT_PALETTE[idx % len(PLOT_PALETTE)]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=3,
            c=PLOT_LIGHT_GREY,
            alpha=0.10,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            color=color,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        _plot_density_contours(ax, coords, mask, color=color)
        annotate_panel(ax, f"{label} (n={int(mask.sum())})")
        style_plot_axes(ax, xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")

    for empty_idx in range(len(selected_labels), nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"figure1_{feature_key}_{method}_small_multiples_interaction.png",
        dpi=220,
    )
    plt.close(fig)


def compute_knn_purity(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(k, len(labels) - 1))
    X = StandardScaler().fit_transform(features.astype(np.float32))
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(X)
    _, nn_idx = neighbors.kneighbors(X, return_distance=True)
    nn_idx = nn_idx[:, 1:]
    neighbor_labels = labels[nn_idx]
    purity = np.mean(neighbor_labels == labels[:, None], axis=1)
    priors = Counter(labels.tolist())
    prior = np.asarray([priors[label] / len(labels) for label in labels], dtype=np.float32)
    enrichment = purity / np.maximum(prior, 1e-8)
    return purity.astype(np.float32), enrichment.astype(np.float32)


def make_figure1_knn_purity(
    records: list[TurnRecord],
    arrays: dict[str, np.ndarray],
    coords: np.ndarray,
    output_dir: Path,
    method: str,
    feature_key: str,
    max_labels: int,
    min_label_count: int,
    k: int,
) -> None:
    raw_labels = get_label_array(records, "interaction")
    selected_labels = set(
        select_plot_labels(
            raw_labels,
            max_labels=max_labels,
            min_label_count=min_label_count,
            include_other=False,
        )
    )
    labels = np.asarray(
        [label if label in selected_labels else "other" for label in raw_labels]
    )
    purity, enrichment = compute_knn_purity(arrays[feature_key], labels, k=k)

    rows = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "episode_id": record.episode_id,
                "turn_num": record.turn_num,
                "speaker": record.speaker,
                "interaction_label": labels[idx],
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "knn_purity": float(purity[idx]),
                "knn_enrichment": float(enrichment[idx]),
            }
        )
    with (output_dir / f"figure1_{feature_key}_{method}_knn_purity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    scatter = axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=purity,
        cmap=PLOT_CMAP,
        s=6,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        vmin=0.0,
        vmax=1.0,
    )
    annotate_panel(axes[0], f"kNN purity (k={k})")
    style_plot_axes(axes[0], xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")
    style_plot_colorbar(
        fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04),
        label="same-label fraction",
    )

    vmax = float(np.percentile(enrichment, 98))
    scatter = axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.clip(enrichment, 0.0, vmax),
        cmap=PLOT_PURPLE_CMAP,
        s=6,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    annotate_panel(axes[1], "Local enrichment over prior")
    style_plot_axes(axes[1], xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")
    style_plot_colorbar(
        fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04),
        label="purity / prior",
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"figure1_{feature_key}_{method}_knn_purity.png", dpi=220)
    plt.close(fig)


def choose_probe_confidence_labels(
    records: list[TurnRecord],
    field: str,
    requested_values: list[str] | None,
    fallback_value: str | None,
    min_label_count: int,
) -> list[str]:
    labels = get_label_array(records, field)
    counts = Counter(labels.tolist())
    chosen: list[str] = []
    candidates = []
    if requested_values:
        candidates.extend(requested_values)
    if fallback_value:
        candidates.append(fallback_value)
    if field == "strategy":
        candidates.append("questioning")
    candidates.extend(label for label, _ in counts.most_common() if label != "other")

    for label in candidates:
        if label in chosen:
            continue
        if counts.get(label, 0) >= min_label_count:
            chosen.append(label)
        if len(chosen) >= 2:
            break
    return chosen


def make_figure1_probe_confidence(
    records: list[TurnRecord],
    arrays: dict[str, np.ndarray],
    coords: np.ndarray,
    output_dir: Path,
    method: str,
    feature_key: str,
    field: str,
    probe_values: list[str],
) -> None:
    if not probe_values:
        return
    labels = get_label_array(records, field)
    X = arrays[feature_key].astype(np.float32)
    probability_columns: dict[str, np.ndarray] = {}
    for value in probe_values:
        direction, intercept, _, _, _ = fit_binary_concept_probe(X, labels, value)
        probability_columns[value] = sigmoid(X @ direction + intercept).astype(np.float32)

    rows = []
    for idx, record in enumerate(records):
        row = {
            "episode_id": record.episode_id,
            "turn_num": record.turn_num,
            "speaker": record.speaker,
            f"{field}_label": labels[idx],
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
        }
        for value, probs in probability_columns.items():
            row[f"prob_{value}"] = float(probs[idx])
        rows.append(row)
    with (output_dir / f"figure1_{feature_key}_{method}_probe_confidence_{field}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ncols = len(probe_values)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.5), squeeze=False)
    for ax, value in zip(axes[0], probe_values):
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=probability_columns[value],
            cmap=PLOT_CMAP,
            s=6,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
            vmin=0.0,
            vmax=1.0,
        )
        annotate_panel(ax, f"P({field}={value} | z)")
        style_plot_axes(ax, xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")
        style_plot_colorbar(
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04),
            label="probe probability",
        )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"figure1_{feature_key}_{method}_probe_confidence_{field}.png",
        dpi=220,
    )
    plt.close(fig)


def _add_covariance_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    color: Any,
    n_std: float = 1.0,
) -> None:
    if len(points) < 5:
        return
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=float(width),
        height=float(height),
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.add_patch(ellipse)


def make_figure1_centroid_arrows(
    records: list[TurnRecord],
    coords: np.ndarray,
    output_dir: Path,
    method: str,
    feature_key: str,
    max_labels: int,
    min_label_count: int,
) -> None:
    labels = get_label_array(records, "interaction")
    selected_labels = select_plot_labels(
        labels,
        max_labels=max_labels,
        min_label_count=min_label_count,
        include_other=False,
    )
    if not selected_labels:
        return

    global_centroid = coords.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=4,
        c=PLOT_LIGHT_GREY,
        alpha=0.14,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        [global_centroid[0]],
        [global_centroid[1]],
        marker="x",
        s=80,
        c=PLOT_DARK_GREY,
        linewidths=2,
        label="global centroid",
    )
    for idx, label in enumerate(selected_labels):
        mask = labels == label
        points = coords[mask]
        centroid = points.mean(axis=0)
        color = PLOT_PALETTE[idx % len(PLOT_PALETTE)]
        ax.annotate(
            "",
            xy=(centroid[0], centroid[1]),
            xytext=(global_centroid[0], global_centroid[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.9),
        )
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=60,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=f"{label} (n={int(mask.sum())})",
            zorder=3,
        )
        _add_covariance_ellipse(ax, points, color=color, n_std=1.0)
    style_plot_axes(ax, xlabel=f"{method.upper()}-1", ylabel=f"{method.upper()}-2")
    style_plot_legend(ax.legend(frameon=True, fontsize=11, loc="best", ncol=2))
    fig.tight_layout()
    fig.savefig(
        output_dir / f"figure1_{feature_key}_{method}_centroid_arrows_interaction.png",
        dpi=220,
    )
    plt.close(fig)


def make_figure1_enhanced(
    records: list[TurnRecord],
    arrays: dict[str, np.ndarray],
    coords: np.ndarray,
    output_dir: Path,
    method: str,
    feature_key: str,
    max_labels: int,
    min_label_count: int,
    knn_k: int,
    concept_field: str,
    concept_value: str | None,
    probe_values: list[str] | None,
) -> None:
    concept_labels = get_label_array(records, concept_field)
    target_label = concept_value
    if not target_label:
        counts = Counter(label for label in concept_labels if label != "other")
        target_label = counts.most_common(1)[0][0] if counts else None

    if target_label:
        make_figure1_density_enrichment(
            records=records,
            coords=coords,
            output_dir=output_dir,
            method=method,
            feature_key=feature_key,
            field=concept_field,
            target_label=target_label,
        )
    make_figure1_small_multiples(
        records=records,
        coords=coords,
        output_dir=output_dir,
        method=method,
        feature_key=feature_key,
        max_labels=6,
        min_label_count=min_label_count,
    )
    make_figure1_knn_purity(
        records=records,
        arrays=arrays,
        coords=coords,
        output_dir=output_dir,
        method=method,
        feature_key=feature_key,
        max_labels=max_labels,
        min_label_count=min_label_count,
        k=knn_k,
    )
    selected_probe_values = choose_probe_confidence_labels(
        records=records,
        field=concept_field,
        requested_values=probe_values,
        fallback_value=target_label,
        min_label_count=min_label_count,
    )
    make_figure1_probe_confidence(
        records=records,
        arrays=arrays,
        coords=coords,
        output_dir=output_dir,
        method=method,
        feature_key=feature_key,
        field=concept_field,
        probe_values=selected_probe_values,
    )
    make_figure1_centroid_arrows(
        records=records,
        coords=coords,
        output_dir=output_dir,
        method=method,
        feature_key=feature_key,
        max_labels=6,
        min_label_count=min_label_count,
    )


def build_probe_targets(records: list[TurnRecord], field: str) -> np.ndarray:
    attr = f"{field}_label"
    return np.asarray([getattr(r, attr) for r in records])


def make_cv_splits(
    y: np.ndarray,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    min_count = min(Counter(y).values())
    max_splits = min_count
    if groups is not None:
        max_splits = min(max_splits, len(np.unique(groups)))
    n_splits = max(2, min(n_splits, max_splits))
    dummy = np.zeros(len(y), dtype=np.int32)

    if (
        groups is not None
        and StratifiedGroupKFold is not None
        and len(np.unique(groups)) >= n_splits
    ):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(dummy, y, groups=groups)), "episode_grouped"

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(dummy, y)), "stratified"


def evaluate_dense_probe_cv(
    X: np.ndarray,
    y_str: np.ndarray,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[dict[str, float], str]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    num_classes = len(label_encoder.classes_)
    splits, split_scheme = make_cv_splits(y, n_splits=n_splits, seed=seed, groups=groups)

    accs = []
    macro_f1s = []
    weighted_f1s = []
    for train_idx, test_idx in splits:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
                n_jobs=None,
            ),
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        macro_f1s.append(f1_score(y[test_idx], preds, average="macro"))
        weighted_f1s.append(f1_score(y[test_idx], preds, average="weighted"))
    return {
        "num_classes": float(num_classes),
        "num_samples": float(len(y)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
    }, split_scheme


def evaluate_sparse_probe_cv(
    texts: list[str],
    y_str: np.ndarray,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[dict[str, float], str]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    num_classes = len(label_encoder.classes_)
    splits, split_scheme = make_cv_splits(y, n_splits=n_splits, seed=seed, groups=groups)

    accs = []
    macro_f1s = []
    weighted_f1s = []
    for train_idx, test_idx in splits:
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        clf = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            n_jobs=None,
        )
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)
        accs.append(accuracy_score(y[test_idx], preds))
        macro_f1s.append(f1_score(y[test_idx], preds, average="macro"))
        weighted_f1s.append(f1_score(y[test_idx], preds, average="weighted"))
    return {
        "num_classes": float(num_classes),
        "num_samples": float(len(y)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
    }, split_scheme


def make_table1(
    records: list[TurnRecord],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    seed: int,
    n_splits: int,
    min_label_count: int,
    num_permutation_runs: int,
    drop_other: bool,
) -> None:
    text_features = [r.probe_text for r in records]
    episode_groups = np.asarray([r.episode_id for r in records])
    results = []
    permutation_rows = []

    feature_sets: dict[str, Any] = {
        "context_hidden": arrays["context_hidden"],
        "z1": arrays["z1"],
        "z2": arrays["z2"],
        "z_concat": arrays["z_concat"],
        "reward_vec": arrays["reward_vec"],
        "tfidf_text": text_features,
    }

    for field in ["intent", "knowledge", "strategy"]:
        y_all = build_probe_targets(records, field)
        keep_labels = {
            label
            for label, count in Counter(y_all).items()
            if count >= min_label_count and (label != "other" or not drop_other)
        }
        keep_mask = np.asarray([label in keep_labels for label in y_all])
        y = y_all[keep_mask]
        groups = episode_groups[keep_mask]
        if len(np.unique(y)) < 2:
            continue

        for feature_name, feature_value in feature_sets.items():
            if feature_name == "tfidf_text":
                filtered_texts = [feature_value[i] for i in np.where(keep_mask)[0]]
                metrics, split_scheme = evaluate_sparse_probe_cv(
                    filtered_texts,
                    y,
                    n_splits=n_splits,
                    seed=seed,
                    groups=groups,
                )
            else:
                X = np.asarray(feature_value)[keep_mask]
                metrics, split_scheme = evaluate_dense_probe_cv(
                    X,
                    y,
                    n_splits=n_splits,
                    seed=seed,
                    groups=groups,
                )

            row = {
                "target": field,
                "feature": feature_name,
                "label_condition": "real",
                "permutation_id": "",
                "split_scheme": split_scheme,
                **metrics,
            }
            results.append(row)

            shuffled_runs = []
            rng = np.random.default_rng(seed)
            for perm_idx in range(num_permutation_runs):
                y_perm = rng.permutation(y)
                if feature_name == "tfidf_text":
                    shuffled_metrics, shuffled_scheme = evaluate_sparse_probe_cv(
                            filtered_texts,
                            y_perm,
                            n_splits=n_splits,
                            seed=seed + perm_idx + 1,
                            groups=groups,
                        )
                    shuffled_runs.append(shuffled_metrics)
                else:
                    shuffled_metrics, shuffled_scheme = evaluate_dense_probe_cv(
                            X,
                            y_perm,
                            n_splits=n_splits,
                            seed=seed + perm_idx + 1,
                            groups=groups,
                    )
                    shuffled_runs.append(shuffled_metrics)
                permutation_rows.append(
                    {
                        "target": field,
                        "feature": feature_name,
                        "label_condition": "shuffled",
                        "permutation_id": perm_idx + 1,
                        "split_scheme": shuffled_scheme,
                        **shuffled_metrics,
                    }
                )
            shuffled_row = {
                "target": field,
                "feature": feature_name,
                "label_condition": "shuffled_summary",
                "permutation_id": "1..{0}".format(num_permutation_runs),
                "split_scheme": split_scheme,
                "num_classes": float(np.mean([r["num_classes"] for r in shuffled_runs])),
                "num_samples": float(np.mean([r["num_samples"] for r in shuffled_runs])),
                "accuracy_mean": float(np.mean([r["accuracy_mean"] for r in shuffled_runs])),
                "accuracy_std": float(np.mean([r["accuracy_std"] for r in shuffled_runs])),
                "macro_f1_mean": float(np.mean([r["macro_f1_mean"] for r in shuffled_runs])),
                "macro_f1_std": float(np.mean([r["macro_f1_std"] for r in shuffled_runs])),
                "weighted_f1_mean": float(np.mean([r["weighted_f1_mean"] for r in shuffled_runs])),
                "weighted_f1_std": float(np.mean([r["weighted_f1_std"] for r in shuffled_runs])),
            }
            results.append(shuffled_row)

    if not results:
        raise ValueError(
            "No probe results were produced. Lower --probe_min_label_count or inspect label extraction."
        )

    csv_path = output_dir / "table1_linear_probe.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    permutation_csv_path = output_dir / "table1_linear_probe_permutations.csv"
    with permutation_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(permutation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(permutation_rows)

    md_lines = [
        "# Table 1: Linear Probe Performance",
        "",
        "| Target | Feature | Labels | Split | Macro-F1 | Accuracy | Classes | Samples |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        md_lines.append(
            "| {target} | {feature} | {label_condition} | {split_scheme} | "
            "{macro_f1_mean:.3f} +/- {macro_f1_std:.3f} | "
            "{accuracy_mean:.3f} +/- {accuracy_std:.3f} | {num_classes:.0f} | {num_samples:.0f} |".format(**row)
        )
    md_lines.extend(
        [
            "",
            "Notes:",
            f"- The shuffled null is saved as a full permutation distribution in `{permutation_csv_path.name}` ({num_permutation_runs} permutations per target/feature pair).",
            "- Heuristic `intent`, `knowledge`, and `strategy` labels share overlapping lexical cues, so probe tasks should be read as correlated coarse views rather than fully disentangled factors.",
            "- When available, the split scheme is episode-grouped cross-validation; otherwise the script falls back to standard stratified folds.",
        ]
    )
    (output_dir / "table1_linear_probe.md").write_text("\n".join(md_lines) + "\n")


def fit_binary_concept_direction(
    X: np.ndarray,
    labels: np.ndarray,
    target_label: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    direction, intercept, y, _, _ = fit_binary_concept_probe(X, labels, target_label)
    return direction, intercept, y


def fit_binary_concept_probe(
    X: np.ndarray,
    labels: np.ndarray,
    target_label: str,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, StandardScaler]:
    y = (labels == target_label).astype(np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Concept label '{target_label}' is degenerate in the current data.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_scaled, y)

    coef_scaled = clf.coef_[0]
    coef_raw = coef_scaled / scaler.scale_
    intercept_raw = float(clf.intercept_[0] - np.sum(coef_scaled * scaler.mean_ / scaler.scale_))
    return (
        coef_raw.astype(np.float32),
        intercept_raw,
        y,
        coef_scaled.astype(np.float32),
        scaler,
    )


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
    filtered_probs = F.softmax(sorted_logits, dim=-1)
    sample = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_idx.gather(-1, sample)
    return next_token.squeeze(-1)


def expand_z_to_prefix(
    model: torch.nn.Module,
    z_tensor: torch.Tensor,
    decoder: Any,
) -> torch.Tensor:
    batch_size = z_tensor.size(0)
    z_belief = z_tensor[:, :model.Z_BELIEF_DIM]
    z_intent = z_tensor[:, model.Z_BELIEF_DIM:model.Z_BELIEF_DIM + model.Z_INTENT_DIM]
    z_thought = z_tensor[:, model.Z_BELIEF_DIM + model.Z_INTENT_DIM:]

    prefix_belief = decoder["z_belief_to_prefix"](z_belief).view(
        batch_size, model.NUM_PREFIX_TOKENS, model.hidden_size
    )
    prefix_intent = decoder["z_intent_to_prefix"](z_intent).view(
        batch_size, model.NUM_PREFIX_TOKENS, model.hidden_size
    )
    prefix_thought = decoder["z_thought_to_prefix"](z_thought).view(
        batch_size, model.NUM_PREFIX_TOKENS, model.hidden_size
    )
    return torch.cat([prefix_belief, prefix_intent, prefix_thought], dim=1)


def decode_mental_text_from_z(
    model: torch.nn.Module,
    tokenizer,
    z: np.ndarray,
    decoder: Any,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed_text: str,
) -> str:
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
    z_prefix = expand_z_to_prefix(model, z_tensor, decoder)

    seed_ids = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if seed_ids.numel() == 0:
        start_id = tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        input_ids = torch.tensor([[start_id]], dtype=torch.long, device=device)
    else:
        input_ids = seed_ids

    for _ in range(max_new_tokens):
        embedding_layer = model.base_model.get_input_embeddings()
        decoder_dtype = next(decoder["cross_attn"].parameters()).dtype
        mental_embeds = embedding_layer(input_ids).to(decoder_dtype)
        attended, _ = decoder["cross_attn"](
            query=mental_embeds,
            key=z_prefix.to(decoder_dtype),
            value=z_prefix.to(decoder_dtype),
        )
        h = decoder["ln"](mental_embeds + attended)
        h = h + decoder["ffn"](h)
        h = decoder["ln2"](h)
        output_embedding = model.base_model.get_output_embeddings()
        if hasattr(output_embedding, "weight"):
            logits = F.linear(
                h.to(output_embedding.weight.dtype),
                output_embedding.weight,
                output_embedding.bias if hasattr(output_embedding, "bias") and output_embedding.bias is not None else None,
            )
        else:
            raise RuntimeError("Model output embedding does not expose weights for mental decoding.")

        logits = logits[0, -1]
        if temperature and temperature > 0:
            logits = logits / temperature
            next_token = top_p_sample(logits, top_p=top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.view(1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        token_id = int(next_token.item())
        if token_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
    return re.sub(r"\s+", " ", text).strip()


def score_response_reward_dims(
    model: torch.nn.Module,
    tokenizer,
    z1: np.ndarray,
    z2: np.ndarray,
    response_text: str,
    device: str,
    max_resp_len: int,
    ensemble_weight: float,
) -> np.ndarray:
    enc = tokenizer(
        response_text if response_text else "[EMPTY]",
        truncation=True,
        max_length=max_resp_len,
        return_tensors="pt",
    )
    resp_ids = enc.input_ids.to(device)
    resp_mask = enc.attention_mask.to(device)
    z1_t = torch.tensor(z1, dtype=torch.float32, device=device).unsqueeze(0)
    z2_t = torch.tensor(z2, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        joint, z1_only, z_combined = model.forward_reward_with_z(z1_t, z2_t, resp_ids, resp_mask)
    joint = joint[0].float().cpu().numpy()
    z1_only = z1_only[0].float().cpu().numpy()
    z_combined = z_combined[0].float().cpu().numpy()
    return ensemble_weight * joint + (1.0 - ensemble_weight) * 0.5 * (z1_only + z_combined)


def response_properties(text: str) -> dict[str, float]:
    text_l = text.lower()
    tokens = re.findall(r"[a-z']+", text_l)
    token_count = max(1, len(tokens))
    token_set = set(tokens)
    uncertainty_hits = sum(tok in UNCERTAINTY_WORDS for tok in tokens)
    uncertainty_hits += sum(phrase in text_l for phrase in UNCERTAINTY_PHRASES)
    return {
        "hedge_score": float(sum(tok in HEDGE_WORDS for tok in tokens) / token_count),
        "uncertainty_score": float(uncertainty_hits / token_count),
        "support_score": float(sum(tok in SUPPORT_WORDS for tok in tokens) / token_count),
        "negotiation_score": float(sum(tok in NEGOTIATION_WORDS for tok in tokens) / token_count),
        "concealment_score": float(sum(tok in CONCEAL_WORDS for tok in tokens) / token_count),
        "coordination_score": float(sum(tok in COORDINATION_WORDS for tok in tokens) / token_count),
        "disclosure_score": float(sum(tok in DISCLOSURE_WORDS for tok in tokens) / token_count),
        "length_tokens": float(token_count),
        "has_offer": float(any(tok in token_set for tok in {"offer", "deal", "price", "compromise"})),
    }


def pick_anchor_indices(
    probs: np.ndarray,
    labels: np.ndarray,
    target_label: str,
    num_examples: int,
) -> np.ndarray:
    opposite_mask = labels != target_label
    candidate_idx = np.where(opposite_mask)[0]
    if len(candidate_idx) < num_examples:
        candidate_idx = np.arange(len(labels))
    margin = np.abs(probs[candidate_idx] - 0.5)
    order = np.argsort(margin)
    return candidate_idx[order[:num_examples]]


def choose_knn_plot_labels(
    labels: np.ndarray,
    concept_value: str,
    max_labels: int = 5,
) -> list[str]:
    counts = Counter(labels.tolist())
    selected = [concept_value]
    if "other" in counts and concept_value != "other":
        selected.append("other")
    for label, _ in counts.most_common():
        if label in selected:
            continue
        selected.append(label)
        if len(selected) >= max_labels:
            break
    return selected


def compute_knn_label_fractions(
    query: np.ndarray,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    tracked_labels: list[str],
    k: int,
    exclude_idx: int | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    distances = np.sum((X_scaled - query) ** 2, axis=1)
    if exclude_idx is not None and 0 <= exclude_idx < len(distances):
        distances[exclude_idx] = np.inf
    k = max(1, min(k, len(distances) - int(exclude_idx is not None)))
    nn_idx = np.argpartition(distances, k - 1)[:k]
    nn_idx = nn_idx[np.argsort(distances[nn_idx])]
    nn_labels = labels[nn_idx]
    fractions = {
        label: float(np.mean(nn_labels == label))
        for label in tracked_labels
    }
    fractions["target"] = float(np.mean(nn_labels == tracked_labels[0]))
    return fractions, nn_idx


def compute_projection_knn_label_fractions(
    query_projection: float,
    projections: np.ndarray,
    labels: np.ndarray,
    tracked_labels: list[str],
    k: int,
    exclude_idx: int | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    distances = np.abs(projections - query_projection)
    if exclude_idx is not None and 0 <= exclude_idx < len(distances):
        distances[exclude_idx] = np.inf
    k = max(1, min(k, len(distances) - int(exclude_idx is not None)))
    nn_idx = np.argpartition(distances, k - 1)[:k]
    nn_idx = nn_idx[np.argsort(distances[nn_idx])]
    nn_labels = labels[nn_idx]
    fractions = {
        label: float(np.mean(nn_labels == label))
        for label in tracked_labels
    }
    fractions["target"] = float(np.mean(nn_labels == tracked_labels[0]))
    return fractions, nn_idx


def concept_direction_subspace_masses(
    direction: np.ndarray,
    z_dim: int,
) -> list[dict[str, float | int | str]]:
    if direction.shape[0] != 2 * z_dim:
        raise ValueError(
            f"Expected concept direction of length {2 * z_dim}, got {direction.shape[0]}."
        )

    subspaces = [
        ("z1_belief", 0, 48),
        ("z1_intent", 48, 88),
        ("z1_thought", 88, 128),
        ("z2_belief", z_dim, z_dim + 48),
        ("z2_intent", z_dim + 48, z_dim + 88),
        ("z2_thought", z_dim + 88, z_dim + 128),
    ]
    total_sq = float(np.sum(direction ** 2) + 1e-12)
    rows: list[dict[str, float | int | str]] = []
    for name, start, end in subspaces:
        coef = direction[start:end]
        sq_norm = float(np.sum(coef ** 2))
        rows.append(
            {
                "subspace": name,
                "start_dim": start,
                "end_dim_exclusive": end,
                "squared_norm": sq_norm,
                "mass": sq_norm / total_sq,
                "l2_norm": float(np.sqrt(sq_norm)),
                "mean_abs_weight": float(np.mean(np.abs(coef))),
            }
        )
    return rows


def make_figure2(
    args: argparse.Namespace,
    records: list[TurnRecord],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    field = args.concept_field
    label_attr = f"{field}_label"
    labels = np.asarray([getattr(r, label_attr) for r in records])

    concept_value = args.concept_value
    if not concept_value:
        counts = Counter(label for label in labels if label != "other")
        if not counts:
            raise ValueError(f"No non-'other' labels available for traversal field '{field}'.")
        concept_value = counts.most_common(1)[0][0]

    X = arrays["z_concat"].astype(np.float32)
    direction, intercept, _, direction_scaled, probe_scaler = fit_binary_concept_probe(
        X, labels, concept_value
    )
    direction_norm = np.linalg.norm(direction) + 1e-8
    unit_direction = direction / direction_norm
    projections = X @ unit_direction
    projection_std = float(np.std(projections))
    base_probs = sigmoid(X @ direction + intercept).astype(np.float32)
    anchor_indices = pick_anchor_indices(base_probs, labels, concept_value, args.num_traversal_examples)

    alphas = np.linspace(args.traversal_min_alpha, args.traversal_max_alpha, args.traversal_steps)
    tracked_labels = choose_knn_plot_labels(labels, concept_value, max_labels=args.traversal_knn_num_labels)
    X_scaled = probe_scaler.transform(X).astype(np.float32)
    direction_scaled_norm = np.linalg.norm(direction_scaled) + 1e-8
    unit_direction_scaled = direction_scaled / direction_scaled_norm
    scaled_projections = X_scaled @ unit_direction_scaled
    scaled_projection_std = float(np.std(scaled_projections))
    direction_mass_rows = concept_direction_subspace_masses(direction_scaled, args.z_dim)
    with (output_dir / f"figure2_direction_mass_{field}_{concept_value}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(direction_mass_rows[0].keys()))
        writer.writeheader()
        writer.writerows(direction_mass_rows)

    examples_md = [
        f"# Figure 2 Probe-Space Traversal for {field}={concept_value}",
        "",
        f"Direction source: linear probe weight on `{field}` over `z_concat`.",
        "Figure 2A plots the binary probe probability along the concept direction.",
        f"Figure 2B plots label enrichment among k={args.traversal_knn_k} nearest neighbors along the concept-axis projection.",
        "Figure 2C plots the squared direction mass in the structured z1/z2 subspaces.",
        "Anchor selection is in-sample: anchors are chosen from the same records used to fit the traversal direction.",
        "The neighbor examples are retrieval diagnostics, not held-out causal evidence.",
        "",
    ]
    csv_rows = []
    probability_traces: list[list[float]] = []
    enrichment_traces: dict[str, list[list[float]]] = {
        label: [] for label in tracked_labels
    }

    for anchor_rank, idx in enumerate(anchor_indices, start=1):
        record = records[idx]
        examples_md.append(f"## Anchor {anchor_rank}")
        examples_md.append(f"- Episode: `{record.episode_id}`")
        examples_md.append(f"- Turn: `{record.turn_num}`")
        examples_md.append(f"- Original labels: intent=`{record.intent_label}`, knowledge=`{record.knowledge_label}`, strategy=`{record.strategy_label}`")
        examples_md.append(f"- Fixed response: `{record.response_text}`")
        examples_md.append(f"- Context excerpt: `{record.context_text[-280:].replace(os.linesep, ' ')}`")
        examples_md.append("")

        anchor_probs = []
        anchor_enrichment: dict[str, list[float]] = {
            label: [] for label in tracked_labels
        }
        for alpha_idx, alpha in enumerate(alphas):
            z_prime_scaled = X_scaled[idx] + alpha * scaled_projection_std * unit_direction_scaled
            z_prime = probe_scaler.inverse_transform(z_prime_scaled.reshape(1, -1)).astype(np.float32)[0]
            probe_prob = float(sigmoid(float(z_prime @ direction + intercept)))
            concept_axis_value = float(z_prime_scaled @ unit_direction_scaled)
            fractions, nn_idx = compute_projection_knn_label_fractions(
                query_projection=concept_axis_value,
                projections=scaled_projections,
                labels=labels,
                tracked_labels=tracked_labels,
                k=args.traversal_knn_k,
                exclude_idx=int(idx),
            )

            row = {
                "anchor_rank": anchor_rank,
                "episode_id": record.episode_id,
                "turn_num": record.turn_num,
                "alpha": float(alpha),
                "probe_probability": probe_prob,
                "base_probability": float(base_probs[idx]),
                "concept_axis_value": concept_axis_value,
                "nearest_neighbor_ids": " ".join(str(int(i)) for i in nn_idx[: min(10, len(nn_idx))]),
                **{f"knn_frac_{label}": fractions[label] for label in tracked_labels},
            }
            csv_rows.append(row)
            anchor_probs.append(probe_prob)
            for label in tracked_labels:
                anchor_enrichment[label].append(fractions[label])
            if alpha_idx in {0, len(alphas) // 2, len(alphas) - 1}:
                examples_md.append(f"- alpha={alpha:.2f}:")
                examples_md.append(f"  P({concept_value}|z)={probe_prob:.3f}")
                examples_md.append(
                    "  kNN fractions: "
                    + ", ".join(f"{label}={fractions[label]:.2f}" for label in tracked_labels)
                )
                for neighbor_rank, nn in enumerate(nn_idx[: args.traversal_knn_examples], start=1):
                    neighbor = records[int(nn)]
                    neighbor_label = getattr(neighbor, label_attr)
                    response_excerpt = re.sub(r"\s+", " ", neighbor.response_text).strip()[:180]
                    examples_md.append(
                        f"  nn{neighbor_rank}: {neighbor_label} | "
                        f"episode=`{neighbor.episode_id}` turn={neighbor.turn_num} | "
                        f"`{response_excerpt}`"
                    )

        probability_traces.append(anchor_probs)
        for label in tracked_labels:
            enrichment_traces[label].append(anchor_enrichment[label])
        examples_md.append("")

    with (output_dir / f"figure2_traversal_{field}_{concept_value}_examples.md").open("w") as f:
        f.write("\n".join(examples_md) + "\n")

    if not csv_rows:
        raise ValueError("No traversal rows were produced for Figure 2.")

    with (output_dir / f"figure2_traversal_{field}_{concept_value}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    prob_arr = np.asarray(probability_traces, dtype=np.float32)
    prob_mean = prob_arr.mean(axis=0)
    prob_std = prob_arr.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.plot(
        alphas,
        prob_mean,
        marker="o",
        color=PLOT_BLUE,
        linewidth=2.4,
        markersize=7,
        label=f"P({concept_value})",
    )
    if len(probability_traces) > 1:
        ax.fill_between(
            alphas,
            np.clip(prob_mean - prob_std, 0.0, 1.0),
            np.clip(prob_mean + prob_std, 0.0, 1.0),
            color=PLOT_BLUE,
            alpha=0.15,
            linewidth=0,
        )
    ax.set_ylim(-0.02, 1.02)
    style_plot_axes(ax, xlabel="alpha", ylabel="Mean probe probability")
    style_plot_legend(ax.legend(frameon=True, fontsize=11, loc="best"))

    ax = axes[1]
    enrichment_colors = label_color_map(tracked_labels)
    for label in tracked_labels:
        enrich_arr = np.asarray(enrichment_traces[label], dtype=np.float32)
        ax.plot(
            alphas,
            enrich_arr.mean(axis=0),
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=enrichment_colors[label],
            label=label,
        )
    ax.set_ylim(-0.02, 1.02)
    style_plot_axes(ax, xlabel="alpha", ylabel="Mean neighbor fraction")
    style_plot_legend(ax.legend(frameon=True, fontsize=11))

    ax = axes[2]
    subspace_names = [str(row["subspace"]) for row in direction_mass_rows]
    masses = [float(row["mass"]) for row in direction_mass_rows]
    colors = [PLOT_BLUE, PLOT_BLUE, PLOT_BLUE, PLOT_PURPLE, PLOT_PURPLE, PLOT_PURPLE]
    ax.barh(subspace_names, masses, color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.01, max(masses) * 1.15))
    style_plot_axes(ax, xlabel="Squared coefficient mass", ylabel=None)

    fig.tight_layout()
    fig.savefig(output_dir / f"figure2_traversal_{field}_{concept_value}.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recursive ToM latents for Sotopia v3.")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/best",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="projects/sotopia/runs/analysis/mental_latent_qwen_v3",
    )
    parser.add_argument(
        "--label_source",
        choices=["response_heuristic", "external"],
        default="response_heuristic",
        help="Use response-derived heuristic labels or an externally annotated JSONL.",
    )
    parser.add_argument(
        "--label_jsonl",
        type=str,
        default=None,
        help="External JSONL with episode_id, turn_num, speaker, and label fields.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_ctx_len", type=int, default=1024)
    parser.add_argument("--max_resp_len", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--ensemble_weight", type=float, default=0.7)
    parser.add_argument("--recompute_latents", action="store_true", default=False)
    parser.add_argument("--recompute_records", action="store_true", default=False)
    parser.add_argument(
        "--use_full_records",
        action="store_true",
        help="Disable balanced-subset analysis and run on all loaded records.",
    )
    parser.add_argument(
        "--balanced_subset_field",
        choices=["interaction", "intent", "knowledge", "strategy"],
        default="interaction",
        help="Primary label family used to build the balanced analysis subset.",
    )
    parser.add_argument(
        "--balanced_subset_per_label",
        type=int,
        default=200,
        help="Target number of records per retained label in the balanced subset.",
    )
    parser.add_argument(
        "--balanced_subset_min_label_count",
        type=int,
        default=40,
        help="Fallback minimum count for retaining a label when exact per-label sampling is impossible.",
    )
    parser.add_argument(
        "--balanced_subset_keep_other",
        action="store_true",
        help="Include the 'other' label in balanced subset selection.",
    )

    parser.add_argument("--figure1_method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--figure1_feature", choices=["z1", "z2", "z_concat", "context_hidden"], default="z_concat")
    parser.add_argument("--figure1_max_labels", type=int, default=10)
    parser.add_argument("--figure1_min_label_count", type=int, default=40)
    parser.add_argument(
        "--figure1_knn_k",
        type=int,
        default=50,
        help="Number of latent nearest neighbors used for Figure 1 local purity/enrichment.",
    )
    parser.add_argument(
        "--figure1_probe_values",
        nargs="*",
        default=None,
        help="Optional label values for Figure 1 probe-confidence maps, e.g. offer_proposal questioning.",
    )

    parser.add_argument("--probe_cv_splits", type=int, default=5)
    parser.add_argument("--probe_min_label_count", type=int, default=40)
    parser.add_argument("--num_permutation_runs", type=int, default=20)
    parser.add_argument(
        "--probe_keep_other_labels",
        dest="probe_drop_other",
        action="store_false",
        help="Keep 'other' labels in probe evaluation instead of dropping them.",
    )
    parser.set_defaults(probe_drop_other=True)

    parser.add_argument("--concept_field", choices=["intent", "knowledge", "strategy"], default="strategy")
    parser.add_argument("--concept_value", type=str, default=None)
    parser.add_argument("--num_traversal_examples", type=int, default=3)
    parser.add_argument("--traversal_min_alpha", type=float, default=-2.0)
    parser.add_argument("--traversal_max_alpha", type=float, default=2.0)
    parser.add_argument("--traversal_steps", type=int, default=5)
    parser.add_argument(
        "--traversal_knn_k",
        type=int,
        default=50,
        help="Number of nearest neighbors used for traversal label-enrichment curves.",
    )
    parser.add_argument(
        "--traversal_knn_num_labels",
        type=int,
        default=5,
        help="Number of label curves to plot in the kNN enrichment panel.",
    )
    parser.add_argument(
        "--traversal_knn_examples",
        type=int,
        default=3,
        help="Number of nearest-neighbor examples written per shown alpha in the traversal markdown.",
    )
    parser.add_argument("--traversal_max_new_tokens", type=int, default=48)
    parser.add_argument("--generation_temperature", type=float, default=0.0)
    parser.add_argument("--generation_top_p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.num_permutation_runs < 20:
        print(
            f"Bumping --num_permutation_runs from {args.num_permutation_runs} to 20 "
            "for a stabler shuffled-label null distribution.",
            flush=True,
        )
        args.num_permutation_runs = 20

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_jsonl = output_dir / "latent_records.jsonl"
    records_meta_json = output_dir / "latent_records_meta.json"
    analysis_records_jsonl = output_dir / "analysis_records_subset.jsonl"
    analysis_subset_stats_json = output_dir / "analysis_subset_stats.json"
    cache_path = output_dir / "latent_arrays.npz"
    cache_meta_path = output_dir / "latent_arrays_meta.json"
    external_labels = None
    if args.label_source == "external":
        if not args.label_jsonl:
            raise ValueError("--label_source external requires --label_jsonl")
        external_labels = load_external_labels(args.label_jsonl)

    expected_records_meta = build_records_cache_meta(args)
    cached_records_meta = load_json(records_meta_json)

    if (
        records_jsonl.exists()
        and not args.recompute_records
        and cached_records_meta == expected_records_meta
    ):
        records = load_records_jsonl(records_jsonl)
        expected_label_source = "external" if args.label_source == "external" else "response_heuristic"
        if any(record.label_source != expected_label_source for record in records):
            records = load_turn_records(
                args.data_path,
                label_source=args.label_source,
                external_labels=external_labels,
            )
            save_records_jsonl(records, records_jsonl)
            save_json(records_meta_json, expected_records_meta)
    else:
        records = load_turn_records(
            args.data_path,
            label_source=args.label_source,
            external_labels=external_labels,
        )
        save_records_jsonl(records, records_jsonl)
        save_json(records_meta_json, expected_records_meta)

    full_record_count = len(records)
    if args.use_full_records:
        selected_records = records
        subset_stats = {
            "subset_enabled": False,
            "reason": "use_full_records",
            "field": args.balanced_subset_field,
            "original_num_records": full_record_count,
            "selected_num_records": full_record_count,
        }
    else:
        selected_records, subset_stats = select_balanced_subset(
            records,
            field=args.balanced_subset_field,
            per_label=args.balanced_subset_per_label,
            min_label_count=args.balanced_subset_min_label_count,
            keep_other=args.balanced_subset_keep_other,
            seed=args.seed,
        )

    save_records_jsonl(selected_records, analysis_records_jsonl)
    save_json(analysis_subset_stats_json, subset_stats)
    print(
        f"Analysis records: {len(selected_records)}/{full_record_count} "
        f"(subset field={args.balanced_subset_field}, "
        f"enabled={subset_stats.get('subset_enabled', False)})",
        flush=True,
    )

    records = selected_records
    summarize_labels(records, output_dir)
    expected_latent_meta = build_latent_cache_meta(args, analysis_records_jsonl, records)
    arrays = load_or_extract_latents(
        args,
        records,
        cache_path,
        cache_meta_path,
        expected_latent_meta,
    )

    figure1_method = args.figure1_method
    if figure1_method == "umap" and umap is None:
        print("UMAP not installed; falling back to t-SNE.", flush=True)
        figure1_method = "tsne"

    figure1_coords = make_figure1(
        records=records,
        arrays=arrays,
        output_dir=output_dir,
        method=figure1_method,
        feature_key=args.figure1_feature,
        max_labels=args.figure1_max_labels,
        min_label_count=args.figure1_min_label_count,
        seed=args.seed,
    )
    make_figure1_enhanced(
        records=records,
        arrays=arrays,
        coords=figure1_coords,
        output_dir=output_dir,
        method=figure1_method,
        feature_key=args.figure1_feature,
        max_labels=args.figure1_max_labels,
        min_label_count=args.figure1_min_label_count,
        knn_k=args.figure1_knn_k,
        concept_field=args.concept_field,
        concept_value=args.concept_value,
        probe_values=args.figure1_probe_values,
    )
    make_table1(
        records=records,
        arrays=arrays,
        output_dir=output_dir,
        seed=args.seed,
        n_splits=args.probe_cv_splits,
        min_label_count=args.probe_min_label_count,
        num_permutation_runs=args.num_permutation_runs,
        drop_other=args.probe_drop_other,
    )
    make_figure2(
        args=args,
        records=records,
        arrays=arrays,
        output_dir=output_dir,
    )

    print(f"Saved analysis artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
