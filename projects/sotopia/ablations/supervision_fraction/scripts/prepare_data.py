#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "sotopia_turn_rewards_v3.jsonl"
DEFAULT_PREWARM = (
    PROJECT_ROOT / "data" / "mental_model_persona_dataset.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data"
SFT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "grpo_agent_qwen_v3" / "sft_warmup"
)
FULL_REWARD_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "coupled_mental_reward_qwen_v3" / "best"
)
FULL_POLICY_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "grpo_agent_qwen_v3" / "step_300"
)
FRACTIONS = (0.25, 0.50)
MENTAL_FIELDS = (
    "partner_belief",
    "strategic_intent",
    "thought_process",
    "second_order_belief",
    "second_order_intent",
    "second_order_thought",
)
REWARD_DIMENSIONS = (
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
)


def read_jsonl(path):
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash(path):
    candidates = (
        path / "adapter_model.safetensors",
        path / "lora_adapter" / "adapter_model.safetensors",
    )
    for candidate in candidates:
        if candidate.exists():
            return sha256_file(candidate)
    raise FileNotFoundError(f"No adapter checkpoint found under {path}")


def speaker_rewards(turn_reward, parsed_episode):
    speaker = turn_reward.get("agent")
    if speaker == parsed_episode.get("agent_1_name", "Agent 1"):
        return turn_reward.get("agent_1_rewards", {})
    if speaker == parsed_episode.get("agent_2_name", "Agent 2"):
        return turn_reward.get("agent_2_rewards", {})
    return None


def annotation_stats(episodes):
    valid_turns = 0
    hard_negative_turns = 0
    complete_mental_turns = 0
    complete_score_turns = 0
    complete_rationale_turns = 0

    for episode in episodes:
        parsed = episode.get("parsed_episode", {})
        turns = parsed.get("turns", [])
        turn_rewards = episode.get("turn_rewards", [])
        if not turns or not turn_rewards:
            continue

        for turn_reward in turn_rewards:
            rewards = speaker_rewards(turn_reward, parsed)
            if not rewards:
                continue
            turn_number = turn_reward.get("turn")
            if not isinstance(turn_number, int) or not 0 <= turn_number < len(turns):
                continue
            response = turns[turn_number].get("content", "")
            if not response.strip():
                continue

            valid_turns += 1
            mental = rewards.get("mental_state", {})
            if str(mental.get("hard_negative_response", "")).strip():
                hard_negative_turns += 1
            if all(str(mental.get(field, "")).strip() for field in MENTAL_FIELDS):
                complete_mental_turns += 1

            dimensions = [rewards.get(name, {}) for name in REWARD_DIMENSIONS]
            if all(isinstance(value, dict) and "score" in value for value in dimensions):
                complete_score_turns += 1
            if all(
                isinstance(value, dict) and str(value.get("reasoning", "")).strip()
                for value in dimensions
            ):
                complete_rationale_turns += 1

    return {
        "episodes": len(episodes),
        "valid_stage1_turns": valid_turns,
        "hard_negative_turns": hard_negative_turns,
        "complete_six_field_mental_turns": complete_mental_turns,
        "complete_seven_score_turns": complete_score_turns,
        "complete_seven_rationale_turns": complete_rationale_turns,
    }


def prewarm_valid_count(rows):
    count = 0
    for row in rows:
        mental = row.get("target_mental_reasoning", {})
        if any(
            str(mental.get(field, "")).strip()
            for field in ("thought", "goal", "belief_state")
        ):
            count += 1
    return count


def strip_augmented_annotations(episodes):
    stripped = []
    for episode in episodes:
        row = {
            "episode_id": episode.get("episode_id"),
            "parsed_episode": copy.deepcopy(episode.get("parsed_episode", {})),
            "original_rewards": copy.deepcopy(episode.get("original_rewards", [])),
            "turn_rewards": [
                {
                    "turn": turn_reward.get("turn"),
                    "agent": turn_reward.get("agent"),
                }
                for turn_reward in episode.get("turn_rewards", [])
            ],
        }
        stripped.append(row)
    return stripped


def nested_subset(rows, selected_indices, count):
    chosen = set(selected_indices[:count])
    return [row for index, row in enumerate(rows) if index in chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prewarm", type=Path, default=DEFAULT_PREWARM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_target_steps", type=int, default=500)
    args = parser.parse_args()

    episodes = read_jsonl(args.source)
    prewarm_rows = read_jsonl(args.prewarm)
    if not episodes:
        raise ValueError("Source dataset is empty")
    if not prewarm_rows:
        raise ValueError("Pre-warm dataset is empty")

    episode_indices = list(range(len(episodes)))
    prewarm_indices = list(range(len(prewarm_rows)))
    random.Random(args.seed).shuffle(episode_indices)
    random.Random(args.seed).shuffle(prewarm_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stripped_path = args.output_dir / "sotopia_base_trajectories_stripped.jsonl"
    stripped_rows = strip_augmented_annotations(episodes)
    write_jsonl(stripped_path, stripped_rows)

    full_stats = annotation_stats(episodes)
    full_batches = math.ceil(full_stats["valid_stage1_turns"] / 4)
    manifest = {
        "protocol": {
            "dataset": "SOTOPIA",
            "seed": args.seed,
            "selection_unit": "episode",
            "nested_subsets": True,
            "stage1_batch_size": 4,
            "stage1_gradient_accumulation": 8,
            "stage1_target_optimizer_steps": args.stage1_target_steps,
            "stage1_reference_full_endpoint_steps": (full_batches // 8) * 7,
            "stage1_scheduler_total_steps": (full_batches * 10) // 8,
            "stage1_scheduler_warmup_steps": int(((full_batches * 10) // 8) * 0.03),
            "stage2_target_grpo_steps": 300,
        },
        "source": {
            "path": str(args.source),
            "sha256": sha256_file(args.source),
            "annotation_stats": full_stats,
        },
        "prewarm_source": {
            "path": str(args.prewarm),
            "sha256": sha256_file(args.prewarm),
            "rows": len(prewarm_rows),
            "valid_rows": prewarm_valid_count(prewarm_rows),
        },
        "stage2_base_trajectories": {
            "path": str(stripped_path),
            "sha256": sha256_file(stripped_path),
            "episodes": len(stripped_rows),
            "generated_annotation_fields_retained": False,
        },
        "conditions": {
            "0": {
                "stage1": None,
                "policy_checkpoint": str(SFT_CHECKPOINT),
                "policy_checkpoint_sha256": checkpoint_hash(SFT_CHECKPOINT),
            },
            "100": {
                "reward_checkpoint": str(FULL_REWARD_CHECKPOINT),
                "reward_checkpoint_sha256": checkpoint_hash(FULL_REWARD_CHECKPOINT),
                "policy_checkpoint": str(FULL_POLICY_CHECKPOINT),
                "policy_checkpoint_sha256": checkpoint_hash(FULL_POLICY_CHECKPOINT),
            },
        },
    }

    selected_episode_ids = {}
    for fraction in FRACTIONS:
        label = str(int(fraction * 100))
        episode_count = math.floor(len(episodes) * fraction)
        prewarm_count = math.floor(len(prewarm_rows) * fraction)
        subset = nested_subset(episodes, episode_indices, episode_count)
        prewarm_subset = nested_subset(prewarm_rows, prewarm_indices, prewarm_count)
        data_path = args.output_dir / f"sotopia_annotations_{label}.jsonl"
        prewarm_path = args.output_dir / f"persona_prewarm_{label}.jsonl"
        write_jsonl(data_path, subset)
        write_jsonl(prewarm_path, prewarm_subset)

        stats = annotation_stats(subset)
        selected_episode_ids[label] = {
            row.get("episode_id") for row in subset
        }
        manifest["conditions"][label] = {
            "fraction": fraction,
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "annotation_stats": stats,
            "prewarm_path": str(prewarm_path),
            "prewarm_sha256": sha256_file(prewarm_path),
            "prewarm_rows": len(prewarm_subset),
            "prewarm_valid_rows": prewarm_valid_count(prewarm_subset),
            "prewarm_epochs": int(round(1 / fraction)),
        }

    if not selected_episode_ids["25"].issubset(selected_episode_ids["50"]):
        raise AssertionError("25% subset is not nested inside the 50% subset")

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
