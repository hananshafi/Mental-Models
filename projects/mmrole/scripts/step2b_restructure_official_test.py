#!/usr/bin/env python3
"""
Step 2b: Restructure MMRole Official Test Set into Per-Turn JSONL
==================================================================
Converts all 294 official test dialogues (inter-role, human-role, commentary)
into the same per-turn belief-centric format used by step 3 annotation.

Handles the difference that test dialogues use "from": "user"/"assistant"
format where the user prompt contains the full character description and
conversation history, and assistant is the single response.
"""

import os
import json
import re
import argparse
from typing import Dict, Optional


def load_profiles(profiles_dir: str) -> Dict[str, str]:
    """Load character profiles from the profiles directory."""
    profiles = {}
    index_path = os.path.join(profiles_dir, "_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for name, filename in index.items():
            # index maps name -> filename (str)
            profile_path = os.path.join(profiles_dir, filename)
            if os.path.exists(profile_path):
                try:
                    with open(profile_path) as f:
                        d = json.load(f)
                    dp = d.get("detailed_profile", d.get("profile", d.get("description", "")))
                    if isinstance(dp, dict):
                        # Flatten dict sections into a single string
                        profiles[name] = "\n".join(
                            f"{k}: {v}" if isinstance(v, str) else f"{k}: {json.dumps(v)}"
                            for k, v in dp.items()
                        )
                    else:
                        profiles[name] = str(dp) if dp else name
                except (json.JSONDecodeError, Exception):
                    profiles[name] = name
            if name not in profiles:
                profiles[name] = name
    return profiles


def extract_conversation_history(user_prompt: str) -> list:
    """Extract conversation history from the user prompt text.

    The user prompt contains lines like:
    [CharName]: utterance text
    """
    history = []
    # Match lines starting with [Name]: text in the conversation history section
    pattern = r'\[([^\]]+)\]:\s*(.+)'
    for match in re.finditer(pattern, user_prompt):
        history.append({
            "turn": len(history),
            "speaker": match.group(1),
            "utterance": match.group(2).strip(),
        })
    return history


def restructure_test_dialogue(dialogue: dict, dialogue_type: str,
                               dist_type: str, profiles: dict) -> list:
    """Convert one official test dialogue into per-turn examples."""

    did = dialogue["id"]
    image = dialogue.get("image", "")
    role = dialogue.get("role", "")
    other_role = dialogue.get("other_role", "")
    conversations = dialogue.get("conversations", [])

    # Resolve image local path
    image_local = ""
    if image:
        fname = os.path.basename(image)
        image_local = f"coco/{fname}"

    # Get profiles
    speaker_profile = profiles.get(role, role)
    partner_profile = profiles.get(other_role, other_role)

    examples = []

    # The test format is: conv[0]=user prompt (with history), conv[1]=assistant response
    # For each response turn, create one per-turn example
    for i in range(0, len(conversations), 2):
        if i + 1 >= len(conversations):
            break

        user_msg = conversations[i].get("value", "")
        assistant_msg = conversations[i + 1].get("value", "")

        # Extract any conversation history from the user prompt
        history = extract_conversation_history(user_msg)

        turn_id = i // 2
        example_id = f"{did}__t{turn_id}"

        example = {
            "example_id": example_id,
            "dialogue_id": did,
            "turn_id": turn_id,
            "agents": {
                "speaker": {
                    "name": role,
                    "profile": speaker_profile[:2000],
                    "role_in_turn": "speaker",
                },
                "partner": {
                    "name": other_role,
                    "profile": partner_profile[:2000],
                    "role_in_turn": "partner",
                },
            },
            "scene": {
                "image": image,
                "image_local": image_local,
            },
            "interaction_context": {
                "dialogue_history": history,
                "current_utterance": assistant_msg,
                "total_turns": len(conversations) // 2,
                "dialogue_type": dialogue_type,
            },
            "test_metadata": {
                "distribution": dist_type,
                "dialogue_type": dialogue_type,
                "original_id": did,
            },
        }
        examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Restructure MMRole official test into per-turn format")
    parser.add_argument("--raw_data_dir", type=str,
                        default="projects/mmrole/raw_data")
    parser.add_argument("--profiles_dir", type=str,
                        default="projects/mmrole/character_profiles")
    parser.add_argument("--output_path", type=str,
                        default="projects/mmrole/mmrole_official_test_per_turn.jsonl")
    args = parser.parse_args()

    profiles = load_profiles(args.profiles_dir)
    print(f"Loaded {len(profiles)} character profiles")

    test_files = []
    for dist in ["in-distribution", "out-of-distribution"]:
        for dtype in ["inter-role", "human-role", "comment"]:
            fname = f"data_test_{dist}_{dtype}_test.jsonl"
            path = os.path.join(args.raw_data_dir, fname)
            if os.path.exists(path):
                test_files.append((path, dtype, dist))

    total_dialogues = 0
    total_turns = 0

    with open(args.output_path, "w") as out_f:
        for path, dtype, dist in test_files:
            count = 0
            with open(path) as f:
                for line in f:
                    try:
                        dialogue = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    examples = restructure_test_dialogue(dialogue, dtype, dist, profiles)
                    for ex in examples:
                        out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        total_turns += 1
                    count += 1
                    total_dialogues += 1

            print(f"  {dist} {dtype}: {count} dialogues")

    print(f"\nTotal: {total_dialogues} dialogues → {total_turns} turns")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
