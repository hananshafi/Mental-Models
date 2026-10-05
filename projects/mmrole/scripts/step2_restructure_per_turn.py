#!/usr/bin/env python3
"""
Step 2: Restructure MMRole into Belief-Centric Per-Turn JSONL
==============================================================
Decomposes each inter-role dialogue into per-turn examples structured
around BELIEF STATE ESTIMATION — not response generation.

Design informed by:
  - nested belief dictionaries (belief[A][B][obj] = state)
  - MMToM-QA: symbolic predicates grounded in scene graphs
  - MuMA-ToM: multi-agent + multimodal belief/goal inference
  - Dynamic Belief Graphs: structured binary belief vectors

Key difference from dialogue-centric format:
  Each example is a snapshot of the belief state at time t, including:
    - What each agent perceives in the image (visual percepts)
    - What each agent believes the other perceives (1st order ToM)
    - What each agent believes the other believes about them (2nd order ToM)
    - Belief probes: structured questions to test the belief state

FOCUS: Inter-role dialogues only (two characters interacting).

Usage:
    python step2_restructure_per_turn.py \
        --input_dir projects/mmrole/raw_data \
        --output_path projects/mmrole/mmrole_per_turn.jsonl \
        --min_utterance_tokens 20

Output format (one line per turn):
{
    "example_id": "Iron_Man_inter_COCO_30_1__t3",
    "dialogue_id": "Iron_Man_inter_COCO_30_1",
    "turn_id": 3,

    "agents": {
        "speaker": {
            "name": "Iron Man",
            "profile": "...",
            "role_in_turn": "speaker"
        },
        "partner": {
            "name": "Hermione Granger",
            "profile": "...",
            "role_in_turn": "partner"
        }
    },

    "scene": {
        "image": "COCO/train2017/000000224155.jpg",
        "image_local": "coco/000000224155.jpg"
    },

    "interaction_context": {
        "dialogue_history": [...],
        "current_utterance": "...",
        "total_turns": 6,
        "dialogue_type": "inter_role"
    }
}
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Character profile extraction from system/user prompts
# ---------------------------------------------------------------------------

def extract_profile_from_instruction(user_msg: str, character_name: str) -> str:
    """Extract character profile text from the MMRole user instruction."""
    profile_parts = []

    for pattern in [
        rf"(?:Character\s+Profile|Background|Description)\s*(?:of\s+)?{re.escape(character_name)}[:\s]+(.*?)(?=Character\s+Profile|Background|Description|\n\n|$)",
        rf"{re.escape(character_name)}(?:'s)?\s+(?:profile|background|description)[:\s]+(.*?)(?=\n\n|$)",
    ]:
        match = re.search(pattern, user_msg, re.IGNORECASE | re.DOTALL)
        if match:
            profile_parts.append(match.group(1).strip())

    if profile_parts:
        return " ".join(profile_parts)

    cleaned = re.sub(r"<img>.*?</img>", "", user_msg)
    cleaned = re.sub(r"Picture \d+:", "", cleaned)
    cleaned = cleaned.strip()

    shoes_match = re.search(
        rf"(?:step into the shoes of|role-?play as|you are)\s+{re.escape(character_name)}[.,:]?\s*(.*)",
        cleaned, re.IGNORECASE | re.DOTALL
    )
    if shoes_match:
        return shoes_match.group(1).strip()[:1000]

    return cleaned[:1000]


def extract_image_path(conversations: List[dict]) -> str:
    """Extract image path from MMRole conversation <img> tags."""
    for conv in conversations:
        val = conv.get("value", "")
        match = re.search(r"<img>(.*?)</img>", val)
        if match:
            return match.group(1)
    return ""


def parse_dialogue_turns(conversations: List[dict]) -> List[Dict[str, str]]:
    """Parse MMRole conversations into clean turn list."""
    turns = []
    for i, conv in enumerate(conversations):
        role = conv.get("from", "")
        value = conv.get("value", "")

        if i == 0 and role == "user":
            if "<img>" in value or "step into" in value.lower() or "role-play" in value.lower():
                continue

        utterance = value.strip()
        utterance = re.sub(r"^\([^)]*\)\s*", "", utterance)
        utterance = utterance.strip('"').strip("'")

        if utterance and len(utterance) > 5:
            turns.append({"role": role, "utterance": utterance})

    return turns


def classify_dialogue(example: dict, source_filename: str = "") -> str:
    """Classify dialogue as commentary, human_role, or inter_role.

    Handles two MMRole formats:
      1. train_85k / RM format: has 'role', 'other_role', 'type' fields
      2. dialogues/ format: only 'id', 'image', 'conversations' — classify by filename or id
    """
    ex_id = example.get("id", "")
    other_role = example.get("other_role", "")
    ex_type = example.get("type", "")

    # Check explicit type field (train_85k format)
    if ex_type:
        type_lower = ex_type.lower()
        if "comment" in type_lower:
            return "commentary"
        if "inter" in type_lower or "role-role" in type_lower:
            return "inter_role"
        if "human" in type_lower:
            return "human_role"

    # Check filename (dialogues/ format)
    fname_lower = source_filename.lower()
    if "inter-role" in fname_lower or "inter_role" in fname_lower:
        return "inter_role"
    if "human-role" in fname_lower or "human_role" in fname_lower:
        return "human_role"
    if "comment" in fname_lower:
        return "commentary"

    # Check id field
    id_lower = ex_id.lower()
    if "role-role" in id_lower or "inter" in id_lower:
        return "inter_role"
    if "comment" in id_lower:
        return "commentary"

    # Check other_role field
    if other_role:
        human_indicators = [
            "a curious human", "a human", "a user", "a fan",
            "a friend", "a stranger", "an interviewer",
            "a reporter", "a journalist", "a student"
        ]
        other_lower = other_role.lower().strip()
        if any(other_lower.startswith(h) or other_lower == h for h in human_indicators):
            return "human_role"
        if other_lower.startswith("a ") or other_lower.startswith("an "):
            return "human_role"
        return "inter_role"

    # Check if conversations use character names (not "user"/"assistant")
    convs = example.get("conversations", [])
    if convs and len(convs) >= 2:
        roles = set()
        for c in convs[:4]:
            r = c.get("role", c.get("from", ""))
            if r and r not in ("user", "assistant", "system"):
                roles.add(r)
        if len(roles) >= 2:
            return "inter_role"

    return "other"


def image_path_to_local(image_path: str) -> str:
    """Convert MMRole image path to local path under images/ dir."""
    filename = os.path.basename(image_path)
    if "COCO" in image_path or "train2017" in image_path:
        return f"coco/{filename}"
    return f"character/{filename}"


# ---------------------------------------------------------------------------
# Belief-centric decomposition
# ---------------------------------------------------------------------------

def decompose_dialogue(example: dict, source_filename: str = "",
                        profile_lookup: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Decompose a single MMRole dialogue into belief-centric per-turn examples.
    Each example represents a belief state snapshot at time t.

    Handles two formats:
      1. train_85k: has 'role', 'other_role', 'system', conversations use 'from': user/assistant
      2. dialogues/: only 'id', 'image', conversations use 'role': "Iron Man" etc.
    """
    dialogue_type = classify_dialogue(example, source_filename)
    if dialogue_type != "inter_role":
        return []

    dialogue_id = example.get("id", "unknown")
    conversations = example.get("conversations", [])
    system_prompt = example.get("system", "")

    if not conversations or len(conversations) < 2:
        return []

    image_path = example.get("image", "") or extract_image_path(conversations)
    image_local = image_path_to_local(image_path) if image_path else ""

    # Detect format: does conversations use 'role' (character names) or 'from' (user/assistant)?
    first_conv = conversations[0]
    uses_character_roles = "role" in first_conv and first_conv.get("role") not in ("user", "assistant", "system")

    if uses_character_roles:
        # Format 2: dialogues/ — character names directly in conversation turns
        # Extract unique character names from conversations
        char_names = []
        seen = set()
        for c in conversations:
            name = c.get("role", "")
            if name and name not in seen:
                char_names.append(name)
                seen.add(name)

        if len(char_names) < 2:
            return []

        role_a = char_names[0]
        role_b = char_names[1]

        # Build named turns directly
        named_turns = []
        for i, conv in enumerate(conversations):
            utterance = conv.get("value", "").strip()
            utterance = re.sub(r"^\([^)]*\)\s*", "", utterance)
            utterance = utterance.strip('"').strip("'")
            speaker = conv.get("role", "")

            if utterance and len(utterance) > 5 and speaker:
                named_turns.append({
                    "turn": len(named_turns),
                    "speaker": speaker,
                    "utterance": utterance,
                })
    else:
        # Format 1: train_85k — uses 'from': user/assistant
        role_a = example.get("role", "Character A")
        role_b = example.get("other_role", "Character B")

        raw_turns = parse_dialogue_turns(conversations)
        if len(raw_turns) < 2:
            return []

        named_turns = []
        for i, turn in enumerate(raw_turns):
            if turn["role"] == "assistant":
                speaker = role_b
            elif turn["role"] == "user":
                speaker = role_a
            else:
                speaker = role_a if i % 2 == 0 else role_b

            named_turns.append({
                "turn": i,
                "speaker": speaker,
                "utterance": turn["utterance"],
            })

    if len(named_turns) < 2:
        return []

    # Get profiles — try profile_lookup, then instruction text, then fallback
    profile_a = ""
    profile_b = ""
    if profile_lookup:
        profile_a = profile_lookup.get(role_a, "")
        profile_b = profile_lookup.get(role_b, "")

    if not profile_a:
        first_user_msg = conversations[0].get("value", "")
        profile_a = extract_profile_from_instruction(first_user_msg, role_a)
    if not profile_b:
        first_user_msg = conversations[0].get("value", "")
        profile_b = extract_profile_from_instruction(first_user_msg, role_b)
    if not profile_a:
        profile_a = f"Character: {role_a}"
    if not profile_b:
        profile_b = f"Character: {role_b}"

    # Generate belief-centric examples
    per_turn_examples = []
    for t_idx, turn in enumerate(named_turns):
        speaker = turn["speaker"]
        partner = role_b if speaker == role_a else role_a
        speaker_profile = profile_a if speaker == role_a else profile_b
        partner_profile = profile_b if speaker == role_a else profile_a

        per_turn_examples.append({
            # Identifiers
            "example_id": f"{dialogue_id}__t{t_idx}",
            "dialogue_id": dialogue_id,
            "turn_id": t_idx,

            # Agent information (structured, not flat)
            "agents": {
                "speaker": {
                    "name": speaker,
                    "profile": speaker_profile,
                    "role_in_turn": "speaker"
                },
                "partner": {
                    "name": partner,
                    "profile": partner_profile,
                    "role_in_turn": "partner"
                }
            },

            # Scene (visual context)
            "scene": {
                "image": image_path,
                "image_local": image_local,
            },

            # Interaction context
            "interaction_context": {
                "dialogue_history": named_turns[:t_idx],
                "current_utterance": turn["utterance"],
                "total_turns": len(named_turns),
                "dialogue_type": dialogue_type,
            },
        })

    return per_turn_examples


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Restructure MMRole into belief-centric per-turn JSONL"
    )
    parser.add_argument("--input_dir", type=str,
                        default="projects/mmrole/raw_data")
    parser.add_argument("--output_path", type=str,
                        default="projects/mmrole/mmrole_per_turn.jsonl")
    parser.add_argument("--min_utterance_tokens", type=int, default=20)
    parser.add_argument("--include_human_role", action="store_true")
    args = parser.parse_args()

    input_files = sorted(Path(args.input_dir).glob("*.jsonl"))
    if not input_files:
        print(f"ERROR: No JSONL files found in {args.input_dir}")
        return

    # Load character profiles for enrichment
    profile_dir = os.path.join(os.path.dirname(args.input_dir), "character_profiles")
    profile_lookup = {}
    if os.path.isdir(profile_dir):
        for pf in Path(profile_dir).glob("*.json"):
            if pf.name == "_index.json":
                continue
            try:
                with open(pf) as f:
                    data = json.load(f)
                name = data.get("name", "")
                # Build a profile string from detailed_profile or instruction_excerpt
                dp = data.get("detailed_profile", {})
                if dp and isinstance(dp, dict):
                    parts = []
                    for key in ["introduction", "personality", "life_story",
                                "main_interpersonal_relationships", "catchphrases"]:
                        if key in dp:
                            parts.append(dp[key] if isinstance(dp[key], str) else str(dp[key]))
                    if parts:
                        profile_lookup[name] = " ".join(parts)[:2000]
                if name and name not in profile_lookup:
                    excerpt = data.get("instruction_excerpt", "")
                    if excerpt:
                        profile_lookup[name] = excerpt[:1000]
            except Exception:
                pass
        print(f"Loaded {len(profile_lookup)} character profiles from {profile_dir}")

    total_dialogues = 0
    total_inter_role = 0
    total_turns = 0
    skipped_short = 0

    with open(args.output_path, "w") as out_f:
        for jsonl_file in input_files:
            print(f"Processing {jsonl_file.name} ...")
            with open(jsonl_file) as f:
                for line in f:
                    example = json.loads(line)
                    if not isinstance(example, dict):
                        continue
                    total_dialogues += 1

                    fname = jsonl_file.name
                    if args.include_human_role:
                        dtype = classify_dialogue(example, fname)
                        if dtype == "commentary":
                            continue

                    per_turn = decompose_dialogue(example, fname, profile_lookup)
                    if not per_turn:
                        continue

                    total_inter_role += 1
                    for turn_ex in per_turn:
                        word_count = len(turn_ex["interaction_context"]["current_utterance"].split())
                        if word_count < args.min_utterance_tokens:
                            skipped_short += 1
                            continue

                        out_f.write(json.dumps(turn_ex, ensure_ascii=False) + "\n")
                        total_turns += 1

    print(f"\n{'='*60}")
    print(f"Step 2 Complete!")
    print(f"{'='*60}")
    print(f"Total dialogues scanned:  {total_dialogues}")
    print(f"Inter-role dialogues:     {total_inter_role}")
    print(f"Per-turn examples:        {total_turns}")
    print(f"Skipped (too short):      {skipped_short}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
