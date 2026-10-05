#!/usr/bin/env python3
"""
Step 1: Download & Setup MMRole Dataset
========================================
Downloads MMRole dataset from HuggingFace, downloads COCO images,
and organizes everything into a clean directory structure.

Usage:
    python step1_download_mmrole.py --output_dir projects/mmrole

Output structure:
    mmrole/
    ├── raw_data/           # Raw HF dataset
    ├── images/
    │   ├── coco/           # COCO train2017 images (only those referenced)
    │   └── character/      # Character-specific images
    ├── character_profiles/  # Extracted character profiles as JSON
    └── scripts/            # This script and others
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set

# ---------------------------------------------------------------------------
# 1. Download MMRole dataset from HuggingFace
# ---------------------------------------------------------------------------

def download_mmrole_dataset(output_dir: str) -> dict:
    """Download MMRole dataset from HuggingFace.

    The MMRole dataset has mismatched columns across files (train has 'type',
    test has 'question'/'eval_model'/etc), so we download the raw JSON files
    directly instead of using load_dataset().
    """
    from huggingface_hub import hf_hub_download, list_repo_tree

    raw_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)

    repo_id = "YanqiDai/MMRole_dataset"
    print(f"Downloading MMRole dataset from {repo_id} ...")

    # List all JSON files in the repo
    json_files = []
    for entry in list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        if entry.path.endswith(".json"):
            json_files.append(entry.path)

    print(f"  Found {len(json_files)} JSON files")

    split_info = {}
    for remote_path in sorted(json_files):
        print(f"  Downloading {remote_path} ...")
        local_path = hf_hub_download(
            repo_id=repo_id, filename=remote_path, repo_type="dataset"
        )

        # Load JSON and save as JSONL
        # Derive a flat name: data/train/train_85k.json -> train_85k.jsonl
        base_name = os.path.basename(remote_path).replace(".json", ".jsonl")
        # Prefix with subdirectory for clarity
        parts = remote_path.replace(".json", "").split("/")
        flat_name = "_".join(parts) + ".jsonl"

        out_path = os.path.join(raw_dir, flat_name)
        print(f"    -> {out_path}")

        with open(local_path, "r") as f_in:
            data = json.load(f_in)

        # Handle both list-of-dicts and single-dict formats
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Some files might be {key: [records]}
            records = []
            for v in data.values():
                if isinstance(v, list):
                    records.extend(v)
            if not records:
                records = [data]
        else:
            print(f"    WARN: unexpected format, skipping")
            continue

        with open(out_path, "w") as f_out:
            for record in records:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        split_info[flat_name] = len(records)
        print(f"    {len(records)} examples")

    return split_info


# ---------------------------------------------------------------------------
# 2. Extract character profiles
# ---------------------------------------------------------------------------

def extract_character_profiles(output_dir: str) -> Dict[str, dict]:
    """
    Parse all dialogues and extract unique character profiles.
    Saves each character profile as a separate JSON file.
    """
    raw_dir = os.path.join(output_dir, "raw_data")
    profile_dir = os.path.join(output_dir, "character_profiles")
    os.makedirs(profile_dir, exist_ok=True)

    characters = {}

    # Also load detailed profiles from profile files
    for jsonl_file in sorted(Path(raw_dir).glob("profiles_*detailed_profiles_*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                if isinstance(data, dict) and "introduction" in data:
                    # Derive character name from filename
                    # e.g., profiles_in-distribution_detailed_profiles_Iron_Man.jsonl
                    fname = jsonl_file.stem  # without .jsonl
                    parts = fname.split("_detailed_profiles_")
                    if len(parts) == 2:
                        char_name = parts[1].replace("_", " ")
                        if char_name not in characters:
                            characters[char_name] = {
                                "name": char_name,
                                "system_prompt": "",
                                "first_seen_in": str(jsonl_file.name),
                                "instruction_excerpt": "",
                                "detailed_profile": data,
                            }

    for jsonl_file in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                example = json.loads(line)

                # Skip non-dict records (roles.json produces strings, profiles.json produces flat dicts)
                if not isinstance(example, dict):
                    continue

                # MMRole stores character info in the system prompt or conversations
                # Try to extract from the structured fields
                role = example.get("role", "")
                other_role = example.get("other_role", "")
                system_prompt = example.get("system", "")

                # Extract from conversations if available
                conversations = example.get("conversations", [])
                if conversations and len(conversations) > 0:
                    user_msg = conversations[0].get("value", "")
                    # The user message typically contains character instructions
                    if role and role not in characters:
                        characters[role] = {
                            "name": role,
                            "system_prompt": system_prompt,
                            "first_seen_in": str(jsonl_file.name),
                            "instruction_excerpt": user_msg[:500] if user_msg else ""
                        }
                    if other_role and other_role not in characters:
                        characters[other_role] = {
                            "name": other_role,
                            "first_seen_in": str(jsonl_file.name),
                            "system_prompt": "",
                            "instruction_excerpt": ""
                        }

    # Save profiles
    for name, profile in characters.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        profile_path = os.path.join(profile_dir, f"{safe_name}.json")
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    # Save a master index
    index_path = os.path.join(profile_dir, "_index.json")
    with open(index_path, "w") as f:
        json.dump(
            {name: f"{name.replace('/', '_').replace(' ', '_')}.json"
             for name in sorted(characters.keys())},
            f, indent=2
        )

    print(f"Extracted {len(characters)} unique characters -> {profile_dir}")
    return characters


# ---------------------------------------------------------------------------
# 3. Collect referenced image paths
# ---------------------------------------------------------------------------

def collect_image_references(output_dir: str) -> Set[str]:
    """Scan all examples and collect unique image paths."""
    raw_dir = os.path.join(output_dir, "raw_data")
    image_refs = set()

    for jsonl_file in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                example = json.loads(line)
                if not isinstance(example, dict):
                    continue
                img = example.get("image", "")
                if img:
                    image_refs.add(img)
                # Also check inside conversations for <img> tags
                for conv in example.get("conversations", []):
                    val = conv.get("value", "")
                    if "<img>" in val:
                        import re
                        for match in re.findall(r"<img>(.*?)</img>", val):
                            image_refs.add(match)

    print(f"Found {len(image_refs)} unique image references")
    return image_refs


def download_coco_images(image_refs: Set[str], output_dir: str):
    """
    Download only the COCO images that are actually referenced.
    MMRole uses COCO train2017 images.
    """
    import urllib.request

    coco_dir = os.path.join(output_dir, "images", "coco")
    os.makedirs(coco_dir, exist_ok=True)

    coco_refs = [ref for ref in image_refs if "COCO" in ref or "train2017" in ref]
    print(f"COCO images to download: {len(coco_refs)}")

    # Extract just the filename from paths like "COCO/train2017/000000224155.jpg"
    coco_base_url = "http://images.cocodataset.org/train2017/"

    downloaded = 0
    skipped = 0
    for ref in sorted(coco_refs):
        filename = os.path.basename(ref)
        dest = os.path.join(coco_dir, filename)

        if os.path.exists(dest):
            skipped += 1
            continue

        url = coco_base_url + filename
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"  Downloaded {downloaded}/{len(coco_refs)} COCO images ...")
        except Exception as e:
            print(f"  WARN: Failed to download {filename}: {e}")

    print(f"COCO images: {downloaded} downloaded, {skipped} already existed")


# ---------------------------------------------------------------------------
# 4. Generate summary stats
# ---------------------------------------------------------------------------

def generate_summary(output_dir: str, split_info: dict, characters: dict, image_refs: set):
    """Write a summary JSON with dataset statistics."""
    raw_dir = os.path.join(output_dir, "raw_data")

    # Count dialogue types
    dialogue_types = {"commentary": 0, "human_role": 0, "inter_role": 0, "other": 0}
    total_turns = 0

    for jsonl_file in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                example = json.loads(line)
                if not isinstance(example, dict):
                    continue
                ex_id = example.get("id", "")
                convs = example.get("conversations", [])
                total_turns += len(convs)

                if "comment" in ex_id.lower():
                    dialogue_types["commentary"] += 1
                elif example.get("other_role", ""):
                    # Has another role = inter-role dialogue
                    other = example["other_role"]
                    if other.startswith("a ") or other.startswith("an ") or other == "a curious human":
                        dialogue_types["human_role"] += 1
                    else:
                        dialogue_types["inter_role"] += 1
                else:
                    dialogue_types["other"] += 1

    summary = {
        "splits": split_info,
        "total_characters": len(characters),
        "total_image_refs": len(image_refs),
        "dialogue_types": dialogue_types,
        "total_conversation_turns": total_turns,
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDataset summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1: Download & setup MMRole dataset")
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole",
                        help="Root output directory")
    parser.add_argument("--skip_coco_download", action="store_true",
                        help="Skip downloading COCO images (if you already have them)")
    parser.add_argument("--coco_images_dir", type=str, default=None,
                        help="Path to existing COCO train2017 images (will symlink instead of download)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Download dataset
    print("=" * 60)
    print("STEP 1a: Downloading MMRole dataset from HuggingFace")
    print("=" * 60)
    split_info = download_mmrole_dataset(args.output_dir)

    # 2. Extract character profiles
    print("\n" + "=" * 60)
    print("STEP 1b: Extracting character profiles")
    print("=" * 60)
    characters = extract_character_profiles(args.output_dir)

    # 3. Collect and download images
    print("\n" + "=" * 60)
    print("STEP 1c: Collecting image references")
    print("=" * 60)
    image_refs = collect_image_references(args.output_dir)

    if args.coco_images_dir and os.path.isdir(args.coco_images_dir):
        # Symlink existing COCO images
        coco_link = os.path.join(args.output_dir, "images", "coco")
        os.makedirs(os.path.dirname(coco_link), exist_ok=True)
        if not os.path.exists(coco_link):
            os.symlink(args.coco_images_dir, coco_link)
            print(f"Symlinked COCO images: {args.coco_images_dir} -> {coco_link}")
    elif not args.skip_coco_download:
        print("\n" + "=" * 60)
        print("STEP 1d: Downloading referenced COCO images")
        print("=" * 60)
        download_coco_images(image_refs, args.output_dir)
    else:
        print("Skipping COCO image download (--skip_coco_download)")

    # 4. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    generate_summary(args.output_dir, split_info, characters, image_refs)

    print("\nStep 1 complete! Next: run step2_restructure_per_turn.py")


if __name__ == "__main__":
    main()
