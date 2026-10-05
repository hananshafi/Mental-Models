#!/usr/bin/env python3
"""
Step 5: Build Training Data from Structured Belief States
===========================================================
Converts validated belief-state annotations into multiple training
formats for different downstream tasks:

  A. BELIEF PREDICTION: (image, agents, context) → belief state
     For training the VAE encoder (z_belief, z_intent, z_thought)

  B. PREFERENCE PAIRS: (image, context, tom_aligned) > (image, context, tom_violation)
     For Bradley-Terry preference learning

  C. PROBE QA: (image, context, question) → answer
     For direct ToM evaluation and reward model fine-tuning

  D. SALIENCE PREDICTION: (image, agent_profile) → per-object salience
     For training visual perspective modules

Character-stratified train/val/test split following MMRole's design.

Usage:
    python step5_build_training_data.py \
        --input_path projects/mmrole/mmrole_annotated_clean.jsonl \
        --output_dir projects/mmrole/training_data
"""

import os
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from itertools import combinations


# ---------------------------------------------------------------------------
# Character split
# ---------------------------------------------------------------------------

def auto_split_characters(examples: List[dict], seed: int = 42) -> Tuple[set, set]:
    """Deterministic character split: ~85% train, ~15% test."""
    all_chars = set()
    for ex in examples:
        agents = ex.get("agents", {})
        all_chars.add(agents.get("speaker", {}).get("name", ""))
        all_chars.add(agents.get("partner", {}).get("name", ""))
    all_chars.discard("")

    sorted_chars = sorted(all_chars)
    rng = random.Random(seed)
    rng.shuffle(sorted_chars)

    n_test = max(1, len(sorted_chars) // 6)
    test_chars = set(sorted_chars[:n_test])
    train_chars = all_chars - test_chars
    return train_chars, test_chars


def load_character_split(profiles_dir: str, seed: int = 42) -> Tuple[set, set]:
    index_path = os.path.join(profiles_dir, "_index.json")
    if not os.path.exists(index_path):
        return set(), set()

    with open(index_path) as f:
        all_chars = set(json.load(f).keys())

    sorted_chars = sorted(all_chars)
    rng = random.Random(seed)
    rng.shuffle(sorted_chars)
    n_test = max(1, len(sorted_chars) // 6)
    return all_chars - set(sorted_chars[:n_test]), set(sorted_chars[:n_test])


# ---------------------------------------------------------------------------
# Format A: Belief Prediction
# ---------------------------------------------------------------------------

def build_belief_prediction_example(ex: dict) -> Dict[str, Any]:
    """
    Training target: predict the full belief state from (image, agents, context).
    This is the primary format for training the VAE z encoder.
    """
    bs = ex["belief_state"]
    agents = ex["agents"]
    ctx = ex["interaction_context"]

    # Flatten 1st-order beliefs into trainable text targets
    fob = bs["first_order_beliefs"]
    sob = bs["second_order_beliefs"]

    return {
        "task": "belief_prediction",
        "example_id": ex["example_id"],
        "image": ex["scene"]["image"],
        "image_local": ex["scene"]["image_local"],

        "speaker_name": agents["speaker"]["name"],
        "speaker_profile": agents["speaker"]["profile"],
        "partner_name": agents["partner"]["name"],
        "partner_profile": agents["partner"]["profile"],

        "dialogue_history": ctx["dialogue_history"],
        "current_utterance": ctx["current_utterance"],

        # ── Targets ──
        # Per-object salience (structured)
        "target_scene_objects": bs["visual_percepts"]["scene_objects"],

        # 1st-order beliefs (text targets for mental decoder)
        "target_speaker_belief": {
            "partner_visual_focus": fob["speaker_believes_about_partner"]["perceived_visual_focus"],
            "partner_intent": fob["speaker_believes_about_partner"]["perceived_intent"],
            "partner_knowledge": fob["speaker_believes_about_partner"]["perceived_knowledge"],
            "partner_emotion": fob["speaker_believes_about_partner"]["perceived_emotion"],
        },

        # 2nd-order beliefs (text targets)
        "target_speaker_2nd_order": {
            "partner_thinks_i_see": sob["speaker_thinks_partner_thinks_speaker"]["sees"],
            "partner_thinks_i_want": sob["speaker_thinks_partner_thinks_speaker"]["wants"],
            "partner_thinks_i_know": sob["speaker_thinks_partner_thinks_speaker"]["knows"],
        },

        # Metadata for filtering/weighting
        "tom_relevance": bs["metadata"]["tom_relevance"],
        "belief_divergence": bs["metadata"]["belief_divergence"],
        "visual_asymmetry": bs["metadata"]["visual_asymmetry_present"],
    }


# ---------------------------------------------------------------------------
# Format B: Preference Pairs
# ---------------------------------------------------------------------------

def build_preference_pairs(ex: dict) -> List[Dict[str, Any]]:
    """
    Generate (preferred, rejected) pairs for Bradley-Terry preference learning.
    Creates multiple pairs from the contrastive responses.
    """
    bs = ex["belief_state"]
    cr = bs["contrastive_responses"]
    agents = ex["agents"]
    ctx = ex["interaction_context"]

    base = {
        "task": "preference",
        "example_id": ex["example_id"],
        "image": ex["scene"]["image"],
        "image_local": ex["scene"]["image_local"],
        "speaker_name": agents["speaker"]["name"],
        "speaker_profile": agents["speaker"]["profile"],
        "partner_name": agents["partner"]["name"],
        "partner_profile": agents["partner"]["profile"],
        "dialogue_history": ctx["dialogue_history"],
    }

    pairs = []
    preferred = cr["tom_aligned"]
    violations = {
        "visual": cr["tom_violation_visual"],
        "belief": cr["tom_violation_belief"],
        "order2": cr["tom_violation_order2"],
        "no_tom": cr["no_tom_baseline"],
    }

    for viol_type, rejected in violations.items():
        if preferred.strip() and rejected.strip() and preferred.strip() != rejected.strip():
            pair = dict(base)
            pair["preferred_response"] = preferred
            pair["rejected_response"] = rejected
            pair["violation_type"] = viol_type
            pair["pair_id"] = f"{ex['example_id']}__{viol_type}"
            pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Format C: Probe QA
# ---------------------------------------------------------------------------

def build_probe_qa(ex: dict) -> List[Dict[str, Any]]:
    """
    Convert belief probes to QA format for evaluation and reward model training.
    """
    bs = ex["belief_state"]
    probes = bs.get("belief_probes", [])
    agents = ex["agents"]
    ctx = ex["interaction_context"]

    qa_examples = []
    for i, probe in enumerate(probes):
        qa_examples.append({
            "task": "probe_qa",
            "example_id": f"{ex['example_id']}__probe{i}",
            "source_example_id": ex["example_id"],
            "image": ex["scene"]["image"],
            "image_local": ex["scene"]["image_local"],
            "speaker_name": agents["speaker"]["name"],
            "partner_name": agents["partner"]["name"],
            "speaker_profile": agents["speaker"]["profile"],
            "partner_profile": agents["partner"]["profile"],
            "dialogue_history": ctx["dialogue_history"],
            "current_utterance": ctx["current_utterance"],

            "question": probe["question"],
            "correct_answer": probe["answer"],
            "wrong_answer": probe["wrong_answer"],
            "probe_type": probe["probe_type"],
            "difficulty": probe["difficulty"],
        })

    return qa_examples


# ---------------------------------------------------------------------------
# Format D: Salience Prediction
# ---------------------------------------------------------------------------

def build_salience_examples(ex: dict) -> List[Dict[str, Any]]:
    """
    Per-agent visual salience prediction: given (image, agent_profile),
    predict salience level for each scene object.
    """
    bs = ex["belief_state"]
    objects = bs["visual_percepts"]["scene_objects"]
    agents = ex["agents"]

    examples = []
    for role in ["speaker", "partner"]:
        agent = agents[role]
        salience_key = f"salience_{role}"
        reason_key = f"salience_reason_{role}"

        target_salience = []
        for obj in objects:
            target_salience.append({
                "object": obj["object"],
                "description": obj["description"],
                "salience": obj.get(salience_key, "medium"),
                "reason": obj.get(reason_key, ""),
            })

        examples.append({
            "task": "salience_prediction",
            "example_id": f"{ex['example_id']}__{role}_salience",
            "source_example_id": ex["example_id"],
            "image": ex["scene"]["image"],
            "image_local": ex["scene"]["image_local"],
            "agent_name": agent["name"],
            "agent_profile": agent["profile"],
            "target_salience": target_salience,
        })

    return examples


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def load_official_test_dialogue_ids(raw_data_dir: str) -> Tuple[set, set]:
    """Load MMRole official test dialogue IDs (stripped to dialogue-level).

    Returns (in_dist_dids, ood_dids) where each ID matches our dialogue_id format.
    Official IDs have an extra _M suffix (conversation index) that we strip.
    """
    in_dist_dids = set()
    ood_dids = set()

    test_files = {
        "in": [
            "data_test_in-distribution_inter-role_test.jsonl",
            "data_test_in-distribution_human-role_test.jsonl",
            "data_test_in-distribution_comment_test.jsonl",
        ],
        "ood": [
            "data_test_out-of-distribution_inter-role_test.jsonl",
            "data_test_out-of-distribution_human-role_test.jsonl",
            "data_test_out-of-distribution_comment_test.jsonl",
        ],
    }

    for split_type, files in test_files.items():
        for fname in files:
            path = os.path.join(raw_data_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        oid = d["id"]
                        # Strip trailing _M conversation index to get dialogue-level ID
                        parts = oid.rsplit("_", 1)
                        did = parts[0] if len(parts) == 2 and parts[1].isdigit() else oid
                        if split_type == "in":
                            in_dist_dids.add(did)
                        else:
                            ood_dids.add(did)
                    except (json.JSONDecodeError, KeyError):
                        continue

    return in_dist_dids, ood_dids


def split_by_character_and_dialogue(
    examples: List[dict],
    train_chars: set, test_chars: set,
    val_ratio: float = 0.1, test_in_ratio: float = 0.05,
    seed: int = 42,
    raw_data_dir: str = "",
) -> Dict[str, List[dict]]:
    """Split examples into train/val/test_in/test_out/official_test.

    Respects MMRole's official test splits:
    - official_test: annotated turns from official in-dist test dialogues
    - test_out: OOD characters (our annotation-based)
    - test_in: held-out in-dist dialogues (our annotation-based)
    - val: validation from in-dist
    - train: everything else
    """

    rng = random.Random(seed)

    # Load official test dialogue IDs
    official_in_dids, official_ood_dids = set(), set()
    if raw_data_dir and os.path.isdir(raw_data_dir):
        official_in_dids, official_ood_dids = load_official_test_dialogue_ids(raw_data_dir)
        print(f"  Official test IDs loaded: {len(official_in_dids)} in-dist, "
              f"{len(official_ood_dids)} OOD")

    # First pass: separate official test, OOD chars, and training pool
    official_test, test_out, train_pool = [], [], []
    for ex in examples:
        did = ex.get("dialogue_id", ex["example_id"])
        agents = ex.get("agents", {})
        s = agents.get("speaker", {}).get("name", "")
        p = agents.get("partner", {}).get("name", "")

        if did in official_in_dids:
            official_test.append(ex)
        elif s in test_chars and p in test_chars:
            test_out.append(ex)
        else:
            train_pool.append(ex)

    if official_test:
        print(f"  Held out {len(official_test)} turns matching MMRole official test dialogues")

    # Split train pool by dialogue (dialogue-level to prevent leakage)
    dialogue_to_examples = defaultdict(list)
    for ex in train_pool:
        dialogue_to_examples[ex.get("dialogue_id", ex["example_id"])].append(ex)

    dids = sorted(dialogue_to_examples.keys())
    rng.shuffle(dids)

    n_val = max(1, int(len(dids) * val_ratio))
    n_test_in = max(1, int(len(dids) * test_in_ratio))

    val_dids = set(dids[:n_val])
    test_in_dids = set(dids[n_val:n_val + n_test_in])
    train_dids = set(dids[n_val + n_test_in:])

    splits = {
        "train": [ex for d in train_dids for ex in dialogue_to_examples[d]],
        "val": [ex for d in val_dids for ex in dialogue_to_examples[d]],
        "test_in": [ex for d in test_in_dids for ex in dialogue_to_examples[d]],
        "test_out": test_out,
    }

    if official_test:
        splits["official_test"] = official_test

    rng.shuffle(splits["train"])
    rng.shuffle(splits["val"])
    return splits


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(examples: List[dict]) -> Dict[str, Any]:
    """Compute dataset statistics."""
    stats = {
        "total_examples": len(examples),
        "tom_relevance": Counter(),
        "belief_divergence": Counter(),
        "visual_asymmetry": Counter(),
        "probe_types": Counter(),
        "probe_difficulties": Counter(),
        "violation_types": Counter(),
        "avg_scene_objects": 0,
    }

    n_objects = []
    for ex in examples:
        bs = ex.get("belief_state", {})
        meta = bs.get("metadata", {})
        stats["tom_relevance"][meta.get("tom_relevance", "?")] += 1
        stats["belief_divergence"][meta.get("belief_divergence", "?")] += 1
        stats["visual_asymmetry"][str(meta.get("visual_asymmetry_present", "?"))] += 1

        objects = bs.get("visual_percepts", {}).get("scene_objects", [])
        n_objects.append(len(objects))

        for probe in bs.get("belief_probes", []):
            stats["probe_types"][probe.get("probe_type", "?")] += 1
            stats["probe_difficulties"][probe.get("difficulty", "?")] += 1

    if n_objects:
        stats["avg_scene_objects"] = sum(n_objects) / len(n_objects)

    # Convert Counters to dicts for JSON
    for k in ["tom_relevance", "belief_divergence", "visual_asymmetry",
              "probe_types", "probe_difficulties"]:
        stats[k] = dict(stats[k])

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 5: Build multi-format training data")
    parser.add_argument("--input_path", type=str,
                        default="projects/mmrole/mmrole_annotated_clean.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/training_data")
    parser.add_argument("--character_profiles_dir", type=str,
                        default="projects/mmrole/character_profiles")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_in_ratio", type=float, default=0.05)
    parser.add_argument("--raw_data_dir", type=str,
                        default="projects/mmrole/raw_data",
                        help="Path to MMRole raw data (for official test split IDs)")
    parser.add_argument("--official_test_annotated", type=str,
                        default="projects/mmrole/mmrole_official_test_annotated_clean.jsonl",
                        help="Annotated official test file (from step2b+step3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print(f"Loading from {args.input_path} ...")
    examples = []
    with open(args.input_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Total: {len(examples)}")

    # Character split
    train_chars, test_chars = load_character_split(args.character_profiles_dir, args.seed)
    if not train_chars:
        train_chars, test_chars = auto_split_characters(examples, args.seed)
    print(f"  Train chars: {len(train_chars)}, Test chars: {len(test_chars)}")

    # Split (respecting MMRole official test)
    splits = split_by_character_and_dialogue(
        examples, train_chars, test_chars,
        args.val_ratio, args.test_in_ratio, args.seed,
        raw_data_dir=args.raw_data_dir,
    )

    # Load separately-annotated official test set (all 294 dialogues)
    if args.official_test_annotated and os.path.exists(args.official_test_annotated):
        official_examples = []
        with open(args.official_test_annotated) as f:
            for line in f:
                official_examples.append(json.loads(line))
        # Replace any auto-split official_test with the full annotated set
        splits["official_test"] = official_examples
        print(f"  Loaded {len(official_examples)} annotated official test examples")

        # Also create sub-splits by distribution and dialogue type
        from collections import defaultdict as dd
        sub = dd(list)
        for ex in official_examples:
            tm = ex.get("test_metadata", {})
            dist = tm.get("distribution", "unknown").replace("-", "_")
            dtype = tm.get("dialogue_type", "unknown").replace("-", "_")
            sub[f"official_test_{dist}_{dtype}"].append(ex)
        for k, v in sorted(sub.items()):
            splits[k] = v
            print(f"    {k}: {len(v)}")

    # Copy official test files for response-generation evaluation
    if os.path.isdir(args.raw_data_dir):
        official_dir = os.path.join(args.output_dir, "official_test_raw")
        os.makedirs(official_dir, exist_ok=True)
        import shutil
        for fname in os.listdir(args.raw_data_dir):
            if fname.startswith("data_test_"):
                src = os.path.join(args.raw_data_dir, fname)
                shutil.copy2(src, os.path.join(official_dir, fname))
        print(f"  Copied MMRole official test files to {official_dir}")

    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} examples")

    # Build all 4 formats for each split
    for split_name, split_data in splits.items():
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # A: Belief prediction
        bp_path = os.path.join(split_dir, "belief_prediction.jsonl")
        bp_count = 0
        with open(bp_path, "w") as f:
            for ex in split_data:
                try:
                    bp = build_belief_prediction_example(ex)
                    f.write(json.dumps(bp, ensure_ascii=False) + "\n")
                    bp_count += 1
                except (KeyError, TypeError) as e:
                    pass

        # B: Preference pairs
        pref_path = os.path.join(split_dir, "preference_pairs.jsonl")
        pref_count = 0
        with open(pref_path, "w") as f:
            for ex in split_data:
                try:
                    for pair in build_preference_pairs(ex):
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        pref_count += 1
                except (KeyError, TypeError):
                    pass

        # C: Probe QA
        qa_path = os.path.join(split_dir, "probe_qa.jsonl")
        qa_count = 0
        with open(qa_path, "w") as f:
            for ex in split_data:
                try:
                    for qa in build_probe_qa(ex):
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                        qa_count += 1
                except (KeyError, TypeError):
                    pass

        # D: Salience prediction
        sal_path = os.path.join(split_dir, "salience_prediction.jsonl")
        sal_count = 0
        with open(sal_path, "w") as f:
            for ex in split_data:
                try:
                    for sal in build_salience_examples(ex):
                        f.write(json.dumps(sal, ensure_ascii=False) + "\n")
                        sal_count += 1
                except (KeyError, TypeError):
                    pass

        # Also save raw annotated data for this split
        raw_path = os.path.join(split_dir, "raw_annotated.jsonl")
        with open(raw_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  {split_name}/: belief={bp_count}, pref={pref_count}, "
              f"qa={qa_count}, salience={sal_count}")

    # Stats
    stats = compute_stats(examples)
    stats["splits"] = {k: len(v) for k, v in splits.items()}
    stats["train_characters"] = sorted(train_chars)
    stats["test_characters"] = sorted(test_chars)

    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Step 5 Complete!")
    print(f"{'='*60}")
    print(f"Output: {args.output_dir}")
    print(f"\nPer-split structure:")
    print(f"  <split>/belief_prediction.jsonl  — VAE encoder targets")
    print(f"  <split>/preference_pairs.jsonl   — Bradley-Terry pairs")
    print(f"  <split>/probe_qa.jsonl           — ToM evaluation probes")
    print(f"  <split>/salience_prediction.jsonl — visual perspective targets")
    print(f"  <split>/raw_annotated.jsonl       — full annotations")
    if "official_test" in splits:
        print(f"\n  official_test/         — {len(splits['official_test'])} turns from MMRole official test dialogues (annotated)")
        print(f"  official_test_raw/     — MMRole original test files (for response generation eval)")

    print(f"\nDataset stats:")
    print(f"  Avg scene objects: {stats['avg_scene_objects']:.1f}")
    print(f"  ToM relevance: {stats['tom_relevance']}")
    print(f"  Belief divergence: {stats['belief_divergence']}")
    print(f"  Visual asymmetry: {stats['visual_asymmetry']}")
    print(f"  Probe types: {stats['probe_types']}")


if __name__ == "__main__":
    main()
