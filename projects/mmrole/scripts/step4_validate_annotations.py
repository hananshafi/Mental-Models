#!/usr/bin/env python3
"""
Step 4: Validate Structured Belief State Annotations
======================================================
Quality checks tailored to the belief-centric format:

  1. Visual percept checks: salience asymmetry, object count
  2. Belief coherence: 1st-order beliefs reference actual scene objects
  3. Probe quality: correct/wrong answers must differ, difficulty distribution
  4. Contrastive response quality: ToM violations must differ from aligned
  5. Belief nesting consistency: 2nd-order beliefs logically follow from 1st-order
  6. Metadata plausibility: asymmetry flag matches actual salience patterns

Usage:
    python step4_validate_annotations.py \
        --input_path projects/mmrole/mmrole_annotated.jsonl \
        --output_clean projects/mmrole/mmrole_annotated_clean.jsonl \
        --output_flagged projects/mmrole/mmrole_annotated_flagged.jsonl
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def check_visual_percepts(bs: dict) -> List[str]:
    """Check visual percept quality."""
    issues = []
    vp = bs.get("visual_percepts", {})
    objects = vp.get("scene_objects", [])

    # Must have 4-8 objects
    if len(objects) < 3:
        issues.append("too_few_scene_objects")
    if len(objects) > 12:
        issues.append("too_many_scene_objects")

    # Check salience asymmetry — at least 1 object must differ between agents
    has_asymmetry = False
    for obj in objects:
        if obj.get("salience_speaker") != obj.get("salience_partner"):
            has_asymmetry = True
            break

    if not has_asymmetry and len(objects) > 0:
        issues.append("no_salience_asymmetry")

    # Check that salience reasons reference character (not generic)
    for obj in objects:
        sr = obj.get("salience_reason_speaker", "").lower()
        pr = obj.get("salience_reason_partner", "").lower()
        if sr and pr and sr == pr:
            issues.append("identical_salience_reasons")
            break

    # Check attention fields not empty
    if not vp.get("speaker_attention", "").strip():
        issues.append("empty_speaker_attention")
    if not vp.get("partner_attention", "").strip():
        issues.append("empty_partner_attention")

    return issues


def check_first_order_beliefs(bs: dict) -> List[str]:
    """Check 1st-order belief quality."""
    issues = []
    fob = bs.get("first_order_beliefs", {})

    for agent_key in ["speaker_believes_about_partner", "partner_believes_about_speaker"]:
        beliefs = fob.get(agent_key, {})
        for field in ["perceived_visual_focus", "perceived_intent", "perceived_knowledge", "perceived_emotion"]:
            val = beliefs.get(field, "")
            if not val or len(val.strip()) < 5:
                issues.append(f"empty_1st_order:{agent_key}.{field}")

    # Check that the two agents' beliefs are not identical
    spb = fob.get("speaker_believes_about_partner", {})
    pbs = fob.get("partner_believes_about_speaker", {})
    if (spb.get("perceived_visual_focus", "").strip().lower() ==
            pbs.get("perceived_visual_focus", "").strip().lower()):
        if spb.get("perceived_visual_focus", "").strip():
            issues.append("symmetric_1st_order_visual_focus")

    return issues


def check_second_order_beliefs(bs: dict) -> List[str]:
    """Check 2nd-order belief quality."""
    issues = []
    sob = bs.get("second_order_beliefs", {})
    nested = sob.get("speaker_thinks_partner_thinks_speaker", {})

    for field in ["sees", "wants", "knows"]:
        val = nested.get(field, "")
        if not val or len(val.strip()) < 5:
            issues.append(f"empty_2nd_order:{field}")

    return issues


def check_belief_probes(bs: dict) -> List[str]:
    """Check belief probe quality."""
    issues = []
    probes = bs.get("belief_probes", [])

    if len(probes) < 4:
        issues.append("too_few_probes")
    if len(probes) > 7:
        issues.append("too_many_probes")

    probe_types_seen = set()
    difficulties_seen = set()

    for i, probe in enumerate(probes):
        q = probe.get("question", "").strip()
        a = probe.get("answer", "").strip()
        wa = probe.get("wrong_answer", "").strip()
        pt = probe.get("probe_type", "")
        diff = probe.get("difficulty", "")

        if not q or len(q) < 10:
            issues.append(f"empty_probe_question:{i}")
        if not a or len(a) < 3:
            issues.append(f"empty_probe_answer:{i}")
        if not wa or len(wa) < 3:
            issues.append(f"empty_probe_wrong_answer:{i}")

        # Correct and wrong answer must differ
        if a.lower() == wa.lower():
            issues.append(f"identical_probe_answers:{i}")

        probe_types_seen.add(pt)
        difficulties_seen.add(diff)

    # Should have variety in probe types
    if len(probe_types_seen) < 2:
        issues.append("low_probe_type_variety")

    # Should have at least one hard probe
    if "hard" not in difficulties_seen and len(probes) >= 4:
        issues.append("no_hard_probes")

    return issues


def check_contrastive_responses(bs: dict) -> List[str]:
    """Check contrastive response quality."""
    issues = []
    cr = bs.get("contrastive_responses", {})

    required = ["tom_aligned", "tom_violation_visual", "tom_violation_belief",
                 "tom_violation_order2", "no_tom_baseline"]

    responses = {}
    for key in required:
        val = cr.get(key, "").strip()
        if not val or len(val) < 15:
            issues.append(f"empty_contrastive:{key}")
        else:
            responses[key] = val.lower()

    # Check that violations differ from aligned
    aligned = responses.get("tom_aligned", "")
    if aligned:
        for viol_key in ["tom_violation_visual", "tom_violation_belief", "tom_violation_order2"]:
            viol = responses.get(viol_key, "")
            if viol and viol == aligned:
                issues.append(f"violation_equals_aligned:{viol_key}")

    # Check violations differ from each other
    viol_keys = ["tom_violation_visual", "tom_violation_belief", "tom_violation_order2"]
    for i in range(len(viol_keys)):
        for j in range(i + 1, len(viol_keys)):
            v1 = responses.get(viol_keys[i], "")
            v2 = responses.get(viol_keys[j], "")
            if v1 and v2 and v1 == v2:
                issues.append(f"duplicate_violations:{viol_keys[i]}={viol_keys[j]}")

    return issues


def check_metadata_consistency(bs: dict) -> List[str]:
    """Check metadata flags against actual annotation content."""
    issues = []
    meta = bs.get("metadata", {})
    vp = bs.get("visual_percepts", {})

    # Check asymmetry flag matches actual salience patterns
    objects = vp.get("scene_objects", [])
    actual_asymmetry = any(
        o.get("salience_speaker") != o.get("salience_partner")
        for o in objects
    )
    flagged_asymmetry = meta.get("visual_asymmetry_present", False)

    if actual_asymmetry and not flagged_asymmetry:
        issues.append("asymmetry_flag_false_but_present")
    if not actual_asymmetry and flagged_asymmetry:
        issues.append("asymmetry_flag_true_but_absent")

    return issues


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> List[str]:
    """Run all validation checks on a single annotated example."""
    bs = example.get("belief_state", {})
    if not bs:
        return ["missing_belief_state"]

    all_issues = []
    all_issues.extend(check_visual_percepts(bs))
    all_issues.extend(check_first_order_beliefs(bs))
    all_issues.extend(check_second_order_beliefs(bs))
    all_issues.extend(check_belief_probes(bs))
    all_issues.extend(check_contrastive_responses(bs))
    all_issues.extend(check_metadata_consistency(bs))
    return all_issues


def main():
    parser = argparse.ArgumentParser(description="Step 4: Validate belief state annotations")
    parser.add_argument("--input_path", type=str,
                        default="projects/mmrole/mmrole_annotated.jsonl")
    parser.add_argument("--output_clean", type=str,
                        default="projects/mmrole/mmrole_annotated_clean.jsonl")
    parser.add_argument("--output_flagged", type=str,
                        default="projects/mmrole/mmrole_annotated_flagged.jsonl")
    parser.add_argument("--max_issues", type=int, default=2,
                        help="Max issues before flagging (2 = more lenient for richer schema)")
    parser.add_argument("--print_examples", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading from {args.input_path} ...")
    examples = []
    with open(args.input_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Total: {len(examples)}")

    clean, flagged = [], []
    issue_counter = Counter()
    char_issues = defaultdict(Counter)

    for ex in examples:
        issues = validate_example(ex)
        ex["_validation_issues"] = issues

        if len(issues) <= args.max_issues:
            clean.append(ex)
        else:
            flagged.append(ex)

        for issue in issues:
            issue_counter[issue] += 1
            speaker = ex.get("agents", {}).get("speaker", {}).get("name", "?")
            char_issues[speaker][issue] += 1

    with open(args.output_clean, "w") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(args.output_flagged, "w") as f:
        for ex in flagged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'='*60}")
    print(f"Validation Results")
    print(f"{'='*60}")
    print(f"  Clean:   {len(clean)} ({100*len(clean)/max(len(examples),1):.1f}%)")
    print(f"  Flagged: {len(flagged)} ({100*len(flagged)/max(len(examples),1):.1f}%)")

    print(f"\nIssue breakdown:")
    for issue, count in issue_counter.most_common(25):
        print(f"  {issue:45s}  {count:5d}  ({100*count/len(examples):.1f}%)")

    print(f"\nTop problematic characters:")
    char_totals = {c: sum(v.values()) for c, v in char_issues.items()}
    for char, total in sorted(char_totals.items(), key=lambda x: -x[1])[:10]:
        print(f"  {char:30s}  {total} issues")

    if flagged and args.print_examples > 0:
        print(f"\nSample flagged ({args.print_examples}):")
        for i, ex in enumerate(flagged[:args.print_examples]):
            print(f"\n  --- #{i+1} {ex.get('example_id','?')} ---")
            print(f"  Issues: {ex['_validation_issues']}")
            bs = ex.get("belief_state", {})
            vp = bs.get("visual_percepts", {})
            n_obj = len(vp.get("scene_objects", []))
            meta = bs.get("metadata", {})
            print(f"  Objects: {n_obj}, Asymmetry: {meta.get('visual_asymmetry_present')}, "
                  f"Divergence: {meta.get('belief_divergence')}, "
                  f"ToM relevance: {meta.get('tom_relevance')}")

    print(f"\nOutputs: {args.output_clean}, {args.output_flagged}")


if __name__ == "__main__":
    main()
