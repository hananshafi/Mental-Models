#!/usr/bin/env python3
"""
Step 6: Theory of Mind Evaluation
====================================
Evaluates model responses on 3 ToM dimensions that complement
MMRole's existing 8 role-playing quality dimensions:

  1. Belief Accuracy (LLM-judge)
     Does the response reflect correct 1st/2nd order belief modeling?
     Uses annotated belief states as reference.

  2. Visual Perspective Taking (LLM-judge)
     Does the response attend to objects salient to the speaker's
     perspective, rather than generic scene description?

  3. ToM Probe Accuracy (automatic)
     Given belief probes as QA, does the model answer correctly?
     Binary accuracy — no LLM judge needed.

Supports two modes:
  A. evaluate_response  — score a model's generated responses
  B. evaluate_probes    — direct QA evaluation on belief probes

Usage:
    # Score generated responses on belief accuracy + visual perspective
    python step6_evaluate_tom.py response \
        --responses_path <model_responses.jsonl> \
        --annotations_path <official_test_annotated.jsonl>

    # Direct probe QA evaluation
    python step6_evaluate_tom.py probe \
        --model_name <your_model> \
        --annotations_path <official_test_annotated.jsonl>
"""

import os
import sys
import json
import time
import base64
import argparse
import threading
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# ---------------------------------------------------------------------------
# Dim 1: Belief Accuracy  (LLM-as-judge)
# ---------------------------------------------------------------------------

BELIEF_ACCURACY_PROMPT = """You are evaluating whether a character's response demonstrates accurate Theory of Mind — correct modeling of what the other character believes, perceives, and intends.

## Context
- **Speaker**: {speaker_name}
- **Partner**: {partner_name}
- **Image**: (attached)
- **Dialogue history**: {history}

## Ground-truth belief state (from expert annotation)
**Speaker believes about partner:**
- Partner's visual focus: {partner_visual_focus}
- Partner's intent: {partner_intent}
- Partner's knowledge: {partner_knowledge}
- Partner's emotion: {partner_emotion}

**Speaker's 2nd-order belief (what speaker thinks partner thinks about speaker):**
- Partner thinks I see: {partner_thinks_i_see}
- Partner thinks I want: {partner_thinks_i_want}
- Partner thinks I know: {partner_thinks_i_know}

## Response to evaluate
{response}

## Task
Score the response on Belief Accuracy (1-10):
- 1-3: Response ignores or contradicts the partner's actual beliefs/perspective
- 4-5: Response is generic, doesn't demonstrate active belief modeling
- 6-7: Response shows awareness of partner's perspective but misses nuances
- 8-9: Response accurately reflects 1st-order beliefs about partner
- 10: Response demonstrates both accurate 1st and 2nd order belief modeling

Output EXACTLY this format:
{{Qualitative Evaluation}}, [Score]: ({{score}})"""


# ---------------------------------------------------------------------------
# Dim 2: Visual Perspective Taking  (LLM-as-judge)
# ---------------------------------------------------------------------------

VISUAL_PERSPECTIVE_PROMPT = """You are evaluating whether a character's response demonstrates perspective-appropriate visual attention — attending to scene elements that are salient from their specific perspective, not just describing the image generically.

## Context
- **Speaker**: {speaker_name} — {speaker_profile_short}
- **Partner**: {partner_name}
- **Image**: (attached)

## Ground-truth visual salience (from expert annotation)
Objects and per-character salience:
{salience_table}

Speaker's visual attention: {speaker_attention}
Partner's visual attention: {partner_attention}

## Response to evaluate
{response}

## Task
Score the response on Visual Perspective Taking (1-10):
- 1-3: Response describes scene generically or mentions objects irrelevant to the speaker's perspective
- 4-5: Response references the image but without character-specific visual framing
- 6-7: Response shows some perspective-appropriate attention to salient objects
- 8-9: Response clearly prioritizes objects that would be salient to this specific character
- 10: Response demonstrates asymmetric visual awareness — focuses on speaker-salient objects while implicitly acknowledging partner may notice different things

Output EXACTLY this format:
{{Qualitative Evaluation}}, [Score]: ({{score}})"""


# ---------------------------------------------------------------------------
# Dim 3: ToM Probe QA  (automatic)
# ---------------------------------------------------------------------------

PROBE_QA_PROMPT = """You are a character in a multimodal conversation. Answer the following question about the mental states, beliefs, and visual percepts of the characters in this scene.

## Context
- **Characters**: {speaker_name} and {partner_name}
- **Image**: (attached)
- **Dialogue so far**: {history}
- **Current utterance by {speaker_name}**: {current_utterance}

## Question
{question}

Answer in 1-2 sentences. Be specific and concise."""


# ---------------------------------------------------------------------------
# MMRole Original 8 Dimensions  (LLM-as-judge, single-model scoring)
# ---------------------------------------------------------------------------

MMROLE_DIMENSIONS = {
    "coherency": "Coherency: Does the response maintain a coherent thread of dialogue without contradicting earlier parts of the conversation or previously established facts?",
    "fluency": "Fluency: Is the response grammatically correct and smoothly articulated?",
    "image_text_relevance": "Image-Text Relevance: Is the response closely related to the visual content of the image?",
    "instruction_adherence": "Instruction Adherence: Does the response accurately adhere to the task instruction, directly role-playing as {speaker_name} and only including words that {speaker_name} should say, without any additional explanatory prefixes or suffixes?",
    "knowledge_consistency": "Knowledge Consistency: Is the response consistent with the factual knowledge that {speaker_name} should possess, including experiences, abilities, and relationships?",
    "personality_consistency": "Personality Consistency: Does the response accurately and sufficiently reflect the personality of {speaker_name}?",
    "response_accuracy": "Response Accuracy: Does the response accurately answer the partner's words or appropriately initiate a conversation based on the image?",
    "tone_consistency": "Tone Consistency: Does the response maintain a consistent tone that aligns with {speaker_name}'s typical manner of speaking and catchphrases, rather than resembling the style of AI assistants?",
}

MMROLE_EVAL_PROMPT = """You are an objective and precise evaluator, specializing in assessing role-playing and multimodal understanding abilities.

## Context
- **Character being role-played**: {speaker_name}
- **Character profile**: {speaker_profile}
- **Talking with**: {partner_name}
- **Image**: (attached)
- **Dialogue history**: {history}

## Response to evaluate
{response}

## Task
{dimension_criteria}

Score from 1 to 10, where 1 = poor and 10 = excellent.

Output EXACTLY this format:
{{Qualitative Evaluation}}, [Score]: ({{score}})"""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_b64(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image(scene: dict, image_dir: str) -> Optional[str]:
    for key in ["image_local", "image"]:
        img = scene.get(key, "")
        if not img:
            continue
        full = os.path.join(image_dir, img)
        if os.path.exists(full):
            return full
        fname = os.path.basename(img)
        for sub in ["coco", "character", ""]:
            cand = os.path.join(image_dir, sub, fname)
            if os.path.exists(cand):
                return cand
    return None


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class ToMJudge:
    """Thread-safe LLM judge for ToM evaluation."""

    def __init__(self, api_key: str, model: str = "gpt-5.4-mini",
                 image_dir: str = "", rate_limit_delay: float = 0.2,
                 max_retries: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.image_dir = image_dir
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "errors": 0}

    def _call_llm(self, prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        content = []
        if image_path:
            b64 = encode_image_b64(image_path)
            if b64:
                ext = os.path.splitext(image_path)[1].lower()
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png"}.get(ext, "image/jpeg")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                })
        content.append({"type": "text", "text": prompt})

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=500,
                    temperature=0.1,
                )
                with self._lock:
                    self.stats["calls"] += 1
                return resp.choices[0].message.content
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower():
                    time.sleep(min(60, 2 ** (attempt + 2)))
                else:
                    with self._lock:
                        self.stats["errors"] += 1
                    if attempt == self.max_retries - 1:
                        print(f"  Judge error: {err[:100]}")
                        return None
                    time.sleep(2 ** attempt)
        return None

    def _parse_score(self, text: str) -> Optional[float]:
        """Extract score from '[Score]: (N)' format."""
        if not text:
            return None
        import re
        # Try [Score]: (N) or [Scores]: (N) or (N)
        m = re.search(r'\[Scores?\]:\s*\((\d+(?:\.\d+)?)\)', text)
        if m:
            return float(m.group(1))
        # Fallback: last number in parentheses
        matches = re.findall(r'\((\d+(?:\.\d+)?)\)', text)
        if matches:
            return float(matches[-1])
        return None

    # ── Dim 1: Belief Accuracy ──

    def score_belief_accuracy(self, annotation: dict, response: str) -> dict:
        bs = annotation["belief_state"]
        agents = annotation["agents"]
        ctx = annotation["interaction_context"]

        fob = bs["first_order_beliefs"]["speaker_believes_about_partner"]
        sob = bs["second_order_beliefs"]["speaker_thinks_partner_thinks_speaker"]

        history_text = "\n".join(
            f"[{t['speaker']}]: {t['utterance']}"
            for t in ctx.get("dialogue_history", [])
        ) or "(start of conversation)"

        prompt = BELIEF_ACCURACY_PROMPT.format(
            speaker_name=agents["speaker"]["name"],
            partner_name=agents["partner"]["name"],
            history=history_text,
            partner_visual_focus=fob["perceived_visual_focus"],
            partner_intent=fob["perceived_intent"],
            partner_knowledge=fob["perceived_knowledge"],
            partner_emotion=fob["perceived_emotion"],
            partner_thinks_i_see=sob["sees"],
            partner_thinks_i_want=sob["wants"],
            partner_thinks_i_know=sob["knows"],
            response=response,
        )

        image_path = resolve_image(annotation.get("scene", {}), self.image_dir)
        result_text = self._call_llm(prompt, image_path)
        score = self._parse_score(result_text)

        return {
            "dimension": "belief_accuracy",
            "score": score,
            "rationale": result_text,
        }

    # ── Dim 2: Visual Perspective Taking ──

    def score_visual_perspective(self, annotation: dict, response: str) -> dict:
        bs = annotation["belief_state"]
        agents = annotation["agents"]
        vp = bs["visual_percepts"]

        # Build salience table
        rows = []
        for obj in vp.get("scene_objects", []):
            rows.append(
                f"- {obj['object']}: speaker={obj.get('salience_speaker','?')}, "
                f"partner={obj.get('salience_partner','?')}"
            )
        salience_table = "\n".join(rows) or "(no objects annotated)"

        prompt = VISUAL_PERSPECTIVE_PROMPT.format(
            speaker_name=agents["speaker"]["name"],
            speaker_profile_short=agents["speaker"].get("profile", "")[:300],
            partner_name=agents["partner"]["name"],
            salience_table=salience_table,
            speaker_attention=vp.get("speaker_attention", "N/A"),
            partner_attention=vp.get("partner_attention", "N/A"),
            response=response,
        )

        image_path = resolve_image(annotation.get("scene", {}), self.image_dir)
        result_text = self._call_llm(prompt, image_path)
        score = self._parse_score(result_text)

        return {
            "dimension": "visual_perspective_taking",
            "score": score,
            "rationale": result_text,
        }

    # ── Dim 3: Probe QA (uses LLM to generate answer, then auto-compares) ──

    def answer_probe(self, annotation: dict, probe: dict) -> dict:
        agents = annotation["agents"]
        ctx = annotation["interaction_context"]

        history_text = "\n".join(
            f"[{t['speaker']}]: {t['utterance']}"
            for t in ctx.get("dialogue_history", [])
        ) or "(start of conversation)"

        prompt = PROBE_QA_PROMPT.format(
            speaker_name=agents["speaker"]["name"],
            partner_name=agents["partner"]["name"],
            history=history_text,
            current_utterance=ctx.get("current_utterance", ""),
            question=probe["question"],
        )

        image_path = resolve_image(annotation.get("scene", {}), self.image_dir)
        model_answer = self._call_llm(prompt, image_path)

        return {
            "question": probe["question"],
            "probe_type": probe.get("probe_type", ""),
            "difficulty": probe.get("difficulty", ""),
            "ground_truth": probe["answer"],
            "wrong_answer": probe["wrong_answer"],
            "model_answer": model_answer,
        }

    # ── MMRole 8 Dimensions ──

    def score_mmrole_dimension(self, annotation: dict, response: str,
                                dim_key: str) -> dict:
        agents = annotation["agents"]
        ctx = annotation["interaction_context"]
        speaker_name = agents["speaker"]["name"]

        criteria = MMROLE_DIMENSIONS[dim_key].format(speaker_name=speaker_name)

        history_text = "\n".join(
            f"[{t['speaker']}]: {t['utterance']}"
            for t in ctx.get("dialogue_history", [])
        ) or "(start of conversation)"

        prompt = MMROLE_EVAL_PROMPT.format(
            speaker_name=speaker_name,
            speaker_profile=agents["speaker"].get("profile", "")[:1000],
            partner_name=agents["partner"]["name"],
            history=history_text,
            response=response,
            dimension_criteria=criteria,
        )

        image_path = resolve_image(annotation.get("scene", {}), self.image_dir)
        result_text = self._call_llm(prompt, image_path)
        score = self._parse_score(result_text)

        return {
            "dimension": dim_key,
            "score": score,
            "rationale": result_text,
        }

    def score_all_mmrole_dimensions(self, annotation: dict, response: str) -> dict:
        results = {}
        for dim_key in MMROLE_DIMENSIONS:
            results[dim_key] = self.score_mmrole_dimension(annotation, response, dim_key)
        return results


# ---------------------------------------------------------------------------
# Probe accuracy scoring (automatic, no judge needed)
# ---------------------------------------------------------------------------

PROBE_MATCH_PROMPT = """Given a question and two candidate answers, determine which one the model's response is closer to.

Question: {question}
Answer A (correct): {correct}
Answer B (wrong): {wrong}
Model's response: {model_answer}

Which answer is the model's response closer to? Reply with EXACTLY "A" or "B"."""


def score_probe_match(judge: ToMJudge, probe_result: dict) -> str:
    """Use LLM to determine if model answer matches correct or wrong answer."""
    prompt = PROBE_MATCH_PROMPT.format(
        question=probe_result["question"],
        correct=probe_result["ground_truth"],
        wrong=probe_result["wrong_answer"],
        model_answer=probe_result["model_answer"],
    )
    result = judge._call_llm(prompt)
    if result and "A" in result.strip()[:5]:
        return "correct"
    elif result and "B" in result.strip()[:5]:
        return "incorrect"
    return "unclear"


# ---------------------------------------------------------------------------
# Mode A: Evaluate generated responses
# ---------------------------------------------------------------------------

def evaluate_responses(args):
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY or pass --api_key")
        sys.exit(1)

    judge = ToMJudge(
        api_key=api_key, model=args.judge_model,
        image_dir=args.image_dir, rate_limit_delay=args.rate_limit_delay,
    )

    # Load annotations (keyed by example_id or dialogue_id)
    annotations = {}
    with open(args.annotations_path) as f:
        for line in f:
            d = json.loads(line)
            annotations[d["example_id"]] = d
            # Also key by original_id if present
            tm = d.get("test_metadata", {})
            if tm.get("original_id"):
                annotations[tm["original_id"]] = d
    print(f"Loaded {len(annotations)} annotations")

    # Load model responses
    responses = []
    with open(args.responses_path) as f:
        for line in f:
            responses.append(json.loads(line))
    print(f"Loaded {len(responses)} responses")

    results = []
    write_lock = threading.Lock()
    progress = {"done": 0}

    run_tom = "tom" in args.eval_dims or "all" in args.eval_dims
    run_mmrole = "mmrole" in args.eval_dims or "all" in args.eval_dims
    dim_label = []
    if run_tom:
        dim_label.append("3 ToM dims")
    if run_mmrole:
        dim_label.append("8 MMRole dims")
    print(f"Evaluating: {' + '.join(dim_label)}")

    def evaluate_one(resp):
        # Match to annotation
        rid = resp.get("example_id", resp.get("id", resp.get("dialogue_id", "")))
        ann = annotations.get(rid)
        if not ann:
            return None

        response_text = resp.get("response", resp.get("generated", resp.get("output", "")))
        if not response_text:
            return None

        result = {
            "example_id": rid,
            "speaker": ann["agents"]["speaker"]["name"],
            "partner": ann["agents"]["partner"]["name"],
            "dialogue_type": ann.get("test_metadata", {}).get("dialogue_type", ""),
            "distribution": ann.get("test_metadata", {}).get("distribution", ""),
        }

        # ToM dimensions
        if run_tom:
            result["belief_accuracy"] = judge.score_belief_accuracy(ann, response_text)
            result["visual_perspective"] = judge.score_visual_perspective(ann, response_text)

        # MMRole 8 dimensions
        if run_mmrole:
            mmrole_scores = judge.score_all_mmrole_dimensions(ann, response_text)
            result.update(mmrole_scores)

        with write_lock:
            progress["done"] += 1
            if progress["done"] % 10 == 0:
                print(f"  Progress: {progress['done']}/{len(responses)} | {judge.stats}")

        return result

    print(f"\nEvaluating with judge={args.judge_model}, workers={args.max_workers} ...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(evaluate_one, r): r for r in responses}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"  Error: {e}")

    # Save results
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    print_response_summary(results)


# ---------------------------------------------------------------------------
# Mode B: Evaluate probes (direct QA)
# ---------------------------------------------------------------------------

def evaluate_probes(args):
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY or pass --api_key")
        sys.exit(1)

    judge = ToMJudge(
        api_key=api_key, model=args.model_name,
        image_dir=args.image_dir, rate_limit_delay=args.rate_limit_delay,
    )

    # Separate judge for scoring (can be cheaper)
    scorer = ToMJudge(
        api_key=api_key, model=args.judge_model,
        image_dir=args.image_dir, rate_limit_delay=args.rate_limit_delay,
    )

    # Load annotations
    annotations = []
    with open(args.annotations_path) as f:
        for line in f:
            annotations.append(json.loads(line))
    print(f"Loaded {len(annotations)} annotations")

    # Flatten probes
    probe_tasks = []
    for ann in annotations:
        bs = ann.get("belief_state", {})
        for probe in bs.get("belief_probes", []):
            probe_tasks.append((ann, probe))

    if args.max_probes > 0:
        probe_tasks = probe_tasks[:args.max_probes]
    print(f"Total probes to evaluate: {len(probe_tasks)}")

    results = []
    write_lock = threading.Lock()
    progress = {"done": 0}

    def evaluate_one_probe(task):
        ann, probe = task

        # Get model's answer
        probe_result = judge.answer_probe(ann, probe)

        # Score match
        match = score_probe_match(scorer, probe_result)
        probe_result["match"] = match
        probe_result["example_id"] = ann["example_id"]

        tm = ann.get("test_metadata", {})
        probe_result["dialogue_type"] = tm.get("dialogue_type", "")
        probe_result["distribution"] = tm.get("distribution", "")

        with write_lock:
            progress["done"] += 1
            if progress["done"] % 50 == 0:
                print(f"  Progress: {progress['done']}/{len(probe_tasks)} | "
                      f"judge={judge.stats} scorer={scorer.stats}")

        return probe_result

    print(f"\nEvaluating probes with model={args.model_name}, "
          f"judge={args.judge_model}, workers={args.max_workers} ...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(evaluate_one_probe, t): t for t in probe_tasks}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"  Error: {e}")

    # Save
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_probe_summary(results)


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def print_response_summary(results: list):
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({len(results)} examples)")
    print(f"{'='*60}")

    # Collect all dimension keys present in results
    all_dims = []
    # ToM dims first
    for dim in ["belief_accuracy", "visual_perspective"]:
        if any(dim in r for r in results):
            all_dims.append(dim)
    # MMRole dims
    for dim in MMROLE_DIMENSIONS:
        if any(dim in r for r in results):
            all_dims.append(dim)

    if not all_dims:
        print("  No scored dimensions found.")
        return

    # ── Per-dimension summary ──
    for dim in all_dims:
        scores = []
        for r in results:
            if dim in r and isinstance(r[dim], dict) and r[dim].get("score") is not None:
                scores.append(r[dim]["score"])
        if not scores:
            continue

        avg = sum(scores) / len(scores)
        print(f"\n  {dim}:")
        print(f"    Mean: {avg:.2f} / 10  ({len(scores)} scored)")

        # By distribution
        for dist in ["in-distribution", "out-of-distribution"]:
            ds = [r[dim]["score"] for r in results
                  if dim in r and isinstance(r[dim], dict)
                  and r[dim].get("score") is not None
                  and r.get("distribution") == dist]
            if ds:
                print(f"    {dist}: {sum(ds)/len(ds):.2f} ({len(ds)})")

        # By dialogue type
        for dtype in ["inter-role", "human-role", "comment"]:
            ds = [r[dim]["score"] for r in results
                  if dim in r and isinstance(r[dim], dict)
                  and r[dim].get("score") is not None
                  and r.get("dialogue_type") == dtype]
            if ds:
                print(f"    {dtype}: {sum(ds)/len(ds):.2f} ({len(ds)})")

    # ── Aggregate table ──
    tom_dims = [d for d in all_dims if d in ("belief_accuracy", "visual_perspective")]
    mmrole_dims = [d for d in all_dims if d in MMROLE_DIMENSIONS]

    if tom_dims:
        tom_scores = []
        for r in results:
            s = [r[d]["score"] for d in tom_dims
                 if d in r and isinstance(r[d], dict) and r[d].get("score") is not None]
            if s:
                tom_scores.append(sum(s) / len(s))
        if tom_scores:
            print(f"\n  ToM Average (2 dims): {sum(tom_scores)/len(tom_scores):.2f}")

    if mmrole_dims:
        mmrole_scores = []
        for r in results:
            s = [r[d]["score"] for d in mmrole_dims
                 if d in r and isinstance(r[d], dict) and r[d].get("score") is not None]
            if s:
                mmrole_scores.append(sum(s) / len(s))
        if mmrole_scores:
            print(f"  MMRole Average (8 dims): {sum(mmrole_scores)/len(mmrole_scores):.2f}")

    if tom_dims and mmrole_dims:
        all_scores = []
        for r in results:
            s = [r[d]["score"] for d in all_dims
                 if d in r and isinstance(r[d], dict) and r[d].get("score") is not None]
            if s:
                all_scores.append(sum(s) / len(s))
        if all_scores:
            print(f"  Overall Average (11 dims): {sum(all_scores)/len(all_scores):.2f}")


def print_probe_summary(results: list):
    print(f"\n{'='*60}")
    print(f"ToM Probe QA Summary ({len(results)} probes)")
    print(f"{'='*60}")

    total = len(results)
    correct = sum(1 for r in results if r["match"] == "correct")
    incorrect = sum(1 for r in results if r["match"] == "incorrect")
    unclear = sum(1 for r in results if r["match"] == "unclear")

    print(f"\n  Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Incorrect: {incorrect}  Unclear: {unclear}")

    # By probe type
    print(f"\n  By probe type:")
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        pt = r.get("probe_type", "unknown")
        by_type[pt]["total"] += 1
        if r["match"] == "correct":
            by_type[pt]["correct"] += 1
    for pt, counts in sorted(by_type.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {pt}: {counts['correct']}/{counts['total']} = {acc:.1f}%")

    # By difficulty
    print(f"\n  By difficulty:")
    by_diff = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        diff = r.get("difficulty", "unknown")
        by_diff[diff]["total"] += 1
        if r["match"] == "correct":
            by_diff[diff]["correct"] += 1
    for diff, counts in sorted(by_diff.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {diff}: {counts['correct']}/{counts['total']} = {acc:.1f}%")

    # By distribution
    print(f"\n  By distribution:")
    by_dist = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        dist = r.get("distribution", "unknown")
        by_dist[dist]["total"] += 1
        if r["match"] == "correct":
            by_dist[dist]["correct"] += 1
    for dist, counts in sorted(by_dist.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {dist}: {counts['correct']}/{counts['total']} = {acc:.1f}%")

    # By dialogue type
    print(f"\n  By dialogue type:")
    by_dtype = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        dtype = r.get("dialogue_type", "unknown")
        by_dtype[dtype]["total"] += 1
        if r["match"] == "correct":
            by_dtype[dtype]["correct"] += 1
    for dtype, counts in sorted(by_dtype.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] else 0
        print(f"    {dtype}: {counts['correct']}/{counts['total']} = {acc:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 6: ToM Evaluation")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── Mode A: Evaluate responses ──
    resp_parser = subparsers.add_parser("response",
        help="Evaluate model-generated responses on belief accuracy + visual perspective")
    resp_parser.add_argument("--responses_path", type=str, required=True,
        help="JSONL with model responses (needs example_id/id + response/generated/output)")
    resp_parser.add_argument("--annotations_path", type=str,
        default="projects/mmrole/training_data/official_test/raw_annotated.jsonl")
    resp_parser.add_argument("--output_path", type=str,
        default="projects/mmrole/eval_tom_response.jsonl")
    resp_parser.add_argument("--image_dir", type=str,
        default="projects/mmrole/images")
    resp_parser.add_argument("--judge_model", type=str, default="gpt-4o")
    resp_parser.add_argument("--eval_dims", type=str, nargs="+", default=["all"],
        choices=["all", "tom", "mmrole"],
        help="Which dimensions to evaluate: 'tom' (3 ToM dims), 'mmrole' (8 original dims), 'all' (both)")
    resp_parser.add_argument("--max_workers", type=int, default=5)
    resp_parser.add_argument("--rate_limit_delay", type=float, default=0.2)
    resp_parser.add_argument("--api_key", type=str, default=None)

    # ── Mode B: Evaluate probes ──
    probe_parser = subparsers.add_parser("probe",
        help="Direct probe QA evaluation on belief probes")
    probe_parser.add_argument("--model_name", type=str, required=True,
        help="Model to test (e.g. gpt-5.4-mini, or local model endpoint)")
    probe_parser.add_argument("--annotations_path", type=str,
        default="projects/mmrole/training_data/official_test/raw_annotated.jsonl")
    probe_parser.add_argument("--output_path", type=str,
        default="projects/mmrole/eval_tom_probes.jsonl")
    probe_parser.add_argument("--image_dir", type=str,
        default="projects/mmrole/images")
    probe_parser.add_argument("--judge_model", type=str, default="gpt-4o",
        help="Model to score probe answer correctness")
    probe_parser.add_argument("--max_workers", type=int, default=5)
    probe_parser.add_argument("--max_probes", type=int, default=-1,
        help="Limit number of probes (-1 = all)")
    probe_parser.add_argument("--rate_limit_delay", type=float, default=0.2)
    probe_parser.add_argument("--api_key", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "response":
        evaluate_responses(args)
    elif args.mode == "probe":
        evaluate_probes(args)


if __name__ == "__main__":
    main()
