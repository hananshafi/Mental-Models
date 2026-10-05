#!/usr/bin/env python3
"""
Step 3: Annotate Visual Theory of Mind — Structured Belief States
==================================================================
Annotates each per-turn example with STRUCTURED BELIEF STATES,
drawing from best practices across ToM benchmarks:

  - nested belief dictionaries for higher-order ToM
  - MMToM-QA: symbolic predicates grounded in visual scenes
  - MuMA-ToM: multi-agent belief/goal inference over video
  - Dynamic Belief Graphs: structured binary belief vectors
  - GridToM: perspective-masked visual inputs

KEY DESIGN: Belief-centric, not dialogue-centric.
  Each annotation captures a BELIEF STATE SNAPSHOT at time t:
    1. Visual percepts: what objects/elements each agent attends to
    2. 1st-order beliefs: what A believes about B's percepts & intent
    3. 2nd-order beliefs: what A believes B believes about A
    4. Belief probes: structured QA pairs to test the belief state
    5. Contrastive belief states: alternative states for preference learning

Resume-safe. Uses OpenAI structured output.

Usage:
    # Pilot (500 turns)
    OPENAI_API_KEY="sk-..." python step3_annotate_visual_tom.py \
        --input_path projects/mmrole/mmrole_per_turn.jsonl \
        --output_path projects/mmrole/mmrole_annotated_pilot.jsonl \
        --image_dir projects/mmrole/images \
        --max_examples 500 --model gpt-5.4-mini

    # Full run (~$60-80 with gpt-5.4-mini)
    OPENAI_API_KEY="sk-..." python step3_annotate_visual_tom.py \
        --input_path projects/mmrole/mmrole_per_turn.jsonl \
        --output_path projects/mmrole/mmrole_annotated.jsonl \
        --image_dir projects/mmrole/images \
        --max_workers 8 --model gpt-5.4-mini
"""

import os
import sys
import json
import base64
import time
import argparse
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# ---------------------------------------------------------------------------
# Annotation schema — structured belief states
# ---------------------------------------------------------------------------
# Uses nested beliefs, MMToM-QA symbolic predicates, and MuMA-ToM QA probes

ANNOTATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "visual_tom_belief_state",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {

                # ── Visual Percepts (what each agent sees) ──────────────
                # Inspired by MMToM-QA scene graphs + GridToM perspective views
                "visual_percepts": {
                    "type": "object",
                    "properties": {
                        "scene_objects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "object": {"type": "string", "description": "Object or element name visible in the image"},
                                    "description": {"type": "string", "description": "Brief visual description (color, position, state)"},
                                    "salience_speaker": {"type": "string", "enum": ["high", "medium", "low", "none"], "description": "How salient this object is to the speaker given their character"},
                                    "salience_partner": {"type": "string", "enum": ["high", "medium", "low", "none"], "description": "How salient this object is to the partner given their character"},
                                    "salience_reason_speaker": {"type": "string", "description": "Why this salience level for speaker (must reference character expertise/personality)"},
                                    "salience_reason_partner": {"type": "string", "description": "Why this salience level for partner (must reference character expertise/personality)"}
                                },
                                "required": ["object", "description", "salience_speaker", "salience_partner", "salience_reason_speaker", "salience_reason_partner"],
                                "additionalProperties": False
                            },
                            "description": "Key objects/elements visible in the image with per-agent salience"
                        },
                        "speaker_attention": {
                            "type": "string",
                            "description": "What the speaker is primarily attending to in this turn and why (grounded in their character)"
                        },
                        "partner_attention": {
                            "type": "string",
                            "description": "What the partner is likely attending to and why (grounded in their character)"
                        }
                    },
                    "required": ["scene_objects", "speaker_attention", "partner_attention"],
                    "additionalProperties": False
                },

                # ── 1st-Order Beliefs (A's model of B) ─────────────────
                # First-order nesting: belief[agent][target]
                "first_order_beliefs": {
                    "type": "object",
                    "properties": {
                        "speaker_believes_about_partner": {
                            "type": "object",
                            "properties": {
                                "perceived_visual_focus": {"type": "string", "description": "What speaker believes partner is looking at / noticing in the image"},
                                "perceived_intent": {"type": "string", "description": "What speaker believes partner's conversational goal is"},
                                "perceived_knowledge": {"type": "string", "description": "What speaker believes partner knows (from dialogue + image)"},
                                "perceived_emotion": {"type": "string", "description": "What speaker believes partner is feeling"}
                            },
                            "required": ["perceived_visual_focus", "perceived_intent", "perceived_knowledge", "perceived_emotion"],
                            "additionalProperties": False
                        },
                        "partner_believes_about_speaker": {
                            "type": "object",
                            "properties": {
                                "perceived_visual_focus": {"type": "string", "description": "What partner believes speaker is looking at / noticing"},
                                "perceived_intent": {"type": "string", "description": "What partner believes speaker's conversational goal is"},
                                "perceived_knowledge": {"type": "string", "description": "What partner believes speaker knows"},
                                "perceived_emotion": {"type": "string", "description": "What partner believes speaker is feeling"}
                            },
                            "required": ["perceived_visual_focus", "perceived_intent", "perceived_knowledge", "perceived_emotion"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["speaker_believes_about_partner", "partner_believes_about_speaker"],
                    "additionalProperties": False
                },

                # ── 2nd-Order Beliefs (A's model of B's model of A) ────
                # Second-order nesting: belief[A][B][A]
                "second_order_beliefs": {
                    "type": "object",
                    "properties": {
                        "speaker_thinks_partner_thinks_speaker": {
                            "type": "object",
                            "properties": {
                                "sees": {"type": "string", "description": "Speaker thinks: partner thinks I'm looking at..."},
                                "wants": {"type": "string", "description": "Speaker thinks: partner thinks my goal is..."},
                                "knows": {"type": "string", "description": "Speaker thinks: partner thinks I know..."}
                            },
                            "required": ["sees", "wants", "knows"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["speaker_thinks_partner_thinks_speaker"],
                    "additionalProperties": False
                },

                # ── Belief Probes (structured QA) ──────────────────────
                # Inspired by MuMA-ToM QA format + FANToM answerability
                "belief_probes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "A specific ToM question about this turn's belief state"},
                            "answer": {"type": "string", "description": "The correct answer"},
                            "wrong_answer": {"type": "string", "description": "A plausible but incorrect answer (for contrastive eval)"},
                            "probe_type": {
                                "type": "string",
                                "enum": ["visual_percept", "first_order_belief", "second_order_belief", "false_belief", "intent_inference"],
                                "description": "What aspect of ToM this probe tests"
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["easy", "medium", "hard"],
                                "description": "easy=observable from image, medium=requires 1st order inference, hard=requires 2nd order"
                            }
                        },
                        "required": ["question", "answer", "wrong_answer", "probe_type", "difficulty"],
                        "additionalProperties": False
                    },
                    "description": "5 probes: 1 visual_percept, 1-2 first_order, 1 second_order, 1 false_belief or intent"
                },

                # ── Contrastive Responses ──────────────────────────────
                # For preference learning (Bradley-Terry loss)
                "contrastive_responses": {
                    "type": "object",
                    "properties": {
                        "tom_aligned": {
                            "type": "string",
                            "description": "A response that demonstrates accurate ToM — correctly accounts for partner's visual perspective and beliefs"
                        },
                        "tom_violation_visual": {
                            "type": "string",
                            "description": "Assumes partner sees/attends to the same things as speaker (ignores visual perspective asymmetry)"
                        },
                        "tom_violation_belief": {
                            "type": "string",
                            "description": "Misattributes beliefs to partner — assumes partner knows/wants something they don't"
                        },
                        "tom_violation_order2": {
                            "type": "string",
                            "description": "Speaker acts on wrong 2nd-order belief — e.g., tries to impress partner with X because they think partner values X, but partner actually doesn't"
                        },
                        "no_tom_baseline": {
                            "type": "string",
                            "description": "Generic in-character response that ignores the image and makes no attempt to model partner"
                        }
                    },
                    "required": ["tom_aligned", "tom_violation_visual", "tom_violation_belief", "tom_violation_order2", "no_tom_baseline"],
                    "additionalProperties": False
                },

                # ── Belief State Metadata ──────────────────────────────
                "metadata": {
                    "type": "object",
                    "properties": {
                        "visual_asymmetry_present": {"type": "boolean", "description": "Do the two characters notice meaningfully different things?"},
                        "belief_divergence": {
                            "type": "string",
                            "enum": ["none", "low", "moderate", "high"],
                            "description": "How much do the agents' belief states diverge?"
                        },
                        "tom_relevance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "How much does accurate ToM matter for this turn? High = response quality depends on modeling partner"
                        }
                    },
                    "required": ["visual_asymmetry_present", "belief_divergence", "tom_relevance"],
                    "additionalProperties": False
                }
            },
            "required": [
                "visual_percepts", "first_order_beliefs", "second_order_beliefs",
                "belief_probes", "contrastive_responses", "metadata"
            ],
            "additionalProperties": False
        }
    }
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_annotation_prompt(example: Dict[str, Any]) -> str:
    """Build the annotation prompt for structured belief state extraction."""

    speaker = example["agents"]["speaker"]
    partner = example["agents"]["partner"]
    ctx = example["interaction_context"]

    history_str = ""
    if ctx["dialogue_history"]:
        for turn in ctx["dialogue_history"]:
            history_str += f"  [{turn['speaker']}]: {turn['utterance']}\n"
    else:
        history_str = "  (First turn — no prior dialogue)\n"

    return f"""You are a Theory of Mind researcher annotating belief states in a multimodal multi-agent dialogue.

An image is attached. Two characters are looking at this image and conversing in character.

═══ SPEAKER ═══
Name: {speaker['name']}
Profile: {speaker['profile']}

═══ PARTNER ═══
Name: {partner['name']}
Profile: {partner['profile']}

═══ DIALOGUE SO FAR ═══
{history_str}
═══ CURRENT TURN ({speaker['name']} says) ═══
"{ctx['current_utterance']}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANNOTATE THE BELIEF STATE AT THIS MOMENT IN TIME.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. VISUAL PERCEPTS
   List 4-8 key objects/elements in the image. For EACH object, rate how salient it is
   to the speaker vs the partner (high/medium/low/none). Salience MUST be character-specific:
   a chef notices food, a soldier notices weapons, an artist notices composition.
   If both characters have identical salience for all objects, you are doing it wrong.

2. FIRST-ORDER BELIEFS (what A believes about B)
   For each agent, state what they believe the OTHER agent:
   - is looking at in the image (visual focus)
   - wants from the conversation (intent)
   - knows so far (accumulated knowledge)
   - is feeling (emotion)

3. SECOND-ORDER BELIEFS (what A thinks B thinks about A)
   State what the speaker believes the partner thinks the speaker:
   - sees in the image
   - wants from the conversation
   - knows

4. BELIEF PROBES — Generate exactly 5 QA pairs:
   - 1 visual_percept probe (easy): "What would [character] notice first?"
   - 1-2 first_order_belief probes (medium): "What does [A] think [B] is interested in?"
   - 1 second_order_belief probe (hard): "What does [A] think [B] believes [A] wants?"
   - 1 false_belief or intent_inference probe: tests whether a model can detect
     belief-reality mismatch or infer hidden intent
   Each probe needs a correct answer AND a plausible wrong answer.

5. CONTRASTIVE RESPONSES — Generate 5 alternative responses:
   - tom_aligned: demonstrates ACCURATE theory of mind about the partner
   - tom_violation_visual: wrongly assumes partner sees what speaker sees
   - tom_violation_belief: misattributes beliefs/knowledge to partner
   - tom_violation_order2: acts on wrong model of partner's model of self
   - no_tom_baseline: generic in-character response, no ToM reasoning

6. METADATA
   - visual_asymmetry_present: do the characters genuinely notice different things?
   - belief_divergence: how different are the agents' belief states?
   - tom_relevance: how much does accurate ToM affect response quality here?

CRITICAL RULES:
- Visual salience MUST differ between characters based on their backgrounds.
- Belief probes must have genuinely different correct vs wrong answers.
- Contrastive responses must be in-character and fluent — the ONLY difference is ToM accuracy.
- 2nd-order beliefs are the hardest — think carefully about recursive modeling."""


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def encode_image_base64(image_path: str) -> Optional[str]:
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image_path(image_local: str, image_dir: str) -> Optional[str]:
    full_path = os.path.join(image_dir, image_local)
    if os.path.exists(full_path):
        return full_path
    filename = os.path.basename(image_local)
    for subdir in ["coco", "character", ""]:
        candidate = os.path.join(image_dir, subdir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class BeliefStateAnnotator:
    """Thread-safe GPT-4o annotator for structured belief states."""

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 image_dir: str = "", rate_limit_delay: float = 0.1,
                 max_retries: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.image_dir = image_dir
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0
        self._lock = threading.Lock()

    def annotate(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Annotate a single example with structured belief state."""

        # Resolve image
        scene = example.get("scene", {})
        image_path = resolve_image_path(scene.get("image_local", ""), self.image_dir)
        if not image_path:
            image_path = resolve_image_path(scene.get("image", ""), self.image_dir)

        prompt_text = build_annotation_prompt(example)

        content_parts = []
        if image_path:
            img_b64 = encode_image_base64(image_path)
            if img_b64:
                ext = os.path.splitext(image_path)[1].lower()
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}", "detail": "low"}
                })
        content_parts.append({"type": "text", "text": prompt_text})

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content_parts}],
                    response_format=ANNOTATION_SCHEMA,
                    max_completion_tokens=4000,
                    temperature=0.3,
                )

                result = json.loads(response.choices[0].message.content)
                with self._lock:
                    self._call_count += 1
                return result

            except json.JSONDecodeError as e:
                print(f"  JSON parse error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower():
                    wait = min(60, 2 ** (attempt + 2))
                    print(f"  Rate limit hit, waiting {wait}s ...")
                    time.sleep(wait)
                elif "content_policy" in error_str.lower():
                    eid = example.get("example_id", "?")
                    print(f"  Content policy rejection: {eid}")
                    return None
                else:
                    print(f"  API error (attempt {attempt+1}): {error_str[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)

        with self._lock:
            self._error_count += 1
        return None

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"calls": self._call_count, "errors": self._error_count}


# ---------------------------------------------------------------------------
# Resume-safe pipeline
# ---------------------------------------------------------------------------

def load_completed_keys(output_path: str) -> Set[str]:
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    completed.add(d["example_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def run_pipeline(
    input_path: str, output_path: str, image_dir: str,
    api_key: str, model: str = "gpt-4o",
    max_workers: int = 5, max_examples: int = -1,
    rate_limit_delay: float = 0.1,
):
    print(f"Loading per-turn examples from {input_path} ...")
    examples = []
    with open(input_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Total: {len(examples)}")

    completed = load_completed_keys(output_path)
    print(f"  Already completed: {len(completed)}")

    remaining = [ex for ex in examples if ex["example_id"] not in completed]
    if max_examples > 0:
        remaining = remaining[:max_examples]
    print(f"  To annotate: {len(remaining)}")

    if not remaining:
        print("Nothing to annotate!")
        return

    annotator = BeliefStateAnnotator(
        api_key=api_key, model=model,
        image_dir=image_dir, rate_limit_delay=rate_limit_delay,
    )

    write_lock = threading.Lock()
    progress = {"done": 0, "failed": 0}

    def process_and_write(example):
        belief_state = annotator.annotate(example)
        if belief_state is not None:
            example["belief_state"] = belief_state
            with write_lock:
                with open(output_path, "a") as out_f:
                    out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                progress["done"] += 1
                if progress["done"] % 50 == 0:
                    total = progress["done"] + progress["failed"]
                    print(f"  Progress: {progress['done']} done, {progress['failed']} failed, "
                          f"{total}/{len(remaining)} | {annotator.stats}")
        else:
            with write_lock:
                progress["failed"] += 1

    print(f"\nAnnotating with {max_workers} workers, model={model} ...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_and_write, ex): ex for ex in remaining}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Worker error: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done: {progress['done']}, Failed: {progress['failed']}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Annotate structured belief states")
    parser.add_argument("--input_path", type=str,
                        default="projects/mmrole/mmrole_per_turn.jsonl")
    parser.add_argument("--output_path", type=str,
                        default="projects/mmrole/mmrole_annotated.jsonl")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--model", type=str, default="gpt-5.4-mini",
                        choices=["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-4o", "gpt-4o-mini"])
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--rate_limit_delay", type=float, default=0.1)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY or pass --api_key")
        sys.exit(1)

    run_pipeline(
        input_path=args.input_path, output_path=args.output_path,
        image_dir=args.image_dir, api_key=api_key, model=args.model,
        max_workers=args.max_workers, max_examples=args.max_examples,
        rate_limit_delay=args.rate_limit_delay,
    )


if __name__ == "__main__":
    main()
