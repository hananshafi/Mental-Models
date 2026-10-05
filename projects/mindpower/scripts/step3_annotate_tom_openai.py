#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.io_utils import read_jsonl
from mindpower.openai_compat import (
    DEFAULT_NRP_MODEL,
    DEFAULT_NRP_PROVIDER_CONFIG,
    OpenAICompatibleWrapper,
    OpenAIProviderConfig,
)
from mindpower.schemas import InterventionExample


TOM_ANNOTATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mindpower_tom_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "perspective": {
                    "type": "object",
                    "properties": {
                        "robot_focus": {"type": "string"},
                        "human_focus": {"type": "string"},
                        "asymmetry": {"type": "string"},
                    },
                    "required": ["robot_focus", "human_focus", "asymmetry"],
                    "additionalProperties": False,
                },
                "first_order_belief": {
                    "type": "object",
                    "properties": {
                        "human_belief": {"type": "string"},
                        "human_goal": {"type": "string"},
                        "human_knowledge": {"type": "string"},
                    },
                    "required": ["human_belief", "human_goal", "human_knowledge"],
                    "additionalProperties": False,
                },
                "second_order_belief": {
                    "type": "object",
                    "properties": {
                        "human_about_robot_goal": {"type": "string"},
                        "human_about_robot_knowledge": {"type": "string"},
                    },
                    "required": ["human_about_robot_goal", "human_about_robot_knowledge"],
                    "additionalProperties": False,
                },
                "hidden_goal": {"type": "string"},
                "false_belief_risk": {"type": "string"},
                "intervention_reason": {"type": "string"},
                "intervention_criticality": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "belief_divergence": {
                    "type": "string",
                    "enum": ["none", "low", "moderate", "high"],
                },
                "visual_asymmetry_present": {"type": "boolean"},
                "tom_relevance": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "belief_probes": {
                    "type": "array",
                    # The prompt asks for exactly 4 probes; enforce it in the
                    # schema as well so the model can't drop below the
                    # downstream training-data expectation.
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "wrong_answer": {"type": "string"},
                            "probe_type": {
                                "type": "string",
                                "enum": [
                                    "visual_asymmetry",
                                    "first_order_belief",
                                    "second_order_belief",
                                    "false_belief_risk",
                                    "hidden_goal",
                                ],
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["easy", "medium", "hard"],
                            },
                        },
                        "required": ["question", "answer", "wrong_answer", "probe_type", "difficulty"],
                        "additionalProperties": False,
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": [
                "perspective",
                "first_order_belief",
                "second_order_belief",
                "hidden_goal",
                "false_belief_risk",
                "intervention_reason",
                "intervention_criticality",
                "belief_divergence",
                "visual_asymmetry_present",
                "tom_relevance",
                "belief_probes",
                "rationale",
            ],
            "additionalProperties": False,
        },
    },
}


def intervention_from_dict(raw: dict[str, Any]) -> InterventionExample:
    return InterventionExample(
        example_id=raw["example_id"],
        episode_id=raw["episode_id"],
        simulator=raw["simulator"],
        scene_id=raw["scene_id"],
        task_name=raw["task_name"],
        natural_language_goal=raw["natural_language_goal"],
        history_actions=raw.get("history_actions", []),
        current_action=raw["current_action"],
        next_action=raw.get("next_action"),
        acting_agent=raw["acting_agent"],
        partner_agent=raw.get("partner_agent"),
        visible_objects=raw.get("visible_objects", []),
        agent_states=raw.get("agent_states", []),
        metadata=raw.get("metadata", {}),
    )


def encode_image_b64(image_path: str) -> Optional[str]:
    path = Path(image_path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_example_image_path(example: InterventionExample) -> Optional[str]:
    for key in ("rgb_path", "image_path", "frame_path"):
        value = example.metadata.get(key)
        if value and Path(value).exists():
            return str(value)
    return None


def _format_state_changes(example: InterventionExample) -> str:
    changes = example.metadata.get("object_state_changes") or []
    if not changes:
        return "- none observed since the previous frame"
    lines = []
    for change in changes[:8]:
        lines.append(
            f"- {change.get('name', 'object')}: "
            f"location {change.get('prev_location')} -> {change.get('curr_location')}, "
            f"state {change.get('prev_state')} -> {change.get('curr_state')}"
        )
    return "\n".join(lines)


def _format_agent_visible_objects(example: InterventionExample, key: str) -> str:
    objects = example.metadata.get(key) or []
    if not objects:
        return "- unavailable"
    lines = []
    for obj in objects[:12]:
        lines.append(
            f"- {obj.get('name', obj.get('object_id', 'object'))} | "
            f"location={obj.get('location', 'unknown')} | "
            f"state={obj.get('state', 'unknown')} | "
            f"reasons={obj.get('attributes', {}).get('visibility_reasons', [])}"
        )
    return "\n".join(lines)


def _format_observation_gap(example: InterventionExample) -> str:
    robot_only = example.metadata.get("robot_only_objects") or []
    human_only = example.metadata.get("human_only_objects") or []
    summary = example.metadata.get("observation_gap_summary")
    if summary:
        return f"- {summary}"
    if robot_only or human_only:
        return (
            f"- robot_only={robot_only[:5] or ['none']}; "
            f"human_only={human_only[:5] or ['none']}"
        )
    return "- no explicit per-agent observation gap available"


def _format_belief_state(example: InterventionExample, key: str) -> str:
    beliefs = example.metadata.get(key) or []
    if not beliefs:
        return "- unavailable"
    lines = []
    for belief in beliefs[:12]:
        lines.append(
            f"- {belief.get('name', belief.get('object_id', 'object'))} | "
            f"believed_location={belief.get('believed_location', 'unknown')} | "
            f"believed_state={belief.get('believed_state', 'unknown')} | "
            f"status={belief.get('belief_status', 'unknown')} | "
            f"staleness_steps={belief.get('staleness_steps', 'unknown')}"
        )
    return "\n".join(lines)


def _format_belief_gap(example: InterventionExample) -> str:
    gap_objects = example.metadata.get("belief_gap_objects") or []
    if gap_objects:
        lines = []
        for gap in gap_objects[:8]:
            lines.append(
                f"- {gap.get('name', 'object')} | gap_type={gap.get('gap_type', 'unknown')} | "
                f"believed=({gap.get('believed_location', 'unknown')}, {gap.get('believed_state', 'unknown')}) | "
                f"world=({gap.get('world_location', 'unknown')}, {gap.get('world_state', 'unknown')}) | "
                f"explanation={gap.get('explanation', '')}"
            )
        return "\n".join(lines)
    summary = example.metadata.get("belief_gap_summary")
    if summary:
        return f"- {summary}"
    return "- no explicit belief-state gap available"


def _format_observer_framing(example: InterventionExample) -> str:
    implicit = bool(example.metadata.get("implicit_robot_observer"))
    if implicit:
        return (
            "Observer role: this trajectory was recorded from a single-agent simulator "
            "episode. Adopt the perspective of an IMPLICIT robot assistant that observes "
            f"'{example.acting_agent}' (the human) with full scene access (god's-eye view). "
            "The robot is not itself acting in this frame; it is reasoning about whether "
            "and when to intervene. Do NOT fabricate robot actions that are not in the "
            "trajectory. Your goal is belief tracking, not action imitation."
        )
    return (
        f"Observer role: reason from the perspective of the robot assistant "
        f"'{example.partner_agent}' who is present in this scene alongside the human "
        f"'{example.acting_agent}'. Both agents may have asymmetric scene access."
    )


def build_annotation_prompt(example: InterventionExample) -> str:
    object_lines = []
    for obj in example.visible_objects[:16]:
        object_lines.append(
            f"- {obj.get('name', obj.get('object_id', 'object'))} | "
            f"location={obj.get('location', 'unknown')} | "
            f"state={obj.get('state', 'unknown')}"
        )

    agent_lines = []
    for agent in example.agent_states:
        agent_lines.append(
            f"- {agent.get('name', 'agent')} | role={agent.get('role', 'unknown')} | "
            f"location={agent.get('location', 'unknown')} | holding={agent.get('holding', [])}"
        )

    history_lines = [f"- {action}" for action in example.history_actions[-8:]]
    tom_reasons = example.metadata.get("tom_worthiness_reasons") or []
    tom_reasons_str = ", ".join(tom_reasons) if tom_reasons else "unspecified"

    return f"""You are a Theory-of-Mind researcher annotating an embodied robot-assistance intervention point.

{_format_observer_framing(example)}

Your job is to perform BELIEF TRACKING for the current moment in a simulator trajectory.

Scene/task context:
- Simulator: {example.simulator}
- Scene: {example.scene_id}
- Task name: {example.task_name}
- Natural-language goal: {example.natural_language_goal}
- Acting agent now: {example.acting_agent}
- Partner agent: {example.partner_agent or "unknown"}
- ToM-worthiness signals flagged upstream: {tom_reasons_str}

Recent action history:
{chr(10).join(history_lines) if history_lines else "- none"}

Current action:
- {example.current_action}

Next action (if known):
- {example.next_action or "unknown"}

World-state objects exported:
{chr(10).join(object_lines) if object_lines else "- none"}

Human-observable objects (derived symbolic view):
{_format_agent_visible_objects(example, "human_visible_objects")}

Robot-observable objects (derived symbolic view):
{_format_agent_visible_objects(example, "robot_visible_objects")}

Per-agent observation gap:
{_format_observation_gap(example)}

Human belief state rolled forward from human-observable evidence:
{_format_belief_state(example, "human_belief_state")}

Robot model of human belief:
{_format_belief_state(example, "robot_model_of_human")}

Belief-state gap versus current world:
{_format_belief_gap(example)}

Object state changes since the previous frame (possible belief-disruption evidence):
{_format_state_changes(example)}

Agent states:
{chr(10).join(agent_lines) if agent_lines else "- none"}

Please infer the robot observer's Theory-of-Mind state at this moment.

Output rules:
1. `perspective.robot_focus` = what the observer (full scene access) should attend to now.
   `perspective.human_focus` = what '{example.acting_agent}' is plausibly attending to, bounded
   by their recent action history. `perspective.asymmetry` = the concrete information the
   observer has that the human does not (or vice versa).
2. `first_order_belief` tracks what the observer thinks '{example.acting_agent}' currently
   believes, wants, and knows -- NOT what is actually true in the scene. Use the rolled
   human belief state and belief-gap evidence above when available.
3. `second_order_belief` tracks what the observer thinks '{example.acting_agent}' expects
   from the robot (goal and knowledge the human attributes to the robot).
4. `hidden_goal` names the latent subgoal implied by current+next action that the human has
   not explicitly uttered.
5. `false_belief_risk` must be grounded. If no plausible belief mismatch exists at this
   frame, set `belief_divergence` to "none" or "low" and `visual_asymmetry_present` to false,
   and write `false_belief_risk` as a single short sentence stating there is no mismatch.
   Do NOT invent a mismatch.
6. `intervention_reason` should explain why ToM matters here, referencing the upstream
   worthiness signals if present.
7. `intervention_criticality` should be `high` only if mistimed help is likely to materially
   hurt task completion or safety.
8. `belief_probes` MUST contain exactly 4 probes, one of each `probe_type`:
   - 1 visual_asymmetry
   - 1 first_order_belief
   - 1 second_order_belief
   - 1 false_belief_risk OR hidden_goal
   For every probe, `answer` and `wrong_answer` MUST be semantically distinct strings.
9. Be concrete and grounded in the provided trajectory state. Avoid generic social-language
   filler. Do not hallucinate objects or agents not listed above.
10. The output must be a single JSON object matching the schema.
"""


def build_pilot_annotation_prompt(example: InterventionExample) -> str:
    object_lines = []
    for obj in example.visible_objects[:12]:
        object_lines.append(
            f"- {obj.get('name', obj.get('object_id', 'object'))} | "
            f"location={obj.get('location', 'unknown')} | "
            f"state={obj.get('state', 'unknown')}"
        )

    agent_lines = []
    for agent in example.agent_states:
        agent_lines.append(
            f"- {agent.get('name', 'agent')} | role={agent.get('role', 'unknown')} | "
            f"location={agent.get('location', 'unknown')} | holding={agent.get('holding', [])}"
        )

    history_lines = [f"- {action}" for action in example.history_actions[-6:]]

    return f"""You are annotating a Theory-of-Mind intervention point for embodied assistance.

{_format_observer_framing(example)}

Infer the observer's belief-tracking state from the simulator context and return JSON only.

Scene/task context:
- Simulator: {example.simulator}
- Scene: {example.scene_id}
- Task name: {example.task_name}
- Natural-language goal: {example.natural_language_goal}
- Acting agent now: {example.acting_agent}
- Partner agent: {example.partner_agent or "unknown"}

Recent action history:
{chr(10).join(history_lines) if history_lines else "- none"}

Current action:
- {example.current_action}

Next action (if known):
- {example.next_action or "unknown"}

World-state objects exported:
{chr(10).join(object_lines) if object_lines else "- none"}

Human-observable objects (derived symbolic view):
{_format_agent_visible_objects(example, "human_visible_objects")}

Robot-observable objects (derived symbolic view):
{_format_agent_visible_objects(example, "robot_visible_objects")}

Per-agent observation gap:
{_format_observation_gap(example)}

Human belief state rolled forward from human-observable evidence:
{_format_belief_state(example, "human_belief_state")}

Robot model of human belief:
{_format_belief_state(example, "robot_model_of_human")}

Belief-state gap versus current world:
{_format_belief_gap(example)}

Object state changes since the previous frame (possible belief-disruption evidence):
{_format_state_changes(example)}

Agent states:
{chr(10).join(agent_lines) if agent_lines else "- none"}

Return exactly one JSON object with these keys:
- perspective: {{robot_focus, human_focus, asymmetry}}
- first_order_belief: {{human_belief, human_goal, human_knowledge}}
- second_order_belief: {{human_about_robot_goal, human_about_robot_knowledge}}
- hidden_goal
- false_belief_risk
- intervention_reason
- intervention_criticality
- belief_divergence
- visual_asymmetry_present
- tom_relevance
- rationale

Constraints:
- Be concrete and specific to this trajectory moment.
- Track what the HUMAN believes (may be stale), NOT what is actually true in the scene.
- Prefer the rolled human belief state and explicit belief-gap evidence over generic action-only inference.
- If no plausible belief mismatch exists, set belief_divergence to "none" or "low" and
  visual_asymmetry_present to false; do NOT invent one.
- Use low|medium|high for intervention_criticality and tom_relevance.
- Use none|low|moderate|high for belief_divergence.
- Use a boolean for visual_asymmetry_present.
- Do not use markdown fences.
- Do not include any text before or after the JSON object.
"""


_REQUIRED_PROBE_TYPES: set[str] = {
    "visual_asymmetry",
    "first_order_belief",
    "second_order_belief",
}
# The fourth probe must cover one of these belief-mismatch / hidden-goal types.
_REQUIRED_PROBE_FOURTH: set[str] = {"false_belief_risk", "hidden_goal"}


class ProbeValidationError(ValueError):
    """Raised when an LLM-produced annotation has unusable belief probes."""


def validate_annotation_probes(result: dict[str, Any]) -> None:
    """Raise ProbeValidationError if probes are missing, degenerate, or incomplete.

    Validates that:
    - Exactly 4 probes are present.
    - Every probe has a non-empty ``answer`` that is semantically distinct from
      ``wrong_answer`` (case-insensitive, whitespace-stripped comparison).
    - The four required probe_type categories are all represented.

    Raising here triggers the existing retry loop in
    :meth:`MindPowerToMAnnotator.annotate`, so we get up to ``max_retries``
    additional attempts to satisfy the constraint before giving up.
    """
    probes = result.get("belief_probes", [])
    if not isinstance(probes, list) or len(probes) != 4:
        raise ProbeValidationError(
            f"expected exactly 4 belief_probes, got {len(probes) if isinstance(probes, list) else 'non-list'}"
        )

    seen_types: set[str] = set()
    for i, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise ProbeValidationError(f"probe[{i}] is not a dict")
        answer = (probe.get("answer") or "").strip()
        wrong = (probe.get("wrong_answer") or "").strip()
        if not answer:
            raise ProbeValidationError(f"probe[{i}] has an empty answer")
        if not wrong:
            raise ProbeValidationError(f"probe[{i}] has an empty wrong_answer")
        if answer.lower() == wrong.lower():
            raise ProbeValidationError(
                f"probe[{i}] has answer == wrong_answer ({answer!r})"
            )
        probe_type = probe.get("probe_type")
        if not isinstance(probe_type, str):
            raise ProbeValidationError(f"probe[{i}] missing probe_type")
        seen_types.add(probe_type)

    missing_required = _REQUIRED_PROBE_TYPES - seen_types
    if missing_required:
        raise ProbeValidationError(
            f"missing required probe_types: {sorted(missing_required)}"
        )
    if not (seen_types & _REQUIRED_PROBE_FOURTH):
        raise ProbeValidationError(
            "missing a false_belief_risk or hidden_goal probe"
        )


def _normalize_string(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_mapping(value: Any, required_keys: list[str], fallback_prefix: str) -> dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    normalized: dict[str, str] = {}
    for key in required_keys:
        normalized[key] = _normalize_string(raw.get(key), f"{fallback_prefix}: unknown")
    return normalized


def normalize_annotation_payload(raw: dict[str, Any], example: InterventionExample) -> dict[str, Any]:
    belief_probes = raw.get("belief_probes", [])
    normalized_probes = []
    if isinstance(belief_probes, list):
        for probe in belief_probes[:4]:
            if not isinstance(probe, dict):
                continue
            normalized_probes.append(
                {
                    "question": _normalize_string(probe.get("question"), "What is the key belief mismatch here?"),
                    "answer": _normalize_string(probe.get("answer"), "Unknown from the current annotation."),
                    "wrong_answer": _normalize_string(probe.get("wrong_answer"), "There is no belief mismatch."),
                    "probe_type": _normalize_string(probe.get("probe_type"), "first_order_belief"),
                    "difficulty": _normalize_string(probe.get("difficulty"), "medium"),
                }
            )

    intervention_criticality = _normalize_string(raw.get("intervention_criticality"), "medium").lower()
    if intervention_criticality not in {"low", "medium", "high"}:
        intervention_criticality = "medium"

    belief_divergence = _normalize_string(raw.get("belief_divergence"), "moderate").lower()
    if belief_divergence not in {"none", "low", "moderate", "high"}:
        belief_divergence = "moderate"

    tom_relevance = _normalize_string(raw.get("tom_relevance"), "high").lower()
    if tom_relevance not in {"low", "medium", "high"}:
        tom_relevance = "high"

    visual_asymmetry_present = raw.get("visual_asymmetry_present")
    if not isinstance(visual_asymmetry_present, bool):
        visual_asymmetry_present = belief_divergence in {"moderate", "high"}

    return {
        "example_id": example.example_id,
        "simulator": example.simulator,
        "task_name": example.task_name,
        "perspective": _normalize_mapping(
            raw.get("perspective"),
            ["robot_focus", "human_focus", "asymmetry"],
            "perspective",
        ),
        "first_order_belief": _normalize_mapping(
            raw.get("first_order_belief"),
            ["human_belief", "human_goal", "human_knowledge"],
            "first_order_belief",
        ),
        "second_order_belief": _normalize_mapping(
            raw.get("second_order_belief"),
            ["human_about_robot_goal", "human_about_robot_knowledge"],
            "second_order_belief",
        ),
        "hidden_goal": _normalize_string(raw.get("hidden_goal"), "Unknown hidden subgoal."),
        "false_belief_risk": _normalize_string(raw.get("false_belief_risk"), "Potential mismatch about object state or location."),
        "intervention_reason": _normalize_string(raw.get("intervention_reason"), "ToM is needed to time help without disrupting the human."),
        "intervention_criticality": intervention_criticality,
        "belief_divergence": belief_divergence,
        "visual_asymmetry_present": visual_asymmetry_present,
        "tom_relevance": tom_relevance,
        "belief_probes": normalized_probes,
        "rationale": _normalize_string(raw.get("rationale"), "Pilot annotation generated from an OpenAI-compatible model."),
    }


def load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            example_id = row.get("example_id")
            if example_id:
                done.add(example_id)
    return done


def write_jsonl_line(output_path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with output_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class MindPowerToMAnnotator:
    def __init__(
        self,
        *,
        wrapper: OpenAICompatibleWrapper,
        rate_limit_delay: float = 0.2,
        max_retries: int = 4,
        max_completion_tokens: int = 3000,
        temperature: float = 0.2,
        simple_json_only: bool = False,
    ) -> None:
        self.wrapper = wrapper
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.simple_json_only = simple_json_only
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "errors": 0}

    def annotate(self, example: InterventionExample) -> dict[str, Any]:
        prompt = build_pilot_annotation_prompt(example) if self.simple_json_only else build_annotation_prompt(example)
        image_path = find_example_image_path(example)
        content: list[dict[str, Any]] = []
        if image_path:
            b64 = encode_image_b64(image_path)
            if b64:
                ext = Path(image_path).suffix.lower()
                mime = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(ext, "image/jpeg")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                    }
                )
        content.append({"type": "text", "text": prompt})

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                if self.simple_json_only:
                    result = self.wrapper.create_plain_json(
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=self.max_completion_tokens,
                        temperature=self.temperature,
                    )
                    result = normalize_annotation_payload(result, example)
                else:
                    result = self.wrapper.create_structured_json(
                        messages=[{"role": "user", "content": content}],
                        json_schema=TOM_ANNOTATION_SCHEMA,
                        max_completion_tokens=self.max_completion_tokens,
                        temperature=self.temperature,
                    )
                # Structured-schema mode enforces probe count/fields at the
                # API level. Both modes still need semantic validation:
                # answer != wrong_answer and the right probe_type coverage.
                # If the model returns a degenerate probe, raise and retry.
                validate_annotation_probes(result)
                with self._lock:
                    self.stats["calls"] += 1
                result["example_id"] = example.example_id
                result["simulator"] = example.simulator
                result["task_name"] = example.task_name
                result["metadata"] = {
                    "annotator": "openai_compatible",
                    "provider": self.wrapper.provider.name,
                    "model": self.wrapper.model,
                    "request_model": self.wrapper.request_model,
                    "annotation_mode": "simple_json_only" if self.simple_json_only else "structured_json_schema",
                    "image_path": image_path or "",
                    "episode_id": example.episode_id,
                    "implicit_robot_observer": bool(
                        example.metadata.get("implicit_robot_observer")
                    ),
                    "tom_worthiness_reasons": example.metadata.get(
                        "tom_worthiness_reasons", []
                    ),
                }
                return result
            except Exception as exc:
                last_error = exc
                with self._lock:
                    self.stats["errors"] += 1
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"Failed to annotate {example.example_id}: {last_error}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM ToM belief-tracking annotator for MindPower.")
    parser.add_argument(
        "--input_path",
        default=str(PROJECT_ROOT / "data" / "intermediate" / "intervention_points.jsonl"),
    )
    parser.add_argument(
        "--output_path",
        # Write to the canonical filename that step4 reads by default. The
        # previous default (tom_annotations_llm.jsonl) silently diverged from
        # step4's default (tom_annotations.jsonl), so LLM annotations were
        # never consumed unless the user knew to pass --tom_input_path.
        default=str(PROJECT_ROOT / "data" / "intermediate" / "tom_annotations.jsonl"),
    )
    parser.add_argument(
        "--provider_config_path",
        default=None,
        help="Optional JSON file with an OpenAI-compatible provider config.",
    )
    parser.add_argument("--api_host", default=DEFAULT_NRP_PROVIDER_CONFIG["settings"]["apiHost"])
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=DEFAULT_NRP_MODEL)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--rate_limit_delay", type=float, default=0.2)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--max_completion_tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--simple_json_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    if args.provider_config_path:
        provider = OpenAIProviderConfig.from_json_file(args.provider_config_path, api_key=args.api_key)
    else:
        provider = OpenAIProviderConfig.from_dict(
            DEFAULT_NRP_PROVIDER_CONFIG,
            api_key=args.api_key,
        )
        provider.api_host = args.api_host

    wrapper = OpenAICompatibleWrapper(
        provider,
        model=args.model,
        api_key=args.api_key,
        enable_thinking=args.enable_thinking,
    )
    annotator = MindPowerToMAnnotator(
        wrapper=wrapper,
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        simple_json_only=args.simple_json_only,
    )

    examples = [intervention_from_dict(row) for row in read_jsonl(args.input_path)]
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    done_ids = set() if args.overwrite else load_done_ids(output_path)
    pending = [example for example in examples if example.example_id not in done_ids]

    file_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(annotator.annotate, example): example for example in pending}
        for future in as_completed(future_map):
            example = future_map[future]
            try:
                row = future.result()
                write_jsonl_line(output_path, row, file_lock)
                print(f"[ok] {example.example_id}")
            except Exception as exc:
                print(f"[error] {example.example_id}: {exc}")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "total_examples": len(examples),
                "pending_examples": len(pending),
                "stats": annotator.stats,
                "provider": provider.name,
                "base_url": provider.base_url,
                "model": args.model,
                "request_model": wrapper.request_model,
                "simple_json_only": args.simple_json_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
