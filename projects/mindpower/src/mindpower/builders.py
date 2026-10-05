from __future__ import annotations

import random
from dataclasses import asdict, replace
from typing import Any

from mindpower.asymmetry import (
    build_agent_observation_views,
    derive_belief_gap_objects,
    derive_world_objects,
    roll_belief_state,
    summarize_belief_gap_objects,
)
from mindpower.heuristics import parse_atomic_action
from mindpower.io_utils import stable_hash
from mindpower.prompting import annotation_to_tagged_text, tom_annotation_to_tagged_text
from mindpower.schemas import (
    BDIAnnotation,
    EventStep,
    InterventionExample,
    SimulatorEpisode,
    ToMAnnotation,
)


IMPLICIT_ROBOT_NAME = "robot_observer"

# Atomic verbs that indicate a concrete assistive hand-off the robot could pre-stage.
_CRITICAL_ASSIST_VERBS = frozenset(
    {
        "grab",
        "open",
        "close",
        "put",
        "putin",
        "putback",
        "give",
        "wipe",
        "find",
        "switchon",
        "switchoff",
        "pour",
        "plugin",
        "plugout",
        "cut",
    }
)


def event_step_from_dict(raw: dict[str, Any]) -> EventStep:
    return EventStep(
        step_index=raw["step_index"],
        action=raw["action"],
        actor=raw["actor"],
        target=raw.get("target"),
        success=raw.get("success", True),
        observation=raw["observation"],
        metadata=raw.get("metadata", {}),
    )


def simulator_episode_from_dict(raw: dict[str, Any]) -> SimulatorEpisode:
    return SimulatorEpisode(
        episode_id=raw["episode_id"],
        simulator=raw["simulator"],
        scene_id=raw["scene_id"],
        task_name=raw["task_name"],
        natural_language_goal=raw["natural_language_goal"],
        agent_names=raw.get("agent_names", []),
        scripted_actions=raw.get("scripted_actions", []),
        events=[event_step_from_dict(event) for event in raw.get("events", [])],
        metadata=raw.get("metadata", {}),
    )


def _object_index(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for obj in objects or []:
        key = obj.get("object_id") or obj.get("name")
        if key is None:
            continue
        index[str(key)] = obj
    return index


def _detect_object_state_changes(
    prev_objects: list[dict[str, Any]],
    curr_objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return objects whose location or state changed between two adjacent frames."""
    prev_index = _object_index(prev_objects)
    changes: list[dict[str, Any]] = []
    for obj in curr_objects or []:
        key = obj.get("object_id") or obj.get("name")
        if key is None:
            continue
        prev = prev_index.get(str(key))
        if prev is None:
            continue
        loc_changed = obj.get("location") != prev.get("location")
        state_changed = obj.get("state") != prev.get("state")
        if loc_changed or state_changed:
            changes.append(
                {
                    "name": obj.get("name", obj.get("object_id", "object")),
                    "prev_location": prev.get("location"),
                    "curr_location": obj.get("location"),
                    "prev_state": prev.get("state"),
                    "curr_state": obj.get("state"),
                }
            )
    return changes


def _is_tom_worthy(
    example: InterventionExample,
    object_state_changes: list[dict[str, Any]],
    observation_views: dict[str, Any],
    belief_gap_objects: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Decide whether an intervention moment carries real ToM signal.

    A moment is kept if at least one of these holds:
      1. Focus shift: next action targets a different object than the current action.
      2. Critical assist verb: the upcoming atomic action is a concrete hand-off the
         robot could pre-stage (Grab/Open/Put/Give/...).
      3. Object state change: an object visible in this frame just changed state or
         location relative to the previous frame (candidate belief-disruption event).
    """
    _, current_target = parse_atomic_action(example.current_action)
    next_verb, next_target = parse_atomic_action(example.next_action or "")
    reasons: list[str] = []

    if (
        next_target
        and next_target != "environment"
        and current_target != next_target
    ):
        reasons.append("focus_shift")

    if next_verb and next_verb.lower() in _CRITICAL_ASSIST_VERBS:
        reasons.append("critical_assist_verb")

    if object_state_changes:
        reasons.append("object_state_change")

    if observation_views.get("visual_asymmetry_present"):
        reasons.append("per_agent_observation_gap")

    if belief_gap_objects:
        reasons.append("belief_state_gap")

    return (bool(reasons), reasons)


def build_intervention_points(
    episode: SimulatorEpisode,
    context_window: int = 4,
    tom_worthy_only: bool = True,
    synthesize_implicit_robot: bool = True,
) -> list[InterventionExample]:
    """Build intervention-point examples from a simulator episode.

    Args:
        episode: Source episode with per-step observations.
        context_window: How many prior actions to surface as ``history_actions``.
        tom_worthy_only: If True, drop frames where no meaningful ToM signal is
            present (no focus shift, no critical assist verb, no object state
            change). Set to False only for smoke tests / debugging.
        synthesize_implicit_robot: If True, when the episode has no secondary
            agent, populate ``partner_agent`` with a synthetic ``robot_observer``
            identity and flag it in metadata. This makes the implicit-observer
            role explicit to downstream prompts.
    """

    examples: list[InterventionExample] = []
    human_belief_state: list[dict[str, Any]] = []
    robot_belief_state: list[dict[str, Any]] = []
    robot_model_of_human: list[dict[str, Any]] = []
    for idx, event in enumerate(episode.events):
        history = [prev.action for prev in episode.events[max(0, idx - context_window):idx]]
        next_action = episode.events[idx + 1].action if idx + 1 < len(episode.events) else None
        obs = event.observation
        example_id = f"{episode.episode_id}__{idx:03d}"

        partner = None
        if len(episode.agent_names) > 1:
            partner = next((name for name in episode.agent_names if name != event.actor), None)
        implicit_robot = False
        if partner is None and synthesize_implicit_robot:
            partner = IMPLICIT_ROBOT_NAME
            implicit_robot = True

        prev_obs = episode.events[idx - 1].observation if idx > 0 else None
        prev_visible = prev_obs["visible_objects"] if prev_obs else []
        object_state_changes = _detect_object_state_changes(
            prev_visible, obs.get("visible_objects", [])
        )
        world_graph = (obs.get("metadata") or {}).get("world_graph")
        observation_views = build_agent_observation_views(
            world_graph,
            human_agent_name=event.actor,
            robot_agent_name=partner,
            fallback_scene=episode.scene_id,
            fallback_visible_objects=obs.get("visible_objects", []),
            implicit_robot_observer=implicit_robot,
        )
        world_objects = derive_world_objects(
            world_graph,
            fallback_scene=episode.scene_id,
            fallback_visible_objects=obs.get("visible_objects", []),
        )
        human_belief_state = roll_belief_state(
            human_belief_state,
            observation_views["human_visible_objects"],
            step_index=idx,
            observer="human",
        )
        robot_belief_state = roll_belief_state(
            robot_belief_state,
            observation_views["robot_visible_objects"],
            step_index=idx,
            observer="robot",
        )
        robot_model_of_human = roll_belief_state(
            robot_model_of_human,
            observation_views["human_visible_objects"],
            step_index=idx,
            observer="robot_model_of_human",
        )
        belief_gap_objects = derive_belief_gap_objects(
            world_objects=world_objects,
            belief_state=human_belief_state,
            agent_visible_objects=observation_views["human_visible_objects"],
            counterpart_visible_objects=observation_views["robot_visible_objects"],
            agent_label="human",
        )
        belief_gap_summary = summarize_belief_gap_objects(belief_gap_objects)

        tmp_example = InterventionExample(
            example_id=example_id,
            episode_id=episode.episode_id,
            simulator=episode.simulator,
            scene_id=episode.scene_id,
            task_name=episode.task_name,
            natural_language_goal=episode.natural_language_goal,
            history_actions=history,
            current_action=event.action,
            next_action=next_action,
            acting_agent=event.actor,
            partner_agent=partner,
            visible_objects=obs.get("visible_objects", []),
            agent_states=obs.get("agent_states", []),
            metadata={},
        )

        worthy, reasons = _is_tom_worthy(
            tmp_example,
            object_state_changes,
            observation_views,
            belief_gap_objects,
        )
        if tom_worthy_only and not worthy:
            continue

        tmp_example.metadata = {
            "frame_index": obs["frame_index"],
            "timestamp": obs["timestamp"],
            "success": event.success,
            "implicit_robot_observer": implicit_robot,
            "tom_worthiness_reasons": reasons,
            "object_state_changes": object_state_changes,
            "human_visible_objects": observation_views["human_visible_objects"],
            "robot_visible_objects": observation_views["robot_visible_objects"],
            "human_only_objects": observation_views["human_only_objects"],
            "robot_only_objects": observation_views["robot_only_objects"],
            "shared_objects": observation_views["shared_objects"],
            "human_room": observation_views["human_room"],
            "robot_room": observation_views["robot_room"],
            "human_view_mode": observation_views["human_view_mode"],
            "robot_view_mode": observation_views["robot_view_mode"],
            "observation_gap_summary": observation_views["observation_gap_summary"],
            "visual_asymmetry_present": observation_views["visual_asymmetry_present"],
            "world_state_objects": world_objects,
            "human_belief_state": [dict(item) for item in human_belief_state],
            "robot_belief_state": [dict(item) for item in robot_belief_state],
            "robot_model_of_human": [dict(item) for item in robot_model_of_human],
            "belief_gap_objects": belief_gap_objects,
            "belief_gap_summary": belief_gap_summary,
            "belief_divergence_present": bool(belief_gap_objects),
            "belief_tracking_source": "symbolic_visibility_rollout",
            "world_graph_available": bool(world_graph),
        }
        examples.append(tmp_example)
    return examples


def build_hierarchy_prediction_record(annotation: BDIAnnotation) -> dict[str, Any]:
    return {
        "task": "hierarchy_prediction",
        "example_id": annotation.example_id,
        "simulator": annotation.simulator,
        "task_name": annotation.task_name,
        "target_text": annotation_to_tagged_text(annotation),
        "annotation": asdict(annotation),
    }


def build_preference_pairs(annotation: BDIAnnotation) -> list[dict[str, Any]]:
    """Construct contrastive pairs by *re-rendering* a modified annotation.

    The previous implementation used ``chosen.replace(<field>, <bad>)`` which
    silently corrupted the string when the field was empty (``str.replace("")``
    inserts between every character) or when the field text happened to collide
    with another substring. We now build a shallow copy of the dataclass with
    just the targeted field overwritten, then re-render via
    :func:`annotation_to_tagged_text`.
    """

    chosen = annotation_to_tagged_text(annotation)

    belief_bad = {
        **annotation.belief,
        "human": "The human already knows everything the robot knows.",
    }
    desire_bad = {
        **annotation.desire,
        "human": "The human has no concrete goal and is wandering.",
    }

    variants: dict[str, BDIAnnotation] = {
        "belief_mismatch": replace(annotation, belief=belief_bad),
        "desire_mismatch": replace(annotation, desire=desire_bad),
        "wrong_decision": replace(annotation, decision="Ignore the human and leave the scene."),
        "wrong_action": replace(annotation, action_plan=["walk_away", "idle"]),
    }

    pairs: list[dict[str, Any]] = []
    for violation_type, bad_annotation in variants.items():
        rejected = annotation_to_tagged_text(bad_annotation)
        if rejected == chosen:
            # Defensive skip: if the field was already degenerate, the variant is a no-op.
            continue
        pairs.append(
            {
                "task": "preference_pair",
                "pair_id": f"{annotation.example_id}__{violation_type}",
                "example_id": annotation.example_id,
                "simulator": annotation.simulator,
                "task_name": annotation.task_name,
                "preferred_response": chosen,
                "rejected_response": rejected,
                "violation_type": violation_type,
            }
        )
    return pairs


def build_probe_qa(annotation: BDIAnnotation) -> list[dict[str, Any]]:
    return [
        {
            "task": "probe_qa",
            "probe_id": f"{annotation.example_id}__belief",
            "example_id": annotation.example_id,
            "question": "What does the robot believe the human is trying to accomplish?",
            "correct_answer": annotation.desire.get("human", ""),
            "wrong_answer": "The human has no task-relevant goal.",
            "probe_type": "desire_inference",
        },
        {
            "task": "probe_qa",
            "probe_id": f"{annotation.example_id}__decision",
            "example_id": annotation.example_id,
            "question": "What should the robot decide to do next?",
            "correct_answer": annotation.decision,
            "wrong_answer": "The robot should disengage and stop assisting.",
            "probe_type": "decision_selection",
        },
        {
            "task": "probe_qa",
            "probe_id": f"{annotation.example_id}__action",
            "example_id": annotation.example_id,
            "question": "What is the best action plan for the robot?",
            "correct_answer": "\n".join(annotation.action_plan),
            "wrong_answer": "walk_away\nidle",
            "probe_type": "action_planning",
        },
    ]


def build_action_target(annotation: BDIAnnotation) -> dict[str, Any]:
    return {
        "task": "action_target",
        "example_id": annotation.example_id,
        "simulator": annotation.simulator,
        "task_name": annotation.task_name,
        "decision": annotation.decision,
        "action_plan": annotation.action_plan,
    }


def build_tom_prediction_record(annotation: ToMAnnotation) -> dict[str, Any]:
    return {
        "task": "tom_prediction",
        "example_id": annotation.example_id,
        "simulator": annotation.simulator,
        "task_name": annotation.task_name,
        "target_text": tom_annotation_to_tagged_text(annotation),
        "annotation": asdict(annotation),
    }


def build_tom_preference_pairs(annotation: ToMAnnotation) -> list[dict[str, Any]]:
    """Construct ToM contrastive pairs by re-rendering a modified annotation.

    Same correctness concern as :func:`build_preference_pairs`: we must avoid
    ``chosen.replace(<field>, <bad>)`` because empty fields corrupt the string.
    """

    chosen = tom_annotation_to_tagged_text(annotation)

    perspective_bad = {
        **annotation.perspective,
        "asymmetry": "There is no meaningful perspective difference between the human and the robot.",
    }
    first_order_bad = {
        **annotation.first_order_belief,
        "human_belief": "The human has no task-relevant belief and is acting randomly.",
    }
    second_order_bad = {
        **annotation.second_order_belief,
        "human_about_robot_goal": "The human expects nothing from the robot and assumes it has no role in the task.",
    }

    variants: dict[str, ToMAnnotation] = {
        "visual_asymmetry_violation": replace(annotation, perspective=perspective_bad),
        "first_order_violation": replace(annotation, first_order_belief=first_order_bad),
        "second_order_violation": replace(annotation, second_order_belief=second_order_bad),
        "false_belief_violation": replace(
            annotation,
            false_belief_risk="There is no possible belief mismatch worth modeling here.",
        ),
    }

    pairs: list[dict[str, Any]] = []
    for violation_type, bad_annotation in variants.items():
        rejected = tom_annotation_to_tagged_text(bad_annotation)
        if rejected == chosen:
            continue
        pairs.append(
            {
                "task": "tom_preference_pair",
                "pair_id": f"{annotation.example_id}__{violation_type}",
                "example_id": annotation.example_id,
                "simulator": annotation.simulator,
                "task_name": annotation.task_name,
                "preferred_response": chosen,
                "rejected_response": rejected,
                "violation_type": violation_type,
            }
        )
    return pairs


def build_tom_probe_qa(annotation: ToMAnnotation) -> list[dict[str, Any]]:
    if annotation.belief_probes:
        return [
            {
                "task": "tom_probe_qa",
                "probe_id": f"{annotation.example_id}__probe{i}",
                "example_id": annotation.example_id,
                "question": probe.get("question", ""),
                "correct_answer": probe.get("answer", ""),
                "wrong_answer": probe.get("wrong_answer", ""),
                "probe_type": probe.get("probe_type", "tom_probe"),
                "difficulty": probe.get("difficulty", "medium"),
            }
            for i, probe in enumerate(annotation.belief_probes)
        ]
    return [
        {
            "task": "tom_probe_qa",
            "probe_id": f"{annotation.example_id}__focus",
            "example_id": annotation.example_id,
            "question": "What should the robot focus on that the human may not be prioritizing yet?",
            "correct_answer": annotation.perspective.get("robot_focus", ""),
            "wrong_answer": annotation.perspective.get("human_focus", ""),
            "probe_type": "visual_asymmetry",
        },
        {
            "task": "tom_probe_qa",
            "probe_id": f"{annotation.example_id}__belief1",
            "example_id": annotation.example_id,
            "question": "What does the robot think the human currently believes about the task?",
            "correct_answer": annotation.first_order_belief.get("human_belief", ""),
            "wrong_answer": "The human has no task-specific belief.",
            "probe_type": "first_order_belief",
        },
        {
            "task": "tom_probe_qa",
            "probe_id": f"{annotation.example_id}__belief2",
            "example_id": annotation.example_id,
            "question": "What does the robot think the human expects from the robot?",
            "correct_answer": annotation.second_order_belief.get("human_about_robot_goal", ""),
            "wrong_answer": "The human expects the robot to remain uninvolved.",
            "probe_type": "second_order_belief",
        },
        {
            "task": "tom_probe_qa",
            "probe_id": f"{annotation.example_id}__false_belief",
            "example_id": annotation.example_id,
            "question": "What possible belief mismatch could cause bad intervention timing?",
            "correct_answer": annotation.false_belief_risk,
            "wrong_answer": "There is no belief mismatch risk in this scene.",
            "probe_type": "false_belief_risk",
        },
    ]


def partition_example_ids(
    example_ids: list[str],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict[str, set[str]]:
    """Deterministically partition a list of example_ids into train/val sets.

    Uses a fixed-seed shuffle on the sorted unique ids so the partition is
    stable across products (BDI, ToM, probes, preference pairs) and across
    reruns. This is the foundation for leak-free splits — every downstream
    product looks up which split its parent example belongs to via these sets.
    """

    unique_ids = sorted(set(example_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    cut = int(len(unique_ids) * train_ratio)
    return {
        "train": set(unique_ids[:cut]),
        "val": set(unique_ids[cut:]),
    }


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.9,
    seed: int = 42,
    split_sets: dict[str, set[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train/val without leaking example_ids across splits.

    If ``split_sets`` is provided (as produced by :func:`partition_example_ids`),
    each record is routed by its ``example_id`` field. Records whose
    ``example_id`` does not appear in any set are silently dropped (should not
    happen in practice — caller should build ``split_sets`` from the union of
    all example_ids first).

    If ``split_sets`` is ``None`` we fall back to building a partition from the
    records' own example_ids. This keeps the function usable standalone but
    callers that mix multiple products should build a single partition first
    and reuse it across calls to keep products aligned.
    """

    items = list(records)
    if split_sets is None:
        ids = [str(row.get("example_id", "")) for row in items if row.get("example_id")]
        split_sets = partition_example_ids(ids, train_ratio=train_ratio, seed=seed)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in items:
        ex_id = str(row.get("example_id", ""))
        if ex_id in split_sets.get("train", set()):
            train.append(row)
        elif ex_id in split_sets.get("val", set()):
            val.append(row)
        # records without a matching example_id are dropped intentionally
    return {"train": train, "val": val}


def derive_reward_vector(annotation: BDIAnnotation) -> dict[str, float]:
    verb, target = parse_atomic_action(annotation.metadata.get("next_action", ""))
    target_bonus = 1.0 if target and target != "environment" else 0.5
    return {
        "belief_match": 1.0,
        "desire_match": 1.0,
        "intention_match": 1.0,
        "decision_match": 1.0 if annotation.decision else 0.0,
        "atomic_local": target_bonus,
        "atomic_global": 1.0 if len(annotation.action_plan) >= 1 else 0.0,
        "executability": 1.0 if verb else 0.5,
    }


def make_manifest_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{stable_hash(list(parts))}"
