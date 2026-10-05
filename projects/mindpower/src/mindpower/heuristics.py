from __future__ import annotations

import re

from mindpower.schemas import BDIAnnotation, InterventionExample, ToMAnnotation


ACTION_RE = re.compile(r"\[(?P<verb>[^\]]+)\]\s*(?:<(?P<target>[^>]+)>)?")


def parse_atomic_action(action: str) -> tuple[str, str]:
    match = ACTION_RE.search(action or "")
    if not match:
        return "Act", "environment"
    verb = match.group("verb") or "Act"
    target = match.group("target") or "environment"
    return verb.strip(), target.strip()


def _object_names(objects: list[dict[str, str]], limit: int = 4) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for obj in objects or []:
        name = str(obj.get("name", "object"))
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= limit:
            break
    return names


def derive_bdi_annotation(example: InterventionExample) -> BDIAnnotation:
    current_verb, current_target = parse_atomic_action(example.current_action)
    next_verb, next_target = parse_atomic_action(example.next_action or "")

    visible_names = [obj.get("name", obj.get("object_id", "object")) for obj in example.visible_objects[:6]]
    visible_summary = ", ".join(visible_names) if visible_names else "no salient objects exported"

    perception = (
        f"The robot observes a {example.task_name} scene with salient objects: {visible_summary}. "
        f"The current embodied event is {current_verb.lower()} on {current_target}."
    )
    belief = {
        "human": (
            f"The human appears to believe the task can progress by focusing on {current_target}. "
            f"The next useful object or location is likely {next_target}."
        ),
        "robot": (
            f"The robot believes the immediate assistance opportunity is around {next_target}, "
            f"because the upcoming action is {next_verb.lower()}."
        ),
    }
    desire = {
        "human": f"The human wants to complete the task: {example.natural_language_goal}",
        "robot": "The robot wants to reduce friction and help the human complete the task safely.",
    }

    if next_verb in {"Grab", "Open", "Put", "Give", "Wipe"}:
        decision = f"Prepare support for the next step involving {next_target}."
        action_plan = [
            f"navigate_to::{next_target}",
            f"assist_with::{next_verb.lower()}::{next_target}",
        ]
    else:
        decision = "Observe and stay ready until a clearer assistive affordance appears."
        action_plan = ["observe_scene", "maintain_safe_distance"]

    intention = (
        f"The robot intends to support the human's next sub-goal by making {next_target} easier "
        f"to access or manipulate."
    )
    rationale = (
        f"Derived from the action trace: current={example.current_action}; next={example.next_action or 'unknown'}."
    )

    return BDIAnnotation(
        example_id=example.example_id,
        simulator=example.simulator,
        task_name=example.task_name,
        perception=perception,
        belief=belief,
        desire=desire,
        intention=intention,
        decision=decision,
        action_plan=action_plan,
        rationale=rationale,
        metadata={
            "heuristic": True,
            "current_action": example.current_action,
            "next_action": example.next_action,
        },
    )

def derive_tom_annotation(example: InterventionExample) -> ToMAnnotation:
    current_verb, current_target = parse_atomic_action(example.current_action)
    next_verb, next_target = parse_atomic_action(example.next_action or "")
    next_focus = next_target if next_target != "environment" else current_target
    human_focus = current_target if current_target != "environment" else next_focus
    human_visible = example.metadata.get("human_visible_objects") or example.visible_objects
    robot_visible = example.metadata.get("robot_visible_objects") or example.visible_objects
    human_visible_names = _object_names(human_visible, limit=5)
    robot_visible_names = _object_names(robot_visible, limit=5)
    robot_only = list(example.metadata.get("robot_only_objects") or [])
    human_only = list(example.metadata.get("human_only_objects") or [])
    observation_gap_summary = example.metadata.get("observation_gap_summary", "")
    belief_gap_objects = list(example.metadata.get("belief_gap_objects") or [])
    belief_gap_summary = str(example.metadata.get("belief_gap_summary", "")).strip()
    primary_belief_gap = belief_gap_objects[0] if belief_gap_objects else None

    if primary_belief_gap:
        asymmetry = (
            f"The robot models a belief gap around {primary_belief_gap.get('name', 'an object')}: "
            f"{primary_belief_gap.get('explanation', 'the human belief does not match the current world state.')}"
        )
    elif robot_only:
        asymmetry = (
            f"The robot can currently observe {', '.join(robot_only[:3])}, while the human is more likely "
            f"focused on {human_focus}. This creates a symbolic visibility gap the robot can reason over."
        )
    elif human_only:
        asymmetry = (
            f"The human can observe {', '.join(human_only[:3])} that is not yet in the robot's masked view, "
            "so the robot may need to act conservatively until it resolves that gap."
        )
    elif next_focus != human_focus:
        asymmetry = (
            f"The human is currently centered on {human_focus}, while the robot should shift "
            f"attention toward {next_focus} to anticipate the next assistive bottleneck."
        )
    else:
        asymmetry = (
            f"Both agents are likely anchored on {human_focus}, so perspective asymmetry is low "
            "but action-readiness still matters."
        )

    human_belief = (
        f"The human likely believes progress depends on {human_focus} and the immediate task step "
        f"is {current_verb.lower()}."
    )
    human_goal = example.natural_language_goal
    if primary_belief_gap:
        human_knowledge = (
            f"The robot models the human's knowledge as stale or incomplete: "
            f"{primary_belief_gap.get('explanation', belief_gap_summary or 'their belief does not fully match the world state.')}"
        )
    elif robot_only:
        human_knowledge = (
            f"The human does not currently observe {', '.join(robot_only[:3])}; their knowledge may therefore "
            "lag behind the robot's scene model."
        )
    else:
        human_knowledge = (
            f"The human may not fully know whether the robot has already recognized that {next_focus} "
            "will become the next useful object or location."
        )

    human_about_robot_goal = (
        f"The robot infers that the human expects it to help once the need around {next_focus} "
        "becomes obvious."
    )
    human_about_robot_knowledge = (
        f"The robot thinks the human assumes it has enough scene context to monitor {human_focus} "
        f"and prepare for {next_focus}."
    )

    hidden_goal = (
        f"The hidden near-term subgoal is to make the transition from {human_focus} to {next_focus} "
        "smooth enough that the task can continue without delay."
    )

    if primary_belief_gap:
        false_belief_risk = belief_gap_summary or primary_belief_gap.get(
            "explanation",
            "A stale human belief could make intervention mistimed.",
        )
    elif robot_only:
        false_belief_risk = (
            f"A likely belief mismatch is that the human cannot yet observe {', '.join(robot_only[:3])}, "
            "so a premature intervention could reveal or act on information the human does not share."
        )
    elif next_focus != human_focus:
        false_belief_risk = (
            f"A likely belief mismatch is that the human may still model {human_focus} as the only "
            f"active bottleneck while the robot should already prepare for {next_focus}."
        )
    else:
        false_belief_risk = (
            f"The main ToM risk is not a strong scene mismatch but whether the human realizes the robot "
            f"is ready to help with {next_focus}."
        )

    critical_verbs = {"grab", "open", "put", "give", "wipe", "find", "switchon"}
    intervention_criticality = "high" if next_verb.lower() in critical_verbs else "medium"
    intervention_reason = (
        f"Assistive intervention matters because the anticipated next step is {next_verb.lower()} "
        f"on {next_focus}, which is easier if the robot proactively aligns with the human's evolving beliefs."
    )
    if belief_gap_objects and belief_gap_summary:
        intervention_reason = f"{intervention_reason} Belief cue: {belief_gap_summary}"
    if observation_gap_summary:
        intervention_reason = f"{intervention_reason} Visibility cue: {observation_gap_summary}"
    rationale = (
        f"Derived from action trace current={example.current_action}; next={example.next_action or 'unknown'}; "
        f"task={example.task_name}."
    )
    if human_visible_names or robot_visible_names:
        rationale += (
            f" human_view={human_visible_names or ['none']}; robot_view={robot_visible_names or ['none']}."
        )

    if primary_belief_gap:
        visual_answer = primary_belief_gap.get("name", next_focus)
        visual_wrong = human_focus
    else:
        visual_answer = ", ".join(robot_only[:3]) if robot_only else next_focus
        visual_wrong = ", ".join(human_only[:3]) if human_only else human_focus
    if visual_wrong == visual_answer:
        visual_wrong = "the current human-visible focus only"

    return ToMAnnotation(
        example_id=example.example_id,
        simulator=example.simulator,
        task_name=example.task_name,
        perspective={
            "robot_focus": next_focus,
            "human_focus": human_focus,
            "asymmetry": asymmetry,
        },
        first_order_belief={
            "human_belief": human_belief,
            "human_goal": human_goal,
            "human_knowledge": human_knowledge,
        },
        second_order_belief={
            "human_about_robot_goal": human_about_robot_goal,
            "human_about_robot_knowledge": human_about_robot_knowledge,
        },
        hidden_goal=hidden_goal,
        false_belief_risk=false_belief_risk,
        intervention_reason=intervention_reason,
        intervention_criticality=intervention_criticality,
        belief_divergence=(
            "high"
            if primary_belief_gap and "stale_belief" in {gap.get("gap_type") for gap in belief_gap_objects}
            else "moderate"
            if (belief_gap_objects or robot_only or human_only or next_focus != human_focus)
            else "low"
        ),
        visual_asymmetry_present=bool(belief_gap_objects or robot_only or human_only or next_focus != human_focus),
        tom_relevance="high" if (intervention_criticality == "high" or belief_gap_objects or robot_only) else "medium",
        belief_probes=[
            {
                "question": "What should the robot attend to before the human explicitly asks for help?",
                "answer": visual_answer,
                "wrong_answer": visual_wrong,
                "probe_type": "visual_asymmetry",
                "difficulty": "medium",
            },
            {
                "question": "What does the robot think the human currently believes is the active bottleneck?",
                "answer": human_belief,
                "wrong_answer": "The human has no active belief about the task.",
                "probe_type": "first_order_belief",
                "difficulty": "medium",
            },
            {
                "question": "What does the robot think the human expects from the robot next?",
                "answer": human_about_robot_goal,
                "wrong_answer": "The human expects the robot to stay passive.",
                "probe_type": "second_order_belief",
                "difficulty": "hard",
            },
            {
                "question": "What belief mismatch could make intervention mistimed?",
                "answer": false_belief_risk,
                "wrong_answer": "There is no meaningful belief mismatch risk.",
                "probe_type": "false_belief_risk",
                "difficulty": "hard",
            },
        ],
        rationale=rationale,
        metadata={
            "heuristic": True,
            "current_action": example.current_action,
            "next_action": example.next_action,
            "human_visible_objects": human_visible_names,
            "robot_visible_objects": robot_visible_names,
            "robot_only_objects": robot_only,
            "human_only_objects": human_only,
            "belief_gap_summary": belief_gap_summary,
        },
    )
