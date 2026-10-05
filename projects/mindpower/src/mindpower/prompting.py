from __future__ import annotations

from mindpower.schemas import BDIAnnotation, InterventionExample, ToMAnnotation


def annotation_to_tagged_text(annotation: BDIAnnotation) -> str:
    belief = annotation.belief
    desire = annotation.desire
    return (
        f"<perception>\n{annotation.perception}\n</perception>\n"
        f"<belief>\n"
        f"Human: {belief.get('human', '')}\n"
        f"Robot: {belief.get('robot', '')}\n"
        f"</belief>\n"
        f"<desire>\n"
        f"Human: {desire.get('human', '')}\n"
        f"Robot: {desire.get('robot', '')}\n"
        f"</desire>\n"
        f"<intention>\n{annotation.intention}\n</intention>\n"
        f"<decision>\n{annotation.decision}\n</decision>\n"
        f"<action>\n" + "\n".join(annotation.action_plan) + "\n</action>"
    )


def tom_annotation_to_tagged_text(annotation: ToMAnnotation) -> str:
    perspective = annotation.perspective
    first_order = annotation.first_order_belief
    second_order = annotation.second_order_belief
    return (
        f"<tom_perception>\n"
        f"Robot focus: {perspective.get('robot_focus', '')}\n"
        f"Human focus: {perspective.get('human_focus', '')}\n"
        f"Asymmetry: {perspective.get('asymmetry', '')}\n"
        f"</tom_perception>\n"
        f"<tom_belief_1st>\n"
        f"Human belief: {first_order.get('human_belief', '')}\n"
        f"Human goal: {first_order.get('human_goal', '')}\n"
        f"Human knowledge: {first_order.get('human_knowledge', '')}\n"
        f"</tom_belief_1st>\n"
        f"<tom_belief_2nd>\n"
        f"Human about robot goal: {second_order.get('human_about_robot_goal', '')}\n"
        f"Human about robot knowledge: {second_order.get('human_about_robot_knowledge', '')}\n"
        f"</tom_belief_2nd>\n"
        f"<tom_hidden_goal>\n{annotation.hidden_goal}\n</tom_hidden_goal>\n"
        f"<tom_false_belief>\n{annotation.false_belief_risk}\n</tom_false_belief>\n"
        f"<tom_intervention>\n"
        f"Reason: {annotation.intervention_reason}\n"
        f"Criticality: {annotation.intervention_criticality}\n"
        f"</tom_intervention>\n"
        f"<tom_metadata>\n"
        f"Belief divergence: {annotation.belief_divergence}\n"
        f"Visual asymmetry present: {annotation.visual_asymmetry_present}\n"
        f"ToM relevance: {annotation.tom_relevance}\n"
        f"</tom_metadata>\n"
        f"<tom_probes>\n"
        + "\n".join(
            f"- [{probe.get('probe_type', 'probe')}] Q: {probe.get('question', '')} | "
            f"A: {probe.get('answer', '')} | Wrong: {probe.get('wrong_answer', '')}"
            for probe in annotation.belief_probes
        )
        + "\n</tom_probes>"
    )


def build_bdi_annotation_prompt(example: InterventionExample) -> str:
    object_lines = [
        f"- {obj.get('name', obj.get('object_id', 'object'))} @ {obj.get('location', 'unknown')}"
        for obj in example.visible_objects[:12]
    ]
    agent_lines = [
        f"- {agent.get('name', 'agent')} ({agent.get('role', 'unknown')}) at {agent.get('location', 'unknown')}"
        for agent in example.agent_states
    ]
    history = "\n".join(f"- {action}" for action in example.history_actions[-6:])

    return (
        "You are annotating a robot-assistance intervention point from an embodied simulator.\n\n"
        f"Task: {example.task_name}\n"
        f"Goal: {example.natural_language_goal}\n"
        f"Simulator: {example.simulator}\n"
        f"Scene: {example.scene_id}\n"
        f"Acting agent: {example.acting_agent}\n"
        f"Partner agent: {example.partner_agent or 'none'}\n\n"
        "Visible objects:\n"
        f"{chr(10).join(object_lines) if object_lines else '- none'}\n\n"
        "Agent states:\n"
        f"{chr(10).join(agent_lines) if agent_lines else '- none'}\n\n"
        "Recent actions:\n"
        f"{history if history else '- none'}\n\n"
        f"Current action: {example.current_action}\n"
        f"Next action: {example.next_action or 'unknown'}\n\n"
        "Produce a structured chain:\n"
        "<perception>...</perception>\n"
        "<belief>...</belief>\n"
        "<desire>...</desire>\n"
        "<intention>...</intention>\n"
        "<decision>...</decision>\n"
        "<action>...</action>"
    )
