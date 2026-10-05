from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class EpisodeRequest:
    request_id: str
    simulator: str
    scene_id: str
    task_name: str
    natural_language_goal: str
    scripted_actions: list[str]
    max_steps: int = 32
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class AgentState:
    name: str
    role: str
    location: str
    holding: list[str] = field(default_factory=list)
    facing: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectState:
    object_id: str
    name: str
    location: str
    state: str = "unknown"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameObservation:
    frame_index: int
    timestamp: float
    visible_objects: list[ObjectState] = field(default_factory=list)
    agent_states: list[AgentState] = field(default_factory=list)
    rgb_path: Optional[str] = None
    segmentation_path: Optional[str] = None
    depth_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventStep:
    step_index: int
    action: str
    actor: str
    target: Optional[str]
    success: bool
    observation: FrameObservation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorEpisode:
    episode_id: str
    simulator: str
    scene_id: str
    task_name: str
    natural_language_goal: str
    agent_names: list[str]
    scripted_actions: list[str]
    events: list[EventStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionExample:
    example_id: str
    episode_id: str
    simulator: str
    scene_id: str
    task_name: str
    natural_language_goal: str
    history_actions: list[str]
    current_action: str
    next_action: Optional[str]
    acting_agent: str
    partner_agent: Optional[str]
    visible_objects: list[dict[str, Any]]
    agent_states: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BDIAnnotation:
    example_id: str
    simulator: str
    task_name: str
    perception: str
    belief: dict[str, str]
    desire: dict[str, str]
    intention: str
    decision: str
    action_plan: list[str]
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToMAnnotation:
    example_id: str
    simulator: str
    task_name: str
    perspective: dict[str, str]
    first_order_belief: dict[str, str]
    second_order_belief: dict[str, str]
    hidden_goal: str
    false_belief_risk: str
    intervention_reason: str
    intervention_criticality: str
    belief_divergence: str = "moderate"
    visual_asymmetry_present: bool = True
    tom_relevance: str = "high"
    belief_probes: list[dict[str, str]] = field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
