from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from mindpower.schemas import (
    AgentState,
    EpisodeRequest,
    EventStep,
    FrameObservation,
    ObjectState,
    SimulatorEpisode,
)
from mindpower.simulators.base import EmbodiedSimulatorAdapter


ACTION_RE = re.compile(r"\[(?P<verb>[^\]]+)\]\s*(?:<(?P<target>[^>]+)>)?")
HOLD_RELATIONS = {"HOLDS_RH", "HOLDS_LH", "HOLDS"}
LOCATION_RELATIONS = {"INSIDE", "ON", "CLOSE", "CLOSETO", "NEAR", "FACING"}


class VirtualHomeAdapter(EmbodiedSimulatorAdapter):
    name = "virtualhome"

    def __init__(self, package_root: str | None = None, unity_binary: str | None = None):
        self.package_root = package_root
        self.unity_binary = unity_binary

    def is_available(self) -> bool:
        if self.package_root and Path(self.package_root).exists():
            return True
        return importlib.util.find_spec("virtualhome") is not None

    def discover_export_root(self, dataset_root: str | Path) -> Path:
        root = Path(dataset_root)
        candidates = [
            root,
            root / "programs_processed_precond_nograb_morepreconds",
            root / "dataset" / "programs_processed_precond_nograb_morepreconds",
        ]
        for candidate in candidates:
            if (candidate / "executable_programs").exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find a VirtualHome export root under {root}. "
            "Expected a directory containing executable_programs/."
        )

    def build_requests_from_export_root(
        self,
        dataset_root: str | Path,
        *,
        limit: int = -1,
        scene_filter: str | None = None,
    ) -> list[EpisodeRequest]:
        export_root = self.discover_export_root(dataset_root)
        exec_root = export_root / "executable_programs"
        state_root = export_root / "state_list"
        init_root = export_root / "initstate"
        no_conds_root = export_root / "withoutconds"

        requests: list[EpisodeRequest] = []
        for script_path in sorted(exec_root.rglob("*.txt")):
            rel_path = script_path.relative_to(exec_root)
            scene_id = rel_path.parts[0] if rel_path.parts else "unknown_scene"
            if scene_filter and scene_filter != scene_id:
                continue
            actions = self._read_script_lines(script_path)
            if not actions:
                continue

            stem = script_path.stem
            state_path = state_root / rel_path.with_suffix(".json")
            initstate_path = self._first_existing(
                init_root / rel_path.with_suffix(".json"),
                init_root / rel_path.with_suffix(".txt"),
            )
            original_script_path = self._first_existing(
                no_conds_root / rel_path.with_suffix(".txt"),
                no_conds_root / rel_path.with_suffix(".json"),
            )

            task_name = self._derive_task_name(scene_id=scene_id, stem=stem)
            goal = f"Complete the household activity: {task_name.replace('_', ' ')}."
            requests.append(
                EpisodeRequest(
                    request_id=f"virtualhome::{scene_id}::{stem}",
                    simulator=self.name,
                    scene_id=scene_id,
                    task_name=task_name,
                    natural_language_goal=goal,
                    scripted_actions=actions,
                    dry_run=False,
                    metadata={
                        "export_root": str(export_root),
                        "script_path": str(script_path),
                        "state_list_path": str(state_path) if state_path.exists() else "",
                        "initstate_path": str(initstate_path) if initstate_path else "",
                        "original_script_path": str(original_script_path) if original_script_path else "",
                    },
                )
            )
            if limit > 0 and len(requests) >= limit:
                break
        return requests

    def collect_episode(self, request: EpisodeRequest, output_dir: str | Path) -> SimulatorEpisode:
        if request.dry_run:
            return self._mock_episode(request)
        if request.metadata.get("script_path"):
            return self._episode_from_exported_files(request)
        raise NotImplementedError(
            "VirtualHome runtime collection is not wired yet. "
            "This adapter now supports importing exported VirtualHome scripts and "
            "state traces via --virtualhome_dataset_root. For live simulator control, "
            "the next step is to connect the installed VirtualHome API and Unity build."
        )

    def _episode_from_exported_files(self, request: EpisodeRequest) -> SimulatorEpisode:
        script_path = Path(request.metadata["script_path"])
        state_list_path = Path(request.metadata["state_list_path"]) if request.metadata.get("state_list_path") else None
        initstate_path = Path(request.metadata["initstate_path"]) if request.metadata.get("initstate_path") else None
        original_script_path = (
            Path(request.metadata["original_script_path"])
            if request.metadata.get("original_script_path") else None
        )

        actions = request.scripted_actions or self._read_script_lines(script_path)
        state_sequence = self._load_state_sequence(state_list_path) if state_list_path and state_list_path.exists() else []

        events: list[EventStep] = []
        detected_agents: list[str] = []
        for step_index, action in enumerate(actions):
            raw_state = state_sequence[min(step_index, len(state_sequence) - 1)] if state_sequence else None
            observation = self._state_to_observation(
                raw_state=raw_state,
                step_index=step_index,
                fallback_scene=request.scene_id,
                fallback_action=action,
            )
            agent_names = [agent.name for agent in observation.agent_states]
            for name in agent_names:
                if name not in detected_agents:
                    detected_agents.append(name)

            _, target = self._parse_action(action)
            actor = detected_agents[0] if detected_agents else "character_0"
            events.append(
                EventStep(
                    step_index=step_index,
                    action=action,
                    actor=actor,
                    target=target,
                    success=True,
                    observation=observation,
                    metadata={
                        "source": "virtualhome_export",
                        "script_path": str(script_path),
                        "state_list_path": str(state_list_path) if state_list_path else "",
                    },
                )
            )

        if not detected_agents:
            detected_agents = ["character_0"]

        initstate = self._load_optional_jsonish(initstate_path)
        original_script = self._read_script_lines(original_script_path) if original_script_path and original_script_path.exists() else []

        return SimulatorEpisode(
            episode_id=request.request_id,
            simulator=self.name,
            scene_id=request.scene_id,
            task_name=request.task_name,
            natural_language_goal=request.natural_language_goal,
            agent_names=detected_agents,
            scripted_actions=list(actions),
            events=events,
            metadata={
                "dry_run": False,
                "source": "virtualhome_export",
                "package_root": self.package_root,
                "unity_binary": self.unity_binary,
                "script_path": str(script_path),
                "state_list_path": str(state_list_path) if state_list_path else "",
                "initstate_path": str(initstate_path) if initstate_path else "",
                "original_script_path": str(original_script_path) if original_script_path else "",
                "initstate": initstate,
                "original_script": original_script,
            },
        )

    def _mock_episode(self, request: EpisodeRequest) -> SimulatorEpisode:
        human = "human_0"
        robot = "robot_0"
        events: list[EventStep] = []
        carried: list[str] = []

        for step_index, action in enumerate(request.scripted_actions):
            match = ACTION_RE.search(action)
            verb = (match.group("verb") if match else "Act").strip()
            target = (match.group("target") if match and match.group("target") else "environment").strip()
            if verb.lower() in {"grab", "take"} and target not in carried:
                carried.append(target)
            if verb.lower() in {"put", "give", "drop"} and target in carried:
                carried.remove(target)

            objects = [
                ObjectState(
                    object_id=f"{target}_{step_index}",
                    name=target,
                    location="scene_anchor" if target not in carried else human,
                    state="visible",
                    attributes={"mentioned_by_action": True},
                ),
                ObjectState(
                    object_id=f"goal_{step_index}",
                    name=request.task_name,
                    location="task_context",
                    state="active",
                    attributes={"goal": request.natural_language_goal},
                ),
            ]
            observation = FrameObservation(
                frame_index=step_index,
                timestamp=float(step_index),
                visible_objects=objects,
                agent_states=[
                    AgentState(name=human, role="human", location=request.scene_id, holding=list(carried)),
                    AgentState(name=robot, role="robot", location=request.scene_id, holding=[]),
                ],
                metadata={"dry_run": True, "simulator": self.name},
            )
            events.append(
                EventStep(
                    step_index=step_index,
                    action=action,
                    actor=human,
                    target=target,
                    success=True,
                    observation=observation,
                    metadata={"verb": verb, "dry_run": True},
                )
            )

        return SimulatorEpisode(
            episode_id=request.request_id,
            simulator=self.name,
            scene_id=request.scene_id,
            task_name=request.task_name,
            natural_language_goal=request.natural_language_goal,
            agent_names=[human, robot],
            scripted_actions=list(request.scripted_actions),
            events=events,
            metadata={
                "dry_run": True,
                "package_root": self.package_root,
                "unity_binary": self.unity_binary,
            },
        )

    @staticmethod
    def _first_existing(*paths: Path) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None

    @staticmethod
    def _read_script_lines(path: Path | None) -> list[str]:
        if path is None or not path.exists():
            return []
        lines: list[str] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "[" not in line or "]" not in line:
                    continue
                lines.append(line)
        return lines

    @staticmethod
    def _derive_task_name(scene_id: str, stem: str) -> str:
        base = stem.lower().replace("-", "_").replace(" ", "_")
        base = re.sub(r"[^a-z0-9_]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("_")
        if not base:
            base = f"{scene_id.lower()}_activity"
        return base

    @staticmethod
    def _parse_action(action: str) -> tuple[str, str]:
        match = ACTION_RE.search(action or "")
        if not match:
            return "Act", "environment"
        verb = (match.group("verb") or "Act").strip()
        target = (match.group("target") or "environment").strip()
        return verb, target

    @staticmethod
    def _load_optional_jsonish(path: Path | None) -> Any:
        if path is None or not path.exists():
            return None
        try:
            with path.open() as f:
                return json.load(f)
        except Exception:
            with path.open() as f:
                return f.read()

    def _load_state_sequence(self, path: Path) -> list[dict[str, Any]]:
        with path.open() as f:
            raw = json.load(f)

        if isinstance(raw, list):
            return [self._coerce_state_graph(item) for item in raw]
        if isinstance(raw, dict):
            for key in ("graph_state_list", "state_list", "graphs", "states"):
                value = raw.get(key)
                if isinstance(value, list):
                    return [self._coerce_state_graph(item) for item in value]
            return [self._coerce_state_graph(raw)]
        return []

    @staticmethod
    def _coerce_state_graph(raw_state: Any) -> dict[str, Any]:
        if isinstance(raw_state, dict):
            if "nodes" in raw_state and "edges" in raw_state:
                return raw_state
            for key in ("graph", "environment_graph", "state"):
                value = raw_state.get(key)
                if isinstance(value, dict) and "nodes" in value:
                    return value
        return {"nodes": [], "edges": [], "raw_state": raw_state}

    def _state_to_observation(
        self,
        *,
        raw_state: dict[str, Any] | None,
        step_index: int,
        fallback_scene: str,
        fallback_action: str,
    ) -> FrameObservation:
        if raw_state is None:
            verb, target = self._parse_action(fallback_action)
            return FrameObservation(
                frame_index=step_index,
                timestamp=float(step_index),
                visible_objects=[
                    ObjectState(
                        object_id=f"{target}_{step_index}",
                        name=target,
                        location=fallback_scene,
                        state="mentioned_by_action",
                        attributes={"fallback": True, "verb": verb},
                    )
                ],
                agent_states=[
                    AgentState(name="character_0", role="human", location=fallback_scene, holding=[]),
                ],
                metadata={"fallback_state": True},
            )

        nodes = raw_state.get("nodes", []) or []
        edges = raw_state.get("edges", []) or []
        id_to_node = {node.get("id"): node for node in nodes if isinstance(node, dict)}
        id_to_name = {
            node_id: self._node_name(node)
            for node_id, node in id_to_node.items()
        }

        visible_objects: list[ObjectState] = []
        agent_states: list[AgentState] = []

        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if self._is_agent_node(node):
                holding = self._holding_for_agent(node_id, edges, id_to_name)
                location = self._location_for_node(node_id, node, edges, id_to_name, fallback_scene)
                role = self._infer_agent_role(node)
                agent_states.append(
                    AgentState(
                        name=self._node_name(node),
                        role=role,
                        location=location,
                        holding=holding,
                        facing=self._facing_for_agent(node_id, edges, id_to_name),
                        metadata={"node_id": node_id},
                    )
                )
                continue
            if self._is_room_node(node):
                continue
            visible_objects.append(
                ObjectState(
                    object_id=str(node_id),
                    name=self._node_name(node),
                    location=self._location_for_node(node_id, node, edges, id_to_name, fallback_scene),
                    state=self._state_label(node),
                    attributes={
                        "class_name": node.get("class_name", ""),
                        "category": node.get("category", ""),
                        "properties": node.get("properties", []),
                    },
                )
            )

        if not agent_states:
            agent_states.append(
                AgentState(name="character_0", role="human", location=fallback_scene, holding=[])
            )

        return FrameObservation(
            frame_index=step_index,
            timestamp=float(step_index),
            visible_objects=visible_objects[:40],
            agent_states=agent_states,
            metadata={
                "source": "virtualhome_export",
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "world_graph": {
                    "nodes": nodes,
                    "edges": edges,
                },
            },
        )

    @staticmethod
    def _node_name(node: dict[str, Any]) -> str:
        for key in ("class_name", "prefab_name", "name", "category"):
            value = node.get(key)
            if value:
                return str(value)
        return f"node_{node.get('id', 'unknown')}"

    @staticmethod
    def _is_agent_node(node: dict[str, Any]) -> bool:
        hay = " ".join(
            str(node.get(key, "")) for key in ("class_name", "category", "prefab_name", "name")
        ).lower()
        return any(token in hay for token in ("character", "agent", "human", "person", "robot"))

    @staticmethod
    def _is_room_node(node: dict[str, Any]) -> bool:
        hay = " ".join(str(node.get(key, "")) for key in ("class_name", "category", "name")).lower()
        return any(token in hay for token in ("room", "kitchen", "bedroom", "bathroom", "livingroom", "living_room"))

    @staticmethod
    def _infer_agent_role(node: dict[str, Any]) -> str:
        hay = " ".join(str(node.get(key, "")) for key in ("class_name", "category", "name")).lower()
        if "robot" in hay:
            return "robot"
        return "human"

    @staticmethod
    def _state_label(node: dict[str, Any]) -> str:
        states = node.get("states")
        if isinstance(states, list) and states:
            return "|".join(str(state) for state in states)
        return "visible"

    @staticmethod
    def _holding_for_agent(node_id: Any, edges: list[dict[str, Any]], id_to_name: dict[Any, str]) -> list[str]:
        held: list[str] = []
        for edge in edges:
            if edge.get("from_id") == node_id and str(edge.get("relation_type", "")).upper() in HOLD_RELATIONS:
                held.append(id_to_name.get(edge.get("to_id"), str(edge.get("to_id"))))
        return held

    @staticmethod
    def _facing_for_agent(node_id: Any, edges: list[dict[str, Any]], id_to_name: dict[Any, str]) -> str | None:
        for edge in edges:
            if edge.get("from_id") == node_id and str(edge.get("relation_type", "")).upper() == "FACING":
                return id_to_name.get(edge.get("to_id"), str(edge.get("to_id")))
        return None

    @staticmethod
    def _location_for_node(
        node_id: Any,
        node: dict[str, Any],
        edges: list[dict[str, Any]],
        id_to_name: dict[Any, str],
        fallback_scene: str,
    ) -> str:
        for edge in edges:
            relation = str(edge.get("relation_type", "")).upper()
            if edge.get("from_id") == node_id and relation in LOCATION_RELATIONS:
                return id_to_name.get(edge.get("to_id"), fallback_scene)
            if edge.get("to_id") == node_id and relation in {"INSIDE", "ON"}:
                return id_to_name.get(edge.get("from_id"), fallback_scene)
        room = node.get("room")
        if room:
            return str(room)
        return fallback_scene
