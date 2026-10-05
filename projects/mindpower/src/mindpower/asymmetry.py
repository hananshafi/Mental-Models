from __future__ import annotations

from typing import Any


_HOLD_RELATIONS = {"HOLDS_RH", "HOLDS_LH", "HOLDS"}
_PROXIMITY_RELATIONS = {"CLOSE", "CLOSETO", "NEAR"}
_VISIBILITY_PARENT_RELATIONS = {"INSIDE", "ON"}
_OPEN_STATES = {"OPEN", "OPENED"}
_CLOSED_STATES = {"CLOSED", "SHUT"}


def _normalize_world_graph(world_graph: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(world_graph, dict):
        return {"nodes": [], "edges": []}
    nodes = world_graph.get("nodes", [])
    edges = world_graph.get("edges", [])
    return {
        "nodes": [node for node in nodes if isinstance(node, dict)],
        "edges": [edge for edge in edges if isinstance(edge, dict)],
    }


def _node_name(node: dict[str, Any]) -> str:
    for key in ("class_name", "prefab_name", "name", "category"):
        value = node.get(key)
        if value:
            return str(value)
    return f"node_{node.get('id', 'unknown')}"


def _state_tokens(node: dict[str, Any]) -> set[str]:
    states = node.get("states")
    if isinstance(states, list):
        return {str(state).upper() for state in states}
    if isinstance(states, str):
        return {states.upper()}
    return set()


def _is_room_node(node: dict[str, Any]) -> bool:
    hay = " ".join(str(node.get(key, "")) for key in ("class_name", "category", "name")).lower()
    return any(token in hay for token in ("room", "kitchen", "bedroom", "bathroom", "livingroom", "living_room"))


def _is_agent_node(node: dict[str, Any]) -> bool:
    hay = " ".join(
        str(node.get(key, "")) for key in ("class_name", "category", "prefab_name", "name")
    ).lower()
    return any(token in hay for token in ("character", "agent", "human", "person", "robot"))


def _infer_role(node: dict[str, Any]) -> str:
    hay = " ".join(str(node.get(key, "")) for key in ("class_name", "category", "name")).lower()
    if "robot" in hay:
        return "robot"
    return "human"


def _build_index(world_graph: dict[str, Any] | None) -> dict[str, Any]:
    graph = _normalize_world_graph(world_graph)
    id_to_node = {node.get("id"): node for node in graph["nodes"]}
    return {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "id_to_node": id_to_node,
    }


def _find_agent_node_id(
    index: dict[str, Any],
    *,
    agent_name: str | None,
    preferred_role: str | None,
) -> Any | None:
    needle = (agent_name or "").strip().lower()
    exact_match = None
    role_match = None
    fallback = None
    for node in index["nodes"]:
        if not _is_agent_node(node):
            continue
        node_id = node.get("id")
        if fallback is None:
            fallback = node_id
        if preferred_role and _infer_role(node) == preferred_role and role_match is None:
            role_match = node_id
        if needle:
            names = {
                str(node.get(key, "")).strip().lower()
                for key in ("class_name", "prefab_name", "name", "category")
                if node.get(key)
            }
            if needle in names:
                exact_match = node_id
                break
    return exact_match or role_match or fallback


def _relation_targets(index: dict[str, Any], node_id: Any, relations: set[str]) -> set[Any]:
    targets: set[Any] = set()
    if node_id is None:
        return targets
    for edge in index["edges"]:
        relation = str(edge.get("relation_type", "")).upper()
        if relation not in relations:
            continue
        if edge.get("from_id") == node_id:
            targets.add(edge.get("to_id"))
        elif edge.get("to_id") == node_id:
            targets.add(edge.get("from_id"))
    return targets


def _held_object_ids(index: dict[str, Any], agent_node_id: Any) -> set[Any]:
    return _relation_targets(index, agent_node_id, _HOLD_RELATIONS)


def _parent_chain(index: dict[str, Any], node_id: Any) -> list[Any]:
    parents: list[Any] = []
    current = node_id
    seen = {node_id}
    while current is not None:
        parent = None
        for edge in index["edges"]:
            relation = str(edge.get("relation_type", "")).upper()
            if relation not in _VISIBILITY_PARENT_RELATIONS:
                continue
            if edge.get("from_id") == current:
                parent = edge.get("to_id")
                break
            if edge.get("to_id") == current:
                parent = edge.get("from_id")
                break
        if parent is None or parent in seen:
            break
        parents.append(parent)
        seen.add(parent)
        current = parent
    return parents


def _room_for_node(index: dict[str, Any], node_id: Any, fallback_scene: str) -> str:
    node = index["id_to_node"].get(node_id)
    if node is None:
        return fallback_scene
    if _is_room_node(node):
        return _node_name(node)
    room = node.get("room")
    if room:
        return str(room)
    for parent_id in _parent_chain(index, node_id):
        parent = index["id_to_node"].get(parent_id)
        if parent and _is_room_node(parent):
            return _node_name(parent)
    return fallback_scene


def _location_label(index: dict[str, Any], node_id: Any, fallback_scene: str) -> str:
    parents = _parent_chain(index, node_id)
    if parents:
        parent = index["id_to_node"].get(parents[0])
        if parent:
            return _node_name(parent)
    return _room_for_node(index, node_id, fallback_scene)


def _hidden_in_closed_container(index: dict[str, Any], node_id: Any, held_ids: set[Any]) -> bool:
    if node_id in held_ids:
        return False
    for parent_id in _parent_chain(index, node_id):
        parent = index["id_to_node"].get(parent_id)
        if parent is None or _is_room_node(parent):
            continue
        states = _state_tokens(parent)
        if _CLOSED_STATES & states and not (_OPEN_STATES & states):
            return True
    return False


def _make_object_record(
    index: dict[str, Any],
    node_id: Any,
    *,
    fallback_scene: str,
    visibility_reasons: list[str],
    observer: str,
) -> dict[str, Any]:
    node = index["id_to_node"][node_id]
    container_path = []
    for parent_id in _parent_chain(index, node_id):
        parent = index["id_to_node"].get(parent_id)
        if parent is None or _is_room_node(parent):
            continue
        container_path.append(_node_name(parent))
    states = node.get("states")
    if isinstance(states, list) and states:
        state = "|".join(str(item) for item in states)
    else:
        state = "visible"
    return {
        "object_id": str(node_id),
        "name": _node_name(node),
        "location": _location_label(index, node_id, fallback_scene),
        "state": state,
        "attributes": {
            "class_name": node.get("class_name", ""),
            "category": node.get("category", ""),
            "properties": node.get("properties", []),
            "room": _room_for_node(index, node_id, fallback_scene),
            "container_path": container_path,
            "visibility_reasons": list(visibility_reasons),
            "observer": observer,
        },
    }


def derive_agent_visible_objects(
    world_graph: dict[str, Any] | None,
    *,
    agent_name: str | None,
    preferred_role: str | None = None,
    fallback_scene: str,
    mode: str = "masked",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _build_index(world_graph)
    if not index["nodes"]:
        return [], {"agent_node_id": None, "room": fallback_scene, "mode": "missing_graph"}

    agent_node_id = _find_agent_node_id(index, agent_name=agent_name, preferred_role=preferred_role)
    agent_room = _room_for_node(index, agent_node_id, fallback_scene) if agent_node_id is not None else fallback_scene
    facing_ids = _relation_targets(index, agent_node_id, {"FACING"})
    nearby_ids = _relation_targets(index, agent_node_id, _PROXIMITY_RELATIONS)
    held_ids = _held_object_ids(index, agent_node_id)

    objects: list[dict[str, Any]] = []
    for node in index["nodes"]:
        if _is_agent_node(node) or _is_room_node(node):
            continue
        node_id = node.get("id")
        if node_id is None:
            continue

        reasons: list[str] = []
        if mode == "god_view":
            reasons.append("god_view")
        else:
            node_room = _room_for_node(index, node_id, fallback_scene)
            if node_id in held_ids:
                reasons.append("held_by_agent")
            elif node_room != agent_room:
                continue
            else:
                reasons.append("same_room")

            if _hidden_in_closed_container(index, node_id, held_ids):
                continue

            if node_id in facing_ids:
                reasons.append("facing")
            if node_id in nearby_ids:
                reasons.append("near")

        objects.append(
            _make_object_record(
                index,
                node_id,
                fallback_scene=fallback_scene,
                visibility_reasons=reasons,
                observer=agent_name or preferred_role or "observer",
            )
        )

    objects.sort(key=lambda item: (item["location"], item["name"], item["object_id"]))
    return objects[:40], {
        "agent_node_id": agent_node_id,
        "room": agent_room,
        "mode": mode,
    }


def _fallback_view(objects: list[dict[str, Any]], observer: str) -> list[dict[str, Any]]:
    copied = []
    for obj in objects[:40]:
        item = dict(obj)
        attributes = dict(item.get("attributes", {}))
        attributes["observer"] = observer
        attributes.setdefault("visibility_reasons", ["fallback_shared_view"])
        item["attributes"] = attributes
        copied.append(item)
    return copied


def _unique_names(objects: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for obj in objects:
        name = str(obj.get("name", "object"))
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def derive_world_objects(
    world_graph: dict[str, Any] | None,
    *,
    fallback_scene: str,
    fallback_visible_objects: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    fallback_visible_objects = list(fallback_visible_objects or [])
    index = _build_index(world_graph)
    if not index["nodes"]:
        return _fallback_view(fallback_visible_objects, "world_state")

    objects: list[dict[str, Any]] = []
    for node in index["nodes"]:
        if _is_agent_node(node) or _is_room_node(node):
            continue
        node_id = node.get("id")
        if node_id is None:
            continue
        objects.append(
            _make_object_record(
                index,
                node_id,
                fallback_scene=fallback_scene,
                visibility_reasons=["world_graph"],
                observer="world_state",
            )
        )
    objects.sort(key=lambda item: (item["location"], item["name"], item["object_id"]))
    return objects[:40]


def _clone_record(record: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(record)
    attributes = record.get("attributes")
    if isinstance(attributes, dict):
        cloned["attributes"] = dict(attributes)
    return cloned


def roll_belief_state(
    previous_state: list[dict[str, Any]] | None,
    observed_objects: list[dict[str, Any]],
    *,
    step_index: int,
    observer: str,
) -> list[dict[str, Any]]:
    previous_state = list(previous_state or [])
    belief_index: dict[str, dict[str, Any]] = {}

    for record in previous_state:
        cloned = _clone_record(record)
        last_seen_step = int(cloned.get("last_seen_step", step_index))
        cloned["currently_visible"] = False
        cloned["staleness_steps"] = max(0, step_index - last_seen_step)
        cloned["belief_status"] = "memory" if cloned["staleness_steps"] > 0 else "fresh"
        belief_index[str(cloned.get("object_id"))] = cloned

    for obj in observed_objects:
        object_id = str(obj.get("object_id"))
        attributes = dict(obj.get("attributes", {}))
        belief_index[object_id] = {
            "object_id": object_id,
            "name": obj.get("name", "object"),
            "believed_location": obj.get("location", "unknown"),
            "believed_state": obj.get("state", "unknown"),
            "room": attributes.get("room", obj.get("location", "unknown")),
            "container_path": list(attributes.get("container_path", [])),
            "last_seen_step": step_index,
            "staleness_steps": 0,
            "currently_visible": True,
            "belief_status": "direct_observation",
            "observer": observer,
        }

    rolled = list(belief_index.values())
    rolled.sort(key=lambda item: (item.get("name", ""), item.get("object_id", "")))
    return rolled


def derive_belief_gap_objects(
    *,
    world_objects: list[dict[str, Any]],
    belief_state: list[dict[str, Any]],
    agent_visible_objects: list[dict[str, Any]],
    counterpart_visible_objects: list[dict[str, Any]] | None = None,
    agent_label: str = "human",
) -> list[dict[str, Any]]:
    world_index = {str(obj.get("object_id")): obj for obj in world_objects}
    belief_index = {str(obj.get("object_id")): obj for obj in belief_state}
    agent_visible_ids = {str(obj.get("object_id")) for obj in agent_visible_objects}
    counterpart_visible_ids = {
        str(obj.get("object_id")) for obj in (counterpart_visible_objects or [])
    }

    gaps: list[dict[str, Any]] = []
    for object_id, belief in belief_index.items():
        world = world_index.get(object_id)
        if world is None:
            continue
        mismatch_fields: list[str] = []
        if belief.get("believed_location") != world.get("location"):
            mismatch_fields.append("location")
        if belief.get("believed_state") != world.get("state"):
            mismatch_fields.append("state")
        if not mismatch_fields:
            continue
        name = str(world.get("name", belief.get("name", "object")))
        explanation = (
            f"{name}: {agent_label} still believes location={belief.get('believed_location')} "
            f"state={belief.get('believed_state')}, but world is location={world.get('location')} "
            f"state={world.get('state')}."
        )
        gaps.append(
            {
                "object_id": object_id,
                "name": name,
                "gap_type": "stale_belief",
                "mismatch_fields": mismatch_fields,
                "believed_location": belief.get("believed_location", "unknown"),
                "believed_state": belief.get("believed_state", "unknown"),
                "world_location": world.get("location", "unknown"),
                "world_state": world.get("state", "unknown"),
                "currently_visible_to_agent": object_id in agent_visible_ids,
                "currently_visible_to_counterpart": object_id in counterpart_visible_ids,
                "staleness_steps": int(belief.get("staleness_steps", 0)),
                "explanation": explanation,
            }
        )

    for obj in counterpart_visible_objects or []:
        object_id = str(obj.get("object_id"))
        if object_id in agent_visible_ids or object_id in belief_index:
            continue
        gaps.append(
            {
                "object_id": object_id,
                "name": obj.get("name", "object"),
                "gap_type": "unobserved_by_agent",
                "mismatch_fields": ["unknown"],
                "believed_location": "unknown",
                "believed_state": "unknown",
                "world_location": obj.get("location", "unknown"),
                "world_state": obj.get("state", "unknown"),
                "currently_visible_to_agent": False,
                "currently_visible_to_counterpart": True,
                "staleness_steps": None,
                "explanation": (
                    f"{obj.get('name', 'object')}: visible to the counterpart but not currently observable "
                    f"to the {agent_label}, so the {agent_label}'s belief may be incomplete."
                ),
            }
        )

    gaps.sort(key=lambda item: (item.get("name", ""), item.get("gap_type", "")))
    return gaps


def summarize_belief_gap_objects(gap_objects: list[dict[str, Any]], *, max_items: int = 3) -> str:
    if not gap_objects:
        return "No explicit belief-state mismatch detected from symbolic rollout."
    explanations = [str(item.get("explanation", "")).strip() for item in gap_objects[:max_items]]
    explanations = [item for item in explanations if item]
    if not explanations:
        return "Belief-state divergence detected, but no textual summary was generated."
    return " ".join(explanations)


def build_agent_observation_views(
    world_graph: dict[str, Any] | None,
    *,
    human_agent_name: str,
    robot_agent_name: str | None,
    fallback_scene: str,
    fallback_visible_objects: list[dict[str, Any]] | None = None,
    implicit_robot_observer: bool = False,
) -> dict[str, Any]:
    fallback_visible_objects = list(fallback_visible_objects or [])
    if not _normalize_world_graph(world_graph)["nodes"]:
        human_visible = _fallback_view(fallback_visible_objects, human_agent_name or "human")
        robot_visible = _fallback_view(
            fallback_visible_objects,
            robot_agent_name or ("robot_observer" if implicit_robot_observer else "robot"),
        )
        human_meta = {"agent_node_id": None, "room": fallback_scene, "mode": "fallback_shared_view"}
        robot_meta = {"agent_node_id": None, "room": fallback_scene, "mode": "fallback_shared_view"}
        robot_view_mode = "fallback_shared_view"
    else:
        human_visible, human_meta = derive_agent_visible_objects(
            world_graph,
            agent_name=human_agent_name,
            preferred_role="human",
            fallback_scene=fallback_scene,
            mode="masked",
        )
        if implicit_robot_observer:
            robot_visible, robot_meta = derive_agent_visible_objects(
                world_graph,
                agent_name=robot_agent_name,
                preferred_role="robot",
                fallback_scene=fallback_scene,
                mode="god_view",
            )
            robot_view_mode = "implicit_god_view"
        else:
            robot_visible, robot_meta = derive_agent_visible_objects(
                world_graph,
                agent_name=robot_agent_name,
                preferred_role="robot",
                fallback_scene=fallback_scene,
                mode="masked",
            )
            robot_view_mode = "agent_masked_view"

    human_ids = {obj["object_id"] for obj in human_visible}
    robot_ids = {obj["object_id"] for obj in robot_visible}
    human_only = [obj for obj in human_visible if obj["object_id"] not in robot_ids]
    robot_only = [obj for obj in robot_visible if obj["object_id"] not in human_ids]

    human_only_names = _unique_names(human_only)
    robot_only_names = _unique_names(robot_only)
    shared_names = _unique_names(
        [obj for obj in robot_visible if obj["object_id"] in human_ids]
    )

    if robot_only_names or human_only_names:
        summary = (
            f"robot_only={robot_only_names[:5] or ['none']}; "
            f"human_only={human_only_names[:5] or ['none']}"
        )
    else:
        summary = "No agent-specific observation gap detected from the symbolic visibility filter."

    return {
        "human_visible_objects": human_visible,
        "robot_visible_objects": robot_visible,
        "human_only_objects": human_only_names,
        "robot_only_objects": robot_only_names,
        "shared_objects": shared_names,
        "human_room": human_meta["room"],
        "robot_room": robot_meta["room"],
        "human_view_mode": human_meta["mode"],
        "robot_view_mode": robot_view_mode,
        "observation_gap_summary": summary,
        "visual_asymmetry_present": bool(human_only_names or robot_only_names),
    }
