from __future__ import annotations

import importlib.util
from pathlib import Path

from mindpower.schemas import EpisodeRequest, SimulatorEpisode
from mindpower.simulators.base import EmbodiedSimulatorAdapter
from mindpower.simulators.virtualhome_adapter import VirtualHomeAdapter


class TDWAdapter(EmbodiedSimulatorAdapter):
    name = "tdw"

    def __init__(self, build_path: str | None = None):
        self.build_path = build_path

    def is_available(self) -> bool:
        if self.build_path and Path(self.build_path).exists():
            return True
        return importlib.util.find_spec("tdw") is not None

    def collect_episode(self, request: EpisodeRequest, output_dir: str | Path) -> SimulatorEpisode:
        if request.dry_run:
            surrogate = VirtualHomeAdapter()
            episode = surrogate._mock_episode(request)
            episode.simulator = self.name
            episode.metadata["tdw_surrogate"] = True
            return episode
        raise NotImplementedError(
            "TDW collection is scaffolded but not wired yet. "
            "Next step: create a TDW controller, spawn the task scene, record object "
            "states and actions, and serialize them into the shared episode schema."
        )
