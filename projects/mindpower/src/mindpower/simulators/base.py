from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mindpower.schemas import EpisodeRequest, SimulatorEpisode


class EmbodiedSimulatorAdapter(ABC):
    name: str = "simulator"

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def collect_episode(self, request: EpisodeRequest, output_dir: str | Path) -> SimulatorEpisode:
        raise NotImplementedError

    def collect_batch(self, requests: list[EpisodeRequest], output_dir: str | Path) -> list[SimulatorEpisode]:
        return [self.collect_episode(request, output_dir) for request in requests]
