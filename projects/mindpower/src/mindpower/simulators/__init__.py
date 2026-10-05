from __future__ import annotations

from mindpower.simulators.tdw_adapter import TDWAdapter
from mindpower.simulators.virtualhome_adapter import VirtualHomeAdapter


def get_simulator_adapter(
    name: str,
    *,
    virtualhome_root: str | None = None,
    virtualhome_binary: str | None = None,
    tdw_build_path: str | None = None,
):
    normalized = name.lower()
    if normalized == "virtualhome":
        return VirtualHomeAdapter(package_root=virtualhome_root, unity_binary=virtualhome_binary)
    if normalized == "tdw":
        return TDWAdapter(build_path=tdw_build_path)
    raise ValueError(f"Unsupported simulator: {name}")
