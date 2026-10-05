#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.config import default_paths
from mindpower.io_utils import ensure_dir
from mindpower.simulators.virtualhome_adapter import VirtualHomeAdapter


def parse_camera_ids(value: str) -> list[int]:
    camera_ids: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        camera_ids.append(int(chunk))
    if not camera_ids:
        raise ValueError("No camera IDs provided.")
    return camera_ids


def load_script_lines(script_path: Path) -> list[str]:
    with script_path.open() as f:
        return [line.strip() for line in f if line.strip()]


def infer_scene_id(script_path: Path | None, explicit_scene_id: int | None) -> int:
    if explicit_scene_id is not None:
        return explicit_scene_id
    if script_path is None:
        raise ValueError("Provide --scene_id when --script_path is omitted.")
    for part in script_path.parts:
        if part.isdigit():
            return int(part)
    raise ValueError(
        f"Could not infer scene_id from {script_path}. "
        "Pass --scene_id explicitly."
    )


def resolve_script_path(
    *,
    script_path_arg: str | None,
    dataset_root_arg: str | None,
) -> Path | None:
    if script_path_arg is None:
        return None

    raw_path = Path(script_path_arg)
    if raw_path.exists():
        return raw_path.resolve()

    if dataset_root_arg is None:
        raise FileNotFoundError(
            f"Could not find script at {raw_path}. "
            "Pass an absolute path or use --dataset_root with a path relative to executable_programs/."
        )

    adapter = VirtualHomeAdapter()
    export_root = adapter.discover_export_root(dataset_root_arg)
    candidate = export_root / "executable_programs" / raw_path
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve script {raw_path} under {export_root / 'executable_programs'}."
    )


def save_camera_image(output_path: Path, image) -> None:
    if isinstance(image, (list, tuple)):
        image = image[0]
    cv2.imwrite(str(output_path), image)


def dump_camera_views(comm, camera_ids: Iterable[int], output_dir: Path, prefix: str) -> list[Path]:
    written: list[Path] = []
    for camera_id in camera_ids:
        ok, image = comm.camera_image([camera_id])
        if not ok or image is None:
            print(f"[warn] Camera {camera_id} unavailable; skipped.")
            continue
        output_path = output_dir / f"{prefix}_cam_{camera_id:03d}.png"
        save_camera_image(output_path, image)
        written.append(output_path)
    return written


def execute_script_best_effort(comm, script_lines: list[str], script_path: Path | None) -> bool:
    errors: list[str] = []

    if script_path is not None and hasattr(comm, "render_script_from_path"):
        try:
            result = comm.render_script_from_path(str(script_path))
            return bool(result[0] if isinstance(result, tuple) else result)
        except Exception as exc:  # pragma: no cover - depends on installed VirtualHome version
            errors.append(f"render_script_from_path: {exc}")

    if hasattr(comm, "render_script"):
        candidate_calls = [
            {"script": script_lines, "processing_time_limit": 60, "find_solution": False},
            {"script": script_lines, "processing_time_limit": 60},
            {"script": script_lines},
        ]
        for kwargs in candidate_calls:
            try:
                result = comm.render_script(**kwargs)
                return bool(result[0] if isinstance(result, tuple) else result)
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - depends on installed VirtualHome version
                errors.append(f"render_script({sorted(kwargs)}): {exc}")

    if errors:
        print("[warn] Script execution did not succeed with the local VirtualHome API:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("[warn] No compatible render_script method found on UnityCommunication.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render camera views from a VirtualHome scene or executable program. "
            "This is useful once MindPower-style VirtualHome scripts are available."
        )
    )
    parser.add_argument(
        "--script_path",
        default=None,
        help=(
            "Path to an executable_programs/*.txt script. "
            "If relative, resolve it under --dataset_root/executable_programs/."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Optional VirtualHome export root used to resolve a relative --script_path.",
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=None,
        help="VirtualHome scene id. If omitted, try to infer it from the script path.",
    )
    parser.add_argument(
        "--unity_binary",
        required=True,
        help="Path to the VirtualHome Unity executable, e.g. VirtualHome.exe.",
    )
    parser.add_argument("--port", default="8080")
    parser.add_argument(
        "--camera_ids",
        default="0",
        help="Comma-separated camera ids to render, e.g. 0,1,2,10.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(default_paths()["logs_dir"] / "virtualhome_renders"),
    )
    parser.add_argument(
        "--execute_script",
        action="store_true",
        help="Best-effort attempt to execute the script before dumping post-execution views.",
    )
    parser.add_argument(
        "--dump_before_execution",
        action="store_true",
        help="Dump camera views immediately after scene reset.",
    )
    parser.add_argument(
        "--dump_after_execution",
        action="store_true",
        help="Dump camera views after script execution. Enabled automatically with --execute_script.",
    )
    args = parser.parse_args()

    script_path = resolve_script_path(
        script_path_arg=args.script_path,
        dataset_root_arg=args.dataset_root,
    )
    script_lines = load_script_lines(script_path) if script_path is not None else []
    scene_id = infer_scene_id(script_path, args.scene_id)
    camera_ids = parse_camera_ids(args.camera_ids)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    try:
        from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "VirtualHome is not importable in this environment. "
            "Install the VirtualHome Python package/repo in the target environment first."
        ) from exc

    comm = UnityCommunication(file_name=args.unity_binary, port=args.port)
    wrote_paths: list[Path] = []
    try:
        ok = comm.reset(scene_id)
        if isinstance(ok, tuple):
            ok = ok[0]
        if not ok:
            raise RuntimeError(f"Failed to reset scene {scene_id}.")

        if args.dump_before_execution:
            wrote_paths.extend(dump_camera_views(comm, camera_ids, output_dir, prefix="before"))

        executed = False
        if args.execute_script:
            if not script_lines:
                raise ValueError("--execute_script requires a valid --script_path.")
            executed = execute_script_best_effort(comm, script_lines, script_path)

        if args.dump_after_execution or executed:
            wrote_paths.extend(dump_camera_views(comm, camera_ids, output_dir, prefix="after"))
    finally:
        comm.close()

    print(f"Scene id: {scene_id}")
    if script_path is not None:
        print(f"Script: {script_path}")
        print(f"Script actions: {len(script_lines)}")
    print(f"Camera ids: {camera_ids}")
    print(f"Wrote {len(wrote_paths)} image(s) to {output_dir}")
    for path in wrote_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
