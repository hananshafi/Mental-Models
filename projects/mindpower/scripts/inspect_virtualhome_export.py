#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mindpower.simulators.virtualhome_adapter import VirtualHomeAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a VirtualHome exported-data root.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    adapter = VirtualHomeAdapter()
    export_root = adapter.discover_export_root(args.dataset_root)
    requests = adapter.build_requests_from_export_root(export_root, limit=-1)

    num_with_states = sum(1 for req in requests if req.metadata.get("state_list_path"))
    num_with_init = sum(1 for req in requests if req.metadata.get("initstate_path"))
    num_with_original = sum(1 for req in requests if req.metadata.get("original_script_path"))

    summary = {
        "export_root": str(export_root),
        "num_requests": len(requests),
        "with_state_list": num_with_states,
        "with_initstate": num_with_init,
        "with_original_script": num_with_original,
        "sample_requests": [req.to_dict() for req in requests[: args.limit]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
