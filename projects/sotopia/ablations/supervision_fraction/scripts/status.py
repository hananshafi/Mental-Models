#!/usr/bin/env python3

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "runs" / "status.json"


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def tail(path, lines=8):
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]


def main():
    if not STATUS_PATH.exists():
        print("Experiment has not been launched.")
        return
    status = json.loads(STATUS_PATH.read_text())
    print(
        f"Experiment: {status.get('state')} "
        f"(updated {status.get('updated_at_utc')})"
    )
    for name, job in sorted(status.get("jobs", {}).items()):
        pid = job.get("pid")
        alive = process_alive(pid)
        print(
            f"\n{name}: state={job.get('state')} pid={pid} "
            f"alive={alive} gpu={job.get('gpus')}"
        )
        for line in tail(Path(job["log"])):
            print(f"  {line}")

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("\nGPUs:")
        print(result.stdout.rstrip())


if __name__ == "__main__":
    main()
