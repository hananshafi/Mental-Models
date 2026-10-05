#!/usr/bin/env python3

import json
import re
from pathlib import Path

from launch import (
    MANIFEST_PATH,
    ROOT,
    STATUS_PATH,
    launch_process,
    now,
    stage2_command,
    wait_for_gpus,
    wait_for_jobs,
    write_status,
)


def selected_checkpoint(label):
    checkpoint = ROOT / "runs" / f"fraction_{label}" / "stage1" / "best"
    adapter = checkpoint / "lora_adapter" / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError(f"Missing intermediate checkpoint: {adapter}")

    log_path = ROOT / "logs" / f"stage1_{label}.log"
    log = log_path.read_text(errors="replace")
    losses = re.findall(r"Best model saved \(train_loss=([0-9.]+)\)", log)
    if not losses:
        raise RuntimeError(f"No completed best-checkpoint event in {log_path}")
    selection = {
        "checkpoint": str(checkpoint),
        "completed_best_epochs": len(losses),
        "selected_train_loss": float(losses[-1]),
        "selection": "latest completed best epoch",
    }
    selection_path = (
        ROOT
        / "runs"
        / f"fraction_{label}"
        / "stage1"
        / "selected_intermediate.json"
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return selection


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    status = (
        json.loads(STATUS_PATH.read_text())
        if STATUS_PATH.exists()
        else {
            "experiment": "sotopia_augmented_supervision_fraction",
            "jobs": {},
            "conditions": ["0", "25", "50", "100"],
            "started_at_utc": now(),
        }
    )
    status["state"] = "running"
    status.pop("error", None)
    status.pop("finished_at_utc", None)
    status["mode"] = "stage2_from_intermediate_best"
    status["intermediate_checkpoints"] = {
        label: selected_checkpoint(label) for label in ("25", "50")
    }
    for label in ("25", "50"):
        job = status["jobs"].setdefault(f"stage1_{label}", {})
        job["state"] = "stopped_at_intermediate"
        job["finished_at_utc"] = now()
    write_status(status)

    try:
        wait_for_gpus([0, 1])
        for label in ("25", "50"):
            gpu_pair = "0,1"
            command = stage2_command(label, manifest, gpu_pair)
            process, log_handle = launch_process(
                f"stage2_{label}",
                command,
                ROOT / "logs" / f"stage2_{label}.log",
                gpu_pair,
                status,
            )
            wait_for_jobs(
                [(f"stage2_{label}", process, log_handle)],
                status,
            )
        status["state"] = "completed"
        status["finished_at_utc"] = now()
        write_status(status)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        status["finished_at_utc"] = now()
        write_status(status)
        raise


if __name__ == "__main__":
    main()
