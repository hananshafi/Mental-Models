#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
PYTHON = Path(sys.executable)
STAGE1_WRAPPER = ROOT / "scripts" / "run_stage1_fraction.py"
STAGE2_WRAPPER = ROOT / "scripts" / "run_stage2_fraction.py"
STAGE2_SOURCE = ROOT / "scripts" / "stage2_grpo_agent_training_v3_utf8.py"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
STATUS_PATH = ROOT / "runs" / "status.json"
SFT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "grpo_agent_qwen_v3" / "sft_warmup"
)
MODEL_NAME = os.environ.get(
    "MENTAL_MODELS_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(status):
    status["updated_at_utc"] = now()
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(STATUS_PATH)


def process_environment(gpus):
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": gpus})
    environment.setdefault(
        "HF_HOME", str(REPO_ROOT / "artifacts" / "huggingface")
    )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    return environment


def launch_process(name, command, log_path, gpus, status):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a")
    log_handle.write(f"\n[{now()}] Launching: {' '.join(command)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=process_environment(gpus),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    status["jobs"][name] = {
        "pid": process.pid,
        "state": "running",
        "gpus": gpus,
        "log": str(log_path),
        "started_at_utc": now(),
    }
    write_status(status)
    return process, log_handle


def wait_for_jobs(jobs, status):
    failures = []
    for name, process, log_handle in jobs:
        return_code = process.wait()
        log_handle.close()
        status["jobs"][name]["return_code"] = return_code
        status["jobs"][name]["state"] = (
            "completed" if return_code == 0 else "failed"
        )
        status["jobs"][name]["finished_at_utc"] = now()
        if return_code != 0:
            failures.append((name, return_code))
        write_status(status)
    if failures:
        raise RuntimeError(f"Training jobs failed: {failures}")


def gpu_memory_mib():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory = {}
    for line in result.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",", maxsplit=1)]
        memory[int(index)] = int(used)
    return memory


def wait_for_gpus(indices, threshold_mib=1024):
    while True:
        memory = gpu_memory_mib()
        if all(memory.get(index, threshold_mib + 1) <= threshold_mib for index in indices):
            return
        print(
            f"[{now()}] Waiting for GPUs {indices}; memory={memory}",
            flush=True,
        )
        time.sleep(60)


def stage1_command(label, condition, protocol, gpu):
    output_dir = ROOT / "runs" / f"fraction_{label}" / "stage1"
    return [
        str(PYTHON),
        str(STAGE1_WRAPPER),
        "--stage1_samples",
        str(condition["annotation_stats"]["valid_stage1_turns"]),
        "--max_opt_steps",
        str(protocol["stage1_target_optimizer_steps"]),
        "--scheduler_total_steps",
        str(protocol["stage1_scheduler_total_steps"]),
        "--scheduler_warmup_steps",
        str(protocol["stage1_scheduler_warmup_steps"]),
        "--fraction_label",
        label,
        "--checkpoint_policy",
        "best",
        "--model_name",
        MODEL_NAME,
        "--data_path",
        condition["data_path"],
        "--output_dir",
        str(output_dir),
        "--batch_size",
        "4",
        "--grad_accum_steps",
        "8",
        "--lr",
        "2e-5",
        "--warmup_ratio",
        "0.03",
        "--max_ctx_len",
        "1024",
        "--max_resp_len",
        "256",
        "--z_dim",
        "128",
        "--lora_r",
        "16",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.05",
        "--num_lora_layers",
        "16",
        "--kl_weight",
        "0.1",
        "--future_weight",
        "0.5",
        "--mental1_weight",
        "0.3",
        "--mental2_weight",
        "0.2",
        "--expl_weight",
        "0.3",
        "--z_only_weight",
        "0.5",
        "--kl_anneal_steps",
        "200",
        "--z2_kl_delay_steps",
        "100",
        "--z2_warmup_steps",
        "100",
        "--mental_prewarm_data",
        condition["prewarm_path"],
        "--mental_prewarm_epochs",
        str(condition["prewarm_epochs"]),
        "--mental_prewarm_lr",
        "2e-4",
        "--head_lr_mult",
        "10.0",
        "--max_grad_norm",
        "5.0",
        "--val_ratio",
        "0",
        "--num_workers",
        "4",
        "--seed",
        "42",
        "--gpu",
        gpu,
    ]


def stage2_command(label, manifest, gpu_pair):
    reward_checkpoint = (
        ROOT / "runs" / f"fraction_{label}" / "stage1" / "best"
    )
    output_dir = ROOT / "runs" / f"fraction_{label}" / "stage2"
    return [
        str(PYTHON),
        str(STAGE2_WRAPPER),
        "--source_script",
        str(STAGE2_SOURCE),
        "--max_steps",
        str(manifest["protocol"]["stage2_target_grpo_steps"]),
        "--fraction_label",
        label,
        "--policy_model_name",
        MODEL_NAME,
        "--reward_model_name",
        MODEL_NAME,
        "--reward_checkpoint_dir",
        str(reward_checkpoint),
        "--reward_version",
        "v3",
        "--data_path",
        manifest["stage2_base_trajectories"]["path"],
        "--output_dir",
        str(output_dir),
        "--preset",
        "qwen",
        "--sft_checkpoint",
        str(SFT_CHECKPOINT),
        "--prompts_per_step",
        "4",
        "--max_gen_len",
        "256",
        "--max_ctx_len",
        "1024",
        "--temperature",
        "0.8",
        "--top_p",
        "0.95",
        "--save_every",
        str(manifest["protocol"]["stage2_target_grpo_steps"]),
        "--seed",
        "42",
        "--gpu",
        gpu_pair,
    ]


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}; run scripts/prepare_data.py first"
        )
    if STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text())
        if previous.get("state") in {"running", "completed"}:
            raise RuntimeError(
                f"Existing experiment state is {previous['state']}; "
                "refusing to launch a duplicate"
            )

    manifest = json.loads(MANIFEST_PATH.read_text())
    status = {
        "experiment": "sotopia_augmented_supervision_fraction",
        "state": "running",
        "started_at_utc": now(),
        "jobs": {},
        "conditions": ["0", "25", "50", "100"],
    }
    write_status(status)

    try:
        stage1_jobs = []
        for label, gpu in (("25", "0"), ("50", "1")):
            command = stage1_command(
                label,
                manifest["conditions"][label],
                manifest["protocol"],
                gpu,
            )
            process, log_handle = launch_process(
                f"stage1_{label}",
                command,
                ROOT / "logs" / f"stage1_{label}.log",
                gpu,
                status,
            )
            stage1_jobs.append((f"stage1_{label}", process, log_handle))
        wait_for_jobs(stage1_jobs, status)

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
