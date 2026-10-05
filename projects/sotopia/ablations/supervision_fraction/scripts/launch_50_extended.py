#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
PYTHON = Path(sys.executable)
STAGE1_SOURCE = PROJECT_ROOT / "scripts" / "stage1_train_coupled_mental_reward_v3.py"
STAGE1_WRAPPER = ROOT / "scripts" / "run_stage1_continuation.py"
STAGE2_SOURCE = (
    ROOT / "scripts" / "stage2_grpo_agent_training_v3_utf8.py"
)
STAGE2_WRAPPER = ROOT / "scripts" / "run_stage2_fraction.py"
MODEL_NAME = os.environ.get(
    "MENTAL_MODELS_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)
SFT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "grpo_agent_qwen_v3" / "sft_warmup"
)
INITIAL_REWARD_CHECKPOINT = (
    ROOT / "runs" / "fraction_50" / "stage1" / "best"
)
RUN_ROOT = ROOT / "runs" / "fraction_50_extended"
STATUS_PATH = RUN_ROOT / "status.json"
LOG_ROOT = ROOT / "logs"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(status):
    status["updated_at_utc"] = now()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
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


def run_job(name, command, log_path, gpus, status):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_handle:
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
        status["state"] = name
        write_status(status)
        return_code = process.wait()

    status["jobs"][name]["return_code"] = return_code
    status["jobs"][name]["state"] = (
        "completed" if return_code == 0 else "failed"
    )
    status["jobs"][name]["finished_at_utc"] = now()
    write_status(status)
    if return_code != 0:
        raise RuntimeError(f"{name} failed with return code {return_code}")


def stage1_command(manifest):
    condition = manifest["conditions"]["50"]
    protocol = manifest["protocol"]
    return [
        str(PYTHON),
        str(STAGE1_WRAPPER),
        "--source_script",
        str(STAGE1_SOURCE),
        "--init_checkpoint",
        str(INITIAL_REWARD_CHECKPOINT),
        "--stage1_samples",
        str(condition["annotation_stats"]["valid_stage1_turns"]),
        "--initial_opt_step",
        "237",
        "--target_opt_step",
        str(protocol["stage1_target_optimizer_steps"]),
        "--scheduler_total_steps",
        str(protocol["stage1_scheduler_total_steps"]),
        "--scheduler_warmup_steps",
        str(protocol["stage1_scheduler_warmup_steps"]),
        "--fraction_label",
        "50_extended",
        "--model_name",
        str(MODEL_NAME),
        "--data_path",
        condition["data_path"],
        "--output_dir",
        str(RUN_ROOT / "stage1"),
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
        "0",
    ]


def stage2_command(manifest):
    return [
        str(PYTHON),
        str(STAGE2_WRAPPER),
        "--source_script",
        str(STAGE2_SOURCE),
        "--max_steps",
        str(manifest["protocol"]["stage2_target_grpo_steps"]),
        "--fraction_label",
        "50_extended",
        "--policy_model_name",
        str(MODEL_NAME),
        "--reward_model_name",
        str(MODEL_NAME),
        "--reward_checkpoint_dir",
        str(RUN_ROOT / "stage1" / "final"),
        "--reward_version",
        "v3",
        "--data_path",
        manifest["stage2_base_trajectories"]["path"],
        "--output_dir",
        str(RUN_ROOT / "stage2"),
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
        "0,1",
    ]


def main():
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    if STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text())
        if previous.get("state") in {
            "stage1_50_extended",
            "stage2_50_extended",
            "completed",
        }:
            raise RuntimeError(
                f"Existing extended-run state is {previous['state']}; "
                "refusing to launch a duplicate"
            )

    status = {
        "experiment": "sotopia_50_percent_extended_stage1",
        "state": "starting",
        "started_at_utc": now(),
        "initial_reward_checkpoint": str(INITIAL_REWARD_CHECKPOINT),
        "stage1_initial_optimizer_step": 237,
        "stage1_target_optimizer_step": 500,
        "stage2_target_grpo_steps": 300,
        "jobs": {},
    }
    write_status(status)

    try:
        run_job(
            "stage1_50_extended",
            stage1_command(manifest),
            LOG_ROOT / "stage1_50_extended.log",
            "0",
            status,
        )
        run_job(
            "stage2_50_extended",
            stage2_command(manifest),
            LOG_ROOT / "stage2_50_extended.log",
            "0,1",
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
