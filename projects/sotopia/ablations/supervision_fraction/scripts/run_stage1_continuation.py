#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel


DEFAULT_STAGE1_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "stage1_train_coupled_mental_reward_v3.py"
)


def cli_value(arguments, name, default=None):
    if name not in arguments:
        return default
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError(f"Missing value for {name}")
    return arguments[index + 1]


def replace_cli_value(arguments, name, value):
    arguments = list(arguments)
    if name in arguments:
        index = arguments.index(name)
        arguments[index + 1] = str(value)
    else:
        arguments.extend([name, str(value)])
    return arguments


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "sotopia_stage1_continuation_source", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LimitedLoader:
    def __init__(self, source, batch_limit):
        self.source = source
        self.batch_limit = batch_limit

    def __iter__(self):
        return itertools.islice(iter(self.source), self.batch_limit)

    def __len__(self):
        return self.batch_limit


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--source_script", type=Path, default=DEFAULT_STAGE1_SCRIPT
    )
    custom.add_argument("--init_checkpoint", type=Path, required=True)
    custom.add_argument("--stage1_samples", type=int, required=True)
    custom.add_argument("--initial_opt_step", type=int, required=True)
    custom.add_argument("--target_opt_step", type=int, required=True)
    custom.add_argument("--scheduler_total_steps", type=int, required=True)
    custom.add_argument("--scheduler_warmup_steps", type=int, required=True)
    custom.add_argument("--fraction_label", required=True)
    custom_args, source_args = custom.parse_known_args()

    if custom_args.target_opt_step <= custom_args.initial_opt_step:
        raise ValueError("Target optimizer step must exceed the initial step")

    init_adapter = (
        custom_args.init_checkpoint
        / "lora_adapter"
        / "adapter_model.safetensors"
    )
    if not init_adapter.exists():
        raise FileNotFoundError(f"Missing initial adapter: {init_adapter}")

    batch_size = int(cli_value(source_args, "--batch_size", 4))
    grad_accum = int(cli_value(source_args, "--grad_accum_steps", 8))
    batches_per_epoch = math.ceil(custom_args.stage1_samples / batch_size)
    optimizer_steps_per_epoch = batches_per_epoch // grad_accum
    additional_steps = (
        custom_args.target_opt_step - custom_args.initial_opt_step
    )
    effective_epochs = math.ceil(
        additional_steps / optimizer_steps_per_epoch
    )
    source_args = replace_cli_value(
        source_args, "--num_epochs", effective_epochs
    )

    output_dir = Path(cli_value(source_args, "--output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "fraction": custom_args.fraction_label,
        "source_script": str(custom_args.source_script),
        "source_script_sha256": sha256_file(custom_args.source_script),
        "initial_checkpoint": str(custom_args.init_checkpoint),
        "initial_adapter_sha256": sha256_file(init_adapter),
        "initial_optimizer_step": custom_args.initial_opt_step,
        "target_optimizer_step": custom_args.target_opt_step,
        "additional_optimizer_steps": additional_steps,
        "stage1_samples": custom_args.stage1_samples,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_per_full_subset_epoch": optimizer_steps_per_epoch,
        "effective_continuation_epochs": effective_epochs,
        "scheduler_total_steps": custom_args.scheduler_total_steps,
        "scheduler_warmup_steps": custom_args.scheduler_warmup_steps,
        "optimizer_state_resumed": False,
        "optimizer_state_note": (
            "The source checkpoint contains model weights only. AdamW state "
            "is reinitialized while the cosine schedule and KL schedule "
            "continue from initial_optimizer_step."
        ),
        "mental_prewarm_repeated": False,
        "checkpoint_policy": "cumulative target endpoint",
    }
    protocol_path = output_dir / "continuation_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    sys.argv = [str(custom_args.source_script)] + source_args
    module = load_module(custom_args.source_script)

    def load_initial_adapter(base_model, _lora_config):
        print(
            f"Continuation: loading trainable LoRA from "
            f"{init_adapter.parent}",
            flush=True,
        )
        return PeftModel.from_pretrained(
            base_model,
            str(init_adapter.parent),
            torch_dtype=torch.bfloat16,
            is_trainable=True,
        )

    module.get_peft_model = load_initial_adapter

    original_model_class = module.RecursiveToMModel

    def load_initial_model(base_model, reward_dim, z_dim):
        model = original_model_class(
            base_model, reward_dim=reward_dim, z_dim=z_dim
        )
        for head_name in module.CUSTOM_HEAD_NAMES:
            head_path = custom_args.init_checkpoint / f"{head_name}.pth"
            if not head_path.exists():
                raise FileNotFoundError(
                    f"Missing initial custom head: {head_path}"
                )
            getattr(model, head_name).load_state_dict(
                torch.load(
                    head_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
        print(
            f"Continuation: loaded {len(module.CUSTOM_HEAD_NAMES)} "
            f"custom heads from {custom_args.init_checkpoint}",
            flush=True,
        )
        return model

    module.RecursiveToMModel = load_initial_model

    original_scheduler = module.get_cosine_schedule_with_warmup

    def continued_scheduler(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1,
    ):
        for parameter_group in optimizer.param_groups:
            parameter_group.setdefault(
                "initial_lr", parameter_group["lr"]
            )
        print(
            "Continuation: overriding scheduler "
            f"{num_warmup_steps}/{num_training_steps} -> "
            f"{custom_args.scheduler_warmup_steps}/"
            f"{custom_args.scheduler_total_steps} at cumulative step "
            f"{custom_args.initial_opt_step}",
            flush=True,
        )
        return original_scheduler(
            optimizer,
            custom_args.scheduler_warmup_steps,
            custom_args.scheduler_total_steps,
            last_epoch=custom_args.initial_opt_step - 1,
        )

    module.get_cosine_schedule_with_warmup = continued_scheduler

    original_train_epoch = module.train_epoch
    continuation_state = {
        "started": False,
        "last_step": custom_args.initial_opt_step,
    }

    def bounded_train_epoch(*arguments, **kwargs):
        source_offset = kwargs.get("global_step_offset", 0)
        if not continuation_state["started"]:
            if source_offset != 0:
                raise RuntimeError(
                    f"Unexpected initial source offset: {source_offset}"
                )
            source_offset = custom_args.initial_opt_step
            continuation_state["started"] = True

        remaining_steps = custom_args.target_opt_step - source_offset
        if remaining_steps <= 0:
            raise RuntimeError(
                "Stage 1 continuation exceeded its optimizer-step target"
            )

        dataloader = arguments[1]
        accumulation = kwargs.get("grad_accum_steps", 1)
        available_steps = len(dataloader) // accumulation
        if available_steps > remaining_steps:
            batch_limit = remaining_steps * accumulation
            print(
                f"Continuation: limiting final epoch to {batch_limit} "
                f"batches ({remaining_steps} optimizer steps)",
                flush=True,
            )
            arguments = list(arguments)
            arguments[1] = LimitedLoader(dataloader, batch_limit)
            arguments = tuple(arguments)

        kwargs["global_step_offset"] = source_offset
        result = original_train_epoch(*arguments, **kwargs)
        continuation_state["last_step"] = result[2]
        if result[2] > custom_args.target_opt_step:
            raise RuntimeError(
                f"Optimizer target exceeded: {result[2]} > "
                f"{custom_args.target_opt_step}"
            )
        return result

    module.train_epoch = bounded_train_epoch

    original_save = module._save_checkpoint

    def endpoint_save(model, save_dir):
        basename = os.path.basename(os.path.normpath(save_dir))
        epoch_match = re.fullmatch(r"epoch_(\d+)", basename)
        if epoch_match:
            cumulative_step = continuation_state["last_step"]
            checkpoint_name = (
                "final"
                if cumulative_step == custom_args.target_opt_step
                else f"step_{cumulative_step}"
            )
            checkpoint_dir = output_dir / checkpoint_name
            print(
                f"Continuation: saving cumulative Step {cumulative_step} "
                f"to {checkpoint_dir}",
                flush=True,
            )
            original_save(model, str(checkpoint_dir))
            return
        if basename == "best":
            print(
                "Continuation: skipping duplicate train-loss best "
                "checkpoint; cumulative endpoints are retained",
                flush=True,
            )
            return
        original_save(model, save_dir)

    module._save_checkpoint = endpoint_save
    module.main()

    final_adapter = (
        output_dir / "final" / "lora_adapter" / "adapter_model.safetensors"
    )
    if not final_adapter.exists():
        raise FileNotFoundError(
            f"Final Stage 1 continuation checkpoint missing: {final_adapter}"
        )
    if continuation_state["last_step"] != custom_args.target_opt_step:
        raise RuntimeError(
            f"Continuation ended at Step {continuation_state['last_step']}, "
            f"expected {custom_args.target_opt_step}"
        )

    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fraction": custom_args.fraction_label,
        "initial_optimizer_step": custom_args.initial_opt_step,
        "optimizer_steps": custom_args.target_opt_step,
        "checkpoint": str(output_dir / "final"),
        "adapter_sha256": sha256_file(final_adapter),
    }
    (output_dir / "completed.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
