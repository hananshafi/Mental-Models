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


class LimitedLoader:
    def __init__(self, source, batch_limit):
        self.source = source
        self.batch_limit = batch_limit

    def __iter__(self):
        return itertools.islice(iter(self.source), self.batch_limit)

    def __len__(self):
        return self.batch_limit


def load_module(path):
    spec = importlib.util.spec_from_file_location("sotopia_stage1_fraction_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--source_script", type=Path, default=DEFAULT_STAGE1_SCRIPT)
    custom.add_argument("--stage1_samples", type=int, required=True)
    custom.add_argument("--max_opt_steps", type=int, required=True)
    custom.add_argument("--scheduler_total_steps", type=int, required=True)
    custom.add_argument("--scheduler_warmup_steps", type=int, required=True)
    custom.add_argument("--fraction_label", required=True)
    custom.add_argument(
        "--checkpoint_policy",
        choices=("best", "final_only"),
        default="best",
    )
    custom_args, source_args = custom.parse_known_args()

    batch_size = int(cli_value(source_args, "--batch_size", 4))
    grad_accum = int(cli_value(source_args, "--grad_accum_steps", 8))
    batches_per_epoch = math.ceil(custom_args.stage1_samples / batch_size)
    optimizer_steps_per_epoch = batches_per_epoch // grad_accum
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("Subset is too small for one optimizer update")
    effective_epochs = math.ceil(
        custom_args.max_opt_steps / optimizer_steps_per_epoch
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
        "stage1_samples": custom_args.stage1_samples,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_per_full_subset_epoch": optimizer_steps_per_epoch,
        "effective_epochs": effective_epochs,
        "target_optimizer_steps": custom_args.max_opt_steps,
        "scheduler_total_steps": custom_args.scheduler_total_steps,
        "scheduler_warmup_steps": custom_args.scheduler_warmup_steps,
        "checkpoint_policy": custom_args.checkpoint_policy,
    }
    with (output_dir / "fraction_protocol.json").open("w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")

    sys.argv = [str(custom_args.source_script)] + source_args
    module = load_module(custom_args.source_script)

    original_scheduler = module.get_cosine_schedule_with_warmup

    def matched_scheduler(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1,
    ):
        print(
            "Fraction protocol: overriding scheduler "
            f"{num_warmup_steps}/{num_training_steps} -> "
            f"{custom_args.scheduler_warmup_steps}/"
            f"{custom_args.scheduler_total_steps}",
            flush=True,
        )
        return original_scheduler(
            optimizer,
            custom_args.scheduler_warmup_steps,
            custom_args.scheduler_total_steps,
            last_epoch=last_epoch,
        )

    module.get_cosine_schedule_with_warmup = matched_scheduler
    original_train_epoch = module.train_epoch

    def bounded_train_epoch(*arguments, **kwargs):
        dataloader = arguments[1]
        global_step_offset = kwargs.get("global_step_offset", 0)
        accumulation = kwargs.get("grad_accum_steps", 1)
        remaining_steps = custom_args.max_opt_steps - global_step_offset
        available_steps = len(dataloader) // accumulation
        if remaining_steps <= 0:
            raise RuntimeError("Stage 1 exceeded its optimizer-step target")
        if available_steps > remaining_steps:
            batch_limit = remaining_steps * accumulation
            print(
                f"Fraction protocol: limiting final epoch to {batch_limit} "
                f"batches ({remaining_steps} optimizer steps)",
                flush=True,
            )
            arguments = list(arguments)
            arguments[1] = LimitedLoader(dataloader, batch_limit)
            arguments = tuple(arguments)
        result = original_train_epoch(*arguments, **kwargs)
        if result[2] > custom_args.max_opt_steps:
            raise RuntimeError(
                f"Optimizer target exceeded: {result[2]} > "
                f"{custom_args.max_opt_steps}"
            )
        return result

    module.train_epoch = bounded_train_epoch
    original_save = module._save_checkpoint
    final_epoch = effective_epochs - 1

    def compact_save(model, save_dir):
        basename = os.path.basename(os.path.normpath(save_dir))
        epoch_match = re.fullmatch(r"epoch_(\d+)", basename)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            if (
                custom_args.checkpoint_policy == "final_only"
                and epoch == final_epoch
            ):
                final_dir = output_dir / "final"
                print(
                    f"Fraction protocol: saving final checkpoint to {final_dir}",
                    flush=True,
                )
                original_save(model, str(final_dir))
            else:
                print(
                    f"Fraction protocol: skipping intermediate checkpoint {basename}",
                    flush=True,
                )
            return
        if basename == "best":
            if custom_args.checkpoint_policy == "best":
                best_dir = output_dir / "best"
                print(
                    f"Fraction protocol: saving current best checkpoint to {best_dir}",
                    flush=True,
                )
                original_save(model, str(best_dir))
                return
            print(
                "Fraction protocol: skipping duplicate best checkpoint; "
                "the matched final endpoint is retained",
                flush=True,
            )
            return
        original_save(model, save_dir)

    module._save_checkpoint = compact_save
    module.main()

    checkpoint_name = (
        "best" if custom_args.checkpoint_policy == "best" else "final"
    )
    checkpoint_dir = output_dir / checkpoint_name
    adapter = checkpoint_dir / "lora_adapter" / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError(f"Final Stage 1 checkpoint missing: {adapter}")
    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fraction": custom_args.fraction_label,
        "optimizer_steps": custom_args.max_opt_steps,
        "checkpoint": str(checkpoint_dir),
        "checkpoint_policy": custom_args.checkpoint_policy,
        "adapter_sha256": sha256_file(adapter),
    }
    with (output_dir / "completed.json").open("w") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
