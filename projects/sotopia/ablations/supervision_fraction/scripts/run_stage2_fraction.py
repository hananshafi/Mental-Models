#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_STAGE2_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "stage2_grpo_agent_training_v3.py"
)


class TargetStepReached(Exception):
    pass


def cli_value(arguments, name, default=None):
    if name not in arguments:
        return default
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError(f"Missing value for {name}")
    return arguments[index + 1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("sotopia_stage2_fraction_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--source_script", type=Path, default=DEFAULT_STAGE2_SCRIPT)
    custom.add_argument("--max_steps", type=int, required=True)
    custom.add_argument("--fraction_label", required=True)
    custom_args, source_args = custom.parse_known_args()

    output_dir = Path(cli_value(source_args, "--output_dir"))
    save_every = int(cli_value(source_args, "--save_every", 50))
    if custom_args.max_steps % save_every:
        raise ValueError("--max_steps must be divisible by --save_every")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "fraction": custom_args.fraction_label,
        "source_script": str(custom_args.source_script),
        "source_script_sha256": sha256_file(custom_args.source_script),
        "target_grpo_steps": custom_args.max_steps,
        "save_every": save_every,
        "checkpoint_policy": "target_step_only",
    }
    with (output_dir / "fraction_protocol.json").open("w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")

    sys.argv = [str(custom_args.source_script)] + source_args
    module = load_module(custom_args.source_script)
    original_save = module.PeftModel.save_pretrained

    def compact_save(model, save_directory, *arguments, **kwargs):
        basename = os.path.basename(os.path.normpath(str(save_directory)))
        step_match = re.fullmatch(r"step_(\d+)", basename)
        if step_match:
            step = int(step_match.group(1))
            if step < custom_args.max_steps:
                print(
                    f"Fraction protocol: skipping intermediate checkpoint {basename}",
                    flush=True,
                )
                return None
            if step == custom_args.max_steps:
                result = original_save(
                    model, save_directory, *arguments, **kwargs
                )
                final_link = output_dir / "final"
                if final_link.is_symlink() or final_link.exists():
                    final_link.unlink()
                final_link.symlink_to(basename, target_is_directory=True)
                adapter = (
                    output_dir / basename / "adapter_model.safetensors"
                )
                completion = {
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "fraction": custom_args.fraction_label,
                    "grpo_steps": step,
                    "checkpoint": str(output_dir / basename),
                    "adapter_sha256": sha256_file(adapter),
                }
                with (output_dir / "completed.json").open("w") as handle:
                    json.dump(completion, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                print(json.dumps(completion, indent=2), flush=True)
                raise TargetStepReached
            raise RuntimeError(
                f"Stage 2 exceeded target: {step} > {custom_args.max_steps}"
            )
        if basename == "best" or re.fullmatch(r"epoch_\d+", basename):
            print(
                f"Fraction protocol: skipping non-target checkpoint {basename}",
                flush=True,
            )
            return None
        return original_save(model, save_directory, *arguments, **kwargs)

    module.PeftModel.save_pretrained = compact_save
    try:
        module.main()
    except TargetStepReached:
        print(
            f"Fraction protocol: stopped cleanly at GRPO step "
            f"{custom_args.max_steps}",
            flush=True,
        )

    adapter = (
        output_dir
        / f"step_{custom_args.max_steps}"
        / "adapter_model.safetensors"
    )
    if not adapter.exists():
        raise FileNotFoundError(f"Final Stage 2 checkpoint missing: {adapter}")


if __name__ == "__main__":
    main()
