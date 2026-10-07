"""Exact-resume support for the training scripts in this directory.

Each training script refreshes <output_dir>/last/ with its trainable weights,
optimizer, scheduler, loop position, and RNG state. Rerunning the same command
with --resume continues from that point. Identical copies of this module live
in each project's scripts/ directory so that the projects stay self-contained.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch

RESUME_DIR = "last"
STATE_FILE = "trainer_state.pt"


def find_resume_dir(output_dir) -> Optional[Path]:
    """Return the newest complete resume directory, if any."""
    # "last.old" only survives when a crash interrupts the swap in save_resume_dir.
    for name in (RESUME_DIR, f"{RESUME_DIR}.old"):
        candidate = Path(output_dir) / name
        if (candidate / STATE_FILE).is_file():
            return candidate
    return None


def require_resume_dir(output_dir) -> Path:
    resume_dir = find_resume_dir(output_dir)
    if resume_dir is None:
        raise FileNotFoundError(
            f"--resume was given but {Path(output_dir) / RESUME_DIR} has no {STATE_FILE}."
        )
    return resume_dir


def save_resume_dir(output_dir, write_weights: Callable[[Path], None], state: dict) -> Path:
    """Write weights and state to a temporary directory, then swap it in as last/."""
    out = Path(output_dir)
    tmp, final, old = out / f"{RESUME_DIR}.tmp", out / RESUME_DIR, out / f"{RESUME_DIR}.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    write_weights(tmp)
    torch.save({**state, "rng": rng_state()}, tmp / STATE_FILE)
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)
    return final


def load_resume_state(resume_dir) -> dict:
    return torch.load(Path(resume_dir) / STATE_FILE, map_location="cpu", weights_only=False)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        for index, cuda_state in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state, index)


def check_resume_args(saved_args: dict, args, keys: Iterable[str]) -> None:
    """Refuse to resume when an argument that shapes training has changed."""
    changed = {
        key: (saved_args.get(key), getattr(args, key))
        for key in keys
        if saved_args.get(key) != getattr(args, key)
    }
    if changed:
        raise ValueError(f"Cannot resume with different training arguments: {changed}")


def epoch_order(num_samples: int, seed: int, epoch: int) -> List[int]:
    """Deterministic per-epoch shuffle, so a resumed run sees the same batches."""
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(num_samples, generator=generator).tolist()


def load_lora_weights(peft_model, adapter_dir) -> None:
    """Load saved LoRA weights into an identically configured PEFT model."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    expected = sum(1 for name, _ in peft_model.named_parameters() if "lora_" in name)
    result = set_peft_model_state_dict(peft_model, weights)
    if result.unexpected_keys or len(weights) != expected:
        raise RuntimeError(
            f"LoRA weights do not match the model: {len(weights)} saved vs {expected} "
            f"expected tensors, unexpected={result.unexpected_keys[:5]}"
        )
