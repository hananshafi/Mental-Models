#!/usr/bin/env python3
"""Render SOTOPIA mental-model experiment commands.

This script prints commands for the SOTOPIA experiment matrix without running
them. Model paths are deliberately flexible: pass CLI flags or set environment
variables such as SOTOPIA_BASE_MODEL, SOTOPIA_POLICY_MODEL, and
SOTOPIA_REWARD_MODEL.
"""

from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path


SOTOPIA_ROOT_DEFAULT = "projects/sotopia"


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    kind: str
    stage1_args: dict[str, str | int | float] = field(default_factory=dict)
    stage2_args: dict[str, str | int | float] = field(default_factory=dict)
    needs_stage1: bool = True
    needs_stage2: bool = True


VARIANTS: dict[str, Variant] = {
    "full_recursive_bit": Variant(
        name="full_recursive_bit",
        kind="mental_reward",
        description="Current full recursive z1/z2 BIT mental-state reward model.",
    ),
    "reward_only_bottleneck": Variant(
        name="reward_only_bottleneck",
        kind="mental_reward",
        description="Same latent architecture, no mental/rationale/future supervision.",
        stage1_args={
            "mental1_weight": 0.0,
            "mental2_weight": 0.0,
            "expl_weight": 0.0,
            "future_weight": 0.0,
        },
    ),
    "first_order_only": Variant(
        name="first_order_only",
        kind="mental_reward",
        description="Keep first-order mental supervision; remove second-order mental loss.",
        stage1_args={"mental2_weight": 0.0},
    ),
    "no_antibypass": Variant(
        name="no_antibypass",
        kind="mental_reward",
        description="Remove z-only reward regularizers.",
        stage1_args={"z_only_weight": 0.0},
    ),
    "no_explanation": Variant(
        name="no_explanation",
        kind="mental_reward",
        description="Remove explanation-conditioned reward head loss.",
        stage1_args={"expl_weight": 0.0},
    ),
    "simple_reward_no_mental": Variant(
        name="simple_reward_no_mental",
        kind="simple_reward",
        description="GRPO with simple reward head; no mental latent model.",
        needs_stage1=False,
    ),
    "zero_shot_base": Variant(
        name="zero_shot_base",
        kind="eval_only",
        description="Evaluate base policy without adapter.",
        needs_stage1=False,
        needs_stage2=False,
    ),
}


REQUIRES_IMPLEMENTATION = {
    "parallel_z1_z2": "Add --latent_arch parallel so z1 and z2 both condition on context only.",
    "reverse_z2_z1": "Add reverse latent ordering as a negative control.",
    "single_latent_256": "Add a single latent with total capacity equal to z1+z2.",
    "merged_subspace": "Decode mental text from the whole latent instead of fixed 48/40/40 subspaces.",
    "flat_mental_summary": "Add a dataset transform that removes BIT tags.",
    "shuffled_mental_labels": "Add deterministic mental-label shuffling by seed.",
}


def q(value: str | os.PathLike[str] | int | float) -> str:
    return shlex.quote(str(value))


def render(cmd: list[str | os.PathLike[str] | int | float]) -> str:
    pieces = [q(part) for part in cmd]
    if len(pieces) <= 6:
        return " ".join(pieces)
    return " \\\n  ".join(pieces)


def add_kv_args(cmd: list[str | os.PathLike[str] | int | float], args: dict[str, str | int | float]) -> None:
    for key, value in args.items():
        cmd.extend([f"--{key}", value])


def variant_stage1_dir(args: argparse.Namespace, variant: str) -> Path:
    return Path(args.run_root) / "stage1" / f"{variant}_seed{args.seed}"


def variant_stage2_dir(args: argparse.Namespace, variant: str) -> Path:
    return Path(args.run_root) / "stage2" / f"{variant}_seed{args.seed}"


def variant_eval_path(args: argparse.Namespace, variant: str) -> Path:
    return (
        Path(args.run_root)
        / "eval"
        / f"{variant}_seed{args.seed}_{args.eval_task}_{args.partner_model}.jsonl"
    )


def variant_analysis_dir(args: argparse.Namespace, variant: str) -> Path:
    return Path(args.run_root) / "analysis" / f"{variant}_seed{args.seed}"


def reward_checkpoint(args: argparse.Namespace, variant: str) -> Path | str:
    if args.reward_checkpoint:
        return args.reward_checkpoint
    return variant_stage1_dir(args, variant) / "best"


def policy_adapter(args: argparse.Namespace, variant: str) -> Path | str:
    if args.policy_adapter:
        return args.policy_adapter
    return variant_stage2_dir(args, variant) / "best"


def stage1_command(args: argparse.Namespace, variant: Variant) -> str:
    out_dir = variant_stage1_dir(args, variant.name)
    cmd: list[str | os.PathLike[str] | int | float] = [
        "python",
        Path(args.sotopia_root) / "stage1_train_coupled_mental_reward_v3.py",
        "--model_name",
        args.base_model,
        "--data_path",
        args.data_path,
        "--output_dir",
        out_dir,
        "--batch_size",
        args.stage1_batch_size,
        "--grad_accum_steps",
        args.stage1_grad_accum_steps,
        "--num_epochs",
        args.stage1_epochs,
        "--lr",
        args.stage1_lr,
        "--seed",
        args.seed,
        "--gpu",
        args.stage1_gpu,
    ]
    add_kv_args(cmd, variant.stage1_args)
    if args.mental_prewarm_data:
        cmd.extend(["--mental_prewarm_data", args.mental_prewarm_data])
        cmd.extend(["--mental_prewarm_epochs", args.mental_prewarm_epochs])
    return render(cmd)


def stage2_command(args: argparse.Namespace, variant: Variant) -> str:
    out_dir = variant_stage2_dir(args, variant.name)
    if variant.kind == "simple_reward":
        cmd: list[str | os.PathLike[str] | int | float] = [
            "python",
            Path(args.sotopia_root) / "stage2_grpo_ablation_no_mental.py",
            "--policy_model_name",
            args.policy_model,
            "--data_path",
            args.data_path,
            "--output_dir",
            out_dir,
            "--group_size",
            args.group_size,
            "--grpo_epochs",
            args.grpo_epochs,
            "--prompts_per_step",
            args.prompts_per_step,
            "--num_ppo_epochs",
            args.num_ppo_epochs,
            "--clip_eps",
            args.clip_eps,
            "--kl_coeff",
            args.kl_coeff,
            "--lr",
            args.grpo_lr,
            "--temperature",
            args.train_temperature,
            "--top_p",
            args.train_top_p,
            "--seed",
            args.seed,
            "--gpu",
            args.stage2_gpu,
            "--save_every",
            args.save_every,
        ]
        return render(cmd)

    cmd = [
        "python",
        Path(args.sotopia_root) / "stage2_grpo_agent_training_v3.py",
        "--policy_model_name",
        args.policy_model,
        "--reward_model_name",
        args.reward_model,
        "--reward_checkpoint_dir",
        reward_checkpoint(args, variant.name),
        "--reward_version",
        "v3",
        "--data_path",
        args.data_path,
        "--output_dir",
        out_dir,
        "--group_size",
        args.group_size,
        "--grpo_epochs",
        args.grpo_epochs,
        "--prompts_per_step",
        args.prompts_per_step,
        "--num_ppo_epochs",
        args.num_ppo_epochs,
        "--clip_eps",
        args.clip_eps,
        "--kl_coeff",
        args.kl_coeff,
        "--lr",
        args.grpo_lr,
        "--temperature",
        args.train_temperature,
        "--top_p",
        args.train_top_p,
        "--seed",
        args.seed,
        "--gpu",
        args.stage2_gpu,
        "--save_every",
        args.save_every,
    ]
    if args.preset:
        cmd.extend(["--preset", args.preset])
    if args.reward_scoring_dims:
        cmd.extend(["--reward_scoring_dims", args.reward_scoring_dims])
    add_kv_args(cmd, variant.stage2_args)
    return render(cmd)


def eval_command(args: argparse.Namespace, variant: Variant) -> str:
    cmd: list[str | os.PathLike[str] | int | float] = [
        "python",
        Path(args.sotopia_root) / "stage3_evaluate_sotopia.py",
        "--policy_model_name",
        args.policy_model,
        "--use_hf",
        "--deduplicate_envs",
        "--task",
        args.eval_task,
        "--output_path",
        variant_eval_path(args, variant.name),
        "--max_turns",
        args.max_turns,
        "--max_episodes",
        args.max_episodes,
        "--policy_agent_index",
        args.policy_agent_index,
        "--partner_model",
        args.partner_model,
        "--judge_model",
        args.judge_model,
        "--temperature",
        args.eval_temperature,
        "--top_p",
        args.eval_top_p,
        "--seed",
        args.seed,
        "--gpu",
        args.eval_gpu,
        "--tag",
        variant.name,
    ]
    if variant.kind == "eval_only":
        cmd.append("--no_adapter")
    else:
        cmd.extend(["--policy_adapter_path", policy_adapter(args, variant.name)])
        cmd.append("--merge_adapter")
    if args.local_partner:
        cmd.append("--local_partner")
        if args.local_partner_model:
            cmd.extend(["--local_partner_model", args.local_partner_model])
        if args.partner_device:
            cmd.extend(["--partner_device", args.partner_device])
    return render(cmd)


def latent_command(args: argparse.Namespace, variant: Variant) -> str:
    if not variant.needs_stage1:
        return f"# {variant.name}: no Stage 1 reward checkpoint; latent analysis is not applicable."
    cmd: list[str | os.PathLike[str] | int | float] = [
        "python",
        Path(args.sotopia_root) / "analyse_tom_latents.py",
        "--base_model_name",
        args.reward_model,
        "--checkpoint_dir",
        reward_checkpoint(args, variant.name),
        "--data_path",
        args.data_path,
        "--output_dir",
        variant_analysis_dir(args, variant.name),
        "--label_source",
        args.label_source,
        "--concept_field",
        args.concept_field,
        "--concept_value",
        args.concept_value,
        "--seed",
        args.seed,
        "--device",
        args.analysis_device,
        "--recompute_latents",
        "--recompute_records",
    ]
    return render(cmd)


def print_variant(args: argparse.Namespace, variant: Variant) -> None:
    print(f"\n# Variant: {variant.name}")
    print(f"# {variant.description}")
    stages = ["stage1", "latent", "stage2", "eval"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "stage1":
            if variant.needs_stage1:
                print("\n# Stage 1: mental/reward model")
                print(stage1_command(args, variant))
            else:
                print("\n# Stage 1: not applicable")
        elif stage == "latent":
            print("\n# Latent analysis")
            print(latent_command(args, variant))
        elif stage == "stage2":
            if variant.needs_stage2:
                print("\n# Stage 2: policy training")
                print(stage2_command(args, variant))
            else:
                print("\n# Stage 2: not applicable")
        elif stage == "eval":
            print("\n# Stage 3: official SOTOPIA evaluation")
            print(eval_command(args, variant))


def parse_args() -> argparse.Namespace:
    env = os.environ
    root = env.get("SOTOPIA_ROOT", SOTOPIA_ROOT_DEFAULT)
    base_model = env.get("SOTOPIA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sotopia-root", default=root)
    parser.add_argument("--run-root", default=env.get("SOTOPIA_RUN_ROOT", str(Path(root) / "experiments" / "runs")))
    parser.add_argument("--data-path", default=env.get("SOTOPIA_DATA_PATH", str(Path(root) / "sotopia_turn_rewards_v3.jsonl")))
    parser.add_argument("--base-model", default=base_model)
    parser.add_argument("--policy-model", default=env.get("SOTOPIA_POLICY_MODEL", base_model))
    parser.add_argument("--reward-model", default=env.get("SOTOPIA_REWARD_MODEL", base_model))
    parser.add_argument("--reward-checkpoint", default=env.get("SOTOPIA_REWARD_CHECKPOINT"))
    parser.add_argument("--policy-adapter", default=env.get("SOTOPIA_POLICY_ADAPTER"))
    parser.add_argument("--variant", default="full_recursive_bit", choices=[*VARIANTS.keys(), "all"])
    parser.add_argument("--stage", default="all", choices=["stage1", "latent", "stage2", "eval", "all"])
    parser.add_argument("--seed", type=int, default=int(env.get("SOTOPIA_SEED", "42")))

    parser.add_argument("--stage1-gpu", default=env.get("SOTOPIA_STAGE1_GPU", "0"))
    parser.add_argument("--stage2-gpu", default=env.get("SOTOPIA_STAGE2_GPU", "0"))
    parser.add_argument("--eval-gpu", default=env.get("SOTOPIA_EVAL_GPU", "0"))
    parser.add_argument("--analysis-device", default=env.get("SOTOPIA_ANALYSIS_DEVICE", "cuda"))

    parser.add_argument("--stage1-epochs", type=int, default=int(env.get("SOTOPIA_STAGE1_EPOCHS", "10")))
    parser.add_argument("--stage1-batch-size", type=int, default=int(env.get("SOTOPIA_STAGE1_BATCH_SIZE", "4")))
    parser.add_argument("--stage1-grad-accum-steps", type=int, default=int(env.get("SOTOPIA_STAGE1_GRAD_ACCUM", "8")))
    parser.add_argument("--stage1-lr", type=float, default=float(env.get("SOTOPIA_STAGE1_LR", "2e-5")))
    parser.add_argument("--mental-prewarm-data", default=env.get("SOTOPIA_MENTAL_PREWARM_DATA"))
    parser.add_argument("--mental-prewarm-epochs", type=int, default=int(env.get("SOTOPIA_MENTAL_PREWARM_EPOCHS", "1")))

    parser.add_argument("--preset", default=env.get("SOTOPIA_PRESET"))
    parser.add_argument("--group-size", type=int, default=int(env.get("SOTOPIA_GROUP_SIZE", "8")))
    parser.add_argument("--grpo-epochs", type=int, default=int(env.get("SOTOPIA_GRPO_EPOCHS", "2")))
    parser.add_argument("--prompts-per-step", type=int, default=int(env.get("SOTOPIA_PROMPTS_PER_STEP", "4")))
    parser.add_argument("--num-ppo-epochs", type=int, default=int(env.get("SOTOPIA_NUM_PPO_EPOCHS", "1")))
    parser.add_argument("--clip-eps", type=float, default=float(env.get("SOTOPIA_CLIP_EPS", "0.2")))
    parser.add_argument("--kl-coeff", type=float, default=float(env.get("SOTOPIA_KL_COEFF", "0.08")))
    parser.add_argument("--grpo-lr", type=float, default=float(env.get("SOTOPIA_GRPO_LR", "2e-6")))
    parser.add_argument("--train-temperature", type=float, default=float(env.get("SOTOPIA_TRAIN_TEMPERATURE", "0.8")))
    parser.add_argument("--train-top-p", type=float, default=float(env.get("SOTOPIA_TRAIN_TOP_P", "0.95")))
    parser.add_argument("--reward-scoring-dims", default=env.get("SOTOPIA_REWARD_SCORING_DIMS"))
    parser.add_argument("--save-every", type=int, default=int(env.get("SOTOPIA_SAVE_EVERY", "50")))

    parser.add_argument("--eval-task", default=env.get("SOTOPIA_EVAL_TASK", "hard"), choices=["all", "hard", "cooperative", "competitive"])
    parser.add_argument("--max-episodes", type=int, default=int(env.get("SOTOPIA_MAX_EPISODES", "-1")))
    parser.add_argument("--max-turns", type=int, default=int(env.get("SOTOPIA_MAX_TURNS", "10")))
    parser.add_argument("--policy-agent-index", type=int, default=int(env.get("SOTOPIA_POLICY_AGENT_INDEX", "0")))
    parser.add_argument("--partner-model", default=env.get("SOTOPIA_PARTNER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge-model", default=env.get("SOTOPIA_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--eval-temperature", type=float, default=float(env.get("SOTOPIA_EVAL_TEMPERATURE", "0.7")))
    parser.add_argument("--eval-top-p", type=float, default=float(env.get("SOTOPIA_EVAL_TOP_P", "0.9")))
    parser.add_argument("--local-partner", action="store_true", default=env.get("SOTOPIA_LOCAL_PARTNER", "0") == "1")
    parser.add_argument("--local-partner-model", default=env.get("SOTOPIA_LOCAL_PARTNER_MODEL"))
    parser.add_argument("--partner-device", default=env.get("SOTOPIA_PARTNER_DEVICE"))

    parser.add_argument("--label-source", default=env.get("SOTOPIA_LABEL_SOURCE", "response_heuristic"), choices=["response_heuristic", "external"])
    parser.add_argument("--concept-field", default=env.get("SOTOPIA_CONCEPT_FIELD", "strategy"), choices=["intent", "knowledge", "strategy"])
    parser.add_argument("--concept-value", default=env.get("SOTOPIA_CONCEPT_VALUE", "offer_proposal"))

    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--list-planned", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_variants:
        for variant in VARIANTS.values():
            print(f"{variant.name}: {variant.description}")
        return
    if args.list_planned:
        for name, note in REQUIRES_IMPLEMENTATION.items():
            print(f"{name}: {note}")
        return

    if args.variant == "all":
        for variant in VARIANTS.values():
            print_variant(args, variant)
    else:
        print_variant(args, VARIANTS[args.variant])


if __name__ == "__main__":
    main()
