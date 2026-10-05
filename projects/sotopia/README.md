# SOTOPIA

Canonical language-agent pipeline for jointly learning recursive mental states
and a reward model, then distilling that signal into a standalone policy with
SFT and GRPO.

## Layout

- `scripts/`: current data generation, Stage 1, Stage 2, Stage 3, and latent analysis.
- `experiments/`: structural, causal, counterfactual, and auxiliary-loss studies.
- `ablations/supervision_fraction/`: matched 0/25/50/100% mental-supervision experiment.
- `legacy/`: older v2 implementations retained for provenance.
- `docs/detailed_pipeline.md`: implementation-level notes from the original workspace.

The SOTOPIA checkout is pinned under `third_party/src/sotopia`. Bootstrap it and
install the local overlay before running this project:

```bash
./tools/bootstrap_third_party.sh sotopia
pip install -e third_party/src/sotopia
```

## 1. Generate annotated episodes

This step downloads SOTOPIA-π episodes and obtains per-turn reward, rationale,
hard-negative, first-order, and second-order annotations.

```bash
export OPENAI_API_KEY=...
python projects/sotopia/scripts/generate_sotopia_full_pipeline.py \
  --output projects/sotopia/data/sotopia_turn_rewards_v3.jsonl \
  --model gpt-4o \
  --limit 1500
```

The command is resumable because completed episode identifiers are read from
the output JSONL.

## 2. Train the coupled mental/reward model

```bash
CUDA_VISIBLE_DEVICES=0 python \
  projects/sotopia/scripts/stage1_train_coupled_mental_reward_v3.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --data_path projects/sotopia/data/sotopia_turn_rewards_v3.jsonl \
  --output_dir projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3 \
  --gpu 0
```

Add `--mental_prewarm_data projects/sotopia/data/mental_model_persona_dataset.jsonl`
when using the optional persona pre-warm set. Stage 1 writes periodic epoch and
best checkpoints containing the LoRA adapter plus all learned mental/reward
heads.

## 3. Train the policy

```bash
CUDA_VISIBLE_DEVICES=0,1 python \
  projects/sotopia/scripts/stage2_grpo_agent_training_v3.py \
  --policy_model_name Qwen/Qwen2.5-7B-Instruct \
  --reward_model_name Qwen/Qwen2.5-7B-Instruct \
  --reward_checkpoint_dir \
    projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/best \
  --data_path projects/sotopia/data/sotopia_turn_rewards_v3.jsonl \
  --output_dir projects/sotopia/checkpoints/grpo_agent_qwen_v3 \
  --preset qwen \
  --gpu 0,1
```

The current Stage 2 script performs its SFT warm-up before GRPO unless
`--no_sft_warmup` is supplied. To reuse an existing warm-up adapter, pass
`--sft_checkpoint`.

## 4. Evaluate with the official SOTOPIA framework

```bash
export OPENAI_API_KEY=...
CUDA_VISIBLE_DEVICES=0 python \
  projects/sotopia/scripts/stage3_evaluate_sotopia.py \
  --policy_model_name Qwen/Qwen2.5-7B-Instruct \
  --policy_adapter_path projects/sotopia/checkpoints/grpo_agent_qwen_v3/best \
  --use_hf \
  --deduplicate_envs \
  --task all \
  --partner_model gpt-4o-mini \
  --judge_model gpt-4o \
  --output_path projects/sotopia/runs/evaluation/qwen_grpo.jsonl \
  --gpu 0
```

Task filters are `all`, `hard`, `cooperative`, and `competitive`. The
`competitive` split is the Craigslist-Bargain transfer evaluation described in
`projects/craigslist_bargain/README.md`.

## Mental-supervision fraction ablation

Place the full annotations and canonical checkpoints at the locations described
in `docs/data-and-checkpoints.md`, then run:

```bash
python projects/sotopia/ablations/supervision_fraction/scripts/prepare_data.py
python projects/sotopia/ablations/supervision_fraction/scripts/launch.py
python projects/sotopia/ablations/supervision_fraction/scripts/status.py
```

`launch.py` uses GPUs 0 and 1 and matches optimizer-step budgets across the 25%
and 50% subsets. Evaluation launchers under `evaluation/` require
`OPENAI_API_KEY` and use GPT-4o-mini as partner, GPT-4o as judge, and seed 42.

## Analyses

`experiments/` contains the paper and rebuttal diagnostics: BIT structure,
role routing, role swap/knockout, latent scrambling, compression controls,
counterfactual preference tests, task-signal probes, and auxiliary-loss
ablations. Generated figures and records belong under
`projects/sotopia/experiments/runs/` and are ignored by Git.
