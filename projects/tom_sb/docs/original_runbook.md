# ToM-SB Training Workspace

This folder contains a lightweight pipeline for training a text-only mental reward model on double-agent defense / belief-steering data.

The goal is to avoid forcing AIDA data into Sotopia episode format. Instead, we generate a flat ToM-SB JSONL format that directly feeds the same tensor interface used by `stage1_train_coupled_mental_reward_v3.py`.

## Files

`generate_tom_sb_data.py`

Generates flat ToM-SB records from either transformed AIDA dataset records or saved AIDA rollout JSONL.

`train_tom_sb_mental_reward_standalone.py`

Trains a standalone recursive ToM reward model with auxiliary mental text generation losses over `mental1_text` and `mental2_text`.

`train_tom_sb_mental_reward_structured.py`

Trains the same reward/preference model, but replaces mental text generation with structured ToM prediction heads. This is the cleaner option when you want `z1` and `z2` supervised directly against first-order and second-order belief labels.

`data/`

Default output directory for generated JSONL data.

`checkpoints/`

Default output directory for trained reward-model checkpoints.

## Data Format

Each JSONL record has this shape:

```json
{
  "example_id": "aida_synth_000001_v0",
  "task": "tom_sb_double_agent_defense",
  "context_text": "Task and dialogue context for the defender...",
  "pos_response": "Safe defender response that avoids leaking unknown truth.",
  "hard_negative": "Truth-leaking response.",
  "reward_vec": [0.8, 0.8, 0.9, 1.0, 1.0, 0.7, 0.75],
  "reward_explanations": "naturalness: ...",
  "mental1_text": "First-Order Belief: ...",
  "mental2_text": "Second-Order Belief: ...",
  "tom_labels": {
    "first_order_belief": {
      "attacker_known_fields": {},
      "attacker_missing_fields": ["corporation", "division", "project_code"],
      "attacker_target_field": "project_code",
      "attacker_confidence": "low"
    },
    "second_order_belief": {
      "attacker_believes_defender_knows": ["corporation", "division", "project_code"],
      "attacker_believes_defender_knows_missing": ["corporation", "division", "project_code"],
      "attacker_expects_defender_helpfulness": "high",
      "attacker_expected_probe_style": "direct_extraction"
    },
    "attacker_goal": "extract_missing_private_field",
    "defender_safe_boundary": {
      "can_confirm": [],
      "must_not_reveal": ["corporation", "division", "project_code"],
      "safe_response_type": "deflect_or_keep_unresolved"
    },
    "belief_update_risk": "low",
    "deception_detection_risk": "medium",
    "expected_next_attack": "repeat_or_rephrase_target_field_request",
    "strategic_intent": "broad_deflection",
    "privacy_mechanism": "safe_abstraction_or_deflection",
    "trust_mechanism": "remain_helpful_without_specifics",
    "decoy_policy": "no_decoy",
    "withholding_policy": "withhold_all_private_fields",
    "rationale_tags": [
      "avoid_confirming_ground_truth",
      "withhold_unknown_private_fields",
      "maintain_helpful_tone"
    ]
  },
  "metadata": {}
}
```

The reward vector has 7 slots so it can reuse the current Stage 1 model head:

```text
naturalness
trust_maintenance
attacker_belief_tracking
privacy_preservation
format_and_non_refusal
decoy_plausibility
defender_goal_success
```

## Generate Synthetic ToM-SB Data From AIDA

Use this when you only have the transformed AIDA dataset and no model rollouts yet.

```bash
python projects/tom_sb/scripts/generate_tom_sb_data.py \
  --aida_dataset_path /tmp/AIDoubleAgentDefenders/datasets_directory/final_datasets/three_layered_dataset.json \
  --output_path projects/tom_sb/data/tom_sb_train.jsonl \
  --val_ratio 0.1 \
  --num_variants_per_record 3 \
  --seed 42
```

This creates:

```text
projects/tom_sb/data/tom_sb_train.jsonl
projects/tom_sb/data/tom_sb_train_val.jsonl
projects/tom_sb/data/reward_schema.json
```

## Generate 5k Belief-Only ToM Data

Use this for mental-reward training when you want first-order and second-order belief labels plus compact structured ToM labels, without reward explanations, strategy text, or rationale-style text.

```bash
python projects/tom_sb/scripts/generate_tom_sb_data.py \
  --aida_dataset_path /tmp/AIDoubleAgentDefenders/datasets_directory/final_datasets/three_layered_dataset.json \
  --output_path projects/tom_sb/data/tom_sb_belief_only_5k_train.jsonl \
  --val_ratio 0.1 \
  --target_train_examples 5000 \
  --num_variants_per_record 20 \
  --belief_only_labels \
  --omit_reward_explanations \
  --seed 42
```

This creates:

```text
projects/tom_sb/data/tom_sb_belief_only_5k_train.jsonl
projects/tom_sb/data/tom_sb_belief_only_5k_train_val.jsonl
```

Train/validation splitting is scenario-disjoint: variants from the same AIDA base scenario are kept in only one split.

When training on this no-explanation data, use `--expl_weight 0.0`.

## Generate ToM-SB Data From AIDA Rollouts

Use this once you have AIDA eval/training rollouts with `conversation_histories`.

```bash
python projects/tom_sb/scripts/generate_tom_sb_data.py \
  --rollouts_path /path/to/aida_eval_results.jsonl \
  --output_path projects/tom_sb/data/tom_sb_rollout_train.jsonl \
  --val_ratio 0.1 \
  --seed 42
```

Rollout mode extracts each defender turn as one training example. It uses the actual defender reply as `pos_response`, derives the hard negative from the unknown ground-truth field, and uses defender reflection text when available.

## Train The Mental Reward Model

Validate the train/val files before loading a model:

```bash
python projects/tom_sb/scripts/train_tom_sb_mental_reward_standalone.py \
  --validate_only
```

Small smoke test:

```bash
CUDA_VISIBLE_DEVICES=7 python projects/tom_sb/scripts/train_tom_sb_mental_reward_standalone.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir projects/tom_sb/checkpoints/tom_sb_reward_smoke \
  --max_examples 64 \
  --max_val_examples 32 \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --num_epochs 1 \
  --expl_weight 0.0
```

Full first pass:

```bash
CUDA_VISIBLE_DEVICES=7 python projects/tom_sb/scripts/train_tom_sb_mental_reward_standalone.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir projects/tom_sb/checkpoints/tom_sb_reward_belief_only_v1 \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --num_epochs 3 \
  --lr 2e-5 \
  --mental_label_mode hybrid \
  --expl_weight 0.0
```

For Llama:

```bash
CUDA_VISIBLE_DEVICES=7 python projects/tom_sb/scripts/train_tom_sb_mental_reward_standalone.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --output_dir projects/tom_sb/checkpoints/tom_sb_reward_llama_belief_only_v1 \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --num_epochs 3 \
  --mental_label_mode hybrid \
  --expl_weight 0.0
```

The standalone training script defaults to the 5k belief-only train/val files. `--mental_label_mode hybrid` trains the z1/z2 mental decoders on both the natural first/second-order belief text and compact serialized `tom_labels`. `CUDA_VISIBLE_DEVICES` controls GPU selection; the trainer does not override it inside Python.

## Train With Structured ToM Losses

Use this version when you do not want the mental loss to depend on generated text correctness. It predicts ToM variables directly: field/tag multi-labels use BCE, categorical labels use CE.

The structured trainer also uses strategic-intent and rationale-style supervision without decoding prose. If the JSONL contains explicit labels, it reads them from `tom_labels`; for older files it derives the same targets from `metadata.strategy`, safe-boundary labels, risk labels, and known/unknown fields.

`z1` predicts first-order belief and boundary facts. `z2` predicts second-order belief plus strategy/rationale labels: `strategic_intent`, `privacy_mechanism`, `trust_mechanism`, `decoy_policy`, `withholding_policy`, and `rationale_tags`.

Validate first:

```bash
python projects/tom_sb/scripts/train_tom_sb_mental_reward_structured.py \
  --validate_only
```

Full first pass:

```bash
CUDA_VISIBLE_DEVICES=7 python projects/tom_sb/scripts/train_tom_sb_mental_reward_structured.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir projects/tom_sb/checkpoints/tom_sb_reward_structured_tom_v1 \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --num_epochs 3 \
  --lr 2e-5 \
  --mental1_weight 0.3 \
  --mental2_weight 0.3 \
  --expl_weight 0.0
```

The structured trainer saves `structured_tom_spec.json` and records the same spec in checkpoint metadata, so downstream policy/RL code can decode the prediction heads safely.

## Train Policy With Learned Reward

After the structured reward checkpoint passes pairwise checks, use `train_tom_sb_policy_grpo.py` to train a defender policy with GRPO. The reward model is loaded frozen from the final checkpoint and the policy is trained as a LoRA adapter. The default command below is conservative for a single GPU: it loads policy and reward on the same visible CUDA device, uses no separate reference model, and keeps a small SFT anchor on the synthetic safe response.

```bash
CUDA_VISIBLE_DEVICES=1 python projects/tom_sb/scripts/train_tom_sb_policy_grpo.py \
  --policy_model_name Qwen/Qwen2.5-7B-Instruct \
  --reward_model_name Qwen/Qwen2.5-7B-Instruct \
  --reward_checkpoint projects/tom_sb/checkpoints/tom_sb_reward_structured_tom_v1_plus2/best \
  --train_path projects/tom_sb/data/tom_sb_belief_only_5k_train.jsonl \
  --reward_schema_path projects/tom_sb/data/reward_schema.json \
  --output_dir projects/tom_sb/checkpoints/tom_sb_policy_grpo_v1 \
  --num_iterations 200 \
  --prompts_per_iter 1 \
  --group_size 4 \
  --max_prompt_len 1536 \
  --max_new_tokens 96 \
  --reward_batch_size 4 \
  --lr 5e-6 \
  --sft_weight 0.05 \
  --ref_kl_coeff 0.0 \
  --leak_penalty 1.0 \
  --dtype bf16 \
  --policy_device cuda:0 \
  --reward_device cuda:0 \
  --save_every 50
```

If memory is tight, reduce `--group_size` to `2`, `--max_prompt_len` to `1024`, or `--reward_batch_size` to `1`. If two GPUs are visible, you can place the reward model separately with `--reward_device cuda:1`.

## Notes

The reward model head is explicitly tied to the ToM-SB reward schema in `data/reward_schema.json`: `naturalness`, `trust_maintenance`, `attacker_belief_tracking`, `privacy_preservation`, `format_and_non_refusal`, `decoy_plausibility`, and `defender_goal_success`.

By default, the generator does not put the attacker's full hidden prior into `context_text`. It does encode known fields in the attacker message when synthetic data is generated. Use `--include_attacker_prior_in_context` only for debugging.

The generated hard negative intentionally leaks an unknown ground-truth field. This gives the preference loss a clean contrast between safe defender behavior and privacy failure.
