# MMRole

Multimodal Theory-of-Mind pipeline for preparing MMRole, generating and
validating mental-state supervision, training visual mental/reward and policy
models, and evaluating role-playing responses.

## 1. Prepare data

The orchestrator downloads MMRole, restructures examples by turn, annotates
belief states, validates annotations, and builds training formats.

```bash
export OPENAI_API_KEY=...
bash projects/mmrole/scripts/run_pipeline.sh --pilot
```

After checking the pilot output, run the full pipeline:

```bash
bash projects/mmrole/scripts/run_pipeline.sh
```

Useful controls:

- `--from-step N` resumes at a specific preparation stage.
- `--skip-coco --coco-dir PATH` reuses an existing COCO image directory.
- `--model MODEL --workers N --max-examples N` controls annotation.

Generated data is written under `projects/mmrole/` in `raw_data/`, `images/`,
`character_profiles/`, and `training_data/`; these directories are ignored.

## 2. Train the visual mental/reward model

```bash
CUDA_VISIBLE_DEVICES=0 python \
  projects/mmrole/scripts/stage0_reward_model_visual_tom.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --output_dir projects/mmrole/checkpoints/stage0_reward_v3 \
  --gpu 0
```

Stage 0 uses belief-prediction and preference-pair files from the full
non-official pool and reserves a deterministic validation holdout.

## 3. SFT

```bash
CUDA_VISIBLE_DEVICES=0 python projects/mmrole/scripts/stage1_sft_visual_tom.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --mental_prefix_checkpoint_dir projects/mmrole/checkpoints/stage0_reward_v3/best \
  --output_dir projects/mmrole/checkpoints/stage1_sft \
  --gpu 0
```

Omit `--mental_prefix_checkpoint_dir` for the plain SFT control. The
LLaVA-NeXT/Mistral variant is in
`stage1_sft_visual_tom_llava_next_mistral.py`.

## 4. GRPO with the learned reward

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python \
  projects/mmrole/scripts/stage2_grpo_learned_reward.py \
  --sft_checkpoint projects/mmrole/checkpoints/stage1_sft/best \
  --reward_checkpoint_dir projects/mmrole/checkpoints/stage0_reward_v3/best \
  --mental_prefix_checkpoint_dir projects/mmrole/checkpoints/stage0_reward_v3/best \
  --output_dir projects/mmrole/checkpoints/stage2_grpo \
  --gpu 0,1,2
```

`stage2_grpo_tom_reward.py` is the structured hand-designed reward baseline;
`stage2_grpo_learned_reward_llava_next_qwen_reward.py` is the cross-backbone
variant.

## 5. Optional contrastive DPO

```bash
CUDA_VISIBLE_DEVICES=0 python projects/mmrole/scripts/stage3_dpo_contrastive.py \
  --grpo_checkpoint projects/mmrole/checkpoints/stage2_grpo/best \
  --output_dir projects/mmrole/checkpoints/stage3_dpo \
  --gpu 0
```

## 6. Evaluate

Generate responses from a registered baseline or local adapter:

```bash
python projects/mmrole/scripts/step6a_generate_responses.py \
  --model qwen2.5-vl-local \
  --output_dir projects/mmrole/runs/responses
```

For a local adapter, also pass `--base_model` and `--adapter_path`. Score the
responses with the ToM and MMRole dimensions:

```bash
export OPENAI_API_KEY=...
python projects/mmrole/scripts/step6_evaluate_tom.py response \
  --responses_path projects/mmrole/runs/responses/qwen2_5_vl_local.jsonl \
  --output_path projects/mmrole/runs/evaluation/qwen2_5_vl_local.jsonl \
  --judge_model gpt-4o \
  --eval_dims all
```

`eval_mmrole_official.py` provides the original answer/review/summary workflow.
