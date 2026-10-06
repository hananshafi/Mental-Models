# BigToM

Text-only mental-model training pipeline and the shared paper evaluation
harness for BigToM, ToMi, and FANToM.

The training pipeline has three sequential stages: Stage 1 trains the mental/reward
model, Stage 2 performs latent-prefix SFT, and Stage 3 performs GRPO. Evaluation
is a separate, unnumbered step.

## Setup

```bash
./tools/bootstrap_third_party.sh bigtom tomi fantom
conda activate mental-models
```

Released benchmark files live under `third_party/src/`. Generated BigToM data,
checkpoints, and run outputs remain under this project.

Download and link the 9,964-example paper training set with:

```bash
python tools/download_data.py
```

Stages 1, 2, and 3 train entirely from this released JSONL and do not require
an OpenAI API key. Official multiple-choice evaluation is also local.

## Optional: Generate and annotate new scenarios

Skip this section when using the released paper training set. An OpenAI API key
is required only when generating a new teacher-annotated dataset.

```bash
export OPENAI_API_KEY=...
ENV_NAME=mental-models \
  bash projects/bigtom/scripts/run_generate_and_annotate.sh
```

Useful overrides include `NUM_NEW_SCENARIOS`, `GEN_MAX_WORKERS`,
`ANN_MAX_WORKERS`, `OUT_CSV`, and `OUT_JSONL`. Default outputs are:

```text
projects/bigtom/data/bigtom_qwen.csv
projects/bigtom/data/bigtom_qwen_annotated.jsonl
```

These generic names are used for newly generated runs. The released paper data
is `projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl`.

## Stage 1: Train the mental/reward model

```bash
CUDA_VISIBLE_DEVICES=0 python projects/bigtom/scripts/stage1_train_mental_reward.py \
  --data projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --out projects/bigtom/checkpoints/stage1_qwen \
  --epochs 3
```

## Stage 2: Latent-prefix SFT

This stage uses the frozen Stage 1 encoder to construct first- and second-order
latent prefixes for the policy.

```bash
CUDA_VISIBLE_DEVICES=0,1 python projects/bigtom/scripts/stage2_policy_sft.py \
  --data projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl \
  --stage1_ckpt projects/bigtom/checkpoints/stage1_qwen/epoch_2 \
  --out projects/bigtom/checkpoints/stage2_qwen
```

## Stage 3: GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python projects/bigtom/scripts/stage3_grpo.py \
  --data projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl \
  --stage1_ckpt projects/bigtom/checkpoints/stage1_qwen/epoch_2 \
  --stage2_ckpt projects/bigtom/checkpoints/stage2_qwen/epoch_1 \
  --out projects/bigtom/checkpoints/stage3_qwen \
  --max_steps 300 \
  --save_every 100
```

## Evaluation

```bash
python projects/bigtom/scripts/evaluate_official_benchmarks.py \
  --datasets bigtom \
  --mode grpo \
  --stage1_ckpt projects/bigtom/checkpoints/stage1_qwen/epoch_2 \
  --policy_ckpt projects/bigtom/checkpoints/stage3_qwen/step_300 \
  --out_dir projects/bigtom/runs/official_bigtom
```

`evaluate_bigtom_official.py` is a narrower BigToM-only implementation of the
released multiple-choice protocol. `evaluate_official_benchmarks.py` is the
preferred unified entrypoint.

## Zero-shot transfer

The BigToM-trained policy is transferred without target-dataset fine-tuning to
ToMi and FANToM.

Validate all paper benchmark paths without loading a model:

```bash
python projects/bigtom/scripts/evaluate_official_benchmarks.py \
  --datasets all \
  --out_dir projects/bigtom/runs/transfer_dry_run \
  --dry_run
```

Evaluate one or both transfer datasets with `--datasets tomi,fantom`. The ToMi
and FANToM project READMEs describe dataset-specific scoring and path overrides.

## Tests and analyses

```bash
pytest -q projects/bigtom/tests
```

`experiments/e1/` contains ToMi belief minimal pairs.
`experiments/posterior/` contains posterior and leave-one-component-out
ablations. Generated artifacts belong in `projects/bigtom/runs/`.
