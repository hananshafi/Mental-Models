# ToM-SB / AIDA extension

Research extension for recursive mental/reward learning in adversarial
information-extraction games. This code is separate from the main paper
pipelines and includes data generation, structured Stage 1 training, policy SFT
and GRPO, pairwise reward evaluation, and an AIDA official-evaluation bridge.

## Environment and upstream code

Most local scripts run in `mental-models`. The released AIDA stack requires its
separate environment:

```bash
./tools/bootstrap_third_party.sh aida
conda env create -f environments/aida.yml
conda activate mental-models-aida
```

`third_party/patches/aida-local.patch` removes mandatory remote logging and
provides portable local defaults.

## Data generation

```bash
python projects/tom_sb/scripts/generate_tom_sb_data.py \
  --output_path projects/tom_sb/data/tom_sb_train.jsonl
```

See `docs/original_runbook.md` for provider and generation options. Generated
JSONL files are ignored.

## Training

The recommended structured reward implementation is:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  projects/tom_sb/scripts/train_tom_sb_mental_reward_structured.py \
  --train_path projects/tom_sb/data/tom_sb_train.jsonl \
  --val_path projects/tom_sb/data/tom_sb_val.jsonl \
  --output_dir projects/tom_sb/checkpoints/mental_reward
```

Then run policy SFT and GRPO:

```bash
CUDA_VISIBLE_DEVICES=0 python projects/tom_sb/scripts/train_tom_sb_policy_sft.py
CUDA_VISIBLE_DEVICES=0 python projects/tom_sb/scripts/train_tom_sb_policy_grpo.py
```

Use explicit `--train_path`, `--reward_checkpoint`, and `--output_dir` values
when your artifact names differ from the defaults.

## AIDA evaluation

Copy `configs/aida_official_eval.example.yaml`, update the local checkpoint
directory, and invoke the pinned AIDA launcher from `third_party/src/aida`.
Keep AIDA outputs under `projects/tom_sb/runs/`.
