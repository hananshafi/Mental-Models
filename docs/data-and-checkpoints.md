# Data and checkpoint conventions

All commands in project READMEs assume execution from the repository root.
Generated artifacts use the following ignored directories:

```text
projects/<project>/data/         downloaded, normalized, or annotated data
projects/<project>/checkpoints/  model, adapter, projector, and reward heads
projects/<project>/runs/         predictions, summaries, figures, and logs
artifacts/huggingface/           shared Hugging Face cache
```

## Released annotations

The paper-ready annotations are published together in
[`hanangani/Mental-Model-Annotation-Dataset`](https://huggingface.co/datasets/hanangani/Mental-Model-Annotation-Dataset).
Run `python tools/download_data.py` to download the pinned release and create
the following local links:

```text
projects/sotopia/data/sotopia_turn_rewards_v3.jsonl
projects/sotopia/data/mental_model_persona_dataset.jsonl
projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl
projects/mmrole/training_data/
```

MMRole images are not redistributed in this release. Run
`python tools/download_mmrole_images.py` to fetch the referenced MMRole
character images and COCO train2017 images into `projects/mmrole/images/`.

## Canonical checkpoint names

### SOTOPIA

```text
projects/sotopia/checkpoints/
├── coupled_mental_reward_qwen_v3/
│   └── best/
└── grpo_agent_qwen_v3/
    ├── sft_warmup/
    ├── step_300/
    └── best/
```

The supervision-fraction ablation writes its own checkpoints under
`projects/sotopia/ablations/supervision_fraction/runs/`.

### BigToM

```text
projects/bigtom/checkpoints/
├── stage1_qwen_5k/best_ckpt/
├── stage2_qwen_5k/epoch_1/
└── stage3_qwen_5k/step_300/
```

Stage 1 contains the mental encoder, LoRA adapter, and learned heads. Stage 2
contains the latent-prefix SFT policy and projector. Stage 3 contains the GRPO
policy adapter and projector.

### MMRole

```text
projects/mmrole/checkpoints/
├── stage0_reward/
├── stage1_sft/
├── stage2_grpo/
└── stage3_dpo/
```

MMRole data preparation also creates `raw_data/`, `images/`,
`character_profiles/`, and `training_data/` under `projects/mmrole/`; all are
ignored.

## External datasets

`tools/bootstrap_third_party.sh` installs released benchmark files under:

```text
third_party/src/bigtom
third_party/src/fantom
third_party/src/tomi
```

The bootstrap also extracts the ToMi test split to
`third_party/src/tomi/tomi_balanced_story_types/` and downloads the
checksum-verified FANToM data to `third_party/src/fantom/data/fantom/`.

Do not train on transfer-test labels. FANToM and ToMi are used by the shared
BigToM evaluator for zero-shot transfer.

## Local path overrides

Most CLIs accept explicit data, checkpoint, and output arguments. FANToM uses
`projects/fantom/configs/models.json`; copy it to `models.local.json`, change
paths there, and pass `--config projects/fantom/configs/models.local.json`.
The local file is ignored by Git.
