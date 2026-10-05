# Data and checkpoint conventions

All commands in project READMEs assume execution from the repository root.
Generated artifacts use the following ignored directories:

```text
projects/<project>/data/         downloaded, normalized, or annotated data
projects/<project>/checkpoints/  model, adapter, projector, and reward heads
projects/<project>/runs/         predictions, summaries, figures, and logs
artifacts/huggingface/           shared Hugging Face cache
```

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
├── stage3_qwen_5k/epoch_1/
└── stage4_qwen_5k/step_300/
```

Stage 1 contains the mental encoder, LoRA adapter, and learned heads. Stage 3
contains the latent-prefix SFT policy and projector. Stage 4 contains the GRPO
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

Do not train on transfer-test labels. FANToM and ToMi are used by the shared
BigToM evaluator for zero-shot transfer.

## Local path overrides

Most CLIs accept explicit data, checkpoint, and output arguments. FANToM uses
`projects/fantom/configs/models.json`; copy it to `models.local.json`, change
paths there, and pass `--config projects/fantom/configs/models.local.json`.
The local file is ignored by Git.
