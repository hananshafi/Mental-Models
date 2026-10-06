# FANToM BigToM Adapter-Loading Audit

## Finding

The original FANToM SFT and GRPO runs referenced different checkpoint
directories and adapter files, but both in-memory policies had freshly
initialized LoRA `B` tensors with norm `0.0`. Their first 1,000 predictions and
multiple-choice scores were therefore identical and are invalid.

The failure occurred when PEFT loaded an adapter after the base policy had been
placed on the second visible GPU (`cuda:1`). `PeftModel.from_pretrained` created
the adapter modules but silently failed to install the saved tensors.

## Checkpoints

| Model | Adapter SHA-256 | Saved LoRA-B norm |
|---|---|---:|
| BigToM SFT epoch 1 | `53b083fe002de54a5004fcd186bf80bc932356c59afd8a98a1b4a79aaee68d39` | 3.397536 |
| BigToM GRPO step 300 | `0e7c5d1524476dc3ee72a9b302de850df0eb6332cee87ed5b80a74aaf1ad3115` | 0.054019 |

Stage-3 checkpoint metadata confirms that GRPO step 300 was initialized from
the configured Stage-2 SFT checkpoint. The two adapter files have different
inodes, hashes, and effective weights.

## Repair

The rebuttal evaluator now explicitly reloads the PEFT state dict onto the
policy device and verifies all 224 adapter tensors exactly against the saved
checkpoint before generating any prediction. The run manifest records:

- canonical adapter directory;
- adapter SHA-256;
- policy device;
- tensor count;
- exact-match status and maximum tensor delta;
- checkpoint and loaded LoRA-B norms.

The fixed first FanToM multiple-choice probe produced:

| Model | Choice scores |
|---|---|
| BigToM SFT epoch 1 | `[-0.2300832123, -0.0223729946]` |
| BigToM GRPO step 300 | `[-0.4935603440, -0.2300296873]` |

The scores are no longer identical. Both affected full runs were restarted
from zero. Invalid outputs were preserved under
`runs/invalid_adapter_load_20260724T235337Z/`.
