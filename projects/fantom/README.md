# FANToM multi-party transfer

Evaluation-only transfer of existing BigToM and SOTOPIA policies to the
multi-party FANToM benchmark. No FANToM labels are used for training,
fine-tuning, prompt selection, or checkpoint selection.

The primary BigToM models encode each FANToM story/question into first- and
second-order latent prefixes at inference. The SOTOPIA adapter is a policy-only
cross-domain control; its mental model was used during training but is not
loaded at deployment.

## Setup

```bash
./tools/bootstrap_third_party.sh fantom
cp projects/fantom/configs/models.json projects/fantom/configs/models.local.json
# Edit models.local.json only if your checkpoints use different names.
```

The tracked registry expects the checkpoint layout in
`docs/data-and-checkpoints.md`. Pass the local registry with
`--config projects/fantom/configs/models.local.json` when customized.

## Compatibility audit

```bash
python projects/fantom/scripts/check_compatibility.py
```

This verifies the released split shape and, when present, adapter/projector
assets. Large checkpoint checks are local and never download or train models.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 bash projects/fantom/scripts/run_smoke.sh bigtom_grpo_step300
```

The smoke run evaluates two questions with local scoring. For the official
scorer, invoke the evaluator directly with `--official`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python projects/fantom/scripts/evaluate_fantom.py \
  --model-id bigtom_grpo_step300 \
  --run-dir projects/fantom/runs/bigtom_grpo_step300 \
  --official \
  --allow-model-download \
  --resume
```

## Full matrix and background launch

```bash
bash projects/fantom/scripts/run_matrix.sh
```

For one tmux session per model:

```bash
bash projects/fantom/scripts/launch_parallel.sh
bash projects/fantom/scripts/monitor_runs.sh
```

GPU assignments and the default model list are explicit in the launch scripts;
edit or override them before using a different machine.

## Partial official scores

```bash
bash projects/fantom/scripts/launch_partial_1000.sh
bash projects/fantom/scripts/monitor_runs.sh
```

`score_partial.py` watches immutable prefixes of each prediction file and calls
the official FANToM scorer. Full and partial summaries are percentages.

## Tests and reports

```bash
pytest -q projects/fantom/tests
```

Compact historical rebuttal reports are preserved in `reports/`. New
predictions, manifests, and summaries are written to the ignored `runs/`
directory.
