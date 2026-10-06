# FANToM Rebuttal Evaluation

This folder contains all new FANToM rebuttal scripts, configuration, reports, and
run artifacts. It deliberately leaves the existing SOTOPIA, BigToM, and MMRole
codebases unchanged.

## Scope

FANToM's official repository states that the benchmark is for evaluation only.
Accordingly, this code performs cross-dataset evaluation and does not train,
fine-tune, select checkpoints, or optimize prompts on FANToM labels.

The primary rebuttal comparison is:

1. Qwen2.5-7B-Instruct base model.
2. BigToM Stage-2 SFT model with the learned first-/second-order latent prefix.
3. BigToM Stage-3 GRPO model with the same latent prefix.
4. SOTOPIA Qwen GRPO policy as a policy-only transfer control.

The BigToM models are the cleanest test of the paper's latent mental-model
mechanism because their inference path constructs `z1` and `z2` from each
FANToM story/question and projects both into the policy prefix. The SOTOPIA
policy is useful as a transfer control, but its mental model was used as a
training-time reward and is not consumed by the policy at inference.

## Layout

- `configs/models.json`: dataset paths and registered local checkpoints.
- `scripts/check_compatibility.py`: static dataset/checkpoint compatibility audit.
- `scripts/evaluate_fantom.py`: resumable inference and official scoring.
- `scripts/summarize_runs.py`: compact Markdown comparison table.
- `scripts/run_smoke.sh`: two-example model-loading smoke test.
- `scripts/run_matrix.sh`: full base/SFT/GRPO/SOTOPIA evaluation matrix.
- `scripts/launch_parallel.sh`: persistent parallel launch on dedicated GPUs.
- `scripts/score_partial.py`: official scoring for immutable prediction prefixes.
- `scripts/launch_partial_1000.sh`: background first-1,000 snapshot watcher.
- `scripts/monitor_runs.sh`: tmux/process status for all full runs.
- `reports/`: committed compatibility and result summaries.
- `runs/`: all newly generated predictions, manifests, summaries, and logs.

Large prediction files and model artifacts under `runs/` are ignored by git.

## Environment

Use the existing ToM environment and local Hugging Face cache:

```bash
export HF_HOME=artifacts/huggingface
export PYTHON=python
```

The latent BigToM runner uses two visible GPUs when available: one for the
Stage-1 encoder and one for the policy. The base and plain-adapter runners use
the first visible GPU.

## Compatibility Audit

```bash
$PYTHON rebuttal/fantom/scripts/check_compatibility.py
```

The audit checks:

- 870 official conversations and 12,832 flattened probes;
- 3--6 distinct speakers per conversation;
- base-model agreement between LoRA adapters and configured base models;
- required Stage-1 heads, latent projector, and adapter weights;
- the prior full official BigToM-to-FANToM result.

## Smoke Test

From the paper repository root:

```bash
CUDA_VISIBLE_DEVICES=1,3 rebuttal/fantom/scripts/run_smoke.sh bigtom_grpo_step300
CUDA_VISIBLE_DEVICES=1 rebuttal/fantom/scripts/run_smoke.sh sotopia_qwen_grpo_best
```

Smoke tests use two examples and skip the official aggregate scorer. They test
model loading, prompt construction, generation/choice scoring, local scoring,
resumption files, and summary serialization.

## Full Evaluation

Run a single model:

```bash
CUDA_VISIBLE_DEVICES=1,3 $PYTHON rebuttal/fantom/scripts/evaluate_fantom.py \
  --model-id bigtom_grpo_step300 \
  --run-dir rebuttal/fantom/runs/bigtom_grpo_step300 \
  --official \
  --allow-model-download
```

Run the registered comparison matrix sequentially:

```bash
CUDA_VISIBLE_DEVICES=1,3 rebuttal/fantom/scripts/run_matrix.sh
```

Launch the four primary runs in persistent parallel tmux sessions:

```bash
rebuttal/fantom/scripts/launch_parallel.sh
rebuttal/fantom/scripts/monitor_runs.sh
```

Completed examples are appended immediately to `predictions.jsonl`, so an
interrupted run can continue with `--resume`.

## First-1,000 Results

Keep the full evaluations running and score an immutable snapshot of the first
1,000 predictions from each model:

```bash
rebuttal/fantom/scripts/launch_partial_1000.sh
rebuttal/fantom/scripts/monitor_runs.sh
```

The watcher runs the official FANToM bridge on CPU as each model reaches 1,000
predictions. It writes per-model artifacts under `runs/<model>/partial_1000/`
and the live comparison table to `reports/partial_1000.md`. It never truncates,
rewrites, or stops the full prediction runs.

These scores are preliminary: the first 1,000 probes are a contiguous prefix,
not a random or stratified sample, and the boundary may cut through one
question set. Use the complete 12,832-probe scores for final claims.

## Summaries

```bash
$PYTHON rebuttal/fantom/scripts/summarize_runs.py \
  --output rebuttal/fantom/reports/results.md
```

Official FANToM aggregate scores are percentages. Partial smoke runs only
produce local row-level metrics and are not suitable for rebuttal claims.
