#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODELS=(
  qwen25_7b_base
  bigtom_sft_epoch1
  bigtom_grpo_step300
  sotopia_qwen_grpo_best
)

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"

for model_id in "${MODELS[@]}"; do
  "${PYTHON}" "${ROOT}/scripts/evaluate_fantom.py" \
    --model-id "${model_id}" \
    --run-dir "${ROOT}/runs/${model_id}" \
    --official \
    --allow-model-download \
    --resume
done

"${PYTHON}" "${ROOT}/scripts/summarize_runs.py"
