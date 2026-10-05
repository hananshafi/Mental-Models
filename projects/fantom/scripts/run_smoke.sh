#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL_ID="${1:-bigtom_grpo_step300}"
RUN_DIR="${ROOT}/runs/smoke_${MODEL_ID}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"

exec "${PYTHON}" "${ROOT}/scripts/evaluate_fantom.py" \
  --model-id "${MODEL_ID}" \
  --run-dir "${RUN_DIR}" \
  --limit 2 \
  --max-new-tokens 32 \
  --log-every 1 \
  --overwrite
