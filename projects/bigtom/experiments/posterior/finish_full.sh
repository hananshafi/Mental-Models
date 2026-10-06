#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PY="${PYTHON:-python}"
D="${PROJECT_ROOT}/runs/posterior"
DATA="${PROJECT_ROOT}/data/bigtom_qwen_5k_annotated.jsonl"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${D}/logs"
echo "[full] Stage-3 GRPO (GPU6,7)"
CUDA_VISIBLE_DEVICES=6,7 "${PY}" "${PROJECT_ROOT}/scripts/stage3_grpo.py" --data "${DATA}" \
  --stage1_ckpt "${D}/ckpt_full" --stage2_ckpt "${D}/stage2_full/epoch_0" \
  --out "${D}/stage3_full" --max_steps 150 --save_every 150 > "${D}/logs/stage3_full.log" 2>&1
echo "[full] Evaluation"
CUDA_VISIBLE_DEVICES=6,7 "${PY}" "${PROJECT_ROOT}/scripts/evaluate_bigtom_official.py" --mode grpo \
  --stage1_ckpt "${D}/ckpt_full" --policy_ckpt "${D}/stage3_full/step_150" \
  --bigtom_csv "${REPO_ROOT}/third_party/src/bigtom/data/bigtom/bigtom.csv" \
  --out_dir "${D}/eval_full" > "${D}/logs/eval_full.log" 2>&1
echo "FULL_DONE"
