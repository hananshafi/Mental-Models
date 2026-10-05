#!/bin/bash
# Usage: run_variant_pipeline.sh TAG GPUA GPUB
# Waits for Stage-1 ckpt, then Stage-3 SFT -> Stage-4 GRPO -> Stage-5 eval (BigToM).
set -euo pipefail
TAG=$1; GA=$2; GB=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PY="${PYTHON:-python}"
D="${PROJECT_ROOT}/runs/posterior"
DATA="${PROJECT_ROOT}/data/bigtom_qwen_5k_annotated.jsonl"
S1="${D}/ckpt_${TAG}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${D}/logs"
LOG(){ echo "[$(date +%H:%M) $TAG] $*"; }

# 1) wait for stage-1 checkpoint
LOG "waiting for Stage-1 ckpt $S1/heads.pt"
while [ ! -f "$S1/heads.pt" ]; do sleep 30; done
sleep 20  # let lora finish flushing
LOG "Stage-1 ready; starting Stage-3 SFT"

# 2) Stage-3 SFT (epochs=1 for a matched, faster warmup)
CUDA_VISIBLE_DEVICES="${GA},${GB}" "${PY}" "${PROJECT_ROOT}/scripts/stage3_policy_sft.py" \
  --data "${DATA}" --stage1_ckpt "${S1}" --out "${D}/stage3_${TAG}" --epochs 2 \
  > "${D}/logs/stage3_${TAG}.log" 2>&1
S3="${D}/stage3_${TAG}/epoch_0"
[ -d "${S3}" ] || S3="${D}/stage3_${TAG}/epoch_1"
LOG "Stage-3 done -> $S3 ; starting Stage-4 GRPO"

# 3) Stage-4 GRPO (150 steps)
CUDA_VISIBLE_DEVICES="${GA},${GB}" "${PY}" "${PROJECT_ROOT}/scripts/stage4_grpo.py" \
  --data "${DATA}" --stage1_ckpt "${S1}" --stage3_ckpt "${S3}" \
  --out "${D}/stage4_${TAG}" --max_steps 300 --save_every 300 \
  > "${D}/logs/stage4_${TAG}.log" 2>&1
S4="${D}/stage4_${TAG}/step_300"
LOG "Stage-4 done -> $S4 ; starting Stage-5 eval (BigToM)"

# 4) Stage-5 eval on official BigToM (TB^FB); rule-based grading (no OpenAI judge)
CUDA_VISIBLE_DEVICES="${GA},${GB}" "${PY}" "${PROJECT_ROOT}/scripts/evaluate_bigtom_official.py" \
  --mode grpo --stage1_ckpt "${S1}" --policy_ckpt "${S4}" \
  --bigtom_csv "${REPO_ROOT}/third_party/src/bigtom/data/bigtom/bigtom.csv" \
  --out_dir "${D}/eval_${TAG}" \
  > "${D}/logs/stage5_${TAG}.log" 2>&1
LOG "Stage-5 done -> $D/eval_$TAG"
echo "PIPELINE_DONE_$TAG"
