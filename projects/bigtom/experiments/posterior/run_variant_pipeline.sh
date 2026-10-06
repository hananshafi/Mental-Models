#!/bin/bash
# Usage: run_variant_pipeline.sh TAG GPUA GPUB
# Waits for Stage-1 ckpt, then Stage-2 SFT -> Stage-3 GRPO -> evaluation (BigToM).
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
LOG "Stage-1 ready; starting Stage-2 SFT"

# Stage 2: latent-prefix SFT (epochs=2 for the matched warmup)
CUDA_VISIBLE_DEVICES="${GA},${GB}" "${PY}" "${PROJECT_ROOT}/scripts/stage2_policy_sft.py" \
  --data "${DATA}" --stage1_ckpt "${S1}" --out "${D}/stage2_${TAG}" --epochs 2 \
  > "${D}/logs/stage2_${TAG}.log" 2>&1
S2="${D}/stage2_${TAG}/epoch_0"
[ -d "${S2}" ] || S2="${D}/stage2_${TAG}/epoch_1"
LOG "Stage-2 done -> $S2 ; starting Stage-3 GRPO"

# Stage 3: GRPO (300 steps)
CUDA_VISIBLE_DEVICES="${GA},${GB}" "${PY}" "${PROJECT_ROOT}/scripts/stage3_grpo.py" \
  --data "${DATA}" --stage1_ckpt "${S1}" --stage2_ckpt "${S2}" \
  --out "${D}/stage3_${TAG}" --max_steps 300 --save_every 300 \
  > "${D}/logs/stage3_${TAG}.log" 2>&1
S3="${D}/stage3_${TAG}/step_300"
LOG "Stage-3 done -> $S3 ; starting evaluation (BigToM)"

# Evaluation on official BigToM (TB^FB); rule-based grading (no OpenAI judge)
CUDA_VISIBLE_DEVICES="${GA},${GB}" "${PY}" "${PROJECT_ROOT}/scripts/evaluate_bigtom_official.py" \
  --mode grpo --stage1_ckpt "${S1}" --policy_ckpt "${S3}" \
  --bigtom_csv "${REPO_ROOT}/third_party/src/bigtom/data/bigtom/bigtom.csv" \
  --out_dir "${D}/eval_${TAG}" \
  > "${D}/logs/eval_${TAG}.log" 2>&1
LOG "Evaluation done -> $D/eval_$TAG"
echo "PIPELINE_DONE_$TAG"
