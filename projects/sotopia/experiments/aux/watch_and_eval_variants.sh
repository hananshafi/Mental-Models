#!/bin/bash
# Watches each individual-component Stage-1 variant; the moment its epoch_0 checkpoint
# saves, evaluates reward-regression correlation and reports. Runs each eval as soon as
# ready (not waiting for the others), on whichever GPU is free at that moment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python}"
D="${PROJECT_ROOT}/runs/aux"
mkdir -p "${D}/logs"
LOG(){ echo "[$(date +%H:%M) watch] $*"; }

pick_gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>22000{print $1; exit}'; }

eval_variant(){
  local dirname=$1 tag=$2
  LOG "waiting for $D/ckpt_s_$dirname/epoch_0/joint_outcome_head.pth"
  while [ ! -f "$D/ckpt_s_$dirname/epoch_0/joint_outcome_head.pth" ]; do sleep 30; done
  sleep 15
  LOG "$tag epoch_0 ready; evaluating"
  local g=""
  while [ -z "$g" ]; do g=$(pick_gpu); [ -z "$g" ] && sleep 30; done
  CUDA_VISIBLE_DEVICES=$g "${PY}" "${SCRIPT_DIR}/sotopia_reward_eval.py" \
    --ckpt $D/ckpt_s_$dirname/epoch_0 --tag ${tag}_ep0 > $D/logs/sreval_${tag}_ep0.log 2>&1
  if [ -f "$D/sreval_${tag}_ep0.json" ]; then
    LOG "RESULT_READY $tag: $(cat $D/sreval_${tag}_ep0.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"corr_mean={d[\"reward_regression_corr_mean\"]:.3f} hardneg_acc={d[\"hardneg_pref_acc_scalar\"]:.1f}")')"
  else
    LOG "EVAL_FAILED $tag"
  fi
}

eval_variant nopref   nopref &
eval_variant nomental nomental &
eval_variant noexpl   noexpl &
wait
LOG "ALL_VARIANT_EVALS_DONE"
