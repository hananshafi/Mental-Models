#!/bin/bash
# noaux (all-aux-removed) Stage-1 -> Stage-2 (SFT+GRPO) -> Stage-3 (official SOTOPIA eval)
# Verifies real output artifacts at each stage (not just log text) before proceeding.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this pipeline}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"
PY="${PYTHON:-python}"
D="${PROJECT_ROOT}/runs/aux"
mkdir -p "${D}/logs"
LOG(){ echo "[$(date +%H:%M) noaux-pipe] $*"; }

# 1) wait for Stage-1 epoch_0 to actually save its head files
LOG "waiting for Stage-1 epoch_0: $D/ckpt_s_noaux/epoch_0/joint_outcome_head.pth"
while [ ! -f "$D/ckpt_s_noaux/epoch_0/joint_outcome_head.pth" ]; do sleep 60; done
sleep 20
LOG "Stage-1 epoch_0 ready"

# 2) Stage-2: SFT warmup + GRPO (single script), needs a free GPU (or pair)
pick_gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>38000{print $1; exit}'; }
LOG "waiting for a free GPU for Stage-2"
G=""
while [ -z "$G" ]; do G=$(pick_gpu); [ -z "$G" ] && sleep 60; done
LOG "Stage-2 (SFT+GRPO) on GPU $G"
env HF_HOME="${HF_HOME}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY}" "${PROJECT_ROOT}/scripts/stage2_grpo_agent_training_v3.py" \
  --reward_checkpoint_dir $D/ckpt_s_noaux/epoch_0 \
  --output_dir $D/grpo_noaux \
  --preset qwen --gpu $G --save_every 50 \
  > $D/logs/stage2_noaux.log 2>&1
# prefer "best", else the highest step_N, else the last epoch_N
if [ -f "$D/grpo_noaux/best/adapter_config.json" ]; then
  POLICY="$D/grpo_noaux/best"
else
  POLICY=$(ls -d $D/grpo_noaux/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
  [ -z "$POLICY" ] && POLICY=$(ls -d $D/grpo_noaux/epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi
if [ -z "$POLICY" ] || [ ! -f "$POLICY/adapter_config.json" ]; then
  LOG "STAGE2_FAILED (no policy checkpoint produced)"; exit 1
fi
LOG "Stage-2 done -> $POLICY"

# 3) Stage-3: official SOTOPIA eval (GPT-4o-mini partner, GPT-4o judge)
LOG "waiting for a free GPU for Stage-3 eval"
G=""
while [ -z "$G" ]; do G=$(pick_gpu); [ -z "$G" ] && sleep 60; done
env HF_HOME="${HF_HOME}" OPENAI_API_KEY="${OPENAI_API_KEY}" "${PY}" "${PROJECT_ROOT}/scripts/stage3_evaluate_sotopia.py" \
  --policy_adapter_path $POLICY \
  --output_path $D/eval_noaux_results.jsonl \
  --tag noaux_official --gpu $G \
  > $D/logs/stage3_noaux.log 2>&1
if [ ! -f "$D/eval_noaux_results.jsonl" ]; then
  LOG "STAGE3_FAILED (no eval output)"; exit 1
fi
LOG "NOAUX_PIPELINE_DONE -> $D/eval_noaux_results.jsonl"
