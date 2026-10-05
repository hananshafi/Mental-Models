#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"

launch() {
  local model_id="$1"
  local gpu_ids="$2"
  local run_dir="${ROOT}/runs/${model_id}"
  local pid_file="${run_dir}/run.pid"
  local log_file="${run_dir}/run.log"
  local session_name="fantom_${model_id}"

  mkdir -p "${run_dir}"
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "${model_id} is already running in tmux session ${session_name}"
    return
  fi

  local command
  printf -v command     'exec env HF_HOME=%q CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %q %q --model-id %q --run-dir %q --official --allow-model-download --resume --log-every 25 >%q 2>&1'     "${HF_HOME}"     "${gpu_ids}"     "${PYTHON}"     "${ROOT}/scripts/evaluate_fantom.py"     "${model_id}"     "${run_dir}"     "${log_file}"

  tmux new-session -d -s "${session_name}" "${command}"
  local pid
  pid="$(tmux display-message -p -t "${session_name}" '#{pane_pid}')"
  echo "${pid}" >"${pid_file}"
  echo "launched ${model_id} on GPU(s) ${gpu_ids}: tmux=${session_name} PID=${pid}"
}

launch qwen25_7b_base 1
launch bigtom_sft_epoch1 3,4
launch bigtom_grpo_step300 5,6
launch sotopia_qwen_grpo_best 7
