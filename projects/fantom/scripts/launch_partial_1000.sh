#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
SESSION_NAME="fantom_partial_1000"
LOG_FILE="${ROOT}/runs/partial_1000.log"
PID_FILE="${ROOT}/runs/partial_1000.pid"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"
mkdir -p "${ROOT}/runs" "${ROOT}/reports"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Partial scorer is already running in tmux session ${SESSION_NAME}"
  exit 0
fi

printf -v command \
  'exec env HF_HOME=%q CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 %q %q --limit 1000 --wait --allow-model-download >%q 2>&1' \
  "${HF_HOME}" \
  "${PYTHON}" \
  "${ROOT}/scripts/score_partial.py" \
  "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "${command}"
pid="$(tmux display-message -p -t "${SESSION_NAME}" '#{pane_pid}')"
echo "${pid}" >"${PID_FILE}"
echo "launched first-1000 scorer on CPU: tmux=${SESSION_NAME} PID=${pid}"
