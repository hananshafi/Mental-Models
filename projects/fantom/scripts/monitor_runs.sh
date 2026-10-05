#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for run_dir in "${ROOT}"/runs/qwen25_7b_base \
               "${ROOT}"/runs/bigtom_sft_epoch1 \
               "${ROOT}"/runs/bigtom_grpo_step300 \
               "${ROOT}"/runs/sotopia_qwen_grpo_best; do
  model_id="$(basename "${run_dir}")"
  pid_file="${run_dir}/run.pid"
  manifest="${run_dir}/run_manifest.json"
  predictions="${run_dir}/predictions.jsonl"
  partial_summary="${run_dir}/partial_1000/summary.json"
  session_name="fantom_${model_id}"
  status="not-started"
  partial_status="waiting"
  predictions_count=0
  pid="-"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    pid="$(tmux display-message -p -t "${session_name}" '#{pane_pid}')"
    status="running"
  elif [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    status="stopped"
  fi

  if [[ -f "${manifest}" ]]; then
    manifest_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "${manifest}")"
    status="${status}/${manifest_status}"
  fi

  if [[ -f "${predictions}" ]]; then
    predictions_count="$(wc -l <"${predictions}")"
  fi
  if [[ -f "${partial_summary}" ]]; then
    partial_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "${partial_summary}")"
  fi

  printf '%-30s pid=%-8s status=%-18s predictions=%-5s partial1000=%s\n' \
    "${model_id}" "${pid}" "${status}" "${predictions_count}" "${partial_status}"
done

if tmux has-session -t fantom_partial_1000 2>/dev/null; then
  partial_pid="$(tmux display-message -p -t fantom_partial_1000 '#{pane_pid}')"
  echo "partial scorer: running (tmux=fantom_partial_1000 pid=${partial_pid})"
elif [[ -f "${ROOT}/runs/partial_1000.pid" ]]; then
  echo "partial scorer: stopped"
else
  echo "partial scorer: not-started"
fi
