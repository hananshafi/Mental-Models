#!/usr/bin/env bash
set -euo pipefail

umask 077

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${ABLATION_ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
EVALUATOR="${EVAL_ROOT}/stage3_evaluate_sotopia_utf8.py"
BASE_MODEL="${MENTAL_MODELS_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUNNER="${EVAL_ROOT}/run_official_fraction.sh"
VERIFIER="${EVAL_ROOT}/verify_evaluator_source.py"
TMUX=(tmux -L sotopia_supervision_eval)

fractions=(0 25 50)
gpus=(0 1 2)
adapters=(
  "${PROJECT_ROOT}/checkpoints/grpo_agent_qwen_v3/sft_warmup"
  "${ABLATION_ROOT}/runs/fraction_25/stage2/step_300"
  "${ABLATION_ROOT}/runs/fraction_50/stage2/step_300"
)

mkdir -p "${EVAL_ROOT}/results" "${EVAL_ROOT}/logs" "${EVAL_ROOT}/pids"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before launching evaluation}"
test -f "${EVALUATOR}"
test -x "${RUNNER}"
command -v tmux >/dev/null
"${PYTHON}" "${VERIFIER}"

for index in "${!fractions[@]}"; do
  fraction="${fractions[$index]}"
  adapter="${adapters[$index]}"
  output="${EVAL_ROOT}/results/fraction_${fraction}_official_all.jsonl"
  summary="${output%.jsonl}_summary.json"
  pid_file="${EVAL_ROOT}/pids/fraction_${fraction}.pid"
  session="sotopia_supervision_${fraction}"

  test -f "${adapter}/adapter_config.json"

  if [[ -s "${output}" || -s "${summary}" ]]; then
    echo "Refusing to overwrite existing output for fraction ${fraction}%." >&2
    exit 1
  fi

  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(cat "${pid_file}")"
    if kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Fraction ${fraction}% is already running as PID ${existing_pid}." >&2
      exit 1
    fi
  fi

  if "${TMUX[@]}" has-session -t "${session}" 2>/dev/null; then
    echo "Fraction ${fraction}% already has tmux session ${session}." >&2
    exit 1
  fi
done

for index in "${!fractions[@]}"; do
  fraction="${fractions[$index]}"
  gpu="${gpus[$index]}"
  output="${EVAL_ROOT}/results/fraction_${fraction}_official_all.jsonl"
  pid_file="${EVAL_ROOT}/pids/fraction_${fraction}.pid"
  session="sotopia_supervision_${fraction}"

  "${TMUX[@]}" new-session -d -s "${session}" "${RUNNER}" "${fraction}" "${gpu}"
  sleep 5

  if ! "${TMUX[@]}" has-session -t "${session}" 2>/dev/null; then
    echo "Fraction ${fraction}% exited during startup; inspect its log." >&2
    exit 1
  fi

  pid="$(cat "${pid_file}")"
  printf 'fraction=%s gpu=%s pid=%s tmux=%s output=%s\n' \
    "${fraction}" "${gpu}" "${pid}" "${session}" "${output}"
done
