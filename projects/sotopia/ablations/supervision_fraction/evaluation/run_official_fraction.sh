#!/usr/bin/env bash
set -euo pipefail

umask 077

fraction="${1:?fraction is required}"
gpu="${2:?GPU index is required}"

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
PROJECT_ROOT="$(cd "${ABLATION_ROOT}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
EVALUATOR="${EVAL_ROOT}/stage3_evaluate_sotopia_utf8.py"
BASE_MODEL="${MENTAL_MODELS_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VERIFIER="${EVAL_ROOT}/verify_evaluator_source.py"

case "${fraction}" in
  0)
    adapter="${PROJECT_ROOT}/checkpoints/grpo_agent_qwen_v3/sft_warmup"
    ;;
  25)
    adapter="${ABLATION_ROOT}/runs/fraction_25/stage2/step_300"
    ;;
  50)
    adapter="${ABLATION_ROOT}/runs/fraction_50/stage2/step_300"
    ;;
  50_extended)
    adapter="${ABLATION_ROOT}/runs/fraction_50_extended/stage2/step_300"
    ;;
  100)
    adapter="${PROJECT_ROOT}/checkpoints/grpo_agent_qwen_v3/step_300"
    ;;
  *)
    echo "Unsupported fraction: ${fraction}" >&2
    exit 2
    ;;
esac

output="${EVAL_ROOT}/results/fraction_${fraction}_official_all.jsonl"
summary="${output%.jsonl}_summary.json"
log="${EVAL_ROOT}/logs/fraction_${fraction}_official_all.log"
pid_file="${EVAL_ROOT}/pids/fraction_${fraction}.pid"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before launching evaluation}"
if [[ -s "${output}" || -s "${summary}" ]]; then
  echo "Refusing to overwrite existing output for fraction ${fraction}." >&2
  exit 1
fi

exec >"${log}" 2>&1
printf '%s\n' "$$" >"${pid_file}"
"${PYTHON}" "${VERIFIER}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/artifacts/huggingface}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${gpu}"

exec "${PYTHON}" "${EVALUATOR}" \
  --policy_model_name "${BASE_MODEL}" \
  --policy_adapter_path "${adapter}" \
  --merge_adapter \
  --use_hf \
  --deduplicate_envs \
  --task all \
  --output_path "${output}" \
  --max_episodes 90 \
  --max_turns 10 \
  --policy_agent_index 0 \
  --partner_model gpt-4o-mini \
  --judge_model gpt-4o \
  --temperature 0.7 \
  --top_p 0.9 \
  --seed 42 \
  --gpu "${gpu}"
