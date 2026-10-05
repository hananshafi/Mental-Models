#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

ENV_NAME="${ENV_NAME:-tom}"

SEED_CSV="${SEED_CSV:-${REPO_ROOT}/third_party/src/bigtom/data/bigtom/bigtom.csv}"
OUT_CSV="${OUT_CSV:-${ROOT_DIR}/data/bigtom_qwen.csv}"
OUT_JSONL="${OUT_JSONL:-${ROOT_DIR}/data/bigtom_qwen_annotated.jsonl}"

NUM_NEW_SCENARIOS="${NUM_NEW_SCENARIOS:-5000}"
K_SHOT="${K_SHOT:-4}"
GEN_BATCH="${GEN_BATCH:-128}"
GEN_MAX_WORKERS="${GEN_MAX_WORKERS:-48}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-1400}"
GEN_SEED="${GEN_SEED:-0}"

ANN_BATCH="${ANN_BATCH:-48}"
ANN_MAX_WORKERS="${ANN_MAX_WORKERS:-24}"
ANN_MAX_TOKENS="${ANN_MAX_TOKENS:-6000}"
ANN_TEMPERATURE="${ANN_TEMPERATURE:-0.4}"
ANN_LIMIT="${ANN_LIMIT:-}"

RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_ANNOTATE="${RUN_ANNOTATE:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
GEN_LOG="${LOG_DIR}/generate_${STAMP}.log"
ANN_LOG="${LOG_DIR}/annotate_${STAMP}.log"

echo "BigToM data pipeline"
echo "  env:             ${ENV_NAME}"
echo "  seed_csv:        ${SEED_CSV}"
echo "  out_csv:         ${OUT_CSV}"
echo "  out_jsonl:       ${OUT_JSONL}"
echo "  new_scenarios:   ${NUM_NEW_SCENARIOS}"
echo "  generate_log:    ${GEN_LOG}"
echo "  annotate_log:    ${ANN_LOG}"

if [[ "${RUN_GENERATE}" == "1" ]]; then
  echo
  echo "[1/2] Generating scenarios"
  conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/generate_scenarios.py" \
    --seed_csv "${SEED_CSV}" \
    --out "${OUT_CSV}" \
    --n "${NUM_NEW_SCENARIOS}" \
    --k_shot "${K_SHOT}" \
    --batch "${GEN_BATCH}" \
    --max_workers "${GEN_MAX_WORKERS}" \
    --temperature "${GEN_TEMPERATURE}" \
    --max_tokens "${GEN_MAX_TOKENS}" \
    --seed "${GEN_SEED}" | tee "${GEN_LOG}"
fi

if [[ "${RUN_ANNOTATE}" == "1" ]]; then
  echo
  echo "[2/2] Annotating scenarios"
  ANN_CMD=(
    conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/annotate_scenarios.py"
    --in "${OUT_CSV}"
    --out "${OUT_JSONL}"
    --batch "${ANN_BATCH}"
    --max_workers "${ANN_MAX_WORKERS}"
    --max_tokens "${ANN_MAX_TOKENS}"
    --temperature "${ANN_TEMPERATURE}"
  )
  if [[ -n "${ANN_LIMIT}" ]]; then
    ANN_CMD+=(--limit "${ANN_LIMIT}")
  fi
  "${ANN_CMD[@]}" | tee "${ANN_LOG}"
fi

echo
echo "Done."
echo "  csv:   ${OUT_CSV}"
echo "  jsonl: ${OUT_JSONL}"
