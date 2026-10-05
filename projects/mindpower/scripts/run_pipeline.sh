#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"
PYTHON="${PYTHON:-python}"

DRY_RUN=false
SIMULATOR="virtualhome"
LIMIT=-1
DATASET_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --simulator)
            SIMULATOR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

DRY_FLAG=""
if [ "$DRY_RUN" = true ]; then
    DRY_FLAG="--dry_run"
fi

DATASET_FLAG=""
if [ -n "$DATASET_ROOT" ]; then
    DATASET_FLAG="--virtualhome_dataset_root ${DATASET_ROOT}"
fi

echo ""
echo "═══ STEP 1: Collect Episodes ═══"
"${PYTHON}" "${SCRIPTS_DIR}/step1_collect_episodes.py" \
    --simulator "${SIMULATOR}" \
    ${DRY_FLAG} \
    ${DATASET_FLAG} \
    --limit "${LIMIT}"

echo ""
echo "═══ STEP 2: Build Intervention Points ═══"
"${PYTHON}" "${SCRIPTS_DIR}/step2_build_intervention_points.py"

echo ""
echo "═══ STEP 3: Annotate BDI ═══"
"${PYTHON}" "${SCRIPTS_DIR}/step3_annotate_bdi.py"

echo ""
echo "═══ STEP 4: Build Training Data ═══"
"${PYTHON}" "${SCRIPTS_DIR}/step4_build_training_data.py" \
    --tom_input_path "${BASE_DIR}/data/intermediate/tom_annotations_heuristic.jsonl"

echo ""
echo "═══ STAGE 0: Reward Scaffold ═══"
"${PYTHON}" "${SCRIPTS_DIR}/stage0_reward_mindpower.py"

echo ""
echo "═══ STAGE 1: SFT Scaffold ═══"
"${PYTHON}" "${SCRIPTS_DIR}/stage1_sft_mindpower.py"

echo ""
echo "═══ STAGE 2: GRPO Scaffold ═══"
"${PYTHON}" "${SCRIPTS_DIR}/stage2_grpo_mindpower.py"

echo ""
echo "Pipeline scaffold complete."
