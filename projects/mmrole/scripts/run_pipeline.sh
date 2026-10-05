#!/bin/bash
# =============================================================================
# MMRole Visual ToM Data Generation Pipeline (Belief-Centric)
# =============================================================================
#
# Generates structured belief state annotations for multimodal Theory of Mind.
# Output is 4 training formats:
#   - belief_prediction: (image, agents, context) → structured belief state
#   - preference_pairs: (tom_aligned, tom_violation) contrastive pairs
#   - probe_qa: structured QA probes (visual_percept, 1st/2nd order, false belief)
#   - salience_prediction: per-agent visual salience maps
#
# Usage:
#   bash run_pipeline.sh                          # Full pipeline
#   bash run_pipeline.sh --pilot                  # 500 turns, gpt-4o-mini (~$5)
#   bash run_pipeline.sh --from-step 3            # Resume from step 3
#   bash run_pipeline.sh --skip-coco --coco-dir /path/to/train2017
# =============================================================================

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"

# Defaults
PILOT_MODE=false
FROM_STEP=1
SKIP_COCO=false
COCO_DIR=""
ANNOTATION_MODEL="gpt-5.4-mini"
ANNOTATION_WORKERS=5
MAX_ANNOTATION_EXAMPLES=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --pilot)
            PILOT_MODE=true
            ANNOTATION_MODEL="gpt-5.4-nano"
            MAX_ANNOTATION_EXAMPLES=500
            shift ;;
        --from-step)  FROM_STEP="$2"; shift 2 ;;
        --skip-coco)  SKIP_COCO=true; shift ;;
        --coco-dir)   COCO_DIR="$2"; shift 2 ;;
        --model)      ANNOTATION_MODEL="$2"; shift 2 ;;
        --workers)    ANNOTATION_WORKERS="$2"; shift 2 ;;
        --max-examples) MAX_ANNOTATION_EXAMPLES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$PILOT_MODE" = true ]; then
    ANNOTATED="${BASE_DIR}/mmrole_annotated_pilot.jsonl"
    CLEAN="${BASE_DIR}/mmrole_annotated_pilot_clean.jsonl"
    FLAGGED="${BASE_DIR}/mmrole_annotated_pilot_flagged.jsonl"
    TRAIN_DIR="${BASE_DIR}/training_data_pilot"
    echo "══════════════════════════════════════════"
    echo "  PILOT MODE: 500 turns, ${ANNOTATION_MODEL}"
    echo "��═════════════════════��═══════════════════"
else
    ANNOTATED="${BASE_DIR}/mmrole_annotated.jsonl"
    CLEAN="${BASE_DIR}/mmrole_annotated_clean.jsonl"
    FLAGGED="${BASE_DIR}/mmrole_annotated_flagged.jsonl"
    TRAIN_DIR="${BASE_DIR}/training_data"
fi

# ── Step 1: Download ──────────────────────────────────────────
if [ "$FROM_STEP" -le 1 ]; then
    echo ""; echo "═══ STEP 1: Download MMRole ═══"
    ARGS="--output_dir ${BASE_DIR}"
    [ "$SKIP_COCO" = true ] && ARGS="${ARGS} --skip_coco_download"
    [ -n "$COCO_DIR" ] && ARGS="${ARGS} --coco_images_dir ${COCO_DIR}"
    python "${SCRIPTS_DIR}/step1_download_mmrole.py" ${ARGS}
fi

# ── Step 2: Restructure ──────────────────────────────────────
if [ "$FROM_STEP" -le 2 ]; then
    echo ""; echo "═══ STEP 2: Belief-Centric Per-Turn Restructuring ═══"
    python "${SCRIPTS_DIR}/step2_restructure_per_turn.py" \
        --input_dir "${BASE_DIR}/raw_data" \
        --output_path "${BASE_DIR}/mmrole_per_turn.jsonl" \
        --min_utterance_tokens 20

    NUM=$(wc -l < "${BASE_DIR}/mmrole_per_turn.jsonl")
    echo "Per-turn examples: ${NUM}"
    [ "$NUM" -lt 100 ] && echo "WARNING: Very few examples. Try --include_human_role"
fi

# ── Step 3: Annotate ─────────────────────────────────────────
if [ "$FROM_STEP" -le 3 ]; then
    echo ""; echo "═══ STEP 3: Structured Belief State Annotation ═══"
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: export OPENAI_API_KEY first"; exit 1
    fi
    python "${SCRIPTS_DIR}/step3_annotate_visual_tom.py" \
        --input_path "${BASE_DIR}/mmrole_per_turn.jsonl" \
        --output_path "${ANNOTATED}" \
        --image_dir "${BASE_DIR}/images" \
        --model "${ANNOTATION_MODEL}" \
        --max_workers "${ANNOTATION_WORKERS}" \
        --max_examples "${MAX_ANNOTATION_EXAMPLES}"
fi

# ── Step 4: Validate ──────────────────────────────────��──────
if [ "$FROM_STEP" -le 4 ]; then
    echo ""; echo "═���═ STEP 4: Validate Belief State Annotations ═══"
    python "${SCRIPTS_DIR}/step4_validate_annotations.py" \
        --input_path "${ANNOTATED}" \
        --output_clean "${CLEAN}" \
        --output_flagged "${FLAGGED}" \
        --max_issues 2 \
        --print_examples 5

    if [ "$PILOT_MODE" = true ]; then
        echo ""
        echo "══════════════════════════════════════════"
        echo "  PILOT COMPLETE — INSPECT BEFORE FULL RUN"
        echo "════════════════��═════════════════════════"
        echo ""
        echo "Check belief quality:"
        echo "  python3 -c \""
        echo "import json"
        echo "with open('${CLEAN}') as f:"
        echo "    for i, line in enumerate(f):"
        echo "        if i >= 3: break"
        echo "        d = json.loads(line)"
        echo "        bs = d['belief_state']"
        echo "        print(f'=== {d[\"example_id\"]} ===')"
        echo "        print(f'  Objects: {len(bs[\"visual_percepts\"][\"scene_objects\"])}')"
        echo "        print(f'  Asymmetry: {bs[\"metadata\"][\"visual_asymmetry_present\"]}')"
        echo "        fob = bs['first_order_beliefs']['speaker_believes_about_partner']"
        echo "        print(f'  Speaker thinks partner sees: {fob[\"perceived_visual_focus\"][:80]}')"
        echo "        print(f'  Probes: {len(bs[\"belief_probes\"])}')"
        echo "        for p in bs['belief_probes'][:2]:"
        echo "            print(f'    [{p[\"probe_type\"]}] Q: {p[\"question\"][:60]}')"
        echo "            print(f'      A: {p[\"answer\"][:40]} | Wrong: {p[\"wrong_answer\"][:40]}')"
        echo "        print()"
        echo "\""
        echo ""
        echo "If good, run full: bash run_pipeline.sh --from-step 3"
        exit 0
    fi
fi

# ── Step 5: Build Training Data ───────���──────────────────────
if [ "$FROM_STEP" -le 5 ]; then
    echo ""; echo "═══ STEP 5: Build Multi-Format Training Data ═══"
    python "${SCRIPTS_DIR}/step5_build_training_data.py" \
        --input_path "${CLEAN}" \
        --output_dir "${TRAIN_DIR}" \
        --character_profiles_dir "${BASE_DIR}/character_profiles" \
        --seed 42
fi

# ── Step 6 (optional): ToM Evaluation ────────────────────────
if [ "$FROM_STEP" -le 6 ] && [ "$FROM_STEP" -ge 6 ]; then
    echo ""; echo "═══ STEP 6: ToM Evaluation ═══"
    echo ""
    echo "Usage examples:"
    echo ""
    echo "  # Evaluate generated responses on belief accuracy + visual perspective:"
    echo "  python ${SCRIPTS_DIR}/step6_evaluate_tom.py response \\"
    echo "      --responses_path <your_model_responses.jsonl> \\"
    echo "      --annotations_path ${BASE_DIR}/mmrole_official_test_annotated_clean.jsonl"
    echo ""
    echo "  # Evaluate probe QA accuracy for a model:"
    echo "  python ${SCRIPTS_DIR}/step6_evaluate_tom.py probe \\"
    echo "      --model_name gpt-5.4-mini \\"
    echo "      --annotations_path ${BASE_DIR}/mmrole_official_test_annotated_clean.jsonl"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════=="
echo ""
echo "${TRAIN_DIR}/"
echo "├── train/"
echo "│   ├── belief_prediction.jsonl    ← VAE z encoder targets"
echo "│   ├─�� preference_pairs.jsonl     ← Bradley-Terry (4 violation types)"
echo "│   ├─�� probe_qa.jsonl             ← ToM evaluation probes"
echo "│   ├── salience_prediction.jsonl  ← visual perspective module"
echo "│   └── raw_annotated.jsonl"
echo "├── val/"
echo "│   └── (same structure)"
echo "├── test_in/"
echo "���   └── (same structure)"
echo "├─�� test_out/"
echo "│   └── (same structure, OOD characters)"
echo "└── dataset_stats.json"
echo ""
