#!/usr/bin/env python3
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[2]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "stage3_evaluate_sotopia.py"
RUNNABLE_PATH = EVAL_ROOT / "stage3_evaluate_sotopia_utf8.py"


source_text = SOURCE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
runnable_text = RUNNABLE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")

if source_text != runnable_text:
    raise SystemExit(
        f"Evaluator drift detected: {RUNNABLE_PATH} no longer matches {SOURCE_PATH}"
    )

compile(runnable_text, str(RUNNABLE_PATH), "exec")
print(f"Evaluator verified against {SOURCE_PATH}")
