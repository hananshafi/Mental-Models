#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
FRACTIONS = (0, 25, 50)


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def count_results(path: Path) -> tuple[int, int]:
    completed = 0
    scored = 0
    if not path.exists():
        return completed, scored

    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            completed += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "agent_1_scores" in record and "agent_2_scores" in record:
                scored += 1
    return completed, scored


def latest_episode(log_path: Path) -> str:
    if not log_path.exists():
        return "-"
    matches = re.findall(
        r"Episode\s+(\d+)/(\d+)", log_path.read_text(errors="ignore")
    )
    return f"{matches[-1][0]}/{matches[-1][1]}" if matches else "-"


def summary_count(summary_path: Path) -> str:
    if not summary_path.exists():
        return "-"
    try:
        summary = json.loads(summary_path.read_text())
        value = summary["policy_agent"]["overall_score"]["n"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return "invalid"
    return str(value)


print("fraction  state     pid       episode  saved  scored  summary_n")
for fraction in FRACTIONS:
    pid_path = EVAL_ROOT / "pids" / f"fraction_{fraction}.pid"
    result_path = EVAL_ROOT / "results" / f"fraction_{fraction}_official_all.jsonl"
    summary_path = result_path.with_name(f"{result_path.stem}_summary.json")
    log_path = EVAL_ROOT / "logs" / f"fraction_{fraction}_official_all.log"

    pid = int(pid_path.read_text().strip()) if pid_path.exists() else 0
    state = "running" if pid and process_is_running(pid) else "stopped"
    saved, scored = count_results(result_path)
    print(
        f"{fraction:>7}%  {state:<9} {pid or '-':<9} "
        f"{latest_episode(log_path):<8} {saved:>5}  {scored:>6}  "
        f"{summary_count(summary_path):>9}"
    )
