import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from official_eval_common import normalize_text, token_f1

SCRIPT_DIR = Path(__file__).resolve().parent


def metric_value(value: float, n: int) -> Dict[str, float]:
    return {"value": float(value), "n": int(n)}


def mean_summary(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return metric_value(0.0, 0)
    return metric_value(sum(values) / len(values), len(values))


def macro_f1_from_labels(gold_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    if not gold_labels:
        return metric_value(0.0, 0)
    labels = sorted({label for label in gold_labels if label is not None} | {label for label in pred_labels if label is not None})
    if not labels:
        return metric_value(0.0, 0)
    per_label = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            per_label.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            per_label.append(0.0)
        else:
            per_label.append(2 * precision * recall / (precision + recall))
    return metric_value(sum(per_label) / len(per_label), len(gold_labels))


def extract_json_payload(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def scorer_supports_wrapper(scorer_path: Path, required_snippets: List[str]) -> bool:
    try:
        text = scorer_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return all(snippet in text for snippet in required_snippets)


def run_python_scorer(
    *,
    scorer_path: Path,
    args: List[str],
    cwd: Optional[Path] = None,
) -> Dict[str, object]:
    command = [sys.executable, str(scorer_path)] + list(args)
    completed = subprocess.run(
        command,
        cwd=str(cwd or scorer_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = extract_json_payload(completed.stdout)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "metrics": parsed if completed.returncode == 0 else None,
    }


def run_official_opentom_scorer(
    *,
    scorer_path: Optional[Path],
    predictions_path: Path,
    location_granularity: str = "coarse",
    perspective: str = "all",
) -> Optional[Dict[str, object]]:
    if scorer_path is None or not scorer_path.exists():
        return None
    if scorer_supports_wrapper(scorer_path, ["--result_path", "json.dumps"]):
        return run_python_scorer(
            scorer_path=scorer_path,
            args=[
                "--result_path", str(predictions_path),
                "--location_granularity", location_granularity,
                "--perspective", perspective,
            ],
            cwd=scorer_path.parent,
        )

    bridge_path = SCRIPT_DIR / "official_eval_bridge_opentom.py"
    benchmark_root = scorer_path.parents[1] if scorer_path.parent.name == "src" else scorer_path.parent
    return run_python_scorer(
        scorer_path=bridge_path,
        args=[
            "--predictions_path", str(predictions_path),
            "--benchmark_root", str(benchmark_root),
            "--location_granularity", location_granularity,
            "--perspective", perspective,
        ],
        cwd=bridge_path.parent,
    )


def run_official_fantom_scorer(
    *,
    scorer_path: Optional[Path],
    predictions_path: Path,
    split_path: Path,
    input_type: str,
    aggregation_target: str = "set",
    embedding_model: str = "sentence-transformers/all-roberta-large-v1",
    allow_model_download: bool = False,
) -> Optional[Dict[str, object]]:
    if scorer_path is None or not scorer_path.exists():
        return None
    if scorer_supports_wrapper(scorer_path, ["--predictions_path", "json.dumps"]):
        return run_python_scorer(
            scorer_path=scorer_path,
            args=[
                "--predictions_path", str(predictions_path),
                "--split_path", str(split_path),
                "--input_type", input_type,
            ],
            cwd=scorer_path.parent,
        )

    bridge_path = SCRIPT_DIR / "official_eval_bridge_fantom.py"
    return run_python_scorer(
        scorer_path=bridge_path,
        args=[
            "--predictions_path", str(predictions_path),
            "--split_path", str(split_path),
            "--input_type", input_type,
            "--aggregation_target", aggregation_target,
            "--embedding_model", embedding_model,
            *(["--allow_model_download"] if allow_model_download else []),
        ],
        cwd=bridge_path.parent,
    )


def run_official_tomi_scorer(
    *,
    scorer_path: Optional[Path],
    predictions_path: Path,
    split_path: Path,
    trace_path: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    if scorer_path is None or not scorer_path.exists():
        return None
    bridge_path = SCRIPT_DIR / "official_eval_bridge_tomi.py"
    return run_python_scorer(
        scorer_path=bridge_path,
        args=[
            "--predictions_path", str(predictions_path),
            "--split_path", str(split_path),
            *(["--trace_path", str(trace_path)] if trace_path else []),
            "--benchmark_root", str(scorer_path.parent),
        ],
        cwd=bridge_path.parent,
    )


def summarize_bigtom_metrics(rows: List[Dict]) -> Dict[str, object]:
    overall = [row["score"] for row in rows]
    by_task: Dict[str, List[float]] = defaultdict(list)
    by_condition: Dict[str, List[float]] = defaultdict(list)
    by_task_condition: Dict[str, List[float]] = defaultdict(list)
    by_init_belief: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        by_task[row["task"]].append(row["score"])
        by_condition[row["condition"]].append(row["score"])
        by_task_condition[f"{row['task']}/{row['condition']}"].append(row["score"])
        by_init_belief[str(row["init_belief"])].append(row["score"])

    return {
        "Accuracy": mean_summary(overall),
        "ByTask": {key: mean_summary(values) for key, values in sorted(by_task.items())},
        "ByCondition": {key: mean_summary(values) for key, values in sorted(by_condition.items())},
        "ByTaskCondition": {key: mean_summary(values) for key, values in sorted(by_task_condition.items())},
        "ByInitBelief": {key: mean_summary(values) for key, values in sorted(by_init_belief.items())},
    }


def summarize_tomi_metrics(rows: List[Dict]) -> Dict[str, object]:
    overall = [row["score"] for row in rows]
    by_question_type: Dict[str, List[float]] = defaultdict(list)
    by_story_type: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("question_type"):
            by_question_type[str(row["question_type"])].append(row["score"])
        if row.get("story_type"):
            by_story_type[str(row["story_type"])].append(row["score"])
    return {
        "Accuracy": mean_summary(overall),
        "ByQuestionType": {key: mean_summary(values) for key, values in sorted(by_question_type.items())},
        "ByStoryType": {key: mean_summary(values) for key, values in sorted(by_story_type.items())},
    }


def summarize_hitom_metrics(rows: List[Dict]) -> Dict[str, object]:
    overall = [row["score"] for row in rows]
    by_question_order: Dict[str, List[float]] = defaultdict(list)
    by_story_length: Dict[str, List[float]] = defaultdict(list)
    by_prompting_type: Dict[str, List[float]] = defaultdict(list)
    by_deception: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("question_order") is not None:
            by_question_order[str(row["question_order"])].append(row["score"])
        if row.get("story_length") is not None:
            by_story_length[str(row["story_length"])].append(row["score"])
        if row.get("prompting_type") is not None:
            by_prompting_type[str(row["prompting_type"])].append(row["score"])
        if row.get("deception") is not None:
            by_deception[str(row["deception"])].append(row["score"])
    return {
        "Accuracy": mean_summary(overall),
        "ByQuestionOrder": {key: mean_summary(values) for key, values in sorted(by_question_order.items())},
        "ByStoryLength": {key: mean_summary(values) for key, values in sorted(by_story_length.items())},
        "ByPromptingType": {key: mean_summary(values) for key, values in sorted(by_prompting_type.items())},
        "ByDeception": {key: mean_summary(values) for key, values in sorted(by_deception.items())},
    }


def summarize_opentom_metrics(rows: List[Dict]) -> Dict[str, object]:
    accuracy_values = [row["score"] for row in rows]
    macro_f1 = macro_f1_from_labels(
        [row.get("gold_label", "") for row in rows],
        [row.get("pred_label", "") for row in rows],
    )
    grouped_fields = {
        "ByQuestionType": "question_type",
        "ByPerspective": "perspective",
        "ByToMOrder": "tom_order",
        "ByNarrativeType": "narrative_type",
        "ByLocationGranularity": "location_granularity",
    }
    grouped = {}
    for metric_name, field_name in grouped_fields.items():
        by_group: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            value = row.get(field_name)
            if value is not None and str(value).strip():
                by_group[str(value)].append(row)
        grouped[metric_name] = {
            key: {
                "Accuracy": mean_summary([item["score"] for item in bucket]),
                "MacroF1": macro_f1_from_labels(
                    [item.get("gold_label", "") for item in bucket],
                    [item.get("pred_label", "") for item in bucket],
                ),
            }
            for key, bucket in sorted(by_group.items())
        }
    return {
        "Accuracy": mean_summary(accuracy_values),
        "MacroF1": macro_f1,
        **grouped,
    }


def _fantom_family(question_type: str) -> str:
    qtype = normalize_text(question_type)
    if "answerability" in qtype:
        return "AnswerabilityQ"
    if "info access" in qtype or "information access" in qtype or "accessibility" in qtype:
        return "Info-AccessQ"
    if "fact" in qtype:
        return "FactQ"
    if "belief" in qtype:
        return "BeliefQ"
    return "Other"


def _fantom_is_yes_no(answer) -> bool:
    return normalize_text(str(answer)) in {"yes", "no", "true", "false"}


def summarize_fantom_metrics(rows: List[Dict]) -> Dict[str, object]:
    all_scores = [row["score"] for row in rows]
    exact_scores = [1.0 if row.get("exact") else 0.0 for row in rows]

    belief_choice = [row["score"] for row in rows if _fantom_family(row.get("question_type", "")) == "BeliefQ" and row.get("choices")]
    belief_dist = [1.0 if row.get("exact") else 0.0 for row in rows if _fantom_family(row.get("question_type", "")) == "BeliefQ" and not row.get("choices")]
    belief_token_f1 = [row.get("token_f1", row["score"]) for row in rows if _fantom_family(row.get("question_type", "")) == "BeliefQ" and not row.get("choices")]

    answerability_rows = [row for row in rows if _fantom_family(row.get("question_type", "")) == "AnswerabilityQ"]
    answerability_list = [row["score"] for row in answerability_rows if isinstance(row.get("answer"), list)]
    answerability_yesno = [row["score"] for row in answerability_rows if _fantom_is_yes_no(row.get("answer"))]

    info_rows = [row for row in rows if _fantom_family(row.get("question_type", "")) == "Info-AccessQ"]
    info_list = [row["score"] for row in info_rows if isinstance(row.get("answer"), list)]
    info_yesno = [row["score"] for row in info_rows if _fantom_is_yes_no(row.get("answer"))]

    fact_scores = [row.get("token_f1", row["score"]) for row in rows if _fantom_family(row.get("question_type", "")) == "FactQ"]

    all_star_rows = [
        row["score"] for row in rows
        if _fantom_family(row.get("question_type", "")) in {"BeliefQ", "AnswerabilityQ", "Info-AccessQ"}
        and not str(row.get("question_type", "")).endswith(":multiple-choice")
    ]

    by_question_type: Dict[str, List[float]] = defaultdict(list)
    by_tom_type: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("question_type"):
            by_question_type[str(row["question_type"])].append(row["score"])
        if row.get("tom_type"):
            by_tom_type[str(row["tom_type"])].append(row["score"])

    return {
        "All*": mean_summary(all_star_rows),
        "All": mean_summary(all_scores),
        "BeliefQ [Choice]": mean_summary(belief_choice),
        "BeliefQ [Dist.]": mean_summary(belief_dist),
        "BeliefQ token-F1": mean_summary(belief_token_f1),
        "All AnswerabilityQ": mean_summary([row["score"] for row in answerability_rows]),
        "AnswerabilityQ [List]": mean_summary(answerability_list),
        "AnswerabilityQ [Y/N]": mean_summary(answerability_yesno),
        "All Info-AccessQ": mean_summary([row["score"] for row in info_rows]),
        "Info-AccessQ [List]": mean_summary(info_list),
        "Info-AccessQ [Y/N]": mean_summary(info_yesno),
        "FactQ token-F1": mean_summary(fact_scores),
        "Exact Match": mean_summary(exact_scores),
        "ByQuestionType": {key: mean_summary(values) for key, values in sorted(by_question_type.items())},
        "ByToMType": {key: mean_summary(values) for key, values in sorted(by_tom_type.items())},
    }
