import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List



def load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def metric_value(value: float, n: int) -> Dict[str, float]:
    return {"value": float(value), "n": int(n)}


def mean_summary(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return metric_value(0.0, 0)
    return metric_value(sum(values) / len(values), len(values))


def _infer_tomi_trace_fields(trace_line: str) -> Dict[str, str]:
    fields = [part.strip() for part in trace_line.split(",") if part.strip()]
    out: Dict[str, str] = {"trace": trace_line.strip()}

    for token in reversed(fields):
        if token in {"true_belief", "false_belief", "second_order_false_belief"}:
            out["story_type"] = token
            break
    for token in reversed(fields):
        if token in {"memory", "reality", "first_order", "second_order", "false_belief", "true_belief", "second_order_false_belief"}:
            out["question_type"] = token
            break
    if "question_type" not in out and fields:
        out["question_type"] = fields[-1]
    return out


def load_tomi_official(test_path: Path, trace_path: Path | None = None) -> List[Dict]:
    rows = []
    story_lines: List[str] = []
    qa_index = 0
    traces: List[Dict[str, str]] = []

    if trace_path and trace_path.exists():
        with open(trace_path) as f:
            traces = [_infer_tomi_trace_fields(line.rstrip("\n")) for line in f if line.strip()]

    with open(test_path) as f:
        for raw_line in f:
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                continue
            try:
                idx_text, remainder = raw_line.split(" ", 1)
                line_idx = int(idx_text)
            except ValueError:
                continue

            if line_idx == 1:
                story_lines = []

            if "\t" in remainder:
                question, answer, *rest = remainder.split("\t")
                row = {
                    "dataset": "tomi",
                    "sample_id": qa_index,
                    "story": " ".join(story_lines).strip(),
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "supporting_facts": rest[0].strip() if rest else "",
                }
                if qa_index < len(traces):
                    row.update(traces[qa_index])
                rows.append(row)
                qa_index += 1
            else:
                story_lines.append(remainder.strip())
    return rows


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


def _build_official_lookup(rows: List[Dict]) -> Dict[str, Dict]:
    lookup = {}
    for row in rows:
        lookup[str(row.get("sample_id"))] = row
    return lookup


def _score_prediction(official_row: Dict, prediction_row: Dict) -> Dict:
    gold = normalize_text(official_row.get("answer", ""))
    pred = normalize_text(prediction_row.get("prediction", ""))
    score = 1.0 if gold == pred or (gold and gold in pred) else 0.0

    record = dict(official_row)
    record.update({
        "prediction": prediction_row.get("prediction", ""),
        "score": score,
    })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--trace_path", default="")
    parser.add_argument("--benchmark_root", default="")
    args = parser.parse_args()

    predictions = load_json_or_jsonl(Path(args.predictions_path))
    official_rows = load_tomi_official(
        Path(args.split_path),
        Path(args.trace_path) if args.trace_path else None,
    )
    official_by_id = _build_official_lookup(official_rows)

    scored_rows = []
    for idx, prediction_row in enumerate(predictions):
        sample_id = str(prediction_row.get("sample_id", idx))
        official_row = official_by_id.get(sample_id)
        if official_row is None:
            if idx < len(official_rows):
                official_row = official_rows[idx]
            else:
                raise KeyError(f"Could not align ToMi prediction to official row: sample_id={sample_id}")
        scored_rows.append(_score_prediction(official_row, prediction_row))

    if not scored_rows:
        raise ValueError("No ToMi predictions were available to score.")

    metrics = summarize_tomi_metrics(scored_rows)
    metrics["Protocol"] = "released_tomi_exact_match_bridge"
    metrics["NumScored"] = len(scored_rows)
    if args.benchmark_root:
        metrics["BenchmarkRoot"] = str(Path(args.benchmark_root))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
