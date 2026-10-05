import argparse
import json
import math
import os
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


def normalize_story(text: str) -> str:
    return "\n".join(line.strip() for line in str(text).strip().splitlines() if line.strip())


def metric(value: float, n: int) -> Dict[str, float]:
    return {"value": float(value), "n": int(n)}


def metric_or_nan(value: float, n: int) -> Dict[str, object]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"value": None, "n": int(n)}
    return metric(value, n)


def split_into_batches(story_keys: List[str], num_batches: int = 5) -> Dict[str, List[str]]:
    chunks = np.array_split(np.array(story_keys, dtype=object), num_batches)
    return {
        f"batch-{idx + 1}": [str(item) for item in chunk.tolist()]
        for idx, chunk in enumerate(chunks)
    }


def build_result_payload(predictions: List[Dict], benchmark_root: Path) -> Tuple[Dict, Dict[str, Dict]]:
    meta_path = benchmark_root / "data" / "opentom_data" / "meta_data.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    narrative_to_key = {}
    plot_to_key = {}
    for key, value in metadata.items():
        narrative_to_key[normalize_story(value["narrative"])] = key
        plot_to_key[normalize_story(value["plot"])] = key

    grouped: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for row in predictions:
        story = normalize_story(row.get("story", ""))
        meta_key = narrative_to_key.get(story) or plot_to_key.get(story)
        if meta_key is None:
            raise KeyError(f"Could not map OpenToM story to metadata key: {story[:200]}")
        grouped.setdefault(meta_key, []).append(row)

    batches = split_into_batches(list(grouped.keys()), num_batches=5)
    result_payload = {batch_name: {} for batch_name in batches}
    for batch_name, keys in batches.items():
        for key in keys:
            result_payload[batch_name][key] = {
                str(idx): {
                    "question": row["question"],
                    "answer": row["answer"],
                    "type": row["question_type"],
                    "prediction": row["prediction"],
                }
                for idx, row in enumerate(grouped[key])
            }
    return result_payload, metadata


def import_official_evaluator(benchmark_root: Path):
    src_dir = benchmark_root / "src"
    evaluate_dir = src_dir / "evaluate"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(evaluate_dir))
    old_cwd = Path.cwd()
    os.chdir(src_dir)
    try:
        from opentom_evaluator import OpenToMEvaluator  # type: ignore
    finally:
        os.chdir(old_cwd)
    return OpenToMEvaluator


def summarize_official_results(result_dict: Dict[str, List[List[Tuple[str, int, int]]]]) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for question_type, batches in result_dict.items():
        if not any(batches):
            continue

        main_acc, main_f1, main_corrupt, main_n = [], [], [], []
        aux_acc, aux_f1, aux_corrupt, aux_n = [], [], [], []
        merged_acc, merged_f1, merged_n = [], [], []

        for batch_result in batches:
            if not batch_result:
                continue

            pred_list, gt_list = [], []
            pred_list2, gt_list2 = [], []
            for entry in batch_result:
                cur_type = entry[0]
                if cur_type == "accessibility":
                    gt_list2.append(entry[1])
                    pred_list2.append(entry[2])
                else:
                    gt_list.append(entry[1])
                    pred_list.append(entry[2])

            valid_pred = [pred for pred in pred_list if pred != -1]
            valid_gt = [gt_list[i] for i in range(len(pred_list)) if pred_list[i] != -1]
            valid_pred = [valid_pred[i] for i in range(len(valid_gt)) if valid_gt[i] is not None]
            valid_gt = [gt for gt in valid_gt if gt is not None]
            corrupt = (len(pred_list) - len(valid_pred)) / len(pred_list) if pred_list else 0.0

            if valid_pred and valid_gt:
                main_acc.append(accuracy_score(valid_gt, valid_pred))
                main_f1.append(f1_score(valid_gt, valid_pred, average="macro"))
                main_corrupt.append(corrupt)
                main_n.append(len(valid_gt))

            if pred_list2:
                valid_pred2 = [pred for pred in pred_list2 if pred != -1]
                valid_gt2 = [gt_list2[i] for i in range(len(pred_list2)) if pred_list2[i] != -1]
                valid_pred2 = [valid_pred2[i] for i in range(len(valid_gt2)) if valid_gt2[i] is not None]
                valid_gt2 = [gt for gt in valid_gt2 if gt is not None]
                corrupt2 = (len(pred_list2) - len(valid_pred2)) / len(pred_list2) if pred_list2 else 0.0

                if valid_pred2 and valid_gt2:
                    aux_acc.append(accuracy_score(valid_gt2, valid_pred2))
                    aux_f1.append(f1_score(valid_gt2, valid_pred2, average="macro"))
                    aux_corrupt.append(corrupt2)
                    aux_n.append(len(valid_gt2))

                    merged_pred = valid_pred + valid_pred2
                    merged_gt = valid_gt + valid_gt2
                    merged_acc.append(accuracy_score(merged_gt, merged_pred))
                    merged_f1.append(f1_score(merged_gt, merged_pred, average="macro"))
                    merged_n.append(len(merged_gt))

        def summarize_bucket(acc_values, f1_values, corrupt_values, counts):
            total_n = sum(counts)
            return {
                "Accuracy": metric_or_nan(float(np.mean(acc_values)) if acc_values else float("nan"), total_n),
                "AccuracyStd": metric_or_nan(float(np.std(acc_values)) if acc_values else float("nan"), len(acc_values)),
                "MacroF1": metric_or_nan(float(np.mean(f1_values)) if f1_values else float("nan"), total_n),
                "MacroF1Std": metric_or_nan(float(np.std(f1_values)) if f1_values else float("nan"), len(f1_values)),
                "CorruptedGenerationPct": metric_or_nan(
                    float(np.mean(corrupt_values) * 100.0) if corrupt_values else float("nan"),
                    len(corrupt_values),
                ),
            }

        if aux_acc:
            metrics[question_type] = {
                "fullness": summarize_bucket(main_acc, main_f1, main_corrupt, main_n),
                "accessibility": summarize_bucket(aux_acc, aux_f1, aux_corrupt, aux_n),
                "overall": {
                    "Accuracy": metric_or_nan(float(np.mean(merged_acc)) if merged_acc else float("nan"), sum(merged_n)),
                    "AccuracyStd": metric_or_nan(float(np.std(merged_acc)) if merged_acc else float("nan"), len(merged_acc)),
                    "MacroF1": metric_or_nan(float(np.mean(merged_f1)) if merged_f1 else float("nan"), sum(merged_n)),
                    "MacroF1Std": metric_or_nan(float(np.std(merged_f1)) if merged_f1 else float("nan"), len(merged_f1)),
                    "CorruptedGenerationPct": metric_or_nan(
                        float(np.mean(main_corrupt + aux_corrupt) * 100.0) if (main_corrupt or aux_corrupt) else float("nan"),
                        len(main_corrupt) + len(aux_corrupt),
                    ),
                },
            }
        else:
            metrics[question_type] = summarize_bucket(main_acc, main_f1, main_corrupt, main_n)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--location_granularity", default="coarse")
    parser.add_argument("--perspective", default="all")
    args = parser.parse_args()

    predictions = load_json_or_jsonl(Path(args.predictions_path))
    benchmark_root = Path(args.benchmark_root)
    result_payload, _ = build_result_payload(predictions, benchmark_root)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(result_payload, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)

    try:
        OpenToMEvaluator = import_official_evaluator(benchmark_root)
        old_cwd = Path.cwd()
        os.chdir(benchmark_root / "src")
        try:
            evaluator = OpenToMEvaluator()
            official_result = evaluator.evaluate(
                str(tmp_path),
                args.location_granularity,
                args.perspective,
            )
        finally:
            os.chdir(old_cwd)
    finally:
        tmp_path.unlink(missing_ok=True)

    metrics = summarize_official_results(official_result)
    metrics["LocationGranularity"] = args.location_granularity
    metrics["Perspective"] = args.perspective
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
