"""
Canonical benchmark-first evaluation entrypoint for the paper models.

This script evaluates `base`, `sft`, or `grpo` checkpoints on the official
BigToM, ToMi, and FANToM benchmark splits.

It prefers benchmark-native inference/scoring behavior:
  - multiple-choice benchmarks use choice scoring when available
  - free-response benchmarks use generation
  - official scorer code paths are invoked when a compatible scorer is vendored
  - local adapters mirror the published metrics when no standalone scorer exists
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from official_eval_common import (
    FANTOM_HEADER,
    build_benchmark_prompt,
    load_policy_bundle,
    normalize_text,
    token_f1,
)
from official_eval_loaders import (
    BenchmarkPaths,
    default_benchmarks_root,
    load_bigtom_official,
    load_fantom_official,
    load_tomi_official,
    resolve_bigtom_paths,
    resolve_fantom_paths,
    resolve_tomi_paths,
    validate_benchmark_paths,
)
from official_eval_scorers import (
    run_official_fantom_scorer,
    run_official_tomi_scorer,
    summarize_bigtom_metrics,
    summarize_fantom_metrics,
    summarize_tomi_metrics,
)


ALL_DATASETS = {"bigtom", "tomi", "fantom"}


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _start_dataset_progress(dataset_name: str, total: int, log_every: int) -> Dict[str, object]:
    state = {
        "dataset_name": dataset_name,
        "total": max(total, 0),
        "log_every": max(log_every, 1),
        "started_at": time.time(),
        "last_logged": 0,
    }
    print(f"[{dataset_name}] starting evaluation on {total} samples", flush=True)
    return state


def _maybe_log_dataset_progress(progress: Dict[str, object], completed: int):
    total = int(progress["total"])
    log_every = int(progress["log_every"])
    last_logged = int(progress["last_logged"])
    if completed <= last_logged:
        return
    should_log = completed == total or completed == 1 or completed % log_every == 0
    if not should_log:
        return

    elapsed = time.time() - float(progress["started_at"])
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate > 0 else None
    pct = (100.0 * completed / total) if total else 100.0
    eta_text = _format_duration(eta) if eta is not None else "n/a"
    rate_text = f"{rate:.2f} ex/s" if rate >= 1.0 else f"{(1.0 / rate):.2f} s/ex" if rate > 0 else "n/a"
    print(
        f"[{progress['dataset_name']}] {completed}/{total} ({pct:.1f}%) "
        f"elapsed={_format_duration(elapsed)} eta={eta_text} rate={rate_text}",
        flush=True,
    )
    progress["last_logged"] = completed


def _print_dataset_stage(dataset_name: str, message: str):
    print(f"[{dataset_name}] {message}", flush=True)


def _ensure_dataset_path(paths: BenchmarkPaths):
    validation = validate_benchmark_paths(paths)
    if not validation["split_exists"]:
        raise FileNotFoundError(
            f"{paths.dataset} split file not found. "
            f"root={paths.root} split_path={paths.split_path}. "
            f"Run ./tools/bootstrap_third_party.sh {paths.dataset} to fetch it."
        )


def _write_predictions_jsonl(path: Path, rows: Iterable[Dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict):
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_json_list(path: Path, rows: Sequence[Dict]):
    with open(path, "w") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def _contains_gold_not_wrong(prediction: str, gold: str, wrong: str = "") -> Optional[bool]:
    pred = normalize_text(prediction)
    gold_n = normalize_text(gold)
    wrong_n = normalize_text(wrong) if wrong else ""
    has_gold = bool(gold_n) and gold_n in pred
    has_wrong = bool(wrong_n) and wrong_n != gold_n and wrong_n in pred
    if has_gold and not has_wrong:
        return True
    if has_wrong and not has_gold:
        return False
    return None


def _normalize_yes_no(text: str) -> Optional[str]:
    norm = normalize_text(text)
    if "yes" in norm or "true" in norm:
        return "yes"
    if "no" in norm or "false" in norm:
        return "no"
    return None


def _resolve_requested_datasets(text: str) -> List[str]:
    if text == "all":
        return sorted(ALL_DATASETS)
    requested = []
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Expected one of {sorted(ALL_DATASETS)}.")
        requested.append(name)
    return requested


def _apply_limit(rows: List[Dict], limit: Optional[int]) -> List[Dict]:
    if limit is None:
        return rows
    return rows[:limit]


def _max_new_tokens_for_dataset(args, dataset_name: str) -> int:
    override = getattr(args, f"{dataset_name}_max_new_tokens", None)
    if override is not None:
        return override
    return args.max_new_tokens


def _official_input_path(out_dir: Path, dataset_name: str, mode: str) -> Path:
    return out_dir / f"{dataset_name}_{mode}_official_input.json"


def _predictions_path(out_dir: Path, dataset_name: str, mode: str) -> Path:
    return out_dir / f"{dataset_name}_{mode}_predictions.jsonl"


def _summary_path(out_dir: Path, dataset_name: str, mode: str) -> Path:
    return out_dir / f"{dataset_name}_{mode}_summary.json"


def _flatten_metrics(metrics: Dict[str, object], prefix: str = "") -> List[Tuple[str, object]]:
    rows: List[Tuple[str, object]] = []
    for key, value in metrics.items():
        metric_name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict) and "value" in value and "n" in value:
            rows.append((metric_name, value))
        elif isinstance(value, dict):
            rows.extend(_flatten_metrics(value, metric_name))
        else:
            rows.append((metric_name, value))
    return rows


def _write_markdown_report(path: Path, combined_summary: Dict):
    lines = [
        f"# Official Benchmark Evaluation ({combined_summary['mode']})",
        "",
        f"Base model: `{combined_summary['base_model']}`",
        "",
    ]
    for dataset_name, summary in combined_summary["datasets"].items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append(f"- Scoring source: `{summary['scoring']['source']}`")
        lines.append(f"- Predictions: `{summary['paths']['predictions_jsonl']}`")
        if summary["scoring"].get("official_scorer_attempted"):
            lines.append(f"- Official scorer attempted: `True`")
        lines.append("")
        lines.append("| Metric | Value | N |")
        lines.append("| --- | ---: | ---: |")
        for metric_name, metric_value in _flatten_metrics(summary["metrics"]):
            if isinstance(metric_value, dict) and "value" in metric_value:
                lines.append(f"| {metric_name} | {metric_value['value']:.4f} | {metric_value['n']} |")
            else:
                lines.append(f"| {metric_name} | {metric_value} |  |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _persist_summary(
    *,
    out_dir: Path,
    dataset_name: str,
    mode: str,
    summary: Dict,
    predictions: Sequence[Dict],
):
    predictions_jsonl = _predictions_path(out_dir, dataset_name, mode)
    official_input_json = _official_input_path(out_dir, dataset_name, mode)
    summary_json = _summary_path(out_dir, dataset_name, mode)
    _write_predictions_jsonl(predictions_jsonl, predictions)
    _write_json_list(official_input_json, predictions)
    summary["paths"]["predictions_jsonl"] = str(predictions_jsonl)
    summary["paths"]["official_input_json"] = str(official_input_json)
    _write_json(summary_json, summary)
    return predictions_jsonl, official_input_json, summary_json


def _finalize_summary(
    *,
    dataset_name: str,
    runner,
    resolved_paths: BenchmarkPaths,
    metrics: Dict[str, object],
    local_metrics: Dict[str, object],
    official_run: Optional[Dict[str, object]],
    scoring_source: str,
    inference_mode: str,
) -> Dict:
    official_metrics = official_run.get("metrics") if official_run else None
    return {
        "dataset": dataset_name,
        "mode": runner.mode,
        "metrics": metrics,
        "local_metrics": local_metrics,
        "official_metrics": official_metrics,
        "paths": {
            "benchmark_root": str(resolved_paths.root),
            "split_path": str(resolved_paths.split_path) if resolved_paths.split_path else None,
            "trace_path": str(resolved_paths.trace_path) if resolved_paths.trace_path else None,
            "scorer_path": str(resolved_paths.scorer_path) if resolved_paths.scorer_path else None,
        },
        "scoring": {
            "source": scoring_source,
            "inference_mode": inference_mode,
            "official_scorer_attempted": bool(official_run),
            "official_scorer_returncode": official_run.get("returncode") if official_run else None,
            "official_scorer_command": official_run.get("command") if official_run else None,
        },
    }


def evaluate_bigtom(rows: List[Dict], runner, args, out_dir: Path, resolved_paths: BenchmarkPaths) -> Dict:
    mc_rng = random.Random(42)
    predictions = []
    max_new_tokens = _max_new_tokens_for_dataset(args, "bigtom")
    progress = _start_dataset_progress("bigtom", len(rows), args.log_every)
    for idx, row in enumerate(rows, start=1):
        if mc_rng.random() < 0.5:
            choices = [row["option_correct"], row["option_wrong"]]
            gold_choice_index = 0
        else:
            choices = [row["option_wrong"], row["option_correct"]]
            gold_choice_index = 1
        if args.bigtom_inference == "choice":
            prompt = build_benchmark_prompt(
                row["story"],
                row["question"],
                dataset_name="BigToM",
                choices=choices,
                extra_instruction="Choose the best answer from the listed options.",
            )
            picked = runner.pick_choice(prompt, choices, story=row["story"], question=row["question"])
            predicted_answer = picked["choice_text"]
            score = 1.0 if picked["choice_index"] == gold_choice_index else 0.0
            detail = {"choice_scores": picked["scores"], "choice_index": picked["choice_index"], "gold_choice_index": gold_choice_index}
        else:
            prompt = build_benchmark_prompt(row["story"], row["question"], dataset_name="BigToM")
            predicted_answer = runner.generate(
                prompt,
                story=row["story"],
                question=row["question"],
                max_new_tokens=max_new_tokens,
            )
            decision = _contains_gold_not_wrong(predicted_answer, row["option_correct"], row["option_wrong"])
            score = 1.0 if decision else 0.0
            detail = {}

        record = dict(row)
        record.update({
            "prediction": predicted_answer,
            "score": score,
            **detail,
        })
        predictions.append(record)
        _maybe_log_dataset_progress(progress, idx)

    local_metrics = summarize_bigtom_metrics(predictions)
    summary = _finalize_summary(
        dataset_name="bigtom",
        runner=runner,
        resolved_paths=resolved_paths,
        metrics=local_metrics,
        local_metrics=local_metrics,
        official_run=None,
        scoring_source="local_adapter",
        inference_mode=args.bigtom_inference,
    )
    _print_dataset_stage("bigtom", "finished local inference; saving predictions")
    _persist_summary(out_dir=out_dir, dataset_name="bigtom", mode=runner.mode, summary=summary, predictions=predictions)
    _print_dataset_stage("bigtom", "completed with scoring_source=local_adapter")
    return summary


def evaluate_tomi(rows: List[Dict], runner, args, out_dir: Path, resolved_paths: BenchmarkPaths) -> Dict:
    predictions = []
    max_new_tokens = _max_new_tokens_for_dataset(args, "tomi")
    progress = _start_dataset_progress("tomi", len(rows), args.log_every)
    for idx, row in enumerate(rows, start=1):
        prompt = build_benchmark_prompt(row["story"], row["question"], dataset_name="ToMi")
        predicted_answer = runner.generate(
            prompt,
            story=row["story"],
            question=row["question"],
            max_new_tokens=max_new_tokens,
        )
        gold = normalize_text(row["answer"])
        pred = normalize_text(predicted_answer)
        score = 1.0 if gold == pred or gold in pred else 0.0

        record = dict(row)
        record.update({"prediction": predicted_answer, "score": score})
        predictions.append(record)
        _maybe_log_dataset_progress(progress, idx)

    local_metrics = summarize_tomi_metrics(predictions)
    _print_dataset_stage("tomi", "finished local inference; saving predictions")
    _, official_input_json, _ = _persist_summary(
        out_dir=out_dir,
        dataset_name="tomi",
        mode=runner.mode,
        summary={
            "dataset": "tomi",
            "mode": runner.mode,
            "metrics": local_metrics,
            "paths": {},
            "scoring": {},
        },
        predictions=predictions,
    )
    _print_dataset_stage("tomi", "running released ToMi protocol bridge")
    official_run = run_official_tomi_scorer(
        scorer_path=resolved_paths.scorer_path,
        predictions_path=official_input_json,
        split_path=resolved_paths.split_path,
        trace_path=resolved_paths.trace_path,
    )
    scoring_source = (
        "official_protocol_bridge"
        if official_run and official_run.get("metrics")
        else "local_adapter"
    )
    summary = _finalize_summary(
        dataset_name="tomi",
        runner=runner,
        resolved_paths=resolved_paths,
        metrics=official_run["metrics"] if official_run and official_run.get("metrics") else local_metrics,
        local_metrics=local_metrics,
        official_run=official_run,
        scoring_source=scoring_source,
        inference_mode="generate",
    )
    summary["paths"]["predictions_jsonl"] = str(_predictions_path(out_dir, "tomi", runner.mode))
    summary["paths"]["official_input_json"] = str(official_input_json)
    _write_json(_summary_path(out_dir, "tomi", runner.mode), summary)
    _print_dataset_stage("tomi", f"completed with scoring_source={scoring_source}")
    return summary


def evaluate_fantom(rows: List[Dict], runner, args, out_dir: Path, resolved_paths: BenchmarkPaths) -> Dict:
    predictions = []
    max_new_tokens = _max_new_tokens_for_dataset(args, "fantom")
    progress = _start_dataset_progress("fantom", len(rows), args.log_every)
    for idx, row in enumerate(rows, start=1):
        choices = row.get("choices") or []
        prompt = FANTOM_HEADER + build_benchmark_prompt(
            row["story"],
            row["question"],
            dataset_name="FANToM",
            choices=choices if choices else None,
        )

        use_choice = bool(choices) and (
            args.fantom_inference == "choice"
            or str(row.get("question_type", "")).endswith(":multiple-choice")
        )

        if use_choice:
            picked = runner.pick_choice(prompt, choices, story=row["story"], question=row["question"])
            predicted_answer = picked["choice_text"]
            choice_scores = picked["scores"]
            predicted_choice_index = picked["choice_index"]
        else:
            predicted_answer = runner.generate(
                prompt,
                story=row["story"],
                question=row["question"],
                max_new_tokens=max_new_tokens,
            )
            choice_scores = None
            predicted_choice_index = None

        gold = row["answer"]
        wrong = row.get("wrong_answer", "")
        qtype = row.get("question_type", "unknown")

        if use_choice and row.get("gold_choice_index") is not None:
            score = 1.0 if predicted_choice_index == row["gold_choice_index"] else 0.0
            exact = bool(score)
            token_f1_value = score
        elif isinstance(gold, list):
            gold_norm = [normalize_text(item) for item in gold if normalize_text(item)]
            wrong_norm = [normalize_text(item) for item in wrong] if isinstance(wrong, list) else []
            pred_norm = normalize_text(predicted_answer)
            found_gold = sum(1 for item in gold_norm if item in pred_norm)
            hit_wrong = any(item and item in pred_norm for item in wrong_norm if item not in gold_norm)
            score = found_gold / len(gold_norm) if gold_norm else 0.0
            exact = found_gold == len(gold_norm) and not hit_wrong
            token_f1_value = score
        elif "binary" in normalize_text(qtype) or _normalize_yes_no(str(gold)) is not None:
            score = 1.0 if _normalize_yes_no(predicted_answer) == _normalize_yes_no(str(gold)) else 0.0
            exact = bool(score)
            token_f1_value = score
        else:
            decision = _contains_gold_not_wrong(predicted_answer, str(gold), str(wrong))
            exact = bool(decision) if decision is not None else normalize_text(str(gold)) == normalize_text(predicted_answer)
            token_f1_value = token_f1(str(gold), predicted_answer)
            score = 1.0 if exact else token_f1_value

        record = dict(row)
        record.update({
            "prediction": predicted_answer,
            "score": score,
            "exact": exact,
            "token_f1": token_f1_value,
        })
        if choice_scores is not None:
            record["choice_scores"] = choice_scores
            record["choice_index"] = predicted_choice_index
        predictions.append(record)
        _maybe_log_dataset_progress(progress, idx)

    local_metrics = summarize_fantom_metrics(predictions)
    _print_dataset_stage("fantom", "finished local inference; saving predictions")
    _, official_input_json, _ = _persist_summary(
        out_dir=out_dir,
        dataset_name="fantom",
        mode=runner.mode,
        summary={
            "dataset": "fantom",
            "mode": runner.mode,
            "metrics": local_metrics,
            "paths": {},
            "scoring": {},
        },
        predictions=predictions,
    )
    _print_dataset_stage("fantom", "running official scorer bridge")
    official_run = run_official_fantom_scorer(
        scorer_path=resolved_paths.scorer_path,
        predictions_path=official_input_json,
        split_path=resolved_paths.split_path,
        input_type=args.fantom_input_type,
        aggregation_target=args.fantom_aggregation_target,
        embedding_model=args.fantom_embedding_model,
        allow_model_download=args.fantom_allow_model_download,
    )
    scoring_source = "official_scorer" if official_run and official_run.get("metrics") else "local_adapter"
    summary = _finalize_summary(
        dataset_name="fantom",
        runner=runner,
        resolved_paths=resolved_paths,
        metrics=official_run["metrics"] if official_run and official_run.get("metrics") else local_metrics,
        local_metrics=local_metrics,
        official_run=official_run,
        scoring_source=scoring_source,
        inference_mode=args.fantom_inference,
    )
    summary["paths"]["predictions_jsonl"] = str(_predictions_path(out_dir, "fantom", runner.mode))
    summary["paths"]["official_input_json"] = str(official_input_json)
    _write_json(_summary_path(out_dir, "fantom", runner.mode), summary)
    _print_dataset_stage("fantom", f"completed with scoring_source={scoring_source}")
    return summary


def _resolve_paths_for_dataset(dataset_name: str, args) -> BenchmarkPaths:
    if dataset_name == "bigtom":
        return resolve_bigtom_paths(split_override=args.bigtom_csv)
    if dataset_name == "tomi":
        return resolve_tomi_paths(
            benchmarks_root=args.benchmarks_root,
            root_override=args.tomi_root,
            split_override=args.tomi_test_path,
            trace_override=args.tomi_trace_path,
        )
    if dataset_name == "fantom":
        return resolve_fantom_paths(
            benchmarks_root=args.benchmarks_root,
            root_override=args.fantom_root,
            split_override=args.fantom_path,
            scorer_override=args.fantom_scorer_path,
            input_type=args.fantom_input_type,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _dry_run_summary(requested: List[str], args, out_dir: Path) -> Dict:
    summary = {
        "mode": args.mode,
        "base_model": args.base_model,
        "benchmarks_root": args.benchmarks_root,
        "datasets": {},
    }
    for dataset_name in requested:
        resolved = _resolve_paths_for_dataset(dataset_name, args)
        validation = validate_benchmark_paths(resolved)
        summary["datasets"][dataset_name] = validation
        if validation["split_exists"]:
            if dataset_name == "bigtom":
                rows = load_bigtom_official(resolved.split_path)
            elif dataset_name == "tomi":
                rows = load_tomi_official(resolved.split_path, resolved.trace_path)
            else:
                rows = load_fantom_official(resolved.split_path, input_type=args.fantom_input_type)
            summary["datasets"][dataset_name]["num_rows"] = len(_apply_limit(rows, args.limit))
    summary_path = out_dir / "dry_run_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="all",
                    help="comma-separated subset of {bigtom,tomi,fantom} or 'all'")
    ap.add_argument("--mode", choices=["base", "sft", "grpo"], default="grpo")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stage1_ckpt", type=str, default="")
    ap.add_argument("--policy_ckpt", type=str, default="")
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--benchmarks_root", type=str, default=str(default_benchmarks_root()))

    ap.add_argument("--bigtom_csv", type=str, default="")
    ap.add_argument("--bigtom_inference", choices=["generate", "choice"], default="choice")

    ap.add_argument("--tomi_root", type=str, default="")
    ap.add_argument("--tomi_test_path", type=str, default="")
    ap.add_argument("--tomi_trace_path", type=str, default="")

    ap.add_argument("--fantom_root", type=str, default="")
    ap.add_argument("--fantom_path", type=str, default="")
    ap.add_argument("--fantom_scorer_path", type=str, default="")
    ap.add_argument("--fantom_inference", choices=["generate", "choice"], default="generate")
    ap.add_argument("--fantom_input_type", choices=["short", "full"], default="short")
    ap.add_argument("--fantom_aggregation_target", choices=["set", "part", "conversation"], default="set")
    ap.add_argument("--fantom_embedding_model", type=str, default="sentence-transformers/all-roberta-large-v1")
    ap.add_argument("--fantom_allow_model_download", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--bigtom_max_new_tokens", type=int, default=None)
    ap.add_argument("--tomi_max_new_tokens", type=int, default=None)
    ap.add_argument("--fantom_max_new_tokens", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = _resolve_requested_datasets(args.datasets)

    if args.dry_run:
        _dry_run_summary(requested, args, out_dir)
        return

    resolved_paths = {dataset_name: _resolve_paths_for_dataset(dataset_name, args) for dataset_name in requested}
    for paths in resolved_paths.values():
        _ensure_dataset_path(paths)

    runner = load_policy_bundle(
        mode=args.mode,
        base_model=args.base_model,
        stage1_ckpt=args.stage1_ckpt or None,
        policy_ckpt=args.policy_ckpt or None,
        z_dim=args.z_dim,
    )

    combined_summary = {
        "mode": args.mode,
        "base_model": args.base_model,
        "benchmarks_root": args.benchmarks_root,
        "datasets": {},
    }

    for dataset_name in requested:
        resolved = resolved_paths[dataset_name]
        if dataset_name == "bigtom":
            rows = _apply_limit(load_bigtom_official(resolved.split_path), args.limit)
            combined_summary["datasets"]["bigtom"] = evaluate_bigtom(rows, runner, args, out_dir, resolved)
        elif dataset_name == "tomi":
            rows = _apply_limit(load_tomi_official(resolved.split_path, resolved.trace_path), args.limit)
            combined_summary["datasets"]["tomi"] = evaluate_tomi(rows, runner, args, out_dir, resolved)
        elif dataset_name == "fantom":
            rows = _apply_limit(load_fantom_official(resolved.split_path, input_type=args.fantom_input_type), args.limit)
            combined_summary["datasets"]["fantom"] = evaluate_fantom(rows, runner, args, out_dir, resolved)

    combined_summary_path = out_dir / f"combined_{args.mode}_summary.json"
    _write_json(combined_summary_path, combined_summary)
    markdown_path = out_dir / f"combined_{args.mode}_summary.md"
    _write_markdown_report(markdown_path, combined_summary)

    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
