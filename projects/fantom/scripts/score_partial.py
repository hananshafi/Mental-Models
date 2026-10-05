#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_RUNS = PROJECT_ROOT / "runs"
DEFAULT_MODELS = (
    "qwen25_7b_base",
    "bigtom_sft_epoch1",
    "bigtom_grpo_step300",
    "sotopia_qwen_grpo_best",
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_project_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def count_jsonl_rows(path: Path, stop_at: int | None = None) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if stop_at is not None and count >= stop_at:
                break
    return count


def read_jsonl_prefix(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
            if len(rows) == limit:
                break
    return rows


def snapshot_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def write_snapshot(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    atomic_write(path, payload)


def score_model(
    *,
    model_id: str,
    config: dict[str, Any],
    runs_dir: Path,
    limit: int,
    allow_model_download: bool,
) -> dict[str, Any]:
    predictions_path = runs_dir / model_id / "predictions.jsonl"
    rows = read_jsonl_prefix(predictions_path, limit)
    if len(rows) != limit:
        raise ValueError(
            f"{model_id} has {len(rows)} predictions; {limit} are required."
        )

    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"{model_id} has a prediction without a sample_id.")
    if len(set(sample_ids)) != limit:
        raise ValueError(f"{model_id} has duplicate sample_ids in its prefix.")

    output_dir = runs_dir / model_id / f"partial_{limit}"
    snapshot_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    write_snapshot(snapshot_path, rows)

    bigtom_scripts = Path(config["shared"]["bigtom_scripts"])
    sys.path.insert(0, str(bigtom_scripts))
    from official_eval_scorers import run_official_fantom_scorer

    scorer_run = run_official_fantom_scorer(
        scorer_path=Path(config["dataset"]["official_scorer_path"]),
        predictions_path=snapshot_path,
        split_path=Path(config["dataset"]["split_path"]),
        input_type=config["dataset"]["default_input_type"],
        aggregation_target="set",
        embedding_model=config["shared"]["embedding_model"],
        allow_model_download=allow_model_download,
    )
    if scorer_run is None:
        raise RuntimeError("The official FANToM scorer could not be located.")

    metrics = scorer_run.get("metrics")
    status = (
        "completed"
        if scorer_run.get("returncode") == 0 and isinstance(metrics, dict)
        else "failed"
    )
    summary = {
        "dataset": "fantom",
        "model_id": model_id,
        "status": status,
        "created_at": utc_now(),
        "sample_count": limit,
        "selection": f"first {limit} flattened probes in official order",
        "complete_split": False,
        "snapshot": {
            "path": str(snapshot_path),
            "sha256": snapshot_digest(rows),
            "first_sample_id": sample_ids[0],
            "last_sample_id": sample_ids[-1],
        },
        "official_metrics": metrics,
        "scoring": {
            "source": "official_fantom_bridge_partial",
            "returncode": scorer_run.get("returncode"),
            "command": scorer_run.get("command"),
            "stderr_tail": str(scorer_run.get("stderr", ""))[-4000:],
        },
        "caveats": [
            "This is a deterministic prefix, not a random or stratified sample.",
            "The final set may be incomplete, so set-level ALL/ALL* are preliminary.",
            "Use the complete 12,832-probe run for final rebuttal claims.",
        ],
    }
    atomic_write(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    if status != "completed":
        raise RuntimeError(
            f"Official scorer failed for {model_id}; see {summary_path}."
        )
    return summary


def format_metric(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "—"


def render_report(
    *,
    model_ids: list[str],
    runs_dir: Path,
    limit: int,
    counts: dict[str, int],
) -> str:
    hashes_to_models: dict[str, list[str]] = {}
    lines = [
        f"# FANToM First-{limit:,} Results",
        "",
        "These are preliminary official-scorer results on the first "
        f"{limit:,} flattened probes. Full evaluations continue independently.",
        "",
        "| Model | Status | N | All* | All | Belief Choice | Belief Dist. | First Order | Second Order | Control First | Control Second |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in model_ids:
        summary_path = runs_dir / model_id / f"partial_{limit}" / "summary.json"
        summary = load_json(summary_path) if summary_path.is_file() else {}
        metrics = summary.get("official_metrics") or {}
        fantom = metrics.get("fantom") or {}
        control = metrics.get("control_task") or {}
        status = summary.get("status", "waiting")
        snapshot_hash = (summary.get("snapshot") or {}).get("sha256")
        if status == "completed" and snapshot_hash:
            hashes_to_models.setdefault(str(snapshot_hash), []).append(model_id)
        lines.append(
            "| `{model}` | {status} | {count} | {all_star} | {all_score} | "
            "{belief_choice} | {belief_dist} | {first_order} | {second_order} | "
            "{control_first} | {control_second} |".format(
                model=model_id,
                status=status,
                count=summary.get("sample_count", counts.get(model_id, 0)),
                all_star=format_metric(
                    fantom.get("inaccessible:set:ALL*")
                ),
                all_score=format_metric(
                    fantom.get("inaccessible:set:ALL")
                ),
                belief_choice=format_metric(
                    fantom.get("inaccessible:belief:multiple-choice")
                ),
                belief_dist=format_metric(
                    fantom.get("inaccessible:belief:distance")
                ),
                first_order=format_metric(
                    fantom.get("inaccessible:first-order")
                ),
                second_order=format_metric(
                    fantom.get("inaccessible:second-order")
                ),
                control_first=format_metric(
                    control.get("accessible:first-order")
                ),
                control_second=format_metric(
                    control.get("accessible:second-order")
                ),
            )
        )

    duplicate_groups = [
        models for models in hashes_to_models.values() if len(models) > 1
    ]
    if duplicate_groups:
        lines.extend(["", "## Diagnostic Warning", ""])
        for models in duplicate_groups:
            model_list = ", ".join(f"`{model}`" for model in models)
            lines.append(
                f"- {model_list} produced byte-identical first-{limit:,} "
                "prediction snapshots, including choice scores. Treat their "
                "equal metrics as behavioral identity, not independent gains."
            )

    lines.extend([
        "",
        "Caveats: this is the first contiguous prefix rather than a random or",
        "stratified sample. The prefix boundary can cut through a question set, so",
        "`All*` and `All` are interim diagnostics rather than final leaderboard",
        "numbers. The immutable snapshots and detailed JSON summaries are stored",
        f"under `runs/<model>/partial_{limit}/`.",
        "",
    ])
    return "\n".join(lines)


def write_report(
    *,
    model_ids: list[str],
    runs_dir: Path,
    output: Path,
    limit: int,
    counts: dict[str, int],
) -> None:
    atomic_write(
        output,
        render_report(
            model_ids=model_ids,
            runs_dir=runs_dir,
            limit=limit,
            counts=counts,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")

    config = load_project_config(args.config)
    model_ids = args.model_ids or list(DEFAULT_MODELS)
    output = args.output or (
        PROJECT_ROOT / "reports" / f"partial_{args.limit}.md"
    )
    pending = set(model_ids)
    counts = {model_id: 0 for model_id in model_ids}
    last_reported_bucket = {model_id: -1 for model_id in model_ids}

    for model_id in list(pending):
        summary_path = (
            args.runs_dir
            / model_id
            / f"partial_{args.limit}"
            / "summary.json"
        )
        if summary_path.is_file():
            summary = load_json(summary_path)
            if summary.get("status") == "completed":
                counts[model_id] = int(summary.get("sample_count", args.limit))
                pending.remove(model_id)

    write_report(
        model_ids=model_ids,
        runs_dir=args.runs_dir,
        output=output,
        limit=args.limit,
        counts=counts,
    )

    while pending:
        ready = []
        for model_id in model_ids:
            if model_id not in pending:
                continue
            path = args.runs_dir / model_id / "predictions.jsonl"
            count = count_jsonl_rows(path, stop_at=args.limit)
            counts[model_id] = count
            bucket = count // 25
            if bucket != last_reported_bucket[model_id]:
                print(
                    f"{model_id}: {count}/{args.limit} predictions",
                    flush=True,
                )
                last_reported_bucket[model_id] = bucket
            if count >= args.limit:
                ready.append(model_id)

        write_report(
            model_ids=model_ids,
            runs_dir=args.runs_dir,
            output=output,
            limit=args.limit,
            counts=counts,
        )

        if not ready:
            if not args.wait:
                break
            time.sleep(args.poll_seconds)
            continue

        for model_id in ready:
            print(f"{model_id}: scoring first {args.limit}", flush=True)
            try:
                score_model(
                    model_id=model_id,
                    config=config,
                    runs_dir=args.runs_dir,
                    limit=args.limit,
                    allow_model_download=args.allow_model_download,
                )
            except Exception as exc:
                print(f"{model_id}: scoring failed: {exc}", flush=True)
            pending.remove(model_id)
            write_report(
                model_ids=model_ids,
                runs_dir=args.runs_dir,
                output=output,
                limit=args.limit,
                counts=counts,
            )

    print(f"Wrote partial results to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
