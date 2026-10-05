#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_RUNS = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "results.md"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_project_config


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "—"


def official_row(
    model_id: str,
    source: str,
    official_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    fantom = official_metrics.get("fantom")
    control = official_metrics.get("control_task")
    if not isinstance(fantom, dict) or not isinstance(control, dict):
        return None
    return {
        "model": model_id,
        "source": source,
        "all_star": fantom.get("inaccessible:set:ALL*"),
        "all": fantom.get("inaccessible:set:ALL"),
        "belief_choice": fantom.get("inaccessible:belief:multiple-choice"),
        "first_order": fantom.get("inaccessible:first-order"),
        "second_order": fantom.get("inaccessible:second-order"),
        "control_first": control.get("accessible:first-order"),
        "control_second": control.get("accessible:second-order"),
    }


def gather_rows(
    config: dict[str, Any],
    runs_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows_by_model: dict[str, dict[str, Any]] = {}
    local_only = []

    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summary = load_json(summary_path)
        model_id = str(summary.get("model", {}).get("id", summary_path.parent.name))
        official_metrics = summary.get("official_metrics")
        if isinstance(official_metrics, dict):
            row = official_row(model_id, "new run", official_metrics)
            if row:
                rows_by_model[model_id] = row
        else:
            local_only.append(
                f"- `{model_id}`: local-only or partial run at `{summary_path}`"
            )

    for model_id, model in config["models"].items():
        prior_path_value = model.get("prior_summary")
        if not prior_path_value or model_id in rows_by_model:
            continue
        prior_path = Path(prior_path_value)
        if not prior_path.is_file():
            continue
        summary = load_json(prior_path)
        official_metrics = summary.get("official_metrics")
        if isinstance(official_metrics, dict):
            row = official_row(model_id, "prior full run", official_metrics)
            if row:
                rows_by_model[model_id] = row

    return list(rows_by_model.values()), local_only


def render_markdown(
    rows: list[dict[str, Any]],
    local_only: list[str],
) -> str:
    lines = [
        "# FANToM Rebuttal Results",
        "",
        "Official FANToM scores are percentages. The primary task contains",
        "information asymmetry; the control task does not.",
        "",
        "| Model | Source | All* | All | Belief Choice | First Order | Second Order | Control First | Control Second |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["model"]):
        lines.append(
            "| {model} | {source} | {all_star} | {all} | {belief_choice} | "
            "{first_order} | {second_order} | {control_first} | {control_second} |".format(
                model=f"`{row['model']}`",
                source=row["source"],
                all_star=format_value(row["all_star"]),
                all=format_value(row["all"]),
                belief_choice=format_value(row["belief_choice"]),
                first_order=format_value(row["first_order"]),
                second_order=format_value(row["second_order"]),
                control_first=format_value(row["control_first"]),
                control_second=format_value(row["control_second"]),
            )
        )

    if local_only:
        lines.extend([
            "",
            "## Partial Runs",
            "",
            *local_only,
        ])

    lines.extend([
        "",
        "The SOTOPIA policy-only control must not be described as latent-conditioned",
        "at inference. Only the BigToM SFT/GRPO entries use the learned `z1`/`z2`",
        "mental prefix when answering FANToM questions.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    rows, local_only = gather_rows(config, args.runs_dir)
    output = render_markdown(rows, local_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} official runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
