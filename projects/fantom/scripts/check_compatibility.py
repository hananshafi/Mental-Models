#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "compatibility.json"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_project_config


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_question_count(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        if isinstance(record.get("factQA"), dict):
            total += 1
        beliefs = record.get("beliefQAs", [])
        if isinstance(beliefs, list):
            total += 2 * sum(isinstance(item, dict) for item in beliefs)
        for field in ("infoAccessibilityQA_list", "answerabilityQA_list"):
            if isinstance(record.get(field), dict):
                total += 1
        for field in ("infoAccessibilityQAs_binary", "answerabilityQAs_binary"):
            values = record.get(field, [])
            if isinstance(values, list):
                total += sum(isinstance(item, dict) for item in values)
    return total


def speaker_count(context: str) -> int:
    names = set()
    for match in re.finditer(r"(?m)^([A-Z][A-Za-z'-]+):\s", context or ""):
        name = match.group(1).strip()
        if name:
            names.add(name)
    return len(names)


def inspect_dataset(dataset_config: dict[str, Any]) -> dict[str, Any]:
    split_path = Path(dataset_config["split_path"])
    result: dict[str, Any] = {
        "split_path": str(split_path),
        "exists": split_path.is_file(),
        "intended_use": dataset_config.get("intended_use"),
    }
    if not split_path.is_file():
        result["compatible"] = False
        result["errors"] = ["Official split is missing."]
        return result

    records = load_json(split_path)
    if not isinstance(records, list):
        result["compatible"] = False
        result["errors"] = ["Official split root must be a JSON list."]
        return result

    counts = [
        speaker_count(str(record.get("short_context", "")))
        for record in records
    ]
    count_histogram = Counter(counts)
    flattened = flatten_question_count(records)
    expected_conversations = int(dataset_config["expected_conversations"])
    expected_questions = int(dataset_config["expected_flattened_questions"])
    errors = []
    if len(records) != expected_conversations:
        errors.append(
            f"Expected {expected_conversations} conversations, found {len(records)}."
        )
    if flattened != expected_questions:
        errors.append(
            f"Expected {expected_questions} flattened questions, found {flattened}."
        )
    if not counts or min(counts) < 3:
        errors.append("FanToM records must contain at least three speakers.")

    result.update({
        "sha256": sha256_file(split_path),
        "conversations": len(records),
        "flattened_questions": flattened,
        "speaker_count_histogram": {
            str(key): value for key, value in sorted(count_histogram.items())
        },
        "min_speakers": min(counts) if counts else None,
        "max_speakers": max(counts) if counts else None,
        "compatible": not errors,
        "errors": errors,
    })
    return result


def inspect_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    result: dict[str, Any] = {
        "path": str(adapter_dir),
        "config_exists": config_path.is_file(),
        "weights_exist": weights_path.is_file(),
    }
    if config_path.is_file():
        config = load_json(config_path)
        result.update({
            "base_model": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "rank": config.get("r"),
        })
    return result


def cache_path_for_model(model_name: str) -> Path:
    hf_home = Path(
        os.environ.get("HF_HOME", str(REPO_ROOT / "artifacts" / "huggingface"))
    )
    return hf_home / "hub" / f"models--{model_name.replace('/', '--')}"


def inspect_model(model_id: str, model: dict[str, Any]) -> dict[str, Any]:
    kind = model["kind"]
    base_model = model["base_model"]
    result: dict[str, Any] = {
        "model_id": model_id,
        "kind": kind,
        "base_model": base_model,
        "training_source": model.get("training_source"),
        "rebuttal_role": model.get("rebuttal_role"),
        "base_model_cached": cache_path_for_model(base_model).is_dir(),
        "limitations": model.get("limitations", []),
    }
    errors: list[str] = []

    if kind == "base":
        result["direct_evaluation"] = True
        result["loader"] = "transformers.AutoModelForCausalLM"
    elif kind == "plain_adapter":
        adapter = inspect_adapter(Path(model["adapter_ckpt"]))
        result["adapter"] = adapter
        if not adapter["config_exists"]:
            errors.append("Adapter config is missing.")
        if not adapter["weights_exist"]:
            errors.append("Adapter weights are missing.")
        if adapter.get("base_model") != base_model:
            errors.append(
                f"Adapter base {adapter.get('base_model')!r} does not match {base_model!r}."
            )
        result["direct_evaluation"] = not errors
        result["loader"] = "plain PEFT policy adapter"
        result["latent_conditioning_at_inference"] = False
    elif kind == "bigtom_latent":
        stage1 = Path(model["stage1_ckpt"])
        policy = Path(model["policy_ckpt"])
        encoder_adapter = inspect_adapter(stage1 / "lora")
        policy_adapter = inspect_adapter(policy / "policy_lora")
        required_files = {
            "stage1_heads": (stage1 / "heads.pt").is_file(),
            "stage1_adapter": encoder_adapter["weights_exist"],
            "policy_adapter": policy_adapter["weights_exist"],
            "latent_projector": (policy / "projector.pt").is_file(),
        }
        result.update({
            "stage1_ckpt": str(stage1),
            "policy_ckpt": str(policy),
            "encoder_adapter": encoder_adapter,
            "policy_adapter": policy_adapter,
            "required_files": required_files,
            "loader": "BigToM z1/z2 latent-prefix policy bundle",
            "latent_conditioning_at_inference": True,
        })
        missing = [name for name, exists in required_files.items() if not exists]
        if missing:
            errors.append(f"Missing required files: {', '.join(missing)}.")
        for label, adapter in (
            ("Stage-1", encoder_adapter),
            ("policy", policy_adapter),
        ):
            if adapter.get("base_model") != base_model:
                errors.append(
                    f"{label} adapter base {adapter.get('base_model')!r} "
                    f"does not match {base_model!r}."
                )
        result["direct_evaluation"] = not errors
    else:
        errors.append(f"Unsupported model kind: {kind}")
        result["direct_evaluation"] = False

    result["compatible"] = not errors
    result["errors"] = errors
    return result


def prior_run_summary(model_id: str, model: dict[str, Any]) -> dict[str, Any] | None:
    summary_value = model.get("prior_summary")
    if not summary_value:
        return None
    path = Path(summary_value)
    result: dict[str, Any] = {
        "model_id": model_id,
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result

    summary = load_json(path)
    official = summary.get("official_metrics") or {}
    fantom = official.get("fantom", {})
    control = official.get("control_task", {})
    scoring = summary.get("scoring", {})
    result.update({
        "official_scorer_succeeded": (
            scoring.get("source") == "official_scorer"
            and scoring.get("official_scorer_returncode") == 0
        ),
        "conversation_input_type": fantom.get("conversation_input_type"),
        "inaccessible_all_star_percent": fantom.get("inaccessible:set:ALL*"),
        "inaccessible_all_percent": fantom.get("inaccessible:set:ALL"),
        "inaccessible_first_order_percent": fantom.get("inaccessible:first-order"),
        "inaccessible_second_order_percent": fantom.get("inaccessible:second-order"),
        "control_first_order_percent": control.get("accessible:first-order"),
        "control_second_order_percent": control.get("accessible:second-order"),
    })
    return result


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    dataset = inspect_dataset(config["dataset"])
    models = {
        model_id: inspect_model(model_id, model)
        for model_id, model in config["models"].items()
    }
    prior_runs = [
        prior
        for model_id, model in config["models"].items()
        if (prior := prior_run_summary(model_id, model)) is not None
    ]
    all_compatible = dataset["compatible"] and all(
        item["compatible"] for item in models.values()
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "models": models,
        "prior_runs": prior_runs,
        "all_registered_assets_compatible": all_compatible,
        "direct_use_conclusion": {
            "bigtom": (
                "Yes. The registered BigToM SFT/GRPO checkpoints can directly "
                "encode FANToM story/question pairs into z1/z2 and answer with "
                "the latent-prefix policy."
            ),
            "sotopia": (
                "Policy adapters are directly loadable for zero-shot QA through "
                "the plain-adapter runner, but their dyadic mental reward model "
                "is not part of inference and should not be described as a "
                "multi-party latent evaluation."
            ),
            "training_policy": (
                "Do not train or select checkpoints on FANToM labels; preserve "
                "the benchmark's evaluation-only intended use."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    report = build_report(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    dataset = report["dataset"]
    print(
        f"FANToM: {dataset.get('conversations')} conversations, "
        f"{dataset.get('flattened_questions')} probes, "
        f"{dataset.get('min_speakers')}-{dataset.get('max_speakers')} speakers"
    )
    for model_id, result in report["models"].items():
        status = "DIRECT" if result["compatible"] else "INCOMPATIBLE"
        latent = "latent-prefix" if result.get("latent_conditioning_at_inference") else "policy-only"
        print(f"{status:12} {model_id:28} {latent}")
        for error in result["errors"]:
            print(f"  - {error}")
    print(f"Wrote {args.output}")
    return 0 if report["all_registered_assets_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
