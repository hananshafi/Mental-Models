#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models.json"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_utils import load_project_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def normalize_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalize_text(reference).split()
    prediction_tokens = normalize_text(prediction).split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    reference_counts = Counter(reference_tokens)
    prediction_counts = Counter(prediction_tokens)
    overlap = sum(
        min(count, prediction_counts.get(token, 0))
        for token, count in reference_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_yes_no(text: str) -> str | None:
    for token in normalize_text(text).split():
        if token in {"yes", "true"}:
            return "yes"
        if token in {"no", "false"}:
            return "no"
    return None


def contains_gold_not_wrong(
    prediction: str,
    gold: str,
    wrong: str = "",
) -> bool | None:
    prediction_normalized = normalize_text(prediction)
    gold_normalized = normalize_text(gold)
    wrong_normalized = normalize_text(wrong) if wrong else ""
    has_gold = bool(gold_normalized) and gold_normalized in prediction_normalized
    has_wrong = (
        bool(wrong_normalized)
        and wrong_normalized != gold_normalized
        and wrong_normalized in prediction_normalized
    )
    if has_gold and not has_wrong:
        return True
    if has_wrong and not has_gold:
        return False
    return None


def score_prediction(
    row: dict[str, Any],
    prediction: str,
    predicted_choice_index: int | None,
) -> dict[str, Any]:
    gold = row["answer"]
    wrong = row.get("wrong_answer", "")
    question_type = str(row.get("question_type", "unknown"))

    if row.get("choices") and row.get("gold_choice_index") is not None:
        score = 1.0 if predicted_choice_index == row["gold_choice_index"] else 0.0
        exact = bool(score)
        token_f1_value = score
    elif isinstance(gold, list):
        gold_normalized = [
            normalize_text(str(item))
            for item in gold
            if normalize_text(str(item))
        ]
        wrong_values = wrong if isinstance(wrong, list) else []
        wrong_normalized = [normalize_text(str(item)) for item in wrong_values]
        prediction_normalized = normalize_text(prediction)
        found_gold = sum(
            1 for item in gold_normalized if item in prediction_normalized
        )
        hit_wrong = any(
            item
            and item in prediction_normalized
            and item not in gold_normalized
            for item in wrong_normalized
        )
        score = found_gold / len(gold_normalized) if gold_normalized else 0.0
        exact = found_gold == len(gold_normalized) and not hit_wrong
        token_f1_value = score
    elif (
        "binary" in normalize_text(question_type)
        or normalize_text(str(gold)) in {"yes", "no", "true", "false"}
    ):
        score = (
            1.0
            if normalize_yes_no(prediction) == normalize_yes_no(str(gold))
            else 0.0
        )
        exact = bool(score)
        token_f1_value = score
    else:
        decision = contains_gold_not_wrong(
            prediction,
            str(gold),
            str(wrong),
        )
        exact = (
            bool(decision)
            if decision is not None
            else normalize_text(str(gold)) == normalize_text(prediction)
        )
        token_f1_value = token_f1(str(gold), prediction)
        score = 1.0 if exact else token_f1_value

    return {
        "score": score,
        "exact": exact,
        "token_f1": token_f1_value,
    }


def add_bigtom_scripts(config: dict[str, Any]) -> Path:
    scripts_path = Path(config["shared"]["bigtom_scripts"])
    if not scripts_path.is_dir():
        raise FileNotFoundError(f"BigToM scripts not found: {scripts_path}")
    sys.path.insert(0, str(scripts_path))
    return scripts_path


def resolve_model_source(model_name: str) -> str:
    direct_path = Path(model_name)
    if direct_path.exists():
        return str(direct_path)

    hf_home = Path(
        os.environ.get("HF_HOME", str(REPO_ROOT / "artifacts" / "huggingface"))
    )
    model_cache = hf_home / "hub" / f"models--{model_name.replace('/', '--')}"
    main_ref = model_cache / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = model_cache / "snapshots" / revision
        if snapshot.is_dir():
            return str(snapshot)

    snapshots = model_cache / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return model_name


def force_load_and_verify_peft_adapter(
    policy,
    adapter_dir: Path,
    policy_device,
    adapter_name: str = "default",
) -> dict[str, Any]:
    import torch
    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        load_peft_weights,
        set_peft_model_state_dict,
    )

    expected = load_peft_weights(str(adapter_dir), device=str(policy_device))
    set_peft_model_state_dict(policy, expected, adapter_name=adapter_name)
    policy.set_adapter(adapter_name)
    loaded = get_peft_model_state_dict(policy, adapter_name=adapter_name)

    expected_keys = set(expected)
    loaded_keys = set(loaded)
    if expected_keys != loaded_keys:
        missing = sorted(expected_keys - loaded_keys)
        extra = sorted(loaded_keys - expected_keys)
        raise RuntimeError(
            "PEFT adapter verification key mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    max_abs_delta = 0.0
    expected_lora_b_sq = 0.0
    loaded_lora_b_sq = 0.0
    all_exact = True
    for key in sorted(expected):
        expected_tensor = expected[key].detach()
        loaded_tensor = loaded[key].detach().to(
            device=expected_tensor.device,
            dtype=expected_tensor.dtype,
        )
        all_exact = all_exact and torch.equal(loaded_tensor, expected_tensor)
        max_abs_delta = max(
            max_abs_delta,
            float((loaded_tensor - expected_tensor).abs().max().item()),
        )
        if ".lora_B." in key:
            expected_lora_b_sq += float(
                expected_tensor.float().pow(2).sum().item()
            )
            loaded_lora_b_sq += float(
                loaded_tensor.float().pow(2).sum().item()
            )

    if not all_exact:
        raise RuntimeError(
            "PEFT adapter tensors do not match the checkpoint after reload; "
            f"max_abs_delta={max_abs_delta}"
        )

    adapter_file = adapter_dir / "adapter_model.safetensors"
    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_file_sha256": (
            sha256_file(adapter_file) if adapter_file.is_file() else None
        ),
        "adapter_name": adapter_name,
        "policy_device": str(policy_device),
        "tensor_count": len(expected),
        "all_tensors_exact": all_exact,
        "max_abs_delta": max_abs_delta,
        "checkpoint_lora_b_norm": expected_lora_b_sq ** 0.5,
        "loaded_lora_b_norm": loaded_lora_b_sq ** 0.5,
    }


def load_runner(model_id: str, model_config: dict[str, Any], z_dim: int):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from official_eval_common import PolicyBundle, load_policy_bundle

    kind = model_config["kind"]
    base_model = model_config["base_model"]
    model_source = resolve_model_source(base_model)
    if kind == "base":
        runner = load_policy_bundle(
            mode="base",
            base_model=model_source,
            stage1_ckpt=None,
            policy_ckpt=None,
            z_dim=z_dim,
        )
    elif kind == "bigtom_latent":
        runner = load_policy_bundle(
            mode="grpo",
            base_model=model_source,
            stage1_ckpt=model_config["stage1_ckpt"],
            policy_ckpt=model_config["policy_ckpt"],
            z_dim=z_dim,
        )
        runner.adapter_audit = force_load_and_verify_peft_adapter(
            runner.policy,
            Path(model_config["policy_ckpt"]) / "policy_lora",
            runner.policy_device,
        )
    elif kind == "plain_adapter":
        if not torch.cuda.is_available():
            raise RuntimeError("Plain-adapter evaluation requires a CUDA GPU.")
        device = torch.device("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy_base = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
        )
        policy = PeftModel.from_pretrained(
            policy_base,
            model_config["adapter_ckpt"],
            is_trainable=False,
        )
        policy.eval()
        runner = PolicyBundle(
            mode=model_id,
            tokenizer=tokenizer,
            policy=policy,
            policy_device=device,
            z_dim=z_dim,
        )
        runner.adapter_audit = force_load_and_verify_peft_adapter(
            runner.policy,
            Path(model_config["adapter_ckpt"]),
            runner.policy_device,
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    runner.mode = model_id
    return runner


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return predictions
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in predictions:
                raise ValueError(
                    f"Duplicate sample_id {sample_id!r} at {path}:{line_number}"
                )
            predictions[sample_id] = record
    return predictions


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_predictions(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_config: dict[str, Any],
    split_path: Path,
) -> dict[str, Any]:
    return {
        "status": "starting",
        "started_at": utc_now(),
        "model_id": args.model_id,
        "model": model_config,
        "dataset": {
            "split_path": str(split_path),
            "sha256": sha256_file(split_path),
            "input_type": args.input_type,
            "expected_conversations": config["dataset"]["expected_conversations"],
            "expected_flattened_questions": config["dataset"]["expected_flattened_questions"],
            "intended_use": config["dataset"]["intended_use"],
        },
        "evaluation": {
            "limit": args.limit,
            "inference": args.inference,
            "max_new_tokens": args.max_new_tokens,
            "official_requested": args.official,
            "aggregation_target": args.aggregation_target,
            "seed": args.seed,
        },
        "environment": {
            "python": sys.executable,
            "hf_home": os.environ.get("HF_HOME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "code": {
            "paper_revision": git_revision(PROJECT_ROOT.parents[1]),
            "argv": sys.argv,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-type", choices=["short", "full"], default=None)
    parser.add_argument("--inference", choices=["generate", "choice"], default="generate")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official", action="store_true")
    parser.add_argument(
        "--aggregation-target",
        choices=["set", "part", "conversation"],
        default="set",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_project_config(args.config)
    if args.model_id not in config["models"]:
        choices = ", ".join(sorted(config["models"]))
        raise KeyError(f"Unknown model {args.model_id!r}. Choose one of: {choices}")
    model_config = config["models"][args.model_id]
    args.input_type = args.input_type or config["dataset"]["default_input_type"]
    args.embedding_model = (
        args.embedding_model or config["shared"]["embedding_model"]
    )

    add_bigtom_scripts(config)
    from official_eval_common import FANTOM_HEADER, build_benchmark_prompt
    from official_eval_loaders import load_fantom_official
    from official_eval_scorers import (
        run_official_fantom_scorer,
        summarize_fantom_metrics,
    )

    split_path = Path(config["dataset"]["split_path"])
    scorer_path = Path(config["dataset"]["official_scorer_path"])
    rows = load_fantom_official(split_path, input_type=args.input_type)
    full_row_count = len(rows)
    if args.limit is not None:
        rows = rows[:args.limit]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.run_dir / "predictions.jsonl"
    official_input_path = args.run_dir / "official_input.json"
    summary_path = args.run_dir / "summary.json"
    manifest_path = args.run_dir / "run_manifest.json"

    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
        official_input_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"{predictions_path} exists; use --resume or --overwrite."
        )

    manifest = build_manifest(args, config, model_config, split_path)
    manifest["dataset"]["loaded_questions"] = full_row_count
    manifest["dataset"]["selected_questions"] = len(rows)
    write_json(manifest_path, manifest)

    if args.dry_run:
        manifest.update({"status": "dry_run_complete", "completed_at": utc_now()})
        write_json(manifest_path, manifest)
        print(
            f"Dry run: model={args.model_id} rows={len(rows)} "
            f"run_dir={args.run_dir}"
        )
        return manifest

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        existing = load_existing_predictions(predictions_path)
        missing_rows = [
            row for row in rows if str(row["sample_id"]) not in existing
        ]
        runner = None
        if missing_rows:
            runner = load_runner(
                args.model_id,
                model_config,
                int(config["shared"]["z_dim"]),
            )
            adapter_audit = getattr(runner, "adapter_audit", None)
            if adapter_audit is not None:
                manifest["model_loading"] = {
                    "peft_adapter": adapter_audit,
                }
                write_json(manifest_path, manifest)
                print(
                    "Verified PEFT adapter: "
                    f"{adapter_audit['adapter_dir']} "
                    f"tensors={adapter_audit['tensor_count']} "
                    f"lora_B_norm={adapter_audit['loaded_lora_b_norm']:.6f}",
                    flush=True,
                )
        started = time.time()

        for index, row in enumerate(rows, start=1):
            sample_id = str(row["sample_id"])
            if sample_id in existing:
                continue
            if runner is None:
                raise RuntimeError("Model runner was not initialized.")

            choices = row.get("choices") or []
            prompt = FANTOM_HEADER + build_benchmark_prompt(
                row["story"],
                row["question"],
                dataset_name="FANToM",
                choices=choices if choices else None,
            )
            use_choice = bool(choices) and (
                args.inference == "choice"
                or str(row.get("question_type", "")).endswith(":multiple-choice")
            )
            if use_choice:
                picked = runner.pick_choice(
                    prompt,
                    choices,
                    story=row["story"],
                    question=row["question"],
                )
                prediction = str(picked["choice_text"])
                predicted_choice_index = int(picked["choice_index"])
                choice_scores = picked["scores"]
            else:
                prediction = runner.generate(
                    prompt,
                    story=row["story"],
                    question=row["question"],
                    max_new_tokens=args.max_new_tokens,
                )
                predicted_choice_index = None
                choice_scores = None

            record = dict(row)
            record.update(
                score_prediction(row, prediction, predicted_choice_index)
            )
            record["prediction"] = prediction
            if predicted_choice_index is not None:
                record["choice_index"] = predicted_choice_index
                record["choice_scores"] = choice_scores
            append_prediction(predictions_path, record)
            existing[sample_id] = record

            if index == 1 or index % args.log_every == 0 or index == len(rows):
                elapsed = time.time() - started
                completed = sum(
                    str(item["sample_id"]) in existing for item in rows
                )
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[fantom] {completed}/{len(rows)} "
                    f"elapsed={elapsed:.0f}s rate={rate:.3f} examples/s",
                    flush=True,
                )

        ordered_predictions = [
            existing[str(row["sample_id"])]
            for row in rows
            if str(row["sample_id"]) in existing
        ]
        if len(ordered_predictions) != len(rows):
            raise RuntimeError(
                f"Expected {len(rows)} predictions, found {len(ordered_predictions)}."
            )

        for record in ordered_predictions:
            record.update(
                score_prediction(
                    record,
                    str(record["prediction"]),
                    record.get("choice_index"),
                )
            )
        rewrite_predictions(predictions_path, ordered_predictions)
        write_json(official_input_path, ordered_predictions)
        local_metrics = summarize_fantom_metrics(ordered_predictions)
        official_run = None
        official_metrics = None
        if args.official:
            expected = int(config["dataset"]["expected_flattened_questions"])
            if args.limit is not None or len(rows) != expected:
                raise ValueError(
                    "Official scoring requires the complete split with no --limit."
                )
            official_run = run_official_fantom_scorer(
                scorer_path=scorer_path,
                predictions_path=official_input_path,
                split_path=split_path,
                input_type=args.input_type,
                aggregation_target=args.aggregation_target,
                embedding_model=args.embedding_model,
                allow_model_download=args.allow_model_download,
            )
            if official_run is not None:
                official_metrics = official_run.get("metrics")

        completed_at = utc_now()
        summary = {
            "dataset": "fantom",
            "model": {
                "id": args.model_id,
                **model_config,
            },
            "evaluation": {
                "input_type": args.input_type,
                "inference": args.inference,
                "questions": len(rows),
                "complete_split": len(rows) == full_row_count,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "metrics": official_metrics or local_metrics,
            "local_metrics": local_metrics,
            "official_metrics": official_metrics,
            "scoring": {
                "source": "official_scorer" if official_metrics else "local_adapter",
                "official_scorer_attempted": args.official,
                "official_scorer_returncode": (
                    official_run.get("returncode")
                    if official_run is not None
                    else None
                ),
                "official_scorer_command": (
                    official_run.get("command")
                    if official_run is not None
                    else None
                ),
                "official_scorer_stderr": (
                    official_run.get("stderr")
                    if official_run is not None
                    else None
                ),
            },
            "paths": {
                "predictions_jsonl": str(predictions_path),
                "official_input_json": str(official_input_path),
                "manifest_json": str(manifest_path),
            },
            "runtime": {
                "started_at": manifest["started_at"],
                "completed_at": completed_at,
                "seconds": time.time() - started,
            },
        }
        write_json(summary_path, summary)
        manifest.update({
            "status": "completed",
            "completed_at": completed_at,
            "summary_path": str(summary_path),
        })
        write_json(manifest_path, manifest)
        print(f"Wrote {summary_path}")
        return summary
    except Exception as error:
        manifest.update({
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
        })
        write_json(manifest_path, manifest)
        raise


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
