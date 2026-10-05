"""
Convenience entrypoint for zero-shot benchmark evaluation.

This wraps `evaluate_official_benchmarks.py` in `base` mode so we can evaluate
an unfine-tuned causal LM across the paper benchmarks without passing the
stage-1 / stage-4 checkpoint arguments used by the BigToM-conditioned modes.
"""
import argparse
import sys
from pathlib import Path

import evaluate_official_benchmarks as official_eval


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _slugify_model_name(model_name: str) -> str:
    return model_name.strip().replace("/", "__").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="all",
                    help="comma-separated subset of {bigtom,tomi,fantom} or 'all'")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out_dir", type=str, default="",
                    help="optional explicit output dir; defaults to eval_results/official_zeroshot/<model_slug>")
    ap.add_argument("--benchmarks_root", type=str, default=str(PROJECT_ROOT / "benchmarks"))

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

    out_dir = args.out_dir
    if not out_dir:
        model_slug = _slugify_model_name(args.base_model)
        out_dir = str(PROJECT_ROOT / "eval_results" / "official_zeroshot" / model_slug)

    forwarded_argv = [
        "evaluate_official_benchmarks.py",
        "--datasets", args.datasets,
        "--mode", "base",
        "--base_model", args.base_model,
        "--out_dir", out_dir,
        "--benchmarks_root", args.benchmarks_root,
        "--bigtom_csv", args.bigtom_csv,
        "--bigtom_inference", args.bigtom_inference,
        "--tomi_root", args.tomi_root,
        "--tomi_test_path", args.tomi_test_path,
        "--tomi_trace_path", args.tomi_trace_path,
        "--fantom_root", args.fantom_root,
        "--fantom_path", args.fantom_path,
        "--fantom_scorer_path", args.fantom_scorer_path,
        "--fantom_inference", args.fantom_inference,
        "--fantom_input_type", args.fantom_input_type,
        "--fantom_aggregation_target", args.fantom_aggregation_target,
        "--fantom_embedding_model", args.fantom_embedding_model,
        "--max_new_tokens", str(args.max_new_tokens),
        "--log_every", str(args.log_every),
    ]
    if args.bigtom_max_new_tokens is not None:
        forwarded_argv.extend(["--bigtom_max_new_tokens", str(args.bigtom_max_new_tokens)])
    if args.tomi_max_new_tokens is not None:
        forwarded_argv.extend(["--tomi_max_new_tokens", str(args.tomi_max_new_tokens)])
    if args.fantom_max_new_tokens is not None:
        forwarded_argv.extend(["--fantom_max_new_tokens", str(args.fantom_max_new_tokens)])

    if args.limit is not None:
        forwarded_argv.extend(["--limit", str(args.limit)])
    if args.dry_run:
        forwarded_argv.append("--dry_run")
    if args.fantom_allow_model_download:
        forwarded_argv.append("--fantom_allow_model_download")

    old_argv = sys.argv
    try:
        sys.argv = forwarded_argv
        official_eval.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
