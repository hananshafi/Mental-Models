import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from official_eval_bridge_tomi import load_tomi_official, summarize_tomi_metrics
from official_eval_common import build_benchmark_prompt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "tomi"
DEFAULT_SPLIT_PATH = DEFAULT_BENCHMARK_ROOT / "tomi_balanced_story_types" / "fb_all_test.txt"
DEFAULT_TRACE_PATH = DEFAULT_BENCHMARK_ROOT / "tomi_balanced_story_types" / "fb_all_test.trace"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

COMPLETION_MODELS = {
    "gpt-3.5-turbo-instruct",
    "text-davinci-003",
    "text-davinci-002",
    "davinci-002",
    "babbage-002",
}

TOMI_EXTRA_INSTRUCTION = (
    "Return only the location identifier copied exactly from the story. "
    "Preserve underscores, and do not add articles, punctuation, or explanation. "
    "Examples: blue_container, green_bucket, closet."
)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _normalize_location(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = text.splitlines()[0].strip()
    text = re.sub(r"^```.*$", "", text).strip()
    text = re.sub(r"\b(in|the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_story_locations(story: str) -> List[str]:
    patterns = [
        r"entered the ([A-Za-z_]+)",
        r"exited the ([A-Za-z_]+)",
        r"is in the ([A-Za-z_]+)",
        r"moved [^.]* to the ([A-Za-z_]+)",
    ]
    candidates: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, story):
            candidate = match.strip().strip(".")
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _canonicalize_tomi_prediction(story: str, prediction: str) -> str:
    raw = (prediction or "").strip()
    if not raw:
        return raw

    first_line = raw.splitlines()[0].strip()
    if first_line.startswith("```"):
        lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("```")]
        if lines:
            first_line = lines[0]

    normalized_pred = _normalize_location(first_line)
    if not normalized_pred:
        normalized_pred = _normalize_location(raw)

    candidates = _extract_story_locations(story)
    exact_matches = [candidate for candidate in candidates if _normalize_location(candidate) == normalized_pred]
    if len(exact_matches) == 1:
        return exact_matches[0]

    contains_matches = [candidate for candidate in candidates if _normalize_location(candidate) in normalized_pred]
    if len(contains_matches) == 1:
        return contains_matches[0]

    return first_line


def _resolve_provider(model: str, provider: str) -> str:
    if provider != "auto":
        return provider
    if model.startswith("gemini-"):
        return "gemini"
    return "openai"


def _resolve_client_kwargs(args) -> tuple[str, Dict[str, str]]:
    provider = _resolve_provider(args.model, args.provider)

    api_key = args.api_key
    if not api_key:
        env_key = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_key)
    if not api_key:
        env_key = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        raise ValueError(
            f"No API key provided. Pass --api_key or set {env_key} in the environment."
        )

    base_url = args.base_url
    if provider == "gemini" and not base_url:
        base_url = GEMINI_OPENAI_BASE_URL

    client_kwargs: Dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return provider, client_kwargs


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    if model in COMPLETION_MODELS:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].text or "").strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _score_rows(rows: List[Dict]) -> Dict[str, object]:
    scored = []
    for row in rows:
        gold = (row.get("answer") or "").strip().lower()
        pred = (row.get("prediction") or "").strip().lower()
        score = 1.0 if gold == pred or (gold and gold in pred) else 0.0
        record = dict(row)
        record["score"] = score
        scored.append(record)

    metrics = summarize_tomi_metrics(scored)
    metrics["Protocol"] = "released_tomi_exact_match_bridge"
    metrics["NumScored"] = len(scored)
    metrics["BenchmarkRoot"] = str(DEFAULT_BENCHMARK_ROOT)
    return {"rows": scored, "metrics": metrics}


def _write_jsonl(path: Path, rows: List[Dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: Dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_predictions(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate an OpenAI-compatible model on ToMi.")
    ap.add_argument("--provider", choices=["auto", "openai", "gemini"], default="auto")
    ap.add_argument("--model", default="gpt-3.5-turbo-instruct")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--openai_api_key", dest="api_key", default=None,
                    help="Deprecated alias for --api_key.")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--openai_base_url", dest="base_url", default=None,
                    help="Deprecated alias for --base_url.")
    ap.add_argument("--split_path", default=str(DEFAULT_SPLIT_PATH))
    ap.add_argument("--trace_path", default=str(DEFAULT_TRACE_PATH))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh and ignore any saved partial/final predictions.",
    )
    args = ap.parse_args()

    provider, client_kwargs = _resolve_client_kwargs(args)
    client = OpenAI(**client_kwargs)

    rows = load_tomi_official(Path(args.split_path), Path(args.trace_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} ToMi examples from {args.split_path}", flush=True)
    print(f"Using provider={provider} model={args.model}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "tomi_openai_predictions.partial.jsonl"
    preds_path = out_dir / "tomi_openai_predictions.jsonl"
    summary_path = out_dir / "tomi_openai_summary.json"

    predictions: List[Dict] = []
    start_idx = 0
    if not args.no_resume:
        resume_path = None
        if partial_path.exists():
            resume_path = partial_path
        elif preds_path.exists():
            resume_path = preds_path
        if resume_path is not None:
            predictions = _load_existing_predictions(resume_path)
            start_idx = len(predictions)
            if predictions:
                print(f"Resuming from {start_idx} saved predictions in {resume_path}", flush=True)
    else:
        for path in (partial_path, preds_path, summary_path):
            if path.exists():
                path.unlink()

    started = time.time()
    for idx, row in enumerate(rows[start_idx:], start=start_idx + 1):
        prompt = build_benchmark_prompt(
            row["story"],
            row["question"],
            dataset_name="ToMi",
            extra_instruction=TOMI_EXTRA_INSTRUCTION,
        )
        raw_prediction = _call_model(
            client,
            args.model,
            prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        prediction = _canonicalize_tomi_prediction(row["story"], raw_prediction)
        rec = dict(row)
        rec["raw_prediction"] = raw_prediction
        rec["prediction"] = prediction
        predictions.append(rec)
        _append_jsonl(partial_path, rec)

        if idx == 1 or idx == len(rows) or idx % args.log_every == 0:
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (len(rows) - idx) / rate if rate > 0 else None
            eta_text = _format_duration(eta) if eta is not None else "n/a"
            print(
                f"[tomi_openai] {idx}/{len(rows)} "
                f"elapsed={_format_duration(elapsed)} eta={eta_text}",
                flush=True,
            )

    scored = _score_rows(predictions)
    _write_jsonl(preds_path, scored["rows"])
    if partial_path.exists():
        partial_path.unlink()

    payload = {
        "dataset": "tomi",
        "mode": "openai_compatible_api",
        "provider": provider,
        "model": args.model,
        "split_path": args.split_path,
        "trace_path": args.trace_path,
        "metrics": scored["metrics"],
        "scoring": {
            "source": "released_tomi_exact_match_bridge",
            "inference_mode": "generate",
            "official_protocol_bridge": True,
        },
        "paths": {
            "predictions_jsonl": str(preds_path),
            "benchmark_root": str(DEFAULT_BENCHMARK_ROOT),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"Predictions: {preds_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
