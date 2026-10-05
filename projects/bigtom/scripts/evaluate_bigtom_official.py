"""
BigToM evaluation following the official protocol from:
https://github.com/cicl-stanford/procedural-evals-tom

Protocol:
- MCQ format: "{question}\\nChoose one of the following:\\na){ans1}\\nb){ans2}"
- random.seed(0) once at start; random.shuffle() per row for choice ordering
- Grade by checking 'a)' or 'b)' in the lowercased model response
- LLM judge (OpenAI) for responses containing neither 'a)' nor 'b)'
- Evaluates: forward_belief, forward_action, backward_belief, percept_to_belief
- Conditions: true_belief, false_belief, true_control, false_control
- Init beliefs: 0 (hide) and 1 (show) initial belief sentence in story
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from official_eval_common import load_policy_bundle

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_BIGTOM_CSV = (
    REPO_ROOT / "third_party" / "src" / "bigtom" / "data" / "bigtom" / "bigtom.csv"
)

SCENARIO_FIELDS = [
    "story", "aware_event", "not_aware_event",
    "action_new_state", "action_init_state",
    "belief_question", "desire_question", "action_question",
    "belief_aware", "desire_aware", "action_aware",
    "belief_not_aware", "desire_not_aware", "action_not_aware",
    "random_event", "aware_random", "not_aware_random",
    "source", "init_belief_idx",
]

VARIABLES = ["forward_belief", "forward_action", "backward_belief", "percept_to_belief"]
CONDITIONS = ["true_belief", "false_belief", "true_control", "false_control"]

EVAL_INSTRUCTION = (
    "Answer the questions based on the context. Keep your answer concise, "
    "few words are enough, maximum one sentence. "
    "Answer as 'Answer:<option>)<answer>'."
)

GRADE_INSTRUCTION = (
    "You are an expert comparing predicted answers with true answers. "
    "You are given a question, the predicted answer, and the true answer. "
    "You are asked to score the predicted answer as either True or False based on its similarity to the true answer. "
    "We only care about semantic similarity, ignore whitespace, typos, punctuation, etc. and focus only on the meaning of the answer. "
    "Mark as False if the answer says lack of information or None of the above. "
    "Mark as False if the answer has irrelevant information. "
    "If a single character is given, match it to the true answer or the false answer and mark as True or False accordingly."
)


def _split_story(story: str) -> List[str]:
    return story.split(".")


def _build_base_story(parts: List[str], init_belief: int) -> str:
    """Build base story for forward/backward variables (without awareness event)."""
    if init_belief == 0:
        return parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[4] + "."
    else:
        return parts[0] + "." + parts[1] + "." + parts[2] + "." + parts[3] + "." + parts[4] + "."


def _build_control_story(parts: List[str], random_event: str, init_belief: int) -> str:
    """Build control story (random distractor event replaces main change event)."""
    if init_belief == 0:
        return ".".join(parts[:3] + [" " + random_event])
    else:
        return ".".join(parts[:4] + [" " + random_event])


def generate_eval_rows(d: Dict, scenario_idx: int) -> List[Dict]:
    """Generate all evaluation rows for a single scenario, mirroring generate_conditions.py."""
    rows = []
    parts = _split_story(d["story"])
    if len(parts) < 5:
        return rows

    for init_belief in (0, 1):
        for variable in VARIABLES:

            if variable == "forward_belief":
                question = d["belief_question"]
                aware_answer = d["belief_aware"]
                not_aware_answer = d["belief_not_aware"]
            elif variable == "forward_action":
                question = d["action_question"]
                aware_answer = d["action_aware"]
                not_aware_answer = d["action_not_aware"]
            elif variable == "backward_belief":
                question = d["belief_question"]
                aware_answer = d["belief_aware"]
                not_aware_answer = d["belief_not_aware"]
            elif variable == "percept_to_belief":
                question = d["belief_question"]
                aware_answer = d["belief_aware"]
                not_aware_answer = d["belief_not_aware"]

            if variable == "percept_to_belief":
                # Only init_belief=1, true_belief condition (as per generate_conditions.py)
                if init_belief == 1:
                    story = parts[0] + "." + parts[1] + "." + parts[2] + "."
                    rows.append({
                        "scenario_idx": scenario_idx,
                        "variable": variable,
                        "condition": "true_belief",
                        "init_belief": init_belief,
                        "story": story,
                        "question": question,
                        "true_answer": aware_answer,
                        "wrong_answer": not_aware_answer,
                    })
                continue

            base_story = _build_base_story(parts, init_belief)
            control_story = _build_control_story(parts, d["random_event"], init_belief)

            for condition in CONDITIONS:
                if condition == "true_belief":
                    full_story = f"{base_story} {d['aware_event']}"
                    true_answer = aware_answer
                    wrong_answer = not_aware_answer
                elif condition == "false_belief":
                    full_story = f"{base_story} {d['not_aware_event']}"
                    true_answer = not_aware_answer
                    wrong_answer = aware_answer
                elif condition == "true_control":
                    full_story = f"{control_story} {d['aware_random']}"
                    true_answer = not_aware_answer
                    wrong_answer = aware_answer
                elif condition == "false_control":
                    full_story = f"{control_story} {d['not_aware_random']}"
                    true_answer = not_aware_answer
                    wrong_answer = aware_answer

                rows.append({
                    "scenario_idx": scenario_idx,
                    "variable": variable,
                    "condition": condition,
                    "init_belief": init_belief,
                    "story": full_story,
                    "question": question,
                    "true_answer": true_answer,
                    "wrong_answer": wrong_answer,
                })

    return rows


def load_bigtom_csv(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for sid, row in enumerate(reader):
            if len(row) < 19:
                continue
            d = {k: v for k, v in zip(SCENARIO_FIELDS, row[:19])}
            rows.extend(generate_eval_rows(d, sid))
    return rows


def build_mcq_prompt(story: str, mcq_question: str) -> str:
    """Build the prompt for the model (chat-0shot style from official evaluate.txt)."""
    return (
        f"{EVAL_INSTRUCTION}\n\n"
        f"Story: {story.strip()}\n"
        f"Question: {mcq_question.strip()}\n"
        f"Answer:"
    )


# Completion-based judge models (official BigToM uses text-davinci-003 via crfmLLM).
# text-davinci-003 is deprecated by OpenAI; gpt-3.5-turbo-instruct is its successor.
COMPLETION_JUDGE_MODELS = {
    "text-davinci-003",
    "text-davinci-002",
    "gpt-3.5-turbo-instruct",
    "davinci-002",
    "babbage-002",
}

JUDGE_STOP_TOKENS = ["Predicted Answer:", "True Answer:", "Response:"]


def _build_judge_prompt(
    question: str,
    predicted_answer: str,
    true_answer_labeled: str,
    wrong_answer_labeled: str,
) -> str:
    """Mirror EvaluateLLM 'eval' method prompt exactly."""
    return (
        f"{GRADE_INSTRUCTION}\n\n"
        f"Here is the question:\n{question}\n"
        f"Here is the true answer:\n{true_answer_labeled}\n"
        f"Here is the false answer:\n{wrong_answer_labeled}\n"
        f"Here is the predicted answer:\n{predicted_answer}\n"
        f"Is the predicted answer close to the true answer compared to the false answer? "
        f"Answer True or False.\n"
        f"A:"
    )


def grade_with_judge(
    client,
    judge_model: str,
    question: str,
    predicted_answer: str,
    true_answer_labeled: str,
    wrong_answer_labeled: str,
) -> str:
    """Call OpenAI judge to grade an ambiguous response.

    Uses Completions API for instruct-style models (matching BigToM's crfmLLM),
    and Chat Completions API for newer chat-only models (gpt-4o, gpt-4o-mini, etc.).
    """
    prompt = _build_judge_prompt(
        question, predicted_answer, true_answer_labeled, wrong_answer_labeled,
    )
    if judge_model in COMPLETION_JUDGE_MODELS:
        response = client.completions.create(
            model=judge_model,
            prompt=prompt,
            max_tokens=10,
            temperature=0,
            stop=JUDGE_STOP_TOKENS,
        )
        return response.choices[0].text.strip()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
        stop=JUDGE_STOP_TOKENS,
    )
    return response.choices[0].message.content.strip()


def grade_response(
    prediction: str,
    true_answer: str,
    wrong_answer: str,
    shuffled_answers: List[str],
    *,
    judge_client=None,
    judge_model: str = "gpt-4o-mini",
    question: str = "",
    verbose: bool = False,
) -> Tuple[float, str, bool]:
    """
    Returns (score, graded_str, used_judge).

    Grading logic mirrors evaluate_conditions.py:
    - shuffled_answers[0] is displayed as a), shuffled_answers[1] as b)
    - If true_answer is at position 0, answer_key='a)'; else answer_key='b)'
    - Check if answer_key in prediction.lower() → True
    - Check if negative_answer_key in prediction.lower() → False
    - Otherwise call LLM judge
    """
    if shuffled_answers[0] == true_answer:
        answer_key = "a)"
        negative_answer_key = "b)"
        true_answer_labeled = "a) " + true_answer
        wrong_answer_labeled = "b) " + wrong_answer
    else:
        answer_key = "b)"
        negative_answer_key = "a)"
        true_answer_labeled = "b) " + true_answer
        wrong_answer_labeled = "a) " + wrong_answer

    pred_lower = prediction.lower()
    if answer_key in pred_lower:
        return 1.0, "True", False
    elif negative_answer_key in pred_lower:
        return 0.0, "False", False
    else:
        if judge_client is None:
            if verbose:
                print(f"  [ambiguous, no judge] pred={prediction!r}")
            return 0.0, "False", False
        graded = grade_with_judge(
            judge_client, judge_model,
            question, prediction,
            true_answer_labeled, wrong_answer_labeled,
        )
        score = 1.0 if "true" in graded.lower() else 0.0
        return score, graded, True


def summarize_results(predictions: List[Dict]) -> Dict:
    overall = [r["score"] for r in predictions]
    by_variable: Dict[str, List[float]] = defaultdict(list)
    by_condition: Dict[str, List[float]] = defaultdict(list)
    by_variable_condition: Dict[str, List[float]] = defaultdict(list)
    by_init_belief: Dict[str, List[float]] = defaultdict(list)

    for r in predictions:
        by_variable[r["variable"]].append(r["score"])
        by_condition[r["condition"]].append(r["score"])
        by_variable_condition[f"{r['variable']}/{r['condition']}"].append(r["score"])
        by_init_belief[str(r["init_belief"])].append(r["score"])

    def _mean(values: List[float]) -> Dict:
        if not values:
            return {"value": 0.0, "n": 0}
        return {"value": sum(values) / len(values), "n": len(values)}

    return {
        "Accuracy": _mean(overall),
        "ByVariable": {k: _mean(v) for k, v in sorted(by_variable.items())},
        "ByCondition": {k: _mean(v) for k, v in sorted(by_condition.items())},
        "ByVariableCondition": {k: _mean(v) for k, v in sorted(by_variable_condition.items())},
        "ByInitBelief": {k: _mean(v) for k, v in sorted(by_init_belief.items())},
    }


def print_summary(summary: Dict):
    def pct(d):
        return f"{d['value']:.2%} (n={d['n']})"

    print("\n" + "=" * 56)
    print("  BigToM Official Protocol Evaluation Results")
    print("=" * 56)
    print(f"Overall Accuracy: {pct(summary['Accuracy'])}")

    print("\nBy Variable:")
    for k, v in summary["ByVariable"].items():
        print(f"  {k:30s} {pct(v)}")

    print("\nBy Condition:")
    for k, v in summary["ByCondition"].items():
        print(f"  {k:30s} {pct(v)}")

    print("\nBy Variable × Condition:")
    for k, v in summary["ByVariableCondition"].items():
        print(f"  {k:46s} {pct(v)}")

    print("\nBy Init Belief:")
    for k, v in summary["ByInitBelief"].items():
        label = "show_belief" if k == "1" else "hide_belief"
        print(f"  {label:30s} {pct(v)}")
    print("=" * 56 + "\n")


def _format_duration(seconds: float) -> str:
    s = max(int(seconds), 0)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser(description="BigToM official-protocol evaluation")

    # Data
    ap.add_argument("--bigtom_csv", default=str(DEFAULT_BIGTOM_CSV),
                    help="Path to bigtom.csv (official 200-scenario file)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of eval rows (for quick testing)")
    ap.add_argument("--variables", default="all",
                    help="Comma-separated list of variables to evaluate, or 'all'")
    ap.add_argument("--conditions", default="all",
                    help="Comma-separated list of conditions to evaluate, or 'all'")

    # Model
    ap.add_argument("--mode", default="base", choices=["base", "sft", "grpo"])
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stage1_ckpt", default=None)
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    # OpenAI judge
    ap.add_argument("--openai_api_key", default=None,
                    help="OpenAI API key for LLM judge (skip judge if not provided)")
    ap.add_argument("--openai_base_url", default=None,
                    help="Optional OpenAI-compatible base URL")
    ap.add_argument("--judge_model", default="gpt-3.5-turbo-instruct",
                    help=("OpenAI judge model. Official BigToM uses text-davinci-003 "
                          "(now deprecated); gpt-3.5-turbo-instruct is the drop-in "
                          "successor using the Completions API. Chat-only models "
                          "like gpt-4o-mini will use the Chat Completions API."))

    # Output
    ap.add_argument("--out_dir", default="eval_bigtom_official",
                    help="Output directory for predictions and summary JSON")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the official-protocol choice-order shuffle. "
                         "Generation remains greedy/deterministic (do_sample=False); "
                         "this only controls random.shuffle() ordering of answer choices.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Filter variables / conditions ---
    selected_vars = (
        VARIABLES if args.variables.strip().lower() == "all"
        else [v.strip() for v in args.variables.split(",")]
    )
    selected_conds = (
        CONDITIONS if args.conditions.strip().lower() == "all"
        else [c.strip() for c in args.conditions.split(",")]
    )

    # --- Load data ---
    csv_path = Path(args.bigtom_csv)
    print(f"Loading BigToM scenarios from {csv_path} …")
    all_rows = load_bigtom_csv(csv_path)
    all_rows = [r for r in all_rows if r["variable"] in selected_vars and r["condition"] in selected_conds]
    if args.limit:
        all_rows = all_rows[: args.limit]
    print(f"Evaluation rows: {len(all_rows)}")

    # --- Load model ---
    print(f"Loading model: mode={args.mode} base={args.base_model}")
    runner = load_policy_bundle(
        mode=args.mode,
        base_model=args.base_model,
        stage1_ckpt=args.stage1_ckpt,
        policy_ckpt=args.policy_ckpt,
        z_dim=args.z_dim,
    )
    print("Model loaded.")

    # --- Set up OpenAI judge ---
    judge_client = None
    if args.openai_api_key:
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": args.openai_api_key}
            if args.openai_base_url:
                client_kwargs["base_url"] = args.openai_base_url
            judge_client = OpenAI(**client_kwargs)
            print(f"OpenAI judge enabled: model={args.judge_model}")
        except ImportError:
            print("WARNING: openai package not installed; running without LLM judge.")
    else:
        print("No OpenAI API key provided; ambiguous responses will be scored as False.")

    # --- Official protocol: set random seed before the evaluation loop ---
    random.seed(args.seed)

    predictions = []
    judge_calls = 0
    start_time = time.time()

    for idx, row in enumerate(all_rows, start=1):
        # Shuffle answers (mirrors random.shuffle in evaluate_conditions.py)
        answers = [row["true_answer"], row["wrong_answer"]]
        random.shuffle(answers)

        mcq_question = (
            f"{row['question']}\n"
            f"Choose one of the following:\n"
            f"a){answers[0]}\n"
            f"b){answers[1]}"
        )
        prompt = build_mcq_prompt(row["story"], mcq_question)

        prediction = runner.generate(
            prompt,
            story=row["story"],
            question=row["question"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        score, graded, used_judge = grade_response(
            prediction,
            row["true_answer"],
            row["wrong_answer"],
            answers,
            judge_client=judge_client,
            judge_model=args.judge_model,
            question=row["question"],
            verbose=args.verbose,
        )
        if used_judge:
            judge_calls += 1

        if args.verbose:
            print(
                f"[{idx}] var={row['variable']} cond={row['condition']} "
                f"init_belief={row['init_belief']} "
                f"pred={prediction!r} graded={graded} score={score}"
            )

        record = {
            "scenario_idx": row["scenario_idx"],
            "variable": row["variable"],
            "condition": row["condition"],
            "init_belief": row["init_belief"],
            "story": row["story"],
            "question": row["question"],
            "true_answer": row["true_answer"],
            "wrong_answer": row["wrong_answer"],
            "shuffled_answers": answers,
            "prediction": prediction,
            "graded": graded,
            "score": score,
            "used_judge": used_judge,
        }
        predictions.append(record)

        if idx == 1 or idx == len(all_rows) or idx % args.log_every == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (len(all_rows) - idx) / rate if rate > 0 else None
            eta_str = _format_duration(eta) if eta is not None else "n/a"
            running_acc = sum(r["score"] for r in predictions) / len(predictions)
            print(
                f"[bigtom_official] {idx}/{len(all_rows)} "
                f"acc={running_acc:.2%} elapsed={_format_duration(elapsed)} eta={eta_str} "
                f"judge_calls={judge_calls}",
                flush=True,
            )

    summary = summarize_results(predictions)
    print_summary(summary)

    # Save outputs
    preds_path = out_dir / f"bigtom_official_{args.mode}_predictions.jsonl"
    summary_path = out_dir / f"bigtom_official_{args.mode}_summary.json"

    with open(preds_path, "w") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    full_output = {
        "mode": args.mode,
        "base_model": args.base_model,
        "bigtom_csv": str(csv_path),
        "judge_model": args.judge_model if judge_client else None,
        "total_judge_calls": judge_calls,
        "n_rows": len(predictions),
        "metrics": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    print(f"Predictions: {preds_path}")
    print(f"Summary:     {summary_path}")
    if judge_client:
        print(f"Total OpenAI judge calls: {judge_calls}")


if __name__ == "__main__":
    main()
