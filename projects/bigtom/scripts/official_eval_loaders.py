import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from stage5_evaluate import load_bigtom_eval


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_BENCHMARKS_ROOT = REPO_ROOT / "third_party" / "src"
DEFAULT_BIGTOM_ROOT = DEFAULT_BENCHMARKS_ROOT / "bigtom"
DEFAULT_BIGTOM_CSV = DEFAULT_BIGTOM_ROOT / "data" / "bigtom" / "bigtom.csv"

TOMI_STORY_TYPES = {
    "true_belief",
    "false_belief",
    "second_order_true_belief",
    "second_order_false_belief",
}

TOMI_QUESTION_TYPES = {
    "memory",
    "reality",
    "first_order",
    "second_order",
}

FANTOM_CONTEXT_KEYS = {
    "short": [
        "conversation_short",
        "short_context",
        "short_input",
        "short_prompt",
        "short_conversation",
        "short_story",
        "story",
        "context",
    ],
    "full": [
        "conversation_full",
        "full_context",
        "full_input",
        "full_prompt",
        "full_conversation",
        "dialogue",
        "story",
        "context",
    ],
}


@dataclass
class BenchmarkPaths:
    dataset: str
    root: Path
    split_path: Optional[Path] = None
    trace_path: Optional[Path] = None
    scorer_path: Optional[Path] = None
    repo_root: Optional[Path] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def default_benchmarks_root() -> Path:
    return DEFAULT_BENCHMARKS_ROOT


def _as_path(value: str) -> Optional[Path]:
    if not value:
        return None
    return Path(value).expanduser()


def _first_existing(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for candidate in candidates:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples", "test", "items", "dataset"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Could not find a list payload in {path}")


def _parse_mc_choices(choices_value) -> List[str]:
    if not choices_value:
        return []
    if isinstance(choices_value, dict):
        ordered = []
        for key in sorted(choices_value):
            value = str(choices_value[key]).strip()
            if value:
                ordered.append(value)
        return ordered
    if isinstance(choices_value, list):
        return [str(choice).strip() for choice in choices_value if str(choice).strip()]

    choices_text = str(choices_value)
    matches = re.findall(r"[A-Z]\.\s*(.*?)(?=(?:,\s*[A-Z]\.\s*)|$)", choices_text)
    if matches:
        return [choice.strip() for choice in matches if choice.strip()]
    return [part.strip() for part in choices_text.split("|||") if part.strip()]


def _coerce_answer(answer_value, choices: List[str]) -> str:
    if answer_value is None:
        return ""
    if isinstance(answer_value, int) and 0 <= answer_value < len(choices):
        return choices[answer_value]
    if isinstance(answer_value, dict):
        for key in ("text", "answer", "label"):
            if key in answer_value:
                return _coerce_answer(answer_value[key], choices)
    text = str(answer_value).strip()
    if not text:
        return ""
    if len(text) == 1 and text.isalpha():
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return text


def _first_present(record: Dict, keys: Iterable[str]):
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalize_fantom_answer(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return str(value).strip()


def _infer_tomi_trace_fields(trace_line: str) -> Dict[str, str]:
    fields = [part.strip() for part in trace_line.split(",") if part.strip()]
    out: Dict[str, str] = {"trace": trace_line.strip()}

    for token in reversed(fields):
        if token in TOMI_STORY_TYPES:
            out["story_type"] = token
            break
    for token in reversed(fields):
        if token in TOMI_QUESTION_TYPES:
            out["question_type"] = token
            break
    if "question_type" not in out and fields:
        out["question_type"] = fields[-1]
    return out


def resolve_bigtom_paths(
    *,
    split_override: Optional[str] = None,
) -> BenchmarkPaths:
    split_path = _as_path(split_override) or DEFAULT_BIGTOM_CSV
    return BenchmarkPaths(
        dataset="bigtom",
        root=DEFAULT_BIGTOM_ROOT,
        split_path=split_path,
        repo_root=DEFAULT_BIGTOM_ROOT,
        metadata={"split_name": "official"},
    )


def resolve_opentom_paths(
    *,
    benchmarks_root: Optional[str] = None,
    root_override: Optional[str] = None,
    split_override: Optional[str] = None,
    scorer_override: Optional[str] = None,
) -> BenchmarkPaths:
    root = _as_path(root_override) or (Path(benchmarks_root or default_benchmarks_root()) / "opentom")
    split_path = _as_path(split_override) or _first_existing(root, [
        "data/opentom.json",
        "data/test.json",
        "data/test.jsonl",
        "data/opentom_test.json",
        "data/OpenToM_test.json",
        "OpenToM_test.json",
        "test.json",
        "test.jsonl",
    ])
    scorer_path = _as_path(scorer_override) or _first_existing(root, [
        "src/evaluate.py",
        "evaluate.py",
        "code/evaluate.py",
        "scripts/evaluate.py",
    ])
    return BenchmarkPaths(
        dataset="opentom",
        root=root,
        split_path=split_path,
        scorer_path=scorer_path,
        repo_root=root,
        metadata={"split_name": "canonical", "uses_official_scorer": True},
    )


def resolve_tomi_paths(
    *,
    benchmarks_root: Optional[str] = None,
    root_override: Optional[str] = None,
    split_override: Optional[str] = None,
    trace_override: Optional[str] = None,
) -> BenchmarkPaths:
    root = _as_path(root_override) or (Path(benchmarks_root or default_benchmarks_root()) / "tomi")
    split_path = _as_path(split_override) or _first_existing(root, [
        "tomi_balanced_story_types/fb_all_test.txt",
        "fb_all_test.txt",
        "test.txt",
        "data/test.txt",
        "tomi/test.txt",
        "tomi/data/test.txt",
        "test/test.txt",
    ])
    trace_path = _as_path(trace_override) or _first_existing(root, [
        "tomi_balanced_story_types/fb_all_test.trace",
        "fb_all_test.trace",
        "trace.txt",
        "data/trace.txt",
        "test_trace.txt",
        "data/test_trace.txt",
        "tomi/trace.txt",
    ])
    scorer_path = _first_existing(root, [
        "main.py",
    ])
    return BenchmarkPaths(
        dataset="tomi",
        root=root,
        split_path=split_path,
        trace_path=trace_path,
        scorer_path=scorer_path,
        repo_root=root,
        metadata={
            "split_name": "canonical",
            "uses_official_scorer": False,
            "uses_official_protocol_bridge": bool(scorer_path),
        },
    )


def resolve_hitom_paths(
    *,
    benchmarks_root: Optional[str] = None,
    root_override: Optional[str] = None,
    split_override: Optional[str] = None,
) -> BenchmarkPaths:
    root = _as_path(root_override) or (Path(benchmarks_root or default_benchmarks_root()) / "hitom")
    split_path = _as_path(split_override) or _first_existing(root, [
        "Hi-ToM_data/Hi-ToM_data.json",
        "Hi-ToM_data.json",
        "data/Hi-ToM_data.json",
        "data/test.json",
        "data/test.jsonl",
        "test.json",
        "test.jsonl",
    ])
    return BenchmarkPaths(
        dataset="hitom",
        root=root,
        split_path=split_path,
        repo_root=root,
        metadata={"split_name": "canonical", "uses_official_scorer": False},
    )


def resolve_fantom_paths(
    *,
    benchmarks_root: Optional[str] = None,
    root_override: Optional[str] = None,
    split_override: Optional[str] = None,
    scorer_override: Optional[str] = None,
    input_type: str = "short",
) -> BenchmarkPaths:
    root = _as_path(root_override) or (Path(benchmarks_root or default_benchmarks_root()) / "fantom")
    if input_type == "short":
        split_candidates = [
            "data/fantom/fantom_v1.json",
            "data/fantom_short.json",
            "data/fantom_short.jsonl",
            "data/test_short.json",
            "FANToM_short.json",
            "fantom_short.json",
            "test_short.json",
        ]
    else:
        split_candidates = [
            "data/fantom/fantom_v1.json",
            "data/fantom_full.json",
            "data/fantom_full.jsonl",
            "data/test_full.json",
            "FANToM_full.json",
            "fantom_full.json",
            "test_full.json",
        ]
    split_path = _as_path(split_override) or _first_existing(root, split_candidates)
    scorer_path = _as_path(scorer_override) or _first_existing(root, [
        "eval_fantom.py",
        "scripts/eval_fantom.py",
        "evaluation/eval_fantom.py",
    ])
    return BenchmarkPaths(
        dataset="fantom",
        root=root,
        split_path=split_path,
        scorer_path=scorer_path,
        repo_root=root,
        metadata={
            "split_name": "canonical",
            "input_type": input_type,
            "uses_official_scorer": True,
        },
    )


def validate_benchmark_paths(paths: BenchmarkPaths) -> Dict[str, object]:
    return {
        "dataset": paths.dataset,
        "root": str(paths.root),
        "root_exists": paths.root.exists(),
        "split_path": str(paths.split_path) if paths.split_path else None,
        "split_exists": paths.split_path.exists() if paths.split_path else False,
        "trace_path": str(paths.trace_path) if paths.trace_path else None,
        "trace_exists": paths.trace_path.exists() if paths.trace_path else False,
        "scorer_path": str(paths.scorer_path) if paths.scorer_path else None,
        "scorer_exists": paths.scorer_path.exists() if paths.scorer_path else False,
        "metadata": dict(paths.metadata),
    }


def load_bigtom_official(csv_path: Path) -> List[Dict]:
    return load_bigtom_eval(csv_path)


def load_opentom_official(path: Path) -> List[Dict]:
    records = _load_json_or_jsonl(path)
    rows = []
    running_index = 0

    def emit_row(base_record: Dict, qa_record: Dict, local_idx: int):
        nonlocal running_index
        story = (
            base_record.get("narrative")
            or base_record.get("plot")
            or base_record.get("story")
            or base_record.get("context")
            or qa_record.get("story")
            or qa_record.get("narrative")
            or qa_record.get("context")
        )
        question = qa_record.get("question") or qa_record.get("query")
        if not story or not question:
            return
        if isinstance(question, dict):
            qa_record = question
            question = qa_record.get("question") or qa_record.get("query")
            if not question:
                return
        choices = _parse_mc_choices(
            qa_record.get("choices")
            or qa_record.get("options")
            or qa_record.get("candidates")
            or qa_record.get("answer_choices")
        )
        answer = _coerce_answer(
            qa_record.get("answer", qa_record.get("gold_answer", qa_record.get("label"))),
            choices,
        )
        question_type = (
            qa_record.get("question_type")
            or qa_record.get("type")
            or qa_record.get("category")
            or base_record.get("question_type")
        )
        plot_info = base_record.get("plot_info", {})
        mover = plot_info.get("mover")
        observer = plot_info.get("observer")
        perspective = qa_record.get("perspective", base_record.get("perspective"))
        if not perspective and mover and observer:
            if f"As {observer}" in question:
                perspective = "observer"
            elif f"As {mover}" in question:
                perspective = "mover"

        tom_order = qa_record.get("tom_order", base_record.get("tom_order"))
        if not tom_order and isinstance(question_type, str):
            if question_type.endswith("-fo"):
                tom_order = "first_order"
            elif question_type.endswith("-so"):
                tom_order = "second_order"

        location_granularity = qa_record.get(
            "location_granularity", base_record.get("location_granularity")
        )
        if not location_granularity and isinstance(question_type, str) and question_type.startswith("location"):
            location_granularity = "fine" if "fine" in str(answer).strip().lower() else "coarse"

        gold_choice_index = None
        if answer and choices:
            for idx, choice in enumerate(choices):
                if answer.strip().lower() == choice.strip().lower():
                    gold_choice_index = idx
                    break

        rows.append({
            "dataset": "opentom",
            "sample_id": qa_record.get("sample_id", base_record.get("sample_id", running_index)),
            "story_id": base_record.get("story_id", base_record.get("id", running_index)),
            "question_id": qa_record.get("question_id", local_idx),
            "story": story.strip(),
            "question": question.strip(),
            "answer": answer,
            "choices": choices,
            "gold_choice_index": gold_choice_index,
            "question_type": question_type,
            "perspective": perspective,
            "tom_order": tom_order,
            "narrative_type": qa_record.get("narrative_type", base_record.get("narrative_type")),
            "location_granularity": location_granularity,
        })
        running_index += 1

    for record in records:
        question_payload = record.get("question")
        if isinstance(question_payload, dict):
            emit_row(record, question_payload, 0)
            continue
        questions = record.get("questions")
        if isinstance(questions, list):
            for local_idx, qa_record in enumerate(questions):
                if isinstance(qa_record, dict):
                    emit_row(record, qa_record, local_idx)
            continue
        if isinstance(record, dict):
            emit_row(record, record, 0)

    return rows


def load_tomi_official(test_path: Path, trace_path: Optional[Path] = None) -> List[Dict]:
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


def load_hitom_official(path: Path) -> List[Dict]:
    records = _load_json_or_jsonl(path)
    rows = []
    for idx, record in enumerate(records):
        row = {
            "dataset": "hitom",
            "sample_id": record.get("sample_id", idx),
            "story": record["story"],
            "question": record["question"],
            "prompt": str(record.get("prompt", "")).strip(),
            "answer": str(record["answer"]).strip(),
            "choices": _parse_mc_choices(record.get("choices", "")),
            "question_order": record.get("question_order"),
            "prompting_type": record.get("prompting_type"),
            "deception": record.get("deception"),
            "story_length": record.get("story_length"),
        }
        rows.append(row)
    return rows


def load_fantom_official(path: Path, input_type: str = "short") -> List[Dict]:
    records = _load_json_or_jsonl(path)
    rows = []
    keys = FANTOM_CONTEXT_KEYS[input_type]
    mc_rng = random.Random(99)

    def append_row(*, base_record: Dict, qa_record: Dict, local_id: str):
        context = _first_present(base_record, keys)
        if not context:
            return
        question = qa_record.get("question") or qa_record.get("probe_question") or qa_record.get("qa_question")
        if not question:
            return
        rows.append({
            "dataset": "fantom",
            "sample_id": qa_record.get("sample_id", f"{base_record.get('set_id', base_record.get('part_id', 'fantom'))}:{local_id}"),
            "set_id": base_record.get("set_id", base_record.get("part_id", base_record.get("dialogue_id", local_id))),
            "story": context,
            "question": question,
            "answer": _normalize_fantom_answer(
                qa_record.get("correct_answer", qa_record.get("answer", qa_record.get("gold_answer", "")))
            ),
            "wrong_answer": _normalize_fantom_answer(qa_record.get("wrong_answer", "")),
            "question_type": qa_record.get("question_type", qa_record.get("type", "unknown")),
            "tom_type": qa_record.get(
                "tom_type",
                qa_record.get("missed_info_accessibility", base_record.get("missed_info_accessibility")),
            ),
            "choices": _parse_mc_choices(qa_record.get("choices", [])),
        })

    def append_mc_belief_row(*, base_record: Dict, qa_record: Dict, local_id: str):
        context = _first_present(base_record, keys)
        if not context:
            return
        question = qa_record.get("question")
        if not question:
            return
        answer_goes_last = mc_rng.choice([True, False])
        wrong_answer = str(qa_record.get("wrong_answer", "")).strip()
        correct_answer = str(qa_record.get("correct_answer", "")).strip()
        if answer_goes_last:
            choices = [wrong_answer, correct_answer]
            gold_choice_index = 1
        else:
            choices = [correct_answer, wrong_answer]
            gold_choice_index = 0
        rows.append({
            "dataset": "fantom",
            "sample_id": f"{base_record.get('set_id', base_record.get('part_id', 'fantom'))}:{local_id}:mc",
            "set_id": base_record.get("set_id", base_record.get("part_id", base_record.get("dialogue_id", local_id))),
            "story": context,
            "question": question,
            "answer": choices[gold_choice_index],
            "wrong_answer": choices[1 - gold_choice_index] if len(choices) > 1 else "",
            "question_type": f"{qa_record.get('question_type', qa_record.get('type', 'unknown'))}:multiple-choice",
            "tom_type": qa_record.get(
                "tom_type",
                qa_record.get("missed_info_accessibility", base_record.get("missed_info_accessibility")),
            ),
            "choices": choices,
            "gold_choice_index": gold_choice_index,
        })

    for idx, record in enumerate(records):
        if "factQA" in record or "beliefQAs" in record:
            if isinstance(record.get("factQA"), dict):
                append_row(base_record=record, qa_record=record["factQA"], local_id=f"{idx}:fact")
            for sub_idx, qa_record in enumerate(record.get("beliefQAs", [])):
                if isinstance(qa_record, dict):
                    append_row(base_record=record, qa_record=qa_record, local_id=f"{idx}:belief:{sub_idx}")
                    append_mc_belief_row(
                        base_record=record,
                        qa_record=qa_record,
                        local_id=f"{idx}:belief:{sub_idx}",
                    )
            for field_name in ("infoAccessibilityQA_list", "answerabilityQA_list"):
                qa_record = record.get(field_name)
                if isinstance(qa_record, dict):
                    append_row(base_record=record, qa_record=qa_record, local_id=f"{idx}:{field_name}")
            for field_name in ("infoAccessibilityQAs_binary", "answerabilityQAs_binary"):
                for sub_idx, qa_record in enumerate(record.get(field_name, [])):
                    if isinstance(qa_record, dict):
                        append_row(base_record=record, qa_record=qa_record, local_id=f"{idx}:{field_name}:{sub_idx}")
            continue

        append_row(base_record=record, qa_record=record, local_id=str(idx))
    return rows
