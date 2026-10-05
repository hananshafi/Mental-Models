"""
ToMi-2 (balanced) loader.

Parses paired (.txt, .trace) splits from
third_party/src/tomi/tomi_balanced_story_types/
into per-question records.

Each .txt block:
  1 Aria entered the front_yard.
  2 Aiden entered the front_yard.
  ...
  6 Noah entered the playroom.
  7 Where will Aria look for the grapefruit?<TAB>blue_container<TAB>1

The corresponding .trace line has comma-separated event tags ending in
question_type and branch_label, e.g.
  enter_agent_0,...,first_order_0_tom,false_belief

We map question_type to:
  belief_order ∈ {0, 1, 2}   (memory/reality → 0)
  agent_idx    ∈ {0, 1, -1}
  requires_tom ∈ {tom, no_tom}

Branch comes from the trace line: true_belief or false_belief (story-level).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_TOMI_DIR = Path(
    "third_party/src/tomi/tomi_balanced_story_types"
)

_LINE_RE = re.compile(r"^(\d+)\s+(.*)$")
_QTYPES_ORDER = {
    "memory": 0,
    "reality": 0,
    "first_order_0_no_tom": 1, "first_order_0_tom": 1,
    "first_order_1_no_tom": 1, "first_order_1_tom": 1,
    "second_order_0_no_tom": 2, "second_order_0_tom": 2,
    "second_order_1_no_tom": 2, "second_order_1_tom": 2,
}


@dataclass
class TomiRecord:
    scenario_id: int
    story: str
    story_lines: tuple[str, ...]
    question: str
    gold_answer: str
    question_type: str
    belief_order: int
    agent_idx: int           # 0 or 1 for ToM questions, -1 for memory/reality
    requires_tom: str        # "tom" or "no_tom"
    branch: str              # "true_belief" or "false_belief"

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "story": self.story,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "question_type": self.question_type,
            "belief_order": self.belief_order,
            "agent_idx": self.agent_idx,
            "requires_tom": self.requires_tom,
            "branch": self.branch,
            # Keys below are aliases the BigToM pipeline expects, populated for
            # compatibility with extract_latents-style helpers.
            "condition": self.branch,
            "init_belief_idx": 0,
            "task": f"order_{self.belief_order}",
        }


def _parse_qtype(qtype: str) -> tuple[int, int, str]:
    if qtype in ("memory", "reality"):
        return _QTYPES_ORDER[qtype], -1, "no_tom"
    m = re.match(r"(first|second)_order_(\d+)_(no_tom|tom)", qtype)
    if not m:
        raise ValueError(f"unrecognised question_type: {qtype}")
    order = 1 if m.group(1) == "first" else 2
    agent = int(m.group(2))
    tom_flag = m.group(3)
    return order, agent, tom_flag


def _iter_blocks(txt_path: Path):
    """Yield (story_lines, question_line_text, gold_answer) blocks."""
    buf: list[tuple[int, str]] = []
    with txt_path.open() as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            m = _LINE_RE.match(raw)
            if not m:
                continue
            n = int(m.group(1))
            content = m.group(2)
            # New block starts when we see "1 " and already have content.
            if n == 1 and buf:
                yield _flush(buf)
                buf = []
            buf.append((n, content))
    if buf:
        yield _flush(buf)


def _flush(buf: list[tuple[int, str]]) -> tuple[tuple[str, ...], str, str]:
    story_lines: list[str] = []
    q_line: Optional[str] = None
    for _, content in buf:
        if "\t" in content:
            q_line = content
        else:
            story_lines.append(content)
    if q_line is None:
        raise ValueError(f"block has no question line: {buf!r}")
    parts = q_line.split("\t")
    question = parts[0].strip()
    gold = parts[1].strip() if len(parts) > 1 else ""
    return tuple(story_lines), question, gold


def load_tomi_records(
    split: str = "test",
    tomi_dir: Path = DEFAULT_TOMI_DIR,
    max_records: Optional[int] = None,
) -> list[TomiRecord]:
    txt_path = tomi_dir / f"fb_all_{split}.txt"
    trace_path = tomi_dir / f"fb_all_{split}.trace"
    if not txt_path.exists() or not trace_path.exists():
        raise FileNotFoundError(f"missing ToMi files for split={split}: {txt_path}, {trace_path}")

    trace_rows = [ln.strip().split(",") for ln in trace_path.open() if ln.strip()]

    blocks = list(_iter_blocks(txt_path))
    if len(blocks) != len(trace_rows):
        raise RuntimeError(
            f"block/trace count mismatch: {len(blocks)} blocks vs {len(trace_rows)} traces"
        )

    # Story-level scenario_id: every 6 consecutive questions share a story setup
    # in the balanced split (memory, fo_0, so_0, reality, fo_1, so_1). We
    # detect new scenarios by comparing the story_lines tuple.
    records: list[TomiRecord] = []
    last_story: Optional[tuple[str, ...]] = None
    sid = -1
    for (story_lines, question, gold), trace in zip(blocks, trace_rows):
        if story_lines != last_story:
            sid += 1
            last_story = story_lines
        if len(trace) < 2:
            continue
        qtype = trace[-2]
        branch = trace[-1]
        order, agent_idx, tom_flag = _parse_qtype(qtype)
        story_text = " ".join(story_lines)
        records.append(TomiRecord(
            scenario_id=sid,
            story=story_text,
            story_lines=story_lines,
            question=question,
            gold_answer=gold,
            question_type=qtype,
            belief_order=order,
            agent_idx=agent_idx,
            requires_tom=tom_flag,
            branch=branch,
        ))
        if max_records and len(records) >= max_records:
            break

    return records


def build_tomi_context_text(rec: dict) -> str:
    """Mirror of bigtom build_encoder_context for ToMi records."""
    return f"{rec['story']}\nQuestion: {rec['question'].strip()}".strip()
