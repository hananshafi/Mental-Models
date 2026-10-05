import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class DummyRunner:
    def __init__(self, mode: str = "base"):
        self.mode = mode

    def _answer_for(self, question: str, choices=None) -> str:
        question_n = question.strip().lower()
        mapping = {
            "where will noah think the marble is?": "red box",
            "what does ava know?": "the key is in the drawer",
            "where is the milk?": "pantry",
            "where will emma look first?": "locker",
            "does sam know when the cafe opens?": "yes",
            "where are the tickets?": "kitchen table",
            "who can answer where the package is?": "Maya and Priya",
            "where will liam think the toy is?": "box",
            "where will liam look for the toy?": "box",
        }
        if question_n in mapping:
            return mapping[question_n]
        if choices:
            return str(choices[0])
        return "yes"

    def generate(
        self,
        prompt: str,
        *,
        story: str,
        question: str,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> str:
        del prompt, story, max_new_tokens, kwargs
        return self._answer_for(question)

    def score_choices(
        self,
        prompt: str,
        choices,
        *,
        story: str,
        question: str,
    ):
        del prompt, story
        target = self._answer_for(question, choices)
        scores = []
        for idx, choice in enumerate(choices):
            score = float(len(choices) - idx)
            if str(choice).strip().lower() == target.strip().lower():
                score = 100.0
            scores.append(score)
        return scores

    def pick_choice(
        self,
        prompt: str,
        choices,
        *,
        story: str,
        question: str,
    ):
        scores = self.score_choices(prompt, choices, story=story, question=question)
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        return {
            "choice_index": best_idx,
            "choice_text": choices[best_idx],
            "scores": scores,
        }


@pytest.fixture
def benchmarks_fixture_root() -> Path:
    return PROJECT_ROOT / "tests" / "fixtures" / "benchmarks"


@pytest.fixture
def bigtom_fixture_csv() -> Path:
    return PROJECT_ROOT / "tests" / "fixtures" / "bigtom_fixture.csv"


@pytest.fixture
def dummy_runner() -> DummyRunner:
    return DummyRunner(mode="base")
