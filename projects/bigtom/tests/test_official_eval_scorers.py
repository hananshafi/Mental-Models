import json

from official_eval_scorers import (
    run_official_fantom_scorer,
    run_official_tomi_scorer,
    summarize_tomi_metrics,
)


def test_run_official_fantom_scorer_parses_metrics(tmp_path, benchmarks_fixture_root):
    predictions_path = tmp_path / "fantom_predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"score": 1.0, "question_type": "Belief Question"},
                {"score": 0.5, "question_type": "Fact Question"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    result = run_official_fantom_scorer(
        scorer_path=benchmarks_fixture_root / "fantom" / "eval_fantom.py",
        predictions_path=predictions_path,
        split_path=benchmarks_fixture_root / "fantom" / "data" / "fantom_short.json",
        input_type="short",
    )

    assert result is not None
    assert result["returncode"] == 0
    assert result["metrics"]["All"]["value"] == 0.75
    assert result["metrics"]["All*"]["n"] == 1


def test_run_official_tomi_scorer_parses_metrics(tmp_path, benchmarks_fixture_root):
    predictions_path = tmp_path / "tomi_predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": 0,
                    "prediction": "pantry",
                },
                {
                    "sample_id": 1,
                    "prediction": "table",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    result = run_official_tomi_scorer(
        scorer_path=benchmarks_fixture_root / "tomi" / "main.py",
        predictions_path=predictions_path,
        split_path=benchmarks_fixture_root / "tomi" / "test.txt",
        trace_path=benchmarks_fixture_root / "tomi" / "trace.txt",
    )

    assert result is not None
    assert result["returncode"] == 0
    assert result["metrics"]["Accuracy"]["value"] == 0.5
    assert result["metrics"]["Accuracy"]["n"] == 2
    assert result["metrics"]["Protocol"] == "released_tomi_exact_match_bridge"


def test_summarize_tomi_metrics_matches_expected():
    summary = summarize_tomi_metrics(
        [
            {"score": 1.0, "question_type": "memory", "story_type": "false_belief"},
            {"score": 0.0, "question_type": "first_order", "story_type": "true_belief"},
        ]
    )

    assert summary["Accuracy"] == {"value": 0.5, "n": 2}
    assert summary["ByQuestionType"]["memory"] == {"value": 1.0, "n": 1}
    assert summary["ByStoryType"]["true_belief"] == {"value": 0.0, "n": 1}
