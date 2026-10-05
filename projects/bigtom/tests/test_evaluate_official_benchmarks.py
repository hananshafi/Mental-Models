import json
import sys
from pathlib import Path

import pytest

import evaluate_official_benchmarks as eval_script


def _invoke_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    eval_script.main()


def test_dry_run_validates_all_benchmarks(
    monkeypatch,
    tmp_path,
    benchmarks_fixture_root,
    bigtom_fixture_csv,
):
    def _should_not_load_model(**kwargs):
        raise AssertionError("load_policy_bundle should not be called during --dry_run")

    monkeypatch.setattr(eval_script, "load_policy_bundle", _should_not_load_model)

    out_dir = tmp_path / "dry-run"
    _invoke_main(
        monkeypatch,
        [
            "evaluate_official_benchmarks.py",
            "--datasets",
            "all",
            "--out_dir",
            str(out_dir),
            "--benchmarks_root",
            str(benchmarks_fixture_root),
            "--bigtom_csv",
            str(bigtom_fixture_csv),
            "--dry_run",
        ],
    )

    payload = json.loads((out_dir / "dry_run_summary.json").read_text(encoding="utf-8"))
    for dataset_name in ("bigtom", "tomi", "fantom"):
        assert payload["datasets"][dataset_name]["split_exists"]
        assert payload["datasets"][dataset_name]["num_rows"] >= 1

    assert payload["datasets"]["fantom"]["scorer_exists"]


@pytest.mark.parametrize(
    ("dataset_name", "expected_scoring_source"),
    [
        ("bigtom", "local_adapter"),
        ("tomi", "official_protocol_bridge"),
        ("fantom", "official_scorer"),
    ],
)
def test_base_mode_smoke_writes_outputs(
    monkeypatch,
    tmp_path,
    benchmarks_fixture_root,
    bigtom_fixture_csv,
    dummy_runner,
    dataset_name,
    expected_scoring_source,
):
    monkeypatch.setattr(eval_script, "load_policy_bundle", lambda **kwargs: dummy_runner)

    out_dir = tmp_path / dataset_name
    _invoke_main(
        monkeypatch,
        [
            "evaluate_official_benchmarks.py",
            "--datasets",
            dataset_name,
            "--mode",
            "base",
            "--out_dir",
            str(out_dir),
            "--benchmarks_root",
            str(benchmarks_fixture_root),
            "--bigtom_csv",
            str(bigtom_fixture_csv),
            "--limit",
            "1",
        ],
    )

    combined_summary_path = out_dir / "combined_base_summary.json"
    markdown_path = out_dir / "combined_base_summary.md"
    dataset_summary_path = out_dir / f"{dataset_name}_base_summary.json"

    assert combined_summary_path.exists()
    assert markdown_path.exists()
    assert dataset_summary_path.exists()

    combined = json.loads(combined_summary_path.read_text(encoding="utf-8"))
    dataset_summary = combined["datasets"][dataset_name]

    assert dataset_summary["scoring"]["source"] == expected_scoring_source
    assert dataset_summary["paths"]["predictions_jsonl"]
    assert Path(dataset_summary["paths"]["predictions_jsonl"]).exists()
    assert Path(dataset_summary["paths"]["official_input_json"]).exists()
