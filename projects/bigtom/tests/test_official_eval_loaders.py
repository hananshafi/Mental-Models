from official_eval_loaders import (
    load_fantom_official,
    load_hitom_official,
    load_opentom_official,
    load_tomi_official,
    resolve_fantom_paths,
    resolve_hitom_paths,
    resolve_opentom_paths,
    resolve_tomi_paths,
    validate_benchmark_paths,
)


def test_benchmark_path_resolution_smoke(benchmarks_fixture_root):
    opentom_paths = resolve_opentom_paths(benchmarks_root=str(benchmarks_fixture_root))
    tomi_paths = resolve_tomi_paths(benchmarks_root=str(benchmarks_fixture_root))
    hitom_paths = resolve_hitom_paths(benchmarks_root=str(benchmarks_fixture_root))
    fantom_paths = resolve_fantom_paths(benchmarks_root=str(benchmarks_fixture_root))

    assert validate_benchmark_paths(opentom_paths)["split_exists"]
    assert validate_benchmark_paths(opentom_paths)["scorer_exists"]
    assert validate_benchmark_paths(tomi_paths)["split_exists"]
    assert validate_benchmark_paths(tomi_paths)["trace_exists"]
    assert validate_benchmark_paths(hitom_paths)["split_exists"]
    assert validate_benchmark_paths(fantom_paths)["split_exists"]
    assert validate_benchmark_paths(fantom_paths)["scorer_exists"]


def test_load_opentom_official_parses_nested_questions(benchmarks_fixture_root):
    rows = load_opentom_official(benchmarks_fixture_root / "opentom" / "data" / "test.json")

    assert len(rows) == 2
    assert rows[0]["story_id"] == "story-1"
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["choices"] == ["red box", "blue box"]
    assert rows[0]["gold_choice_index"] == 0
    assert rows[0]["perspective"] == "third_person"
    assert rows[1]["answer"] == "the key is in the drawer"


def test_load_tomi_official_aligns_trace_metadata(benchmarks_fixture_root):
    rows = load_tomi_official(
        benchmarks_fixture_root / "tomi" / "test.txt",
        benchmarks_fixture_root / "tomi" / "trace.txt",
    )

    assert len(rows) == 2
    assert rows[0]["answer"] == "pantry"
    assert rows[0]["story_type"] == "false_belief"
    assert rows[0]["question_type"] == "memory"
    assert rows[1]["story_type"] == "true_belief"
    assert rows[1]["question_type"] == "first_order"


def test_load_hitom_official_parses_choices(benchmarks_fixture_root):
    rows = load_hitom_official(benchmarks_fixture_root / "hitom" / "Hi-ToM_data.json")

    assert len(rows) == 1
    assert rows[0]["choices"] == ["locker", "desk"]
    assert rows[0]["question_order"] == 1
    assert rows[0]["story_length"] == "short"


def test_load_fantom_official_short_variant(benchmarks_fixture_root):
    rows = load_fantom_official(
        benchmarks_fixture_root / "fantom" / "data" / "fantom_short.json",
        input_type="short",
    )

    assert len(rows) == 3
    assert rows[0]["story"].startswith("Alex told Sam")
    assert rows[0]["answer"] == "yes"
    assert rows[1]["question_type"] == "Fact Question"
    assert rows[2]["answer"] == ["Maya", "Priya"]
    assert rows[2]["tom_type"] == "second_order"
