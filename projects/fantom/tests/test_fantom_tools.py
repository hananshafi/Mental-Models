import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compatibility = load_script("check_compatibility")
evaluator = load_script("evaluate_fantom")
partial = load_script("score_partial")


class CompatibilityTests(unittest.TestCase):
    def test_official_split_shape(self):
        config = json.loads(
            (PROJECT_ROOT / "configs" / "models.json").read_text(encoding="utf-8")
        )
        split_path = Path(config["dataset"]["split_path"])
        if not split_path.is_file():
            self.skipTest("Run tools/bootstrap_third_party.sh fantom first.")
        result = compatibility.inspect_dataset(config["dataset"])
        self.assertTrue(result["compatible"])
        self.assertEqual(result["conversations"], 870)
        self.assertEqual(result["flattened_questions"], 12832)
        self.assertEqual(result["min_speakers"], 3)
        self.assertEqual(result["max_speakers"], 6)

    def test_primary_checkpoint_assets(self):
        if os.environ.get("MENTAL_MODELS_RUN_ASSET_TESTS") != "1":
            self.skipTest("Set MENTAL_MODELS_RUN_ASSET_TESTS=1 for checkpoint tests.")
        config = json.loads(
            (PROJECT_ROOT / "configs" / "models.json").read_text(encoding="utf-8")
        )
        for model_id in (
            "bigtom_sft_epoch1",
            "bigtom_grpo_step300",
            "sotopia_qwen_grpo_best",
        ):
            result = compatibility.inspect_model(
                model_id,
                config["models"][model_id],
            )
            self.assertTrue(result["compatible"], result["errors"])


class ScoringTests(unittest.TestCase):
    def test_multiple_choice_scoring(self):
        row = {
            "answer": "blue",
            "wrong_answer": "red",
            "choices": ["red", "blue"],
            "gold_choice_index": 1,
            "question_type": "tom:belief:inaccessible:multiple-choice",
        }
        result = evaluator.score_prediction(row, "blue", 1)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["exact"])

    def test_list_scoring_rejects_wrong_character(self):
        row = {
            "answer": ["Alice", "Bob"],
            "wrong_answer": ["Carol"],
            "choices": [],
            "question_type": "tom:answerability:list",
        }
        result = evaluator.score_prediction(
            row,
            "Alice, Bob, and Carol",
            None,
        )
        self.assertEqual(result["score"], 1.0)
        self.assertFalse(result["exact"])

    def test_binary_scoring(self):
        row = {
            "answer": "yes",
            "wrong_answer": "no",
            "choices": [],
            "question_type": "tom:answerability:binary",
        }
        result = evaluator.score_prediction(row, "Yes.", None)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["exact"])

    def test_nonbinary_knowledge_answer_is_not_yes_no(self):
        row = {
            "answer": "Gianna will not know about Snowflake.",
            "wrong_answer": "Gianna knows about Snowflake.",
            "choices": [],
            "question_type": "tom:belief:inaccessible",
        }
        result = evaluator.score_prediction(
            row,
            "Gianna knows about Snowflake.",
            None,
        )
        self.assertLess(result["score"], 1.0)
        self.assertFalse(result["exact"])


class PartialScoringTests(unittest.TestCase):
    def test_jsonl_prefix_is_immutable_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"sample_id": str(index)}) + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            rows = partial.read_jsonl_prefix(path, 3)
            self.assertEqual(
                [row["sample_id"] for row in rows],
                ["0", "1", "2"],
            )
            self.assertEqual(partial.count_jsonl_rows(path, stop_at=3), 3)


if __name__ == "__main__":
    unittest.main()
