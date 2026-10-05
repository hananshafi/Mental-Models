import argparse
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


def load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path) as f:
        return json.load(f)


def compute_f1(ground_truth: str, model_response: str) -> float:
    ground_truth_tokens = ground_truth.split()
    model_response_tokens = model_response.split()
    common = Counter(ground_truth_tokens) & Counter(model_response_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(model_response_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


class TransformerSentenceEncoder:
    def __init__(self, model_name: str, *, local_files_only: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        last_hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].detach().cpu().numpy()


def flatten_fantom(split_path: Path, input_type: str) -> List[Dict]:
    data = load_json_or_jsonl(split_path)
    rows: List[Dict] = []
    rng = random.Random(99)

    for record in data:
        context = record["short_context"].strip() if input_type == "short" else record["full_context"].strip()
        set_id = record["set_id"]
        fact_q = record["factQA"]["question"]
        fact_a = record["factQA"]["correct_answer"]

        fact_row = dict(record["factQA"])
        fact_row["set_id"] = set_id
        fact_row["context"] = context
        fact_row["match_question"] = fact_row["question"]
        rows.append(fact_row)

        for belief in record["beliefQAs"]:
            belief_row = dict(belief)
            belief_row["set_id"] = set_id
            belief_row["context"] = context
            belief_row["match_question"] = belief_row["question"]
            rows.append(belief_row)

            mc_row = dict(belief)
            option_a = mc_row["wrong_answer"]
            option_b = mc_row["correct_answer"]
            answer_goes_last = rng.choice([True, False])
            if answer_goes_last:
                choices = [option_a, option_b]
                answer = 1
            else:
                choices = [option_b, option_a]
                answer = 0
            mc_row["set_id"] = set_id
            mc_row["context"] = context
            mc_row["question_type"] = mc_row["question_type"] + ":multiple-choice"
            mc_row["choices_list"] = choices
            mc_row["correct_answer"] = answer
            mc_row["match_question"] = belief["question"]
            rows.append(mc_row)

        answerability_list = dict(record["answerabilityQA_list"])
        answerability_list["fact_question"] = fact_q
        answerability_list["set_id"] = set_id
        answerability_list["context"] = context
        answerability_list["match_question"] = answerability_list["question"]
        if input_type == "full" and len(answerability_list["wrong_answer"]) > 0:
            answerability_list["missed_info_accessibility"] = "inaccessible"
        rows.append(answerability_list)

        if input_type == "full":
            missed_info_accessibility_for_full = record["answerabilityQAs_binary"][0]["missed_info_accessibility"]
            for qa in record["answerabilityQAs_binary"]:
                if qa["correct_answer"] != "yes":
                    missed_info_accessibility_for_full = "inaccessible"
        for qa in record["answerabilityQAs_binary"]:
            qa_row = dict(qa)
            qa_row["fact_question"] = fact_q
            qa_row["set_id"] = set_id
            qa_row["context"] = context
            qa_row["match_question"] = qa_row["question"]
            if input_type == "full":
                qa_row["missed_info_accessibility"] = missed_info_accessibility_for_full
            rows.append(qa_row)

        info_list = dict(record["infoAccessibilityQA_list"])
        info_list["fact_question"] = fact_q
        info_list["fact_answer"] = fact_a
        info_list["set_id"] = set_id
        info_list["context"] = context
        info_list["match_question"] = info_list["question"]
        if input_type == "full" and len(info_list["wrong_answer"]) > 0:
            info_list["missed_info_accessibility"] = "inaccessible"
        rows.append(info_list)

        if input_type == "full":
            missed_info_accessibility_for_full = record["infoAccessibilityQAs_binary"][0]["missed_info_accessibility"]
            for qa in record["infoAccessibilityQAs_binary"]:
                if qa["correct_answer"] != "yes":
                    missed_info_accessibility_for_full = "inaccessible"
        for qa in record["infoAccessibilityQAs_binary"]:
            qa_row = dict(qa)
            qa_row["fact_question"] = fact_q
            qa_row["fact_answer"] = fact_a
            qa_row["set_id"] = set_id
            qa_row["context"] = context
            qa_row["match_question"] = qa_row["question"]
            if input_type == "full":
                qa_row["missed_info_accessibility"] = missed_info_accessibility_for_full
            rows.append(qa_row)

    return rows


def map_binary_answer_to_int(model_response: str) -> int:
    model_answer = model_response.lower().strip("'").strip('"')
    if " yes," in model_answer or " yes " in model_answer or model_answer.startswith("yes") or " yes." in model_answer or " knows " in model_answer or model_answer.startswith("true"):
        return 1
    if " no," in model_answer or " no " in model_answer or model_answer.startswith("no") or " no." in model_answer or " does not know " in model_answer or " doesn't know " in model_answer or model_answer.startswith("false"):
        return 0
    return -1


def yesno_to_int(yesno_str: str) -> int:
    return {"yes": 1, "no": 0, "no:long": 0, "error": -1}[yesno_str]


def evaluate_mc_belief_q(qa: Dict, prediction_row: Dict) -> bool:
    # Compare by text: predicted_answer is the raw choice text from inference,
    # choices_list and correct_answer (int index) are in the bridge's own ordering.
    # Using choice_index from the prediction row is wrong because inference and bridge
    # use different randomization functions, so the same seed produces different
    # choice orderings — making inference-side indices incompatible with bridge-side indices.
    response = str(prediction_row.get("prediction", "")).lower()
    for idx, choice in enumerate(qa["choices_list"]):
        if str(choice).strip().lower() == response.strip():
            return idx == int(qa["correct_answer"])

    int_to_alphabet = {0: "a", 1: "b", 2: "c", 3: "d"}
    answer = int_to_alphabet[int(qa["correct_answer"])]
    return (
        response.startswith(f"({answer})")
        or response.startswith(f"{answer})")
        or response.startswith(f"{answer}.")
        or response.startswith(f"{answer}:")
        or response.startswith(f"{answer},")
        or f"({answer})" in response
        or answer == response
    )


def evaluate_list_q(qa: Dict, model_response: str):
    excluded_aware_character = any(character.lower() not in model_response.lower() for character in qa["correct_answer"])
    included_unaware_character = any(character.lower() in model_response.lower() for character in qa["wrong_answer"])
    return not (excluded_aware_character or included_unaware_character), excluded_aware_character, included_unaware_character


def evaluate_fact_q(qa: Dict, model_response: str) -> float:
    return compute_f1(str(qa["correct_answer"]).lower(), model_response.lower())


def build_prediction_lookup(predictions: List[Dict]) -> Dict[Tuple[str, str, str], Dict]:
    lookup = {}
    for row in predictions:
        key = (
            str(row.get("set_id", "")),
            str(row.get("question_type", "")),
            str(row.get("question", "")).strip(),
        )
        lookup[key] = row
    return lookup


def run_reports(df: pd.DataFrame, aggregation_target: str, input_type: str) -> Dict[str, Dict]:
    if input_type == "short":
        df = df.copy()
        df.drop(df[(df["question_type"].str.endswith(":binary")) & (df["correct_answer"] == "no:long")].index, inplace=True)

    df["conversation_id"] = df["set_id"].map(lambda x: x.split("-")[0])
    df["part_id"] = df["set_id"].map(lambda x: "-".join(x.split("-")[:2]))

    def score_and_analyze(target_scenario: str) -> Dict[str, object]:
        report = {"conversation_input_type": input_type}
        agg_col = aggregation_target + "_id"
        tom_df = df[df["question_type"].str.startswith("tom")].copy()
        target_df = tom_df[tom_df["missed_info_accessibility"] == target_scenario].copy()

        if target_scenario == "accessible":
            _target_df = tom_df[tom_df["missed_info_accessibility"] == target_scenario].copy()
            set_ids = _target_df["set_id"].unique()
            target_sets = []
            for set_id in set_ids:
                if tom_df[tom_df["set_id"] == set_id]["missed_info_accessibility"].eq(target_scenario).all():
                    target_sets.append(set_id)
        else:
            target_sets = target_df["set_id"].unique()

        report[target_scenario + ":set:ALL*"] = target_df[target_df["set_id"].isin(target_sets)].groupby(agg_col)["result"].all().mean()

        target_question_for_all = [
            "tom:belief:" + target_scenario + ":multiple-choice",
            "tom:answerability:list",
            "tom:answerability:binary",
            "tom:info_accessibility:list",
            "tom:info_accessibility:binary",
        ]
        report[target_scenario + ":set:ALL"] = target_df[
            target_df["question_type"].isin(target_question_for_all) & target_df["set_id"].isin(target_sets)
        ].groupby(agg_col)["result"].all().mean()

        report[target_scenario + ":belief:multiple-choice"] = target_df[
            target_df["question_type"].str.endswith(":multiple-choice")
        ]["result"].mean()
        report[target_scenario + ":belief:distance"] = target_df[
            target_df["question_type"] == "tom:belief:" + target_scenario
        ]["result"].mean()
        report[target_scenario + ":belief_true_word-f1"] = target_df[
            (target_df["question_type"] == "tom:belief:" + target_scenario) & (target_df["result"] == True)
        ]["word_overlap"].mean()

        report[target_scenario + ":answerability:set:ALL"] = target_df[
            target_df["question_type"].str.startswith("tom:answerability")
        ].groupby(agg_col)["result"].all().mean()
        report[target_scenario + ":answerability:list"] = target_df[
            target_df["question_type"] == "tom:answerability:list"
        ]["result"].mean()
        answerability_model_responses = target_df[
            target_df["question_type"] == "tom:answerability:binary"
        ]["binarized_model_answer"].to_list()
        answerability_references = target_df[
            target_df["question_type"] == "tom:answerability:binary"
        ]["correct_answer"].map(yesno_to_int).to_list()
        report[target_scenario + ":answerability:binary-f1"] = f1_score(
            answerability_references,
            answerability_model_responses,
            pos_label=0,
            average="weighted",
        )

        report[target_scenario + ":info_accessibility:set:ALL"] = target_df[
            target_df["question_type"].str.startswith("tom:info_accessibility")
        ].groupby(agg_col)["result"].all().mean()
        report[target_scenario + ":info_accessibility:list"] = target_df[
            target_df["question_type"] == "tom:info_accessibility:list"
        ]["result"].mean()
        accessibility_model_responses = target_df[
            target_df["question_type"] == "tom:info_accessibility:binary"
        ]["binarized_model_answer"].to_list()
        accessibility_references = target_df[
            target_df["question_type"] == "tom:info_accessibility:binary"
        ]["correct_answer"].map(yesno_to_int).to_list()
        report[target_scenario + ":info_accessibility:binary-f1"] = f1_score(
            accessibility_references,
            accessibility_model_responses,
            pos_label=0,
            average="weighted",
        )

        report["fact_word-f1"] = df[df["question_type"].str.startswith("fact")]["result"].mean()

        list_wrong = target_df[
            (target_df["question_type"] == "tom:answerability:list") & (target_df["result"] == False)
        ][["excluded_aware_character", "included_unaware_character"]].copy()
        list_wrong["both"] = list_wrong["excluded_aware_character"] & list_wrong["included_unaware_character"]
        list_wrong["reason"] = list_wrong.apply(
            lambda x: "did_both"
            if x["both"]
            else "excluded_aware_character"
            if x["excluded_aware_character"]
            else "included_unaware_character",
            axis=1,
        )
        report[target_scenario + ":tom:lists:wrong_reasons:freq"] = list_wrong["reason"].value_counts(normalize=False).to_dict()

        binary_wrong_reasons = target_df[
            (target_df["question_type"].str.endswith(":binary")) & (target_df["result"] == False)
        ]["binarized_model_answer"].value_counts(normalize=False).to_dict()
        if 0 in binary_wrong_reasons:
            binary_wrong_reasons["false_negative"] = binary_wrong_reasons.pop(0)
        if 1 in binary_wrong_reasons:
            binary_wrong_reasons["false_positive"] = binary_wrong_reasons.pop(1)
        if -1 in binary_wrong_reasons:
            binary_wrong_reasons["irrelevant_response"] = binary_wrong_reasons.pop(-1)
        report[target_scenario + ":tom:binary:wrong_reasons:freq"] = binary_wrong_reasons

        belief_df = tom_df[tom_df["question_type"] == ("tom:belief:" + target_scenario)].copy()
        belief_df["tom_order"] = belief_df["tom_type"].map(lambda x: x.split(":")[0] if isinstance(x, str) else str(x))
        tom_order_results = belief_df.groupby("tom_order")["result"].value_counts(normalize=True)
        for idx in tom_order_results.index:
            if idx[1] is True:
                report[target_scenario + ":" + idx[0]] = tom_order_results[idx]

        belief_results = belief_df.groupby("tom_type")["result"].value_counts(normalize=True)
        for idx in belief_results.index:
            if idx[1] is True:
                report[target_scenario + ":" + idx[0]] = belief_results[idx]

        binary_qas = target_df[target_df["question_type"].str.endswith(":binary")].copy()
        binary_qas["target_character"] = binary_qas["question"].map(lambda x: x.removeprefix("Does ").split(" know")[0].lower())
        belief_qas = target_df[target_df["question_type"].str.startswith("tom:belief")].copy()
        belief_qas["target_character"] = belief_qas["question"].map(lambda x: x.lower().split("does ")[1].split()[0].lower())
        answerability_list_qas = target_df[target_df["question_type"].str.endswith("answerability:list")].set_index(agg_col, drop=False)
        accessibility_list_qas = target_df[target_df["question_type"].str.endswith("info_accessibility:list")].set_index(agg_col, drop=False)

        binary_answerability_qas = binary_qas[binary_qas["question_type"].str.startswith("tom:answerability:")]
        tiled_answerability_list_qas = binary_answerability_qas[[agg_col, "target_character", "correct_answer"]].join(
            answerability_list_qas[["prediction", agg_col]], on=agg_col, how="inner", lsuffix="-binary"
        )
        tiled_answerability_list_qas["binarized_model_answer"] = tiled_answerability_list_qas.apply(
            lambda x: x["target_character"].lower()
            in (x["prediction"].lower() if isinstance(x["prediction"], str) else ""),
            axis=1,
        )
        tiled_answerability_list_qas["binarized_correct_answer"] = tiled_answerability_list_qas["correct_answer"].map(
            lambda x: True if x == "yes" else False
        )
        tiled_answerability_list_qas["result"] = tiled_answerability_list_qas.apply(
            lambda x: x["binarized_model_answer"] == x["binarized_correct_answer"], axis=1
        )

        binary_accessibility_qas = binary_qas[binary_qas["question_type"].str.startswith("tom:info_accessibility:")]
        tiled_accessibility_list_qas = binary_accessibility_qas[[agg_col, "target_character", "correct_answer"]].join(
            accessibility_list_qas[["prediction", agg_col]], on=agg_col, how="inner", lsuffix="-binary"
        )
        tiled_accessibility_list_qas["binarized_model_answer"] = tiled_accessibility_list_qas.apply(
            lambda x: x["target_character"].lower()
            in (x["prediction"].lower() if isinstance(x["prediction"], str) else ""),
            axis=1,
        )
        tiled_accessibility_list_qas["binarized_correct_answer"] = tiled_accessibility_list_qas["correct_answer"].map(
            lambda x: True if x == "yes" else False
        )
        tiled_accessibility_list_qas["result"] = tiled_accessibility_list_qas.apply(
            lambda x: x["binarized_model_answer"] == x["binarized_correct_answer"], axis=1
        )

        df_for_all_character_metric = pd.concat([
            binary_qas[["target_character", agg_col, "result"]],
            belief_qas[["target_character", agg_col, "result"]],
            tiled_answerability_list_qas[["target_character", agg_col, "result"]],
            tiled_accessibility_list_qas[["target_character", agg_col, "result"]],
        ])
        report[target_scenario + ":set:ALL_character"] = df_for_all_character_metric.groupby(
            [agg_col, "target_character"]
        )["result"].all().mean()

        df_for_character_consistency = pd.concat([
            binary_qas[["target_character", agg_col, "binarized_model_answer"]],
            tiled_answerability_list_qas[["target_character", agg_col, "binarized_model_answer"]],
            tiled_accessibility_list_qas[["target_character", agg_col, "binarized_model_answer"]],
        ])
        report[target_scenario + ":set:character_answer_consistency"] = df_for_character_consistency.groupby(
            [agg_col, "target_character"]
        )["binarized_model_answer"].nunique().eq(1).mean()

        normalized_report = {}
        for key, value in report.items():
            if isinstance(value, float):
                normalized_report[key] = round(value, 3) * 100
            else:
                normalized_report[key] = value
        return normalized_report

    return {
        "fantom": score_and_analyze("inaccessible"),
        "control_task": score_and_analyze("accessible"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--input_type", default="short", choices=["short", "full"])
    parser.add_argument("--aggregation_target", default="set", choices=["set", "part", "conversation"])
    parser.add_argument("--embedding_model", default="sentence-transformers/all-roberta-large-v1")
    parser.add_argument("--allow_model_download", action="store_true")
    args = parser.parse_args()

    try:
        encoder = TransformerSentenceEncoder(
            args.embedding_model,
            local_files_only=not args.allow_model_download,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FANToM embedding model '{args.embedding_model}'. "
            "Provide a cached local copy or rerun the bridge with --allow_model_download."
        ) from exc

    predictions = load_json_or_jsonl(Path(args.predictions_path))
    flattened = flatten_fantom(Path(args.split_path), args.input_type)
    lookup = build_prediction_lookup(predictions)

    @lru_cache(maxsize=None)
    def embed(text: str):
        return encoder.encode(text)

    evaluated_rows = []
    for qa in flattened:
        key = (
            str(qa.get("set_id", "")),
            str(qa.get("question_type", "")),
            str(qa.get("match_question", qa.get("question", ""))).strip(),
        )
        pred_row = lookup.get(key)
        if pred_row is None:
            continue

        prediction = str(pred_row.get("prediction", "")).strip()
        qa_eval = dict(qa)
        qa_eval["prediction"] = prediction

        if qa_eval["question_type"].startswith("tom:belief:") and qa_eval["question_type"].endswith(":multiple-choice"):
            qa_eval["result"] = evaluate_mc_belief_q(qa_eval, pred_row)
        elif qa_eval["question_type"].startswith("tom:belief:"):
            wrong_emb = embed(str(qa_eval["wrong_answer"]))
            correct_emb = embed(str(qa_eval["correct_answer"]))
            response_emb = embed(prediction)
            similarity_wrong = cosine_similarity(response_emb.reshape(1, -1), wrong_emb.reshape(1, -1))[0][0]
            similarity_correct = cosine_similarity(response_emb.reshape(1, -1), correct_emb.reshape(1, -1))[0][0]
            if similarity_wrong >= similarity_correct:
                qa_eval["result"] = False
                qa_eval["word_overlap"] = compute_f1(str(qa_eval["wrong_answer"]), prediction)
            else:
                qa_eval["result"] = True
                qa_eval["word_overlap"] = compute_f1(str(qa_eval["correct_answer"]), prediction)
        elif qa_eval["question_type"].endswith(":list"):
            result, excluded_aware_character, included_unaware_character = evaluate_list_q(qa_eval, prediction)
            qa_eval["result"] = result
            qa_eval["excluded_aware_character"] = excluded_aware_character
            qa_eval["included_unaware_character"] = included_unaware_character
        elif qa_eval["question_type"].endswith(":binary"):
            qa_eval["binarized_model_answer"] = map_binary_answer_to_int(prediction)
            qa_eval["result"] = yesno_to_int(qa_eval["correct_answer"]) == qa_eval["binarized_model_answer"]
        elif qa_eval["question_type"].startswith("fact"):
            qa_eval["result"] = evaluate_fact_q(qa_eval, prediction)
        else:
            raise NotImplementedError(f"Unsupported FANToM question type: {qa_eval['question_type']}")

        evaluated_rows.append(qa_eval)

    reports = run_reports(pd.DataFrame(evaluated_rows), args.aggregation_target, args.input_type)
    print(json.dumps(reports, ensure_ascii=False))


if __name__ == "__main__":
    main()
