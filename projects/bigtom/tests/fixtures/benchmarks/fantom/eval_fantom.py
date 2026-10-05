import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_path", required=True)
    ap.add_argument("--split_path", required=True)
    ap.add_argument("--input_type", default="short")
    args = ap.parse_args()

    with open(args.predictions_path) as f:
        rows = json.load(f)

    n = len(rows)
    all_score = sum(float(row.get("score", 0.0)) for row in rows) / n if n else 0.0
    star_rows = [
        row for row in rows
        if any(
            token in str(row.get("question_type", "")).lower()
            for token in ("belief", "answerability", "info access")
        )
    ]
    star_n = len(star_rows)
    star_score = (
        sum(float(row.get("score", 0.0)) for row in star_rows) / star_n
        if star_n
        else 0.0
    )
    payload = {
        "All": {"value": all_score, "n": n},
        "All*": {"value": star_score, "n": star_n},
        "InputType": args.input_type,
        "SplitPath": args.split_path,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
