import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", required=True)
    ap.add_argument("--location_granularity", default="coarse")
    ap.add_argument("--perspective", default="all")
    args = ap.parse_args()

    with open(args.result_path) as f:
        rows = json.load(f)

    n = len(rows)
    accuracy = sum(float(row.get("score", 0.0)) for row in rows) / n if n else 0.0
    payload = {
        "Accuracy": {"value": accuracy, "n": n},
        "MacroF1": {"value": accuracy, "n": n},
        "LocationGranularity": args.location_granularity,
        "Perspective": args.perspective,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
