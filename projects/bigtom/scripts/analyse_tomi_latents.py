"""
Phase-1 pilot: zero-shot transfer of BigToM-trained recursive ToM latents
to ToMi-2 (balanced) higher-order belief questions.

Question this script answers: does z2 separate true/false belief items
*at order 2* better than z1 does, on a benchmark the model never saw
during stage-1 training?

Pipeline:
  1. Load Stage-1 RecursiveToMModel from ckpt.
  2. Load ToMi-2 records (default: test split), encode z1/z2/context_hidden
     for each (story, question) record.
  3. Run linear probes for these targets:
       - branch                 (true_belief vs false_belief, all orders)
       - belief_order           ({0, 1, 2}, multiclass)
       - requires_tom           (tom vs no_tom)
       - branch_order_1         (slice: order==1, true vs false)
       - branch_order_2         (slice: order==2, true vs false)
       - branch_order_2_tom     (slice: order==2 AND requires_tom==tom)
     across features {z1, z2, z_concat, context_hidden, tfidf_text}, with
     scenario-grouped 5-fold CV and shuffled-label permutation null.
  4. Save table1_linear_probe.{csv,md} + permutations + arrays + records.

No traversal in Phase 1 — that's Phase 2 if z2 cleanly beats z1 here.

Usage:
  python scripts/analyse_tomi_latents.py \
      --ckpt projects/bigtom/checkpoints/stage1_qwen_5k/step_300 \
      --output_dir projects/bigtom/tomi2_latent_analysis_step300 \
      --max_records 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyse_tom_latents import (  # noqa: E402
    extract_latents, probe_dense, probe_tfidf,
)
from probe_mental_decoder_qwen5k import load_model  # noqa: E402
from tomi_loader import (  # noqa: E402
    build_tomi_context_text, load_tomi_records,
)


# ── ToMi-specific probe target builder ───────────────────────────────────────
def make_targets(records: list[dict]) -> dict[str, dict]:
    """Return target_name -> {"y": np.ndarray, "mask": np.ndarray | None}.

    A None mask means use all records; otherwise the probe operates only
    on the masked subset (sliced labels and features).
    """
    branch = np.array([r["branch"] for r in records])
    order = np.array([r["belief_order"] for r in records])
    tom = np.array([r["requires_tom"] for r in records])

    # Binary collapsing: any-false (false or second_order_false) vs true.
    branch_any_false = np.where(branch == "true_belief", "true", "false")
    # Binary: second-order false specifically vs everything else.
    branch_so = np.where(branch == "second_order_false_belief", "so_false", "other")

    targets: dict[str, dict] = {
        "branch_3way":          {"y": branch,           "mask": None},
        "branch_any_false":     {"y": branch_any_false, "mask": None},
        "branch_so_false":      {"y": branch_so,        "mask": None},
        "belief_order":         {"y": order.astype(str), "mask": None},
        "requires_tom":         {"y": tom,              "mask": None},
        "branch_3way_order_1":  {"y": branch,           "mask": order == 1},
        "branch_3way_order_2":  {"y": branch,           "mask": order == 2},
        "branch_so_order_2":    {"y": branch_so,        "mask": order == 2},
        "branch_so_order_2_tom": {"y": branch_so,       "mask": (order == 2) & (tom == "tom")},
    }
    return targets


def make_table1_tomi(records, arrays, output_dir, n_splits, seed, n_perm):
    targets = make_targets(records)
    groups = np.asarray([r["scenario_id"] for r in records])
    feats = {
        "z1": arrays["z1"],
        "z2": arrays["z2"],
        "z_concat": arrays["z_concat"],
        "context_hidden": arrays["context_hidden"],
        "reward_vec": arrays["reward_vec"],
    }
    texts = [build_tomi_context_text(r) for r in records]
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    perm_rows: list[dict] = []
    for tname, spec in targets.items():
        y_full = spec["y"]
        mask = spec["mask"]
        if mask is None:
            idx = np.arange(len(records))
        else:
            idx = np.where(mask)[0]
        if len(idx) < 30:
            print(f"  [skip] target={tname} too few samples ({len(idx)})")
            continue
        y = y_full[idx]
        if len(np.unique(y)) < 2:
            print(f"  [skip] target={tname} only {len(np.unique(y))} class(es) after slicing")
            continue
        g = groups[idx]
        sub_texts = [texts[i] for i in idx]

        for fname, X in feats.items():
            X_sub = X[idx]
            real = probe_dense(X_sub, y, n_splits, seed, g)
            real |= {"target": tname, "feature": fname, "label_condition": "real"}
            rows.append(real)
            for k in range(n_perm):
                yp = rng.permutation(y)
                p = probe_dense(X_sub, yp, n_splits, seed, g)
                perm_rows.append({
                    "target": tname, "feature": fname, "label_condition": "shuffled",
                    "permutation_id": k + 1, **p,
                })
        real_t = probe_tfidf(sub_texts, y, n_splits, seed, g)
        real_t |= {"target": tname, "feature": "tfidf_text", "label_condition": "real"}
        rows.append(real_t)
        for k in range(n_perm):
            yp = rng.permutation(y)
            p = probe_tfidf(sub_texts, yp, n_splits, seed, g)
            perm_rows.append({
                "target": tname, "feature": "tfidf_text", "label_condition": "shuffled",
                "permutation_id": k + 1, **p,
            })

    # Write CSVs (mirror format of analyse_tom_latents)
    csv_lines = ["target,feature,label_condition,split_scheme,num_classes,num_samples,acc_mean,acc_std,f1_mean,f1_std"]
    for r in rows:
        csv_lines.append(
            f"{r['target']},{r['feature']},{r['label_condition']},{r['split_scheme']},"
            f"{r['num_classes']},{r['num_samples']},{r['acc_mean']:.4f},{r['acc_std']:.4f},"
            f"{r['f1_mean']:.4f},{r['f1_std']:.4f}"
        )

    perm_csv = ["target,feature,label_condition,permutation_id,split_scheme,num_classes,num_samples,acc_mean,acc_std,f1_mean,f1_std"]
    perm_acc_by_key: dict[tuple, list[float]] = {}
    perm_f1_by_key: dict[tuple, list[float]] = {}
    for r in perm_rows:
        k = (r["target"], r["feature"])
        perm_acc_by_key.setdefault(k, []).append(r["acc_mean"])
        perm_f1_by_key.setdefault(k, []).append(r["f1_mean"])
        perm_csv.append(
            f"{r['target']},{r['feature']},{r['label_condition']},{r['permutation_id']},"
            f"{r['split_scheme']},{r['num_classes']},{r['num_samples']},"
            f"{r['acc_mean']:.4f},{r['acc_std']:.4f},{r['f1_mean']:.4f},{r['f1_std']:.4f}"
        )
    for k, accs in perm_acc_by_key.items():
        f1s = perm_f1_by_key[k]
        csv_lines.append(
            f"{k[0]},{k[1]},shuffled_summary,episode_grouped,-,-,"
            f"{np.mean(accs):.4f},{np.std(accs):.4f},{np.mean(f1s):.4f},{np.std(f1s):.4f}"
        )

    (output_dir / "table1_linear_probe.csv").write_text("\n".join(csv_lines))
    (output_dir / "table1_linear_probe_permutations.csv").write_text("\n".join(perm_csv))

    md = ["# Linear probe (ToMi-2 zero-shot transfer)", "",
          "| target | feature | labels | F1 | Acc | N |",
          "|---|---|---|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['target']} | {r['feature']} | real | "
            f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f} | "
            f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f} | {r['num_samples']} |"
        )
    for k, accs in perm_acc_by_key.items():
        f1s = perm_f1_by_key[k]
        md.append(
            f"| {k[0]} | {k[1]} | shuffled (n={n_perm}) | "
            f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f} | "
            f"{np.mean(accs):.3f} ± {np.std(accs):.3f} | - |"
        )
    (output_dir / "table1_linear_probe.md").write_text("\n".join(md))
    print(f"Wrote {output_dir / 'table1_linear_probe.csv'}")


def save_records_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tomi_split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_records", type=int, default=1500)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model …", flush=True)
    model, tok = load_model(device, Path(args.ckpt))

    print(f"Loading ToMi-2 split={args.tomi_split} …", flush=True)
    raw = load_tomi_records(split=args.tomi_split)
    records = [r.to_dict() for r in raw]
    print(f"  {len(records)} (story, question) records before subsampling")
    if args.max_records and len(records) > args.max_records:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(records), size=args.max_records, replace=False)
        records = [records[i] for i in sorted(idx)]
        print(f"  subsampled to {len(records)}")

    n_by_order = {o: sum(1 for r in records if r["belief_order"] == o) for o in (0, 1, 2)}
    n_by_branch = {b: sum(1 for r in records if r["branch"] == b) for b in ("true_belief", "false_belief")}
    print(f"  by belief_order: {n_by_order}")
    print(f"  by branch:       {n_by_branch}")

    print("Encoding latents …", flush=True)
    arrays = extract_latents(
        model, tok, records, device,
        context_text_fn=build_tomi_context_text,
    )
    np.savez(output_dir / "latent_arrays.npz", **arrays)
    save_records_jsonl(records, output_dir / "analysis_records.jsonl")
    (output_dir / "latent_arrays_meta.json").write_text(json.dumps({
        "ckpt": args.ckpt,
        "benchmark": "ToMi-2 balanced",
        "tomi_split": args.tomi_split,
        "num_records": len(records),
        "z1_dim": int(arrays["z1"].shape[1]),
        "z2_dim": int(arrays["z2"].shape[1]),
        "context_dim": int(arrays["context_hidden"].shape[1]),
    }, indent=2))

    print("Running linear probes …", flush=True)
    make_table1_tomi(records, arrays, output_dir, args.n_splits, args.seed, args.n_perm)
    print("Done.")


if __name__ == "__main__":
    main()
