"""
Sweep across stage1_qwen_5k checkpoints to find one whose mental decoder
hasn't collapsed onto branch-prototypes. Decodes m1/m2 for the same set of
scenarios at each ckpt and reports cross-scenario diversity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from probe_mental_decoder_qwen5k import (  # noqa: E402
    DATA, generate_mental, load_model,
)
from stage1_train_mental_reward import (  # noqa: E402
    build_task_context, build_encoder_context,
)


def diversity_stats(texts: list[str]) -> dict:
    n = len(texts)
    if n == 0:
        return {"n": 0, "unique": 0, "unique_frac": 0.0, "avg_uniq_trigrams": 0.0}
    uniq = len(set(t.strip() for t in texts))
    tri_uniqs = []
    for t in texts:
        toks = t.split()
        if len(toks) < 3:
            tri_uniqs.append(0.0)
            continue
        tris = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        tri_uniqs.append(len(set(tris)) / max(1, len(tris)))
    return {
        "n": n,
        "unique": uniq,
        "unique_frac": uniq / n,
        "avg_uniq_trigrams": sum(tri_uniqs) / n,
    }


def pick_scenarios(by_sid: dict, n: int) -> list:
    paired = sorted(s for s, d in by_sid.items() if "aware" in d and "not_aware" in d)
    step = max(1, len(paired) // n)
    return paired[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpts", nargs="+",
        default=["step_500", "step_1000", "step_1500", "step_2000", "epoch_0", "best_ckpt"],
    )
    ap.add_argument("--n_scenarios", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--out", type=str, default=str(REPO / "checkpoints/stage1_qwen_5k/decoder_sweep.md"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    by_sid: dict = {}
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r
    sids = pick_scenarios(by_sid, args.n_scenarios)
    print(f"Probing scenarios: {sids}", flush=True)

    out_lines = ["# stage1_qwen_5k mental-decoder sweep", ""]
    summary_table = [
        "| ckpt | step | kl1 | kl2 | m1 CE | m2 CE | m1 unique/N | m1 trig-uniq | m2 unique/N | m2 trig-uniq |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    base_dir = REPO / "checkpoints/stage1_qwen_5k"
    for tag in args.ckpts:
        ckpt_dir = base_dir / tag
        if not ckpt_dir.exists():
            print(f"skip missing {tag}")
            continue
        ck = torch.load(ckpt_dir / "heads.pt", map_location="cpu", weights_only=False)
        am = ck.get("avg_metrics", {})

        try:
            model, tok = load_model(device, ckpt_dir)
        except Exception as e:
            print(f"{tag}: load failed: {e}")
            continue

        m1_texts: list[str] = []
        m2_texts: list[str] = []
        per_section = [f"## ckpt = {tag}", ""]
        for sid in sids:
            for cond in ("aware", "not_aware"):
                row = by_sid[sid][cond]
                ctx = build_task_context(row["story"], 0, percept=row["percept"])
                ctx = build_encoder_context(ctx, row["belief_question"])
                enc = tok(ctx, truncation=True, max_length=768, return_tensors="pt").to(device)
                mu1, mu2 = model.encode_z1_z2_deterministic(enc.input_ids, enc.attention_mask)
                t1 = generate_mental(model, tok, mu1, model.mental1_decoder, args.max_new_tokens)
                t2 = generate_mental(model, tok, mu2, model.mental2_decoder, args.max_new_tokens)
                m1_texts.append(t1)
                m2_texts.append(t2)
                per_section += [
                    f"- sid={sid} {cond}",
                    f"  - gold m1: `{row['first_order_belief']}`",
                    f"  - dec m1:  `{t1[:140]}`",
                    f"  - gold m2: `{row['second_order_belief']}`",
                    f"  - dec m2:  `{t2[:140]}`",
                ]
        s1 = diversity_stats(m1_texts)
        s2 = diversity_stats(m2_texts)
        per_section += [
            "",
            f"**m1 diversity**: unique {s1['unique']}/{s1['n']} = {s1['unique_frac']:.2f}, "
            f"avg-trigram-uniq {s1['avg_uniq_trigrams']:.2f}",
            f"**m2 diversity**: unique {s2['unique']}/{s2['n']} = {s2['unique_frac']:.2f}, "
            f"avg-trigram-uniq {s2['avg_uniq_trigrams']:.2f}",
            "",
        ]
        summary_table.append(
            f"| {tag} | {ck.get('global_step')} | "
            f"{am.get('kl1',0):.2f} | {am.get('kl2',0):.2f} | "
            f"{am.get('m1',0):.2f} | {am.get('m2',0):.2f} | "
            f"{s1['unique']}/{s1['n']} | {s1['avg_uniq_trigrams']:.2f} | "
            f"{s2['unique']}/{s2['n']} | {s2['avg_uniq_trigrams']:.2f} |"
        )
        print(f"\n=== {tag} ===")
        print(f"m1 unique {s1['unique']}/{s1['n']} trig-uniq {s1['avg_uniq_trigrams']:.2f} | "
              f"m2 unique {s2['unique']}/{s2['n']} trig-uniq {s2['avg_uniq_trigrams']:.2f}")
        for line in per_section[2:14]:
            print(line)

        out_lines += per_section
        del model
        torch.cuda.empty_cache()

    out_lines = ["# stage1_qwen_5k mental-decoder sweep", "",
                 "## Summary"] + summary_table + [""] + out_lines[2:]
    Path(args.out).write_text("\n".join(out_lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
