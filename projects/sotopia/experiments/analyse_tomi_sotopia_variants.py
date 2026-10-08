#!/usr/bin/env python3
"""Zero-shot ToMi-2 transfer of SOTOPIA-trained mental-supervision variants.

For each variant checkpoint (BIT / flat / shuffled), encode ToMi-2 stories
through the SOTOPIA encoder, extract z1/z2/z_concat/context_hidden, and run
the same linear-probe table used for BigToM in
analyse_tomi_latents.py.

Targets (binary unless noted):
  branch_3way                 (3-class: true_belief / false_belief / so_false)
  branch_any_false            (true vs any-false)
  branch_so_false             (so_false vs other)
  belief_order                (multiclass 0/1/2)
  requires_tom                (tom vs no_tom)
  branch_3way_order_1         (sliced)
  branch_3way_order_2         (sliced)
  branch_so_order_2           (sliced — recursive false-belief detection)
  branch_so_order_2_tom       (sliced — ToM-required recursive false belief)

Usage:
  python experiments/analyse_tomi_sotopia_variants.py \
    --variants \
      BIT=.../coupled_mental_reward_checkpoint_qwen_v3/epoch_5 \
      flat=.../runs/stage1/flat_mental_summary_qwen7b_seed42/best \
      shuffled=.../runs/stage1/shuffled_mental_qwen7b_seed42/best \
    --output_dir projects/sotopia/experiments/runs/stage1/tomi_transfer \
    --gpu 7
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

SOTOPIA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOTOPIA_ROOT))
sys.path.insert(0, "projects/bigtom/scripts")  # for tomi_loader

from stage1_train_coupled_mental_reward_v3 import (  # noqa: E402
    CUSTOM_HEAD_NAMES, REWARD_DIM, RecursiveToMModel,
)
from experiments.eval_mental_variants import load_variant_checkpoint  # noqa: E402
from experiments.probe_recursive_decomposition_empirical import (  # noqa: E402
    regression_probe_cv,
)
from tomi_loader import load_tomi_records  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter


# ── ToMi → SOTOPIA-shaped context ────────────────────────────────────────────
def build_sotopia_tomi_context(rec: dict) -> str:
    """Mirror SOTOPIA's _format_context so the encoder sees a familiar shape."""
    return (
        f"Scenario: {rec['story']}\n"
        f"Background: \n"
        f"Goal: Reason about each agent's beliefs.\n"
        f"Secret: None\n"
        f"Dialogue History:\n"
        f"Q: {rec['question'].strip()}\n"
        f"Turn 1 | Reasoner:"
    )


# ── Latent extraction (no dataset wrapper — direct tokenize) ────────────────
@torch.no_grad()
def extract_tomi_latents(
    model: RecursiveToMModel, tokenizer, records: list[dict],
    device: torch.device, batch_size: int, max_ctx_len: int,
) -> dict[str, np.ndarray]:
    z1_chunks, z2_chunks, ctx_chunks = [], [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        contexts = [build_sotopia_tomi_context(r) for r in batch]
        enc = tokenizer(contexts, truncation=True, max_length=max_ctx_len,
                        padding=True, return_tensors="pt")
        ctx_ids = enc.input_ids.to(device)
        ctx_mask = enc.attention_mask.to(device)
        with torch.amp.autocast(enabled=device.type == "cuda",
                                device_type="cuda", dtype=torch.bfloat16):
            ctx_last = model._encode_context(ctx_ids, ctx_mask)
            mu1 = model.z1_mu(ctx_last.to(model.z1_mu.weight.dtype))
            z2_inp = torch.cat([ctx_last.to(mu1.dtype), mu1], dim=1)
            mu2 = model.z2_mu(z2_inp)
        z1_chunks.append(mu1.float().cpu().numpy())
        z2_chunks.append(mu2.float().cpu().numpy())
        ctx_chunks.append(ctx_last.float().cpu().numpy())
        if (start // batch_size) % 50 == 0:
            print(f"  encoded {start}/{len(records)}", flush=True)

    z1 = np.concatenate(z1_chunks, axis=0)
    z2 = np.concatenate(z2_chunks, axis=0)
    return {
        "z1": z1.astype(np.float32),
        "z2": z2.astype(np.float32),
        "z_concat": np.concatenate([z1, z2], axis=1).astype(np.float32),
        "context_hidden": np.concatenate(ctx_chunks, axis=0).astype(np.float32),
    }


# ── Probe machinery (mirrors analyse_tomi_latents) ──────────────────────────
def make_cv_splits(y, n_splits, seed, groups):
    min_count = min(Counter(y).values())
    n_splits = max(2, min(n_splits, min_count))
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
    dummy = np.zeros(len(y), dtype=np.int32)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        sp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sp.split(dummy, y, groups=groups)), "episode_grouped"
    sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sp.split(dummy, y)), "stratified"


def probe_dense(X, y_str, n_splits, seed, groups):
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    splits, scheme = make_cv_splits(y, n_splits, seed, groups)
    accs, f1s = [], []
    for tr, te in splits:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return {
        "num_classes": int(len(le.classes_)),
        "num_samples": int(len(y)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "split_scheme": scheme,
    }


def make_targets(records: list[dict]) -> dict[str, dict]:
    branch = np.array([r["branch"] for r in records])
    order = np.array([r["belief_order"] for r in records])
    tom = np.array([r["requires_tom"] for r in records])
    branch_any_false = np.where(branch == "true_belief", "true", "false")
    branch_so = np.where(branch == "second_order_false_belief", "so_false", "other")
    return {
        "branch_3way":           {"y": branch,           "mask": None},
        "branch_any_false":      {"y": branch_any_false, "mask": None},
        "branch_so_false":       {"y": branch_so,        "mask": None},
        "belief_order":          {"y": order.astype(str), "mask": None},
        "requires_tom":          {"y": tom,              "mask": None},
        "branch_3way_order_1":   {"y": branch,           "mask": order == 1},
        "branch_3way_order_2":   {"y": branch,           "mask": order == 2},
        "branch_so_order_2":     {"y": branch_so,        "mask": order == 2},
        "branch_so_order_2_tom": {"y": branch_so,        "mask": (order == 2) & (tom == "tom")},
    }


def run_probe_table(
    variant_name: str, records: list[dict], arrays: dict[str, np.ndarray],
    n_splits: int, seed: int, n_perm: int, rng: np.random.Generator,
) -> list[dict]:
    targets = make_targets(records)
    groups = np.asarray([r["scenario_id"] for r in records])
    feats = {"z1": arrays["z1"], "z2": arrays["z2"],
             "z_concat": arrays["z_concat"], "context_hidden": arrays["context_hidden"]}

    rows: list[dict] = []
    for tname, spec in targets.items():
        y_full = spec["y"]; mask = spec["mask"]
        idx = np.arange(len(records)) if mask is None else np.where(mask)[0]
        if len(idx) < 30 or len(np.unique(y_full[idx])) < 2:
            print(f"  [skip] target={tname} n={len(idx)} (insufficient)")
            continue
        y = y_full[idx]; g = groups[idx]
        for fname, X in feats.items():
            X_sub = X[idx]
            real = probe_dense(X_sub, y, n_splits, seed, g)
            rows.append({"variant": variant_name, "target": tname, "feature": fname,
                         "label_condition": "real", **real})
            shuf_f1s, shuf_accs = [], []
            for _ in range(n_perm):
                yp = rng.permutation(y)
                p = probe_dense(X_sub, yp, n_splits, seed, g)
                shuf_f1s.append(p["f1_mean"]); shuf_accs.append(p["acc_mean"])
            rows.append({"variant": variant_name, "target": tname, "feature": fname,
                         "label_condition": "shuffled_summary",
                         "f1_mean": float(np.mean(shuf_f1s)),
                         "f1_std": float(np.std(shuf_f1s)),
                         "acc_mean": float(np.mean(shuf_accs)),
                         "acc_std": float(np.std(shuf_accs)),
                         "num_classes": real["num_classes"],
                         "num_samples": real["num_samples"],
                         "split_scheme": real["split_scheme"]})
    return rows


# ── Output ───────────────────────────────────────────────────────────────────
def write_outputs(out_dir: Path, all_rows: list[dict]) -> None:
    fieldnames = ["variant", "target", "feature", "label_condition",
                  "split_scheme", "num_classes", "num_samples",
                  "acc_mean", "acc_std", "f1_mean", "f1_std"]
    with (out_dir / "tomi_transfer_probes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    targets = sorted({r["target"] for r in all_rows})
    features = ["z1", "z2", "z_concat", "context_hidden"]
    variants = sorted({r["variant"] for r in all_rows})

    md = ["# ToMi-2 zero-shot transfer (SOTOPIA-trained variants)", ""]
    for tgt in targets:
        md += [f"## target = {tgt}", "",
               "| feature | labels | " + " | ".join(variants) + " |",
               "|---|---|" + "|".join(["---:"] * len(variants)) + "|"]
        for feat in features:
            for cond in ("real", "shuffled_summary"):
                cells = []
                for v in variants:
                    hit = next((r for r in all_rows
                                if r["variant"] == v and r["target"] == tgt
                                and r["feature"] == feat
                                and r["label_condition"] == cond), None)
                    cells.append(f"{hit['f1_mean']:.3f}" if hit else "—")
                md.append(f"| {feat} | {cond} | " + " | ".join(cells) + " |")
        md.append("")
    (out_dir / "tomi_transfer_probes.md").write_text("\n".join(md))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", required=True,
                   help="tag=ckpt_path entries")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tomi_split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max_records", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_ctx_len", type=int, default=512)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_perm", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    variant_specs = []
    for spec in args.variants:
        if "=" not in spec:
            raise ValueError(f"--variants expects tag=path, got {spec!r}")
        tag, path = spec.split("=", 1)
        variant_specs.append((tag.strip(), Path(path.strip())))

    print(f"Loading ToMi-2 split={args.tomi_split} …", flush=True)
    raw = load_tomi_records(split=args.tomi_split)
    records = [r.to_dict() for r in raw]
    if args.max_records and len(records) > args.max_records:
        idx = rng.choice(len(records), size=args.max_records, replace=False)
        records = [records[i] for i in sorted(idx)]
    print(f"  using {len(records)} records", flush=True)
    print(f"  by belief_order: { {o: sum(1 for r in records if r['belief_order']==o) for o in (0,1,2)} }")
    print(f"  by branch:       { Counter(r['branch'] for r in records) }")

    all_rows: list[dict] = []
    for tag, ckpt_dir in variant_specs:
        if not ckpt_dir.exists():
            print(f"[skip] {tag}: missing {ckpt_dir}", flush=True)
            continue
        print(f"\n=== {tag} ({ckpt_dir.name}) ===", flush=True)
        tokenizer, model = load_variant_checkpoint(args.model_name, ckpt_dir, device, args.z_dim)
        arrays = extract_tomi_latents(model, tokenizer, records, device,
                                      args.batch_size, args.max_ctx_len)
        np.savez_compressed(out_dir / f"latents_{tag}.npz", **arrays)
        del model; torch.cuda.empty_cache()
        rows = run_probe_table(tag, records, arrays, args.n_splits, args.seed,
                                args.n_perm, rng)
        all_rows.extend(rows)

    write_outputs(out_dir, all_rows)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"\nWrote {out_dir}/tomi_transfer_probes.md", flush=True)


if __name__ == "__main__":
    main()
