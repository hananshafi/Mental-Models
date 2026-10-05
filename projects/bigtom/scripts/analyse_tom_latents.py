"""
BigToM port of the sotopia ToM-latent analysis.

Pipeline:
  1. Load Stage-1 RecursiveToMModel from a checkpoint dir.
  2. For every paired (aware, not_aware) BigToM scenario, encode z1/z2/context_hidden
     deterministically, save records jsonl + latent arrays npz.
  3. Linear-probe table: target ∈ {branch, init_belief, task} × feature ∈
     {z1, z2, z_concat, context_hidden, tfidf_text} with episode-grouped CV
     and a shuffled-label null distribution.
  4. UMAP/t-SNE of z_concat colored by branch.
  5. Latent traversal: fit a logistic-regression direction on z_concat for
     branch=aware, sweep alpha, decode mental1/mental2 with the AR decoder,
     and trace belief_classifier prob + z1_only / z_combined reward heads.

Usage:
  python scripts/analyse_tom_latents.py \
      --ckpt projects/bigtom/checkpoints/stage1_qwen_5k/step_1000 \
      --output_dir projects/bigtom/runs/analysis/mental_latent_qwen_step1000 \
      --max_records 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from stage1_train_mental_reward import (  # noqa: E402
    build_encoder_context, build_task_context,
)
from probe_mental_decoder_qwen5k import (  # noqa: E402
    DATA, generate_mental, load_model,
)


# ── Phase 1: data + latent extraction ────────────────────────────────────────
def load_paired_scenarios(data_path: Path) -> list[dict]:
    by_sid: dict = {}
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r
    out: list[dict] = []
    for sid in sorted(by_sid):
        d = by_sid[sid]
        if "aware" not in d or "not_aware" not in d:
            continue
        for cond in ("aware", "not_aware"):
            for init_flag in (0, 1):
                for task in ("forward_belief", "forward_action", "backward_belief"):
                    pos = d[cond]
                    out.append({
                        "scenario_id": sid,
                        "condition": cond,
                        "init_belief_idx": init_flag,
                        "task": task,
                        "story": pos["story"],
                        "percept": pos.get("percept", ""),
                        "gold_action": pos.get("gold_action", ""),
                        "belief_question": pos["belief_question"],
                        "action_question": pos.get("action_question", ""),
                        "first_order_belief": pos["first_order_belief"],
                        "second_order_belief": pos["second_order_belief"],
                        "gold_belief": pos.get("gold_belief", ""),
                    })
    return out


def build_context_text(rec: dict) -> str:
    if rec["task"] == "backward_belief":
        ctx = build_task_context(rec["story"], rec["init_belief_idx"], action=rec["gold_action"])
        question = rec["belief_question"]
    elif rec["task"] == "forward_belief":
        ctx = build_task_context(rec["story"], rec["init_belief_idx"], percept=rec["percept"])
        question = rec["belief_question"]
    else:
        ctx = build_task_context(rec["story"], rec["init_belief_idx"], percept=rec["percept"])
        question = rec["action_question"] or rec["belief_question"]
    return build_encoder_context(ctx, question)


@torch.no_grad()
def extract_latents(
    model, tok, records: list[dict], device: str, max_ctx_len: int = 768,
    context_text_fn=None,
) -> dict[str, np.ndarray]:
    if context_text_fn is None:
        context_text_fn = build_context_text
    z1_list: list[np.ndarray] = []
    z2_list: list[np.ndarray] = []
    ctx_list: list[np.ndarray] = []
    z1r_list: list[float] = []
    zcr_list: list[float] = []

    for i, rec in enumerate(records):
        text = context_text_fn(rec)
        enc = tok(text, truncation=True, max_length=max_ctx_len, return_tensors="pt").to(device)
        ctx_last, _ = model._encode(enc.input_ids, enc.attention_mask)
        mu1 = model.z1_mu(ctx_last.to(model.z1_mu.weight.dtype))
        z2_inp = torch.cat([ctx_last.to(mu1.dtype), mu1], dim=1)
        mu2 = model.z2_mu(z2_inp)

        z1_list.append(mu1[0].float().cpu().numpy())
        z2_list.append(mu2[0].float().cpu().numpy())
        ctx_list.append(ctx_last[0].float().cpu().numpy())

        z1_only = model.z1_only_reward_head(
            mu1.to(model.z1_only_reward_head[0].weight.dtype)
        )
        z_combined = model.z_combined_reward_head(
            torch.cat([mu1, mu2], dim=1).to(model.z_combined_reward_head[0].weight.dtype)
        )
        z1r_list.append(float(z1_only.item()))
        zcr_list.append(float(z_combined.item()))

        if (i + 1) % 200 == 0:
            print(f"  encoded {i + 1}/{len(records)}", flush=True)

    return {
        "z1": np.asarray(z1_list, dtype=np.float32),
        "z2": np.asarray(z2_list, dtype=np.float32),
        "z_concat": np.concatenate(
            [np.asarray(z1_list, dtype=np.float32), np.asarray(z2_list, dtype=np.float32)],
            axis=1,
        ),
        "context_hidden": np.asarray(ctx_list, dtype=np.float32),
        "reward_vec": np.stack(
            [np.asarray(z1r_list, dtype=np.float32), np.asarray(zcr_list, dtype=np.float32)],
            axis=1,
        ),
    }


# ── Phase 2: linear probes ───────────────────────────────────────────────────
def make_cv_splits(y: np.ndarray, n_splits: int, seed: int, groups: np.ndarray | None):
    min_count = min(Counter(y).values())
    max_splits = min_count
    if groups is not None:
        max_splits = min(max_splits, len(np.unique(groups)))
    n_splits = max(2, min(n_splits, max_splits))
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


def probe_tfidf(texts, y_str, n_splits, seed, groups):
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    splits, scheme = make_cv_splits(y, n_splits, seed, groups)
    accs, f1s = [], []
    for tr, te in splits:
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xtr, y[tr])
        p = clf.predict(Xte)
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return {
        "num_classes": int(len(le.classes_)),
        "num_samples": int(len(y)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "split_scheme": scheme,
    }


def make_table1(records, arrays, output_dir, n_splits, seed, n_perm):
    targets = {
        "branch": np.asarray([r["condition"] for r in records]),
        "init_belief": np.asarray([str(r["init_belief_idx"]) for r in records]),
        "task": np.asarray([r["task"] for r in records]),
    }
    groups = np.asarray([r["scenario_id"] for r in records])
    feats = {
        "z1": arrays["z1"],
        "z2": arrays["z2"],
        "z_concat": arrays["z_concat"],
        "context_hidden": arrays["context_hidden"],
        "reward_vec": arrays["reward_vec"],
    }
    texts = [build_context_text(r) for r in records]
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    perm_rows: list[dict] = []
    for tname, y in targets.items():
        for fname, X in feats.items():
            real = probe_dense(X, y, n_splits, seed, groups)
            real |= {"target": tname, "feature": fname, "label_condition": "real"}
            rows.append(real)
            for k in range(n_perm):
                yp = rng.permutation(y)
                p = probe_dense(X, yp, n_splits, seed, groups)
                perm_rows.append({
                    "target": tname, "feature": fname, "label_condition": "shuffled",
                    "permutation_id": k + 1, **p,
                })
        real_t = probe_tfidf(texts, y, n_splits, seed, groups)
        real_t |= {"target": tname, "feature": "tfidf_text", "label_condition": "real"}
        rows.append(real_t)
        for k in range(n_perm):
            yp = rng.permutation(y)
            p = probe_tfidf(texts, yp, n_splits, seed, groups)
            perm_rows.append({
                "target": tname, "feature": "tfidf_text", "label_condition": "shuffled",
                "permutation_id": k + 1, **p,
            })

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

    md = ["# Linear probe (BigToM stage1)", "", "| target | feature | labels | F1 | Acc | N |",
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


# ── Phase 3: 2-D embedding ───────────────────────────────────────────────────
def make_figure1(records, arrays, output_dir):
    try:
        import umap
    except Exception as e:
        print(f"umap-learn not available ({e}); skipping UMAP")
        return
    Xs = StandardScaler().fit_transform(arrays["z_concat"])
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=20, min_dist=0.1)
    emb = reducer.fit_transform(Xs)
    branch = np.asarray([r["condition"] for r in records])
    fig, ax = plt.subplots(figsize=(7, 6))
    for cond, color in [("aware", "#2E86AB"), ("not_aware", "#E76F51")]:
        m = branch == cond
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.55, c=color, label=cond, edgecolors="none")
    ax.set_title("UMAP of z_concat colored by branch (BigToM stage1)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_z_concat_umap.png", dpi=180)
    np.savetxt(
        output_dir / "figure1_z_concat_umap_points.csv",
        np.concatenate([emb, branch[:, None].astype(object)], axis=1),
        fmt="%s", delimiter=",", header="umap_x,umap_y,branch", comments="",
    )
    print(f"Wrote {output_dir / 'figure1_z_concat_umap.png'}")


# ── Phase 4: traversal with AR decoder ───────────────────────────────────────
def fit_concept_direction(X, y_bin):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xs, y_bin)
    coef = clf.coef_[0] / sc.scale_
    intercept = float(clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_))
    return coef.astype(np.float32), intercept


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def _trained_head_outputs(model, mu1_t, mu2_t):
    """Returns (belief_class0_prob, z1_only_reward, z_combined_reward)."""
    blogits = model.belief_classifier(
        mu1_t.to(model.belief_classifier[0].weight.dtype)
    )[0].float().cpu().numpy()
    bprob = float(np.exp(blogits[0]) / (np.exp(blogits[0]) + np.exp(blogits[1])))
    z1r = float(model.z1_only_reward_head(
        mu1_t.to(model.z1_only_reward_head[0].weight.dtype)
    ).item())
    zcr = float(model.z_combined_reward_head(
        torch.cat([mu1_t, mu2_t], dim=1).to(model.z_combined_reward_head[0].weight.dtype)
    ).item())
    return bprob, z1r, zcr


@torch.no_grad()
def make_figure2(args, records, arrays, model, tok, output_dir, device):
    """
    Held-out traversal:
      - Split scenarios into train/test (by scenario_id, paired branches stay together).
      - Fit logistic-regression branch direction on train split z_concat.
      - Pick anchors near decision boundary on the *test* split.
      - For each anchor, traverse along (a) the belief direction and
        (b) k random unit directions of matched perturbation L2 (= |α|·σ_proj_train).
      - Record trained-head responses (belief classifier, z1_only / z_combined reward).
        Mental decoding is run only on the belief-direction traversal.
    """
    branch = np.asarray([r["condition"] for r in records])
    sids = np.asarray([r["scenario_id"] for r in records])
    rng = np.random.default_rng(args.seed)

    # Group-aware train/test split over scenario_ids
    unique_sids = np.unique(sids)
    rng.shuffle(unique_sids)
    n_train_sids = max(1, int(len(unique_sids) * (1.0 - args.holdout_frac)))
    train_sids = set(unique_sids[:n_train_sids].tolist())
    train_mask = np.array([s in train_sids for s in sids])
    test_mask = ~train_mask

    y_train = (branch[train_mask] == "aware").astype(np.int64)
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        raise RuntimeError("Train split is degenerate after grouping; widen holdout_frac.")

    direction, intercept = fit_concept_direction(arrays["z_concat"][train_mask], y_train)
    unit_dir = direction / (np.linalg.norm(direction) + 1e-8)
    # α scale: use σ along the direction over all data — unit choice, not label leakage
    # since the direction is fit on train only.
    proj_all = arrays["z_concat"] @ unit_dir
    proj_std = float(np.std(proj_all))

    base_probs_test = sigmoid(arrays["z_concat"][test_mask] @ direction + intercept)
    test_indices = np.where(test_mask)[0]

    # Pick anchors near the boundary on the test split, balanced across branches.
    aware_te = (branch[test_mask] == "aware")
    aware_idx = test_indices[aware_te]
    naw_idx = test_indices[~aware_te]
    aware_margin = np.abs(base_probs_test[aware_te] - 0.5)
    naw_margin = np.abs(base_probs_test[~aware_te] - 0.5)
    half = max(1, args.num_traversal_examples // 2)
    aware_pick = aware_idx[np.argsort(aware_margin)[:half]]
    naw_pick = naw_idx[np.argsort(naw_margin)[:args.num_traversal_examples - half]]
    anchor_idx = np.concatenate([naw_pick, aware_pick])[: args.num_traversal_examples]

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    n_random = args.num_random_directions
    z_dim = arrays["z1"].shape[1]
    full_dim = arrays["z_concat"].shape[1]

    rows: list[dict] = []
    md = [
        "# Held-out branch-direction traversal",
        "",
        f"Train scenarios: {len(train_sids)} (paired records: {int(train_mask.sum())}).",
        f"Test scenarios: {len(unique_sids) - len(train_sids)} (paired records: {int(test_mask.sum())}).",
        "Direction = logistic-regression coefficients on z_concat for branch=aware,",
        "fit on train scenarios only.",
        "",
        f"Anchors: {args.num_traversal_examples} test-split records nearest the boundary.",
        f"α units: σ_proj along held-out direction over all data = {proj_std:.3f}.",
        f"Random control: {n_random} unit directions, perturbation L2 matched at α·σ_proj.",
        "",
    ]

    for rank, idx in enumerate(anchor_idx, 1):
        rec = records[idx]
        z1 = arrays["z1"][idx].copy()
        z2 = arrays["z2"][idx].copy()
        md += [
            f"## Anchor {rank}: sid={rec['scenario_id']} cond={rec['condition']} task={rec['task']}",
            f"- gold first-order:  `{rec['first_order_belief']}`",
            f"- gold second-order: `{rec['second_order_belief']}`",
            f"- base p(aware|z) = {float(sigmoid(np.concatenate([z1, z2]) @ direction + intercept)):.3f}",
            "",
        ]

        # Sample random directions per anchor (independent draws)
        rand_dirs = rng.normal(size=(n_random, full_dim)).astype(np.float32)
        rand_dirs /= (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-8)

        for alpha in alphas:
            mag = float(alpha) * proj_std

            # ── Belief-direction traversal (with mental decoding) ──
            delta = mag * unit_dir
            z1b, z2b = z1 + delta[:z_dim], z2 + delta[z_dim:]
            mu1_t = torch.tensor(z1b, dtype=torch.float32, device=device).unsqueeze(0)
            mu2_t = torch.tensor(z2b, dtype=torch.float32, device=device).unsqueeze(0)
            m1_text = generate_mental(model, tok, mu1_t, model.mental1_decoder, args.max_new_tokens)
            m2_text = generate_mental(model, tok, mu2_t, model.mental2_decoder, args.max_new_tokens)
            bprob_b, z1r_b, zcr_b = _trained_head_outputs(model, mu1_t, mu2_t)
            p_aware_b = float(sigmoid(np.concatenate([z1b, z2b]) @ direction + intercept))

            rows.append({
                "anchor_rank": rank, "scenario_id": rec["scenario_id"],
                "condition": rec["condition"], "task": rec["task"],
                "direction": "belief", "rand_id": -1, "alpha": float(alpha),
                "p_branch_aware": p_aware_b, "p_belief_class0": bprob_b,
                "z1_only_reward": z1r_b, "z_combined_reward": zcr_b,
                "decoded_mental1": m1_text, "decoded_mental2": m2_text,
            })

            # ── Random-direction control (matched ‖α·σ‖, no decoding) ──
            for ri in range(n_random):
                delta_r = mag * rand_dirs[ri]
                z1r, z2r = z1 + delta_r[:z_dim], z2 + delta_r[z_dim:]
                mu1_r = torch.tensor(z1r, dtype=torch.float32, device=device).unsqueeze(0)
                mu2_r = torch.tensor(z2r, dtype=torch.float32, device=device).unsqueeze(0)
                bprob_r, z1r_r, zcr_r = _trained_head_outputs(model, mu1_r, mu2_r)
                p_aware_r = float(sigmoid(np.concatenate([z1r, z2r]) @ direction + intercept))
                rows.append({
                    "anchor_rank": rank, "scenario_id": rec["scenario_id"],
                    "condition": rec["condition"], "task": rec["task"],
                    "direction": "random", "rand_id": ri, "alpha": float(alpha),
                    "p_branch_aware": p_aware_r, "p_belief_class0": bprob_r,
                    "z1_only_reward": z1r_r, "z_combined_reward": zcr_r,
                    "decoded_mental1": "", "decoded_mental2": "",
                })

            md.append(
                f"- α={alpha:+.2f}  belief→ p_cls0={bprob_b:.3f}  r_z1={z1r_b:.2f}  r_zc={zcr_b:.2f}\n"
                f"  - m1: `{m1_text[:140]}`\n  - m2: `{m2_text[:140]}`"
            )
        md.append("")

    csv_path = output_dir / "figure2_traversal_branch_aware.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                json.dumps(r[k]) if isinstance(r[k], str) else f"{r[k]}"
                for k in keys
            ) + "\n")
    (output_dir / "figure2_traversal_branch_aware.md").write_text("\n".join(md))

    # Plot: classifier probability for belief vs random, per anchor
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    palette = ["#2E86AB", "#1E8449", "#7B3F99", "#E07B00"]
    for rank in range(1, args.num_traversal_examples + 1):
        c = palette[(rank - 1) % len(palette)]
        belief_rows = sorted(
            [r for r in rows if r["anchor_rank"] == rank and r["direction"] == "belief"],
            key=lambda r: r["alpha"],
        )
        rand_rows = [r for r in rows if r["anchor_rank"] == rank and r["direction"] == "random"]
        # Aggregate random: mean ± std over random_id at each alpha
        alpha_values = sorted({r["alpha"] for r in rand_rows})
        rand_cls_mean = []
        rand_cls_std = []
        rand_zr_mean = []
        rand_zr_std = []
        rand_zc_mean = []
        rand_zc_std = []
        for a in alpha_values:
            sub = [r for r in rand_rows if r["alpha"] == a]
            rand_cls_mean.append(np.mean([r["p_belief_class0"] for r in sub]))
            rand_cls_std.append(np.std([r["p_belief_class0"] for r in sub]))
            rand_zr_mean.append(np.mean([r["z1_only_reward"] for r in sub]))
            rand_zr_std.append(np.std([r["z1_only_reward"] for r in sub]))
            rand_zc_mean.append(np.mean([r["z_combined_reward"] for r in sub]))
            rand_zc_std.append(np.std([r["z_combined_reward"] for r in sub]))
        xs_b = [r["alpha"] for r in belief_rows]
        cls_b = [r["p_belief_class0"] for r in belief_rows]
        z1r_b = [r["z1_only_reward"] for r in belief_rows]
        zcr_b = [r["z_combined_reward"] for r in belief_rows]
        axes[0].plot(xs_b, cls_b, "-o", color=c, linewidth=1.8, markersize=6,
                     label=f"anchor {rank} belief-dir")
        axes[0].fill_between(
            alpha_values,
            np.array(rand_cls_mean) - np.array(rand_cls_std),
            np.array(rand_cls_mean) + np.array(rand_cls_std),
            color=c, alpha=0.10,
        )
        axes[0].plot(alpha_values, rand_cls_mean, ":", color=c, linewidth=1.0, alpha=0.65,
                     label=f"anchor {rank} random ({n_random})")

        axes[1].plot(xs_b, z1r_b, "-o", color=c, linewidth=1.6, markersize=5,
                     label=f"anchor {rank} z1_only belief")
        axes[1].plot(xs_b, zcr_b, "--s", color=c, linewidth=1.4, markersize=4, alpha=0.85,
                     label=f"anchor {rank} z_comb belief")
        axes[1].plot(alpha_values, rand_zr_mean, ":", color=c, linewidth=0.9, alpha=0.55)
        axes[1].plot(alpha_values, rand_zc_mean, "-.", color=c, linewidth=0.9, alpha=0.45)
    axes[0].axhline(0.5, color="k", linestyle=":", alpha=0.4)
    axes[0].set_xlabel("α"); axes[0].set_ylabel("p(belief class 0)")
    axes[0].set_title("Trained belief classifier vs traversal direction")
    axes[1].set_xlabel("α"); axes[1].set_ylabel("reward")
    axes[1].set_title("Reward heads vs traversal direction")
    axes[0].legend(fontsize=6.5, loc="best", ncol=2)
    axes[1].legend(fontsize=6.5, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_traversal_branch_aware.png", dpi=160)
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'figure2_traversal_branch_aware.png'}")


# ── orchestration ────────────────────────────────────────────────────────────
def save_records_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default=str(DATA))
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_records", type=int, default=1500,
                    help="Subsample to at most this many records (uniform over scenarios).")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_traversal_examples", type=int, default=4)
    ap.add_argument("--alpha_min", type=float, default=-2.5)
    ap.add_argument("--alpha_max", type=float, default=2.5)
    ap.add_argument("--alpha_steps", type=int, default=7)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--skip_traversal", action="store_true")
    ap.add_argument("--holdout_frac", type=float, default=0.3,
                    help="Fraction of unique scenarios reserved as held-out for traversal anchors.")
    ap.add_argument("--num_random_directions", type=int, default=8,
                    help="Random unit-direction control samples per anchor for traversal.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model …", flush=True)
    model, tok = load_model(device, Path(args.ckpt))

    print("Loading scenarios …", flush=True)
    records = load_paired_scenarios(Path(args.data))
    print(f"  {len(records)} task-records before subsampling")
    if args.max_records and len(records) > args.max_records:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(records), size=args.max_records, replace=False)
        records = [records[i] for i in sorted(idx)]
        print(f"  subsampled to {len(records)}")

    print("Encoding latents …", flush=True)
    arrays = extract_latents(model, tok, records, device)
    np.savez(output_dir / "latent_arrays.npz", **arrays)
    save_records_jsonl(records, output_dir / "analysis_records.jsonl")
    (output_dir / "latent_arrays_meta.json").write_text(json.dumps({
        "ckpt": args.ckpt, "num_records": len(records),
        "z1_dim": int(arrays["z1"].shape[1]), "z2_dim": int(arrays["z2"].shape[1]),
        "context_dim": int(arrays["context_hidden"].shape[1]),
    }, indent=2))

    print("Running linear probes …", flush=True)
    make_table1(records, arrays, output_dir, args.n_splits, args.seed, args.n_perm)

    print("Building UMAP figure …", flush=True)
    make_figure1(records, arrays, output_dir)

    if not args.skip_traversal:
        print("Running traversal + AR decoding …", flush=True)
        make_figure2(args, records, arrays, model, tok, output_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()
