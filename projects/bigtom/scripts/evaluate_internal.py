"""
BigToM internal sanity-check evaluation.

Evaluates any of:
    - `base`  : plain base model, no ToM encoder
    - `sft`   : Stage 2 z-conditioned SFT checkpoint
    - `grpo`  : Stage 3 z-conditioned GRPO checkpoint

on:
    - BigToM   (forward_belief, forward_action, backward_belief; true/false/control)
    - ToMi     (external JSONL, optional)
    - FANToM   (external JSONL, optional)

Metric: answer matching with optional Qwen-as-judge fallback.

For paper-ready benchmark reporting, prefer `evaluate_official_benchmarks.py`,
which targets the official benchmark test splits and benchmark-native metrics.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_client import QwenClient
from stage1_train_mental_reward import build_encoder_context, build_presented_story
from stage2_policy_sft import (
    MENTAL_PREFIX_LEN,
    MentalPrefixProjector,
    build_prompt,
    load_stage1_encoder,
)


SCENARIO_FIELDS = [
    "story", "aware_event", "not_aware_event",
    "action_new_state", "action_init_state",
    "belief_question", "desire_question", "action_question",
    "belief_aware", "desire_aware", "action_aware",
    "belief_not_aware", "desire_not_aware", "action_not_aware",
    "random_event", "aware_random", "not_aware_random",
    "source", "init_belief_idx",
]


def split_story_sentences(story: str) -> List[str]:
    return [s.strip() for s in story.split(".") if s.strip()]


def build_control_story(story: str, random_event: str, init_belief_flag: int) -> str:
    parts = split_story_sentences(story)
    if len(parts) < 5:
        return f"{story.strip()} {random_event.strip()}".strip()
    shown = parts[:4] if init_belief_flag else parts[:3]
    return ". ".join(shown + [random_event.strip()]) + "."


def load_bigtom_eval(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for sid, row in enumerate(reader):
            if len(row) < 19:
                continue
            d = {k: v for k, v in zip(SCENARIO_FIELDS, row[:19])}
            if len(split_story_sentences(d["story"])) < 5:
                continue

            for init_belief_flag in (0, 1):
                base_story = build_presented_story(d["story"], init_belief_flag)
                control_story = build_control_story(d["story"], d["random_event"], init_belief_flag)

                rows.extend([
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_belief",
                        "condition": "true_belief",
                        "init_belief": init_belief_flag,
                        "story": f"{base_story} {d['aware_event']}",
                        "question": d["belief_question"],
                        "answer": d["belief_aware"],
                        "option_correct": d["belief_aware"],
                        "option_wrong": d["belief_not_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_belief",
                        "condition": "false_belief",
                        "init_belief": init_belief_flag,
                        "story": f"{base_story} {d['not_aware_event']}",
                        "question": d["belief_question"],
                        "answer": d["belief_not_aware"],
                        "option_correct": d["belief_not_aware"],
                        "option_wrong": d["belief_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_belief",
                        "condition": "true_control",
                        "init_belief": init_belief_flag,
                        "story": f"{control_story} {d['aware_random']}",
                        "question": d["belief_question"],
                        "answer": d["belief_not_aware"],
                        "option_correct": d["belief_not_aware"],
                        "option_wrong": d["belief_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_belief",
                        "condition": "false_control",
                        "init_belief": init_belief_flag,
                        "story": f"{control_story} {d['not_aware_random']}",
                        "question": d["belief_question"],
                        "answer": d["belief_not_aware"],
                        "option_correct": d["belief_not_aware"],
                        "option_wrong": d["belief_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_action",
                        "condition": "true_belief",
                        "init_belief": init_belief_flag,
                        "story": f"{base_story} {d['aware_event']}",
                        "question": d["action_question"],
                        "answer": d["action_aware"],
                        "option_correct": d["action_aware"],
                        "option_wrong": d["action_not_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_action",
                        "condition": "false_belief",
                        "init_belief": init_belief_flag,
                        "story": f"{base_story} {d['not_aware_event']}",
                        "question": d["action_question"],
                        "answer": d["action_not_aware"],
                        "option_correct": d["action_not_aware"],
                        "option_wrong": d["action_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_action",
                        "condition": "true_control",
                        "init_belief": init_belief_flag,
                        "story": f"{control_story} {d['aware_random']}",
                        "question": d["action_question"],
                        "answer": d["action_not_aware"],
                        "option_correct": d["action_not_aware"],
                        "option_wrong": d["action_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "forward_action",
                        "condition": "false_control",
                        "init_belief": init_belief_flag,
                        "story": f"{control_story} {d['not_aware_random']}",
                        "question": d["action_question"],
                        "answer": d["action_not_aware"],
                        "option_correct": d["action_not_aware"],
                        "option_wrong": d["action_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "backward_belief",
                        "condition": "true_belief",
                        "init_belief": init_belief_flag,
                        "story": f"{base_story} {d['action_aware']}",
                        "question": d["belief_question"],
                        "answer": d["belief_aware"],
                        "option_correct": d["belief_aware"],
                        "option_wrong": d["belief_not_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "backward_belief",
                        "condition": "false_belief",
                        "init_belief": init_belief_flag,
                        "story": f"{base_story} {d['action_not_aware']}",
                        "question": d["belief_question"],
                        "answer": d["belief_not_aware"],
                        "option_correct": d["belief_not_aware"],
                        "option_wrong": d["belief_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "backward_belief",
                        "condition": "true_control",
                        "init_belief": init_belief_flag,
                        "story": f"{control_story} {d['action_not_aware']}",
                        "question": d["belief_question"],
                        "answer": d["belief_not_aware"],
                        "option_correct": d["belief_not_aware"],
                        "option_wrong": d["belief_aware"],
                    },
                    {
                        "dataset": "bigtom",
                        "sid": sid,
                        "task": "backward_belief",
                        "condition": "false_control",
                        "init_belief": init_belief_flag,
                        "story": f"{control_story} {d['action_not_aware']}",
                        "question": d["belief_question"],
                        "answer": d["belief_not_aware"],
                        "option_correct": d["belief_not_aware"],
                        "option_wrong": d["belief_aware"],
                    },
                ])
    return rows


def load_jsonl_eval(path: Path, dataset_name: str) -> List[Dict]:
    rows = []
    if not path.exists():
        print(f"  (transfer) {dataset_name}: file not found at {path}, skipping")
        return rows
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "dataset": dataset_name,
                "story": r["story"],
                "question": r["question"],
                "answer": r["answer"],
                "option_correct": r["answer"],
                "option_wrong": r.get("wrong_answer", ""),
                "task": r.get("task", dataset_name),
                "condition": r.get("condition", "default"),
                "init_belief": r.get("init_belief", -1),
            })
    return rows


def load_base_policy(base_model_name: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    model.eval()
    return model


def load_lora_policy(base_model_name: str, ckpt_dir: Path, device: torch.device):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    lora_dir = ckpt_dir / "policy_lora"
    if not lora_dir.exists():
        raise FileNotFoundError(f"{lora_dir} not found")
    model = PeftModel.from_pretrained(base, str(lora_dir), is_trainable=False)
    model.eval()
    return model


@torch.no_grad()
def generate_answer(
    policy, tokenizer, device,
    prompt: str,
    mental_prefix_embeds: Optional[torch.Tensor] = None,
    max_new_tokens: int = 128,
) -> str:
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    )["input_ids"].to(device)

    if mental_prefix_embeds is None:
        out = policy.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = out[0, prompt_ids.size(1):]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    embed = policy.get_input_embeddings()
    tok_embeds = embed(prompt_ids)
    full_embeds = torch.cat([mental_prefix_embeds, tok_embeds], dim=1)
    attn = torch.ones(full_embeds.size()[:2], dtype=torch.long, device=device)

    out = policy.generate(
        inputs_embeds=full_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_candidate_answers(completion: str) -> List[str]:
    raw = completion.strip()
    if not raw:
        return []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    candidates = [raw]
    if lines:
        candidates.append(lines[-1])
    lower = raw.lower()
    for marker in ("final answer:", "answer:"):
        idx = lower.rfind(marker)
        if idx >= 0:
            candidates.append(raw[idx + len(marker):].strip())
    return [_norm(c) for c in candidates if c.strip()]


def rule_match(completion: str, correct: str, wrong: str) -> Optional[bool]:
    corr = _norm(correct)
    wr = _norm(wrong) if wrong else ""
    candidates = _extract_candidate_answers(completion)
    if not candidates:
        return None

    has_correct = any(corr and (cand == corr or corr in cand) for cand in candidates)
    has_wrong = any(wr and wr != corr and (cand == wr or wr in cand) for cand in candidates)
    if has_correct and not has_wrong:
        return True
    if has_wrong and not has_correct:
        return False
    return None


def qwen_judge(completion: str, correct: str, wrong: str, client: QwenClient) -> bool:
    prompt = (
        "You are judging whether a model answer matches the correct reference.\n"
        f"Correct answer: {correct}\n"
        f"Incorrect alternative: {wrong}\n"
        f"Model answer: {completion}\n\n"
        "Reply with a single word: CORRECT or INCORRECT."
    )
    out = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=10,
    )
    return "correct" in (out or "").lower() and "incorrect" not in (out or "").lower()


def summary_keys(row: Dict) -> List[str]:
    keys = [f"dataset/{row['dataset']}"]
    if row["dataset"] == "bigtom":
        keys.append(f"bigtom/{row['task']}")
        keys.append(f"bigtom/{row['task']}/{row['condition']}")
        keys.append(f"bigtom/{row['task']}/{row['condition']}/init{row['init_belief']}")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "sft", "grpo"], required=True)
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stage1_ckpt", type=str,
                    default="projects/bigtom/checkpoints/stage1/epoch_2")
    ap.add_argument("--policy_ckpt", type=str, default=None,
                    help="Stage 2 SFT or Stage 3 GRPO checkpoint dir (has policy_lora/ + projector.pt)")
    ap.add_argument("--bigtom_csv", type=str,
                    default="third_party/src/bigtom/data/bigtom/bigtom.csv")
    ap.add_argument("--tomi_path", type=str, default="")
    ap.add_argument("--fantom_path", type=str, default="")
    ap.add_argument("--limit", type=int, default=None, help="max rows per dataset")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--use_qwen_judge", action="store_true")
    args = ap.parse_args()

    ngpus = torch.cuda.device_count()
    enc_device = torch.device("cuda:0")
    policy_device = torch.device(f"cuda:{1 if ngpus > 1 else 0}")
    print(f"encoder={enc_device}  policy={policy_device}  mode={args.mode}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    encoder = None
    projector = None
    if args.mode in ("sft", "grpo"):
        encoder = load_stage1_encoder(
            args.base_model, Path(args.stage1_ckpt), args.z_dim, enc_device,
        )
        policy = load_lora_policy(args.base_model, Path(args.policy_ckpt), policy_device)
        projector = MentalPrefixProjector(
            z_dim=args.z_dim, hidden_size=policy.config.hidden_size, num_prefix=MENTAL_PREFIX_LEN,
        ).to(policy_device).float()
        proj_pt = Path(args.policy_ckpt) / "projector.pt"
        projector.load_state_dict(torch.load(proj_pt, map_location=policy_device)["projector"])
        projector.eval()
    else:
        policy = load_base_policy(args.base_model, policy_device)

    all_rows: List[Dict] = []
    all_rows.extend(load_bigtom_eval(Path(args.bigtom_csv)))
    if args.tomi_path:
        all_rows.extend(load_jsonl_eval(Path(args.tomi_path), "tomi"))
    if args.fantom_path:
        all_rows.extend(load_jsonl_eval(Path(args.fantom_path), "fantom"))

    if args.limit is not None:
        by_ds: Dict[str, List[Dict]] = defaultdict(list)
        for row in all_rows:
            by_ds[row["dataset"]].append(row)
        limited = []
        for rows in by_ds.values():
            limited.extend(rows[: args.limit])
        all_rows = limited
    print(f"Total evaluation rows: {len(all_rows)}")

    judge = QwenClient() if args.use_qwen_judge else None

    os.makedirs(Path(args.out).parent, exist_ok=True)
    summary = defaultdict(lambda: [0, 0])
    with open(args.out, "w") as fout:
        for i, row in enumerate(all_rows):
            prompt = build_prompt(row["story"], row["question"], row.get("task"))

            mental = None
            if projector is not None and encoder is not None:
                ctx = build_encoder_context(row["story"], row["question"])
                ctx_enc = tok(ctx, return_tensors="pt", truncation=True, max_length=768)
                with torch.no_grad():
                    mu1, mu2 = encoder.encode_z1_z2_deterministic(
                        ctx_enc["input_ids"].to(enc_device),
                        ctx_enc["attention_mask"].to(enc_device),
                    )
                z1 = mu1.to(policy_device, dtype=torch.float32)
                z2 = mu2.to(policy_device, dtype=torch.float32)
                mental = projector(z1, z2).to(torch.bfloat16)

            completion = generate_answer(
                policy, tok, policy_device, prompt,
                mental_prefix_embeds=mental,
                max_new_tokens=args.max_new_tokens,
            )

            match = rule_match(completion, row["option_correct"], row.get("option_wrong", ""))
            if match is None and judge is not None:
                match = qwen_judge(completion, row["option_correct"], row.get("option_wrong", ""), judge)
            if match is None:
                match = False

            entry = dict(row)
            entry["completion"] = completion
            entry["correct"] = bool(match)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()

            for key in summary_keys(row):
                summary[key][0] += int(match)
                summary[key][1] += 1

            if (i + 1) % 20 == 0:
                running = " ".join(
                    f"{k}={v[0]/v[1]:.3f}({v[1]})"
                    for k, v in summary.items()
                    if k.startswith("dataset/")
                )
                print(f"  [{i + 1}/{len(all_rows)}] {running}", flush=True)

    print("\n=== FINAL ===")
    for key in sorted(summary):
        correct, total = summary[key]
        print(f"  {key:45s} {correct:4d}/{total:<4d} acc={correct / total:.4f}")


if __name__ == "__main__":
    main()
