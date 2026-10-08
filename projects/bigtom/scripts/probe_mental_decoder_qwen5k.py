"""
Quick generation test for the bigtom stage1 mental decoder (autoregressive).

Loads Qwen2.5-7B-Instruct + LoRA + heads from
projects/bigtom/checkpoints/stage1_qwen/best_ckpt
and decodes mental1 / mental2 for a handful of BigToM scenarios via greedy.

Usage:
    python probe_mental_decoder_qwen5k.py [--n 6] [--max_new_tokens 48]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stage1_train_mental_reward import (  # noqa: E402
    RecursiveToMModel,
    build_task_context,
    build_encoder_context,
)

DEFAULT_CKPT = REPO / "checkpoints/stage1_qwen/best_ckpt"
DATA = REPO / "data/bigtom_qwen_5k_annotated.jsonl"


def load_model(device: str, ckpt_dir: Path):
    print(f"Loading base + LoRA + heads from {ckpt_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    base.config.pad_token_id = tok.pad_token_id

    lora_dir = ckpt_dir / "lora"
    if lora_dir.exists():
        base = PeftModel.from_pretrained(base, str(lora_dir))
    else:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none",
        )
        base = get_peft_model(base, lora)

    model = RecursiveToMModel(base, z_dim=128).to(device)
    for name, p in model.named_parameters():
        if not name.startswith("base_model.") and not name.startswith("transformer."):
            p.data = p.data.float()

    ck = torch.load(ckpt_dir / "heads.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    head_missing = [k for k in missing if not (k.startswith("base_model.") or k.startswith("transformer."))]
    if head_missing:
        print(f"WARNING: missing head keys: {head_missing[:5]} (total={len(head_missing)})")
    if unexpected:
        print(f"NOTE: unexpected keys (non-fatal): {unexpected[:3]} (total={len(unexpected)})")
    bm = ck.get('best_metric')
    print(f"checkpoint best_metric={bm if bm is None else f'{bm:.4f}'} step={ck.get('global_step')} "
          f"avg_metrics m1={ck['avg_metrics']['m1']:.3f} m2={ck['avg_metrics']['m2']:.3f} "
          f"kl1={ck['avg_metrics'].get('kl1',0):.2f} kl2={ck['avg_metrics'].get('kl2',0):.3f}",
          flush=True)
    model.eval()
    return model, tok


@torch.no_grad()
def generate_mental(model, tok, z, decoder_bundle, max_new_tokens: int = 48) -> str:
    """
    Greedy AR decode mirroring training's _decode_mental exactly:
      memory = cat(prefix, embeds_of_content_so_far)
      out[P-1+i] predicts content_token_i from prefix + content_{<i}
    No BOS prepended — training never saw one at the prefix boundary.
    """
    device = z.device
    num_prefix = int(decoder_bundle.num_prefix.item())
    prefix = decoder_bundle["z_to_prefix"](
        z.to(decoder_bundle["z_to_prefix"].weight.dtype)
    ).view(1, num_prefix, model.hidden_size)

    embed_layer = model.base_model.get_input_embeddings()
    lm_weight = embed_layer.weight
    content_ids: list[int] = []

    for _ in range(max_new_tokens):
        if not content_ids:
            memory = prefix
        else:
            content_t = torch.tensor([content_ids], dtype=torch.long, device=device)
            emb_content = embed_layer(content_t).to(prefix.dtype)
            memory = torch.cat([prefix, emb_content], dim=1)
        tgt_len = memory.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1,
        )
        out = decoder_bundle["decoder"](tgt=memory, memory=memory, tgt_mask=causal_mask)
        out = decoder_bundle["out_norm"](out)
        last = out[:, -1].to(lm_weight.dtype)
        logits = F.linear(last, lm_weight)
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id == tok.eos_token_id:
            break
        content_ids.append(next_id)

    if not content_ids:
        return ""
    text = tok.decode(content_ids, skip_special_tokens=True)
    return text.strip()


def detect_repetition(text: str) -> bool:
    toks = text.split()
    if len(toks) < 6:
        return False
    n3 = len(toks) - 2
    trigrams = [tuple(toks[i:i + 3]) for i in range(n3)]
    return len(set(trigrams)) <= max(2, n3 // 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT),
                    help="Path to checkpoint dir containing heads.pt and lora/")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out", type=str, default=None,
                    help="Output markdown path; defaults to <ckpt>/decoder_probe.md")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = args.out or str(ckpt_path / "decoder_probe.md")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(device, ckpt_path)

    by_sid: dict = {}
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r
    paired_sids = sorted(s for s, d in by_sid.items() if "aware" in d and "not_aware" in d)
    if paired_sids:
        step = max(1, len(paired_sids) // args.n)
        paired_sids = paired_sids[::step][: args.n]

    out_lines = ["# BigToM stage1 mental decoder probe", ""]
    n_repeating = 0
    n_total = 0

    for sid in paired_sids:
        for cond in ("aware", "not_aware"):
            row = by_sid[sid][cond]
            ctx = build_task_context(row["story"], 0, percept=row["percept"])
            ctx = build_encoder_context(ctx, row["belief_question"])

            enc = tok(ctx, truncation=True, max_length=768, return_tensors="pt").to(device)
            mu1, mu2 = model.encode_z1_z2_deterministic(enc.input_ids, enc.attention_mask)

            m1_text = generate_mental(model, tok, mu1, model.mental1_decoder, args.max_new_tokens)
            m2_text = generate_mental(model, tok, mu2, model.mental2_decoder, args.max_new_tokens)

            n_total += 2
            if detect_repetition(m1_text):
                n_repeating += 1
            if detect_repetition(m2_text):
                n_repeating += 1

            out_lines += [
                f"## sid={sid} condition={cond}",
                f"- **gold first-order:** `{row['first_order_belief']}`",
                f"- **decoded mental1:** `{m1_text}`",
                f"- **gold second-order:** `{row['second_order_belief']}`",
                f"- **decoded mental2:** `{m2_text}`",
                "",
            ]
            print(f"[sid={sid}/{cond}] m1: {m1_text[:120]}", flush=True)
            print(f"[sid={sid}/{cond}] m2: {m2_text[:120]}", flush=True)

    summary = (
        f"\n## Summary\n"
        f"- Decodings: {n_total}, with degenerate repetition heuristic flagging "
        f"{n_repeating}/{n_total} = {n_repeating / max(1, n_total):.0%}.\n"
        f"- avg_metrics from ckpt: m1=0.748, m2=0.700 (CE on held data).\n"
    )
    out_lines.append(summary)
    Path(out_path).write_text("\n".join(out_lines))
    print(summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
