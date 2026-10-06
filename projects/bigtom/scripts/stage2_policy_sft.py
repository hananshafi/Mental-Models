"""
BigToM Stage 2 — z-conditioned QA SFT.

Loads the frozen recursive ToM encoder from Stage 1, encodes each prompt into
(z1, z2), projects both into a learned mental soft-prompt prefix, and fine-tunes
a separate LoRA adapter on the same base LM to answer BigToM questions.

Supported supervised tasks from the annotated JSONL:
    - forward_action
    - forward_belief
    - backward_belief

For each task, we train both init_belief=0 and init_belief=1 prompt variants so
the policy matches the official BigToM condition structure used at evaluation.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

from stage1_train_mental_reward import (
    RecursiveToMModel,
    build_encoder_context,
    build_task_context,
    get_transformer_from_peft,
)


MENTAL_PREFIX_LEN = 16  # total slots; split 8/8 between z1 and z2

TASK_LABELS = {
    "forward_action": "forward action prediction",
    "forward_belief": "forward belief prediction",
    "backward_belief": "backward belief inference",
}

SYSTEM_PROMPT = (
    "You answer questions about a short story and the character's beliefs. "
    "Return only the answer to the question."
)


# ──────────────────────────────────────────────────────────────────────────────
# Mental prefix projector
# ──────────────────────────────────────────────────────────────────────────────
class MentalPrefixProjector(nn.Module):
    """Map (z1, z2) -> [N_prefix, hidden_size] soft-prompt tokens."""

    def __init__(self, z_dim: int, hidden_size: int, num_prefix: int = MENTAL_PREFIX_LEN):
        super().__init__()
        assert num_prefix % 2 == 0
        self.num_prefix = num_prefix
        self.half = num_prefix // 2
        self.z1_to_prefix = nn.Linear(z_dim, self.half * hidden_size)
        self.z2_to_prefix = nn.Linear(z_dim, self.half * hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        for m in (self.z1_to_prefix, self.z2_to_prefix):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        hidden_size = self.z1_to_prefix.out_features // self.half
        p1 = self.z1_to_prefix(z1).view(batch_size, self.half, hidden_size)
        p2 = self.z2_to_prefix(z2).view(batch_size, self.half, hidden_size)
        return self.norm(torch.cat([p1, p2], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Frozen encoder loader
# ──────────────────────────────────────────────────────────────────────────────
def load_stage1_encoder(
    base_model_name: str, stage1_ckpt: Path, z_dim: int, device: torch.device,
) -> RecursiveToMModel:
    print(f"Loading stage-1 encoder base: {base_model_name}")
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    base.config.pad_token_id = tok.pad_token_id

    # The Stage 1 heads were trained on LoRA-adapted hidden states; loading them
    # on the plain base model silently yields a mismatched encoder and reward.
    lora_dir = stage1_ckpt / "lora"
    if not (lora_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Stage 1 checkpoint {stage1_ckpt} has no LoRA adapter at {lora_dir}. "
            "Retrain Stage 1 with the current stage1_train_mental_reward.py, which "
            "saves lora/ next to heads.pt."
        )
    base = PeftModel.from_pretrained(base, str(lora_dir), is_trainable=False)

    model = RecursiveToMModel(base, z_dim=z_dim).to(device)
    heads_pt = stage1_ckpt / "heads.pt"
    ckpt = torch.load(heads_pt, map_location=device)
    state_dict = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    backbone_prefixes = ("base_model.", "transformer.")
    missing_heads = [k for k in missing if not k.startswith(backbone_prefixes)]
    if missing_heads or unexpected:
        raise RuntimeError(
            f"Stage 1 heads in {heads_pt} do not match the model: "
            f"missing={missing_heads[:10]} unexpected={unexpected[:10]}"
        )
    print(f"  loaded stage1 LoRA from {lora_dir} and {len(state_dict)} head tensors")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(context: str, question: str, task: Optional[str] = None) -> str:
    task_line = ""
    if task:
        task_line = f"Task: {TASK_LABELS.get(task, task)}\n"
    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"{task_line}"
        f"Story: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_target(answer: str) -> str:
    return f" {answer.strip()}"


def iter_training_examples(row: Dict[str, str]) -> Iterable[Dict[str, str]]:
    for init_belief in (0, 1):
        forward_context = build_task_context(
            row["story"], init_belief, percept=row["percept"],
        )
        backward_context = build_task_context(
            row["story"], init_belief, action=row["gold_action"],
        )
        yield {
            "task": "forward_action",
            "context": forward_context,
            "question": row["action_question"],
            "answer": row["gold_action"],
        }
        yield {
            "task": "forward_belief",
            "context": forward_context,
            "question": row["belief_question"],
            "answer": row["gold_belief"],
        }
        yield {
            "task": "backward_belief",
            "context": backward_context,
            "question": row["belief_question"],
            "answer": row["gold_belief"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class BigToMSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_ctx_len: int = 768, max_total_len: int = 1280):
        self.tok = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_total_len = max_total_len
        self.samples = []
        task_counts = {k: 0 for k in TASK_LABELS}

        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if not (row.get("gold_action") and row.get("gold_belief")):
                    continue
                for ex in iter_training_examples(row):
                    self.samples.append(ex)
                    task_counts[ex["task"]] += 1

        print(f"Loaded {len(self.samples)} SFT samples from {path}")
        for task, count in task_counts.items():
            print(f"  {task}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        encoder_text = build_encoder_context(sample["context"], sample["question"])
        prompt_text = build_prompt(sample["context"], sample["question"], sample["task"])
        target_text = build_target(sample["answer"])

        ctx = self.tok(
            encoder_text, truncation=True, max_length=self.max_ctx_len,
            return_tensors="pt", add_special_tokens=True,
        )

        prompt_ids = self.tok(prompt_text, add_special_tokens=True)["input_ids"]
        full = self.tok(
            prompt_text + target_text,
            truncation=True, max_length=self.max_total_len,
            return_tensors="pt", add_special_tokens=True,
        )
        input_ids = full["input_ids"][0]
        attn_mask = full["attention_mask"][0]
        labels = input_ids.clone()
        cutoff = min(len(prompt_ids), input_ids.size(0))
        labels[:cutoff] = -100
        labels[attn_mask == 0] = -100

        return {
            "ctx_ids": ctx["input_ids"][0],
            "ctx_mask": ctx["attention_mask"][0],
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


def collate(batch, pad_id: int):
    def pad(seqs, pad_val):
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, : seq.size(0)] = seq
        return out

    return {
        "ctx_ids": pad([b["ctx_ids"] for b in batch], pad_id),
        "ctx_mask": pad([b["ctx_mask"] for b in batch], 0),
        "input_ids": pad([b["input_ids"] for b in batch], pad_id),
        "attention_mask": pad([b["attention_mask"] for b in batch], 0),
        "labels": pad([b["labels"] for b in batch], -100),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Policy forward with mental prefix
# ──────────────────────────────────────────────────────────────────────────────
def policy_forward_with_prefix(
    policy,
    mental_prefix_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
):
    """Prepend the mental prefix to the policy's input embeddings."""
    embed = policy.get_input_embeddings()
    tok_embeds = embed(input_ids)
    full_embeds = torch.cat([mental_prefix_embeds, tok_embeds], dim=1)

    batch_size, prefix_len, _ = mental_prefix_embeds.shape
    prefix_mask = torch.ones(batch_size, prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    prefix_labels = torch.full((batch_size, prefix_len), -100, device=labels.device, dtype=labels.dtype)
    full_labels = torch.cat([prefix_labels, labels], dim=1)

    out = policy(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        labels=full_labels,
        use_cache=False,
    )
    return out.loss


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default="projects/bigtom/data/bigtom_qwen_5k_annotated.jsonl")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stage1_ckpt", type=str,
                    default="projects/bigtom/checkpoints/stage1/epoch_2")
    ap.add_argument("--out", type=str,
                    default="projects/bigtom/checkpoints/stage2")
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--projector_lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_ctx_len", type=int, default=768)
    ap.add_argument("--max_total_len", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    encoder_device = torch.device("cuda:0")
    policy_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"Encoder on {encoder_device}, policy on {policy_device}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    encoder = load_stage1_encoder(
        args.base_model, Path(args.stage1_ckpt), args.z_dim, encoder_device,
    )
    hidden_size = encoder.hidden_size

    print(f"Loading policy base: {args.base_model}")
    policy_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": policy_device},
    )
    policy_base.config.pad_token_id = tok.pad_token_id
    policy_base.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    )
    policy = get_peft_model(policy_base, lora_cfg)
    policy.print_trainable_parameters()

    projector = MentalPrefixProjector(
        z_dim=args.z_dim, hidden_size=hidden_size, num_prefix=MENTAL_PREFIX_LEN,
    ).to(policy_device).float()

    dataset = BigToMSFTDataset(
        args.data, tok, max_ctx_len=args.max_ctx_len, max_total_len=args.max_total_len,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate(batch, pad_id=tok.pad_token_id),
        num_workers=2, drop_last=True,
    )

    policy_params = [p for p in policy.parameters() if p.requires_grad]
    projector_params = list(projector.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_params, "lr": args.lr},
            {"params": projector_params, "lr": args.projector_lr},
        ],
        weight_decay=0.01,
    )
    total_steps = max(1, (len(loader) // args.grad_accum) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps,
    )

    os.makedirs(args.out, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        policy.train()
        projector.train()
        running_loss = 0.0
        running_n = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            with torch.no_grad():
                mu1, mu2 = encoder.encode_z1_z2_deterministic(
                    batch["ctx_ids"].to(encoder_device),
                    batch["ctx_mask"].to(encoder_device),
                )
            z1 = mu1.to(policy_device, dtype=torch.float32)
            z2 = mu2.to(policy_device, dtype=torch.float32)
            mental_prefix = projector(z1, z2).to(torch.bfloat16)

            input_ids = batch["input_ids"].to(policy_device)
            attn = batch["attention_mask"].to(policy_device)
            labels = batch["labels"].to(policy_device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = policy_forward_with_prefix(
                    policy, mental_prefix, input_ids, attn, labels,
                )

            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            running_n += 1

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_params + projector_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (i + 1) % (args.grad_accum * 20) == 0:
                print(
                    f"[epoch {epoch} step {global_step}] "
                    f"sft_loss={running_loss / max(1, running_n):.4f}",
                    flush=True,
                )
                running_loss = 0.0
                running_n = 0

        ckpt = Path(args.out) / f"epoch_{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(ckpt / "policy_lora")
        tok.save_pretrained(ckpt / "policy_lora")
        torch.save(
            {"projector": projector.state_dict(), "args": vars(args)},
            ckpt / "projector.pt",
        )
        print(f"Saved epoch {epoch} to {ckpt}")


if __name__ == "__main__":
    main()
