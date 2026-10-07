#!/usr/bin/env python3
"""
Stage 1: Visual ToM SFT for LLaVA-NeXT-Mistral-7B
=================================================
Fine-tunes a LLaVA-NeXT-Mistral backbone with LoRA on the chain-of-belief format:
  <perception> → <belief_1st> → <belief_2nd> → <response>

Uses belief_prediction.jsonl + salience_prediction.jsonl training data,
with optional auxiliary salience validation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python stage1_sft_visual_tom_llava_next_mistral.py \
        --base_model llava-hf/llava-v1.6-mistral-7b-hf \
        --output_dir projects/mmrole/checkpoints/stage1_sft_llava_next_mistral
"""

import os
import sys
import json
import gc
import time
import argparse
import random
from typing import Optional, List, Dict

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "fileno"):
    try:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    except Exception:
        pass

print(">> stage1_sft_visual_tom_llava_next_mistral.py starting...", flush=True)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from PIL import Image

# Shared model utilities (supports Qwen2.5-VL + Qwen-VL-Chat)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    detect_model_type, load_base_model, get_tokenizer,
    resolve_image, load_and_resize_image,
    build_full_chat_text, build_prompt_only_text,
    tokenize_batch, compute_prompt_length, create_labels,
)
from resume_state import (
    check_resume_args, epoch_order, load_lora_weights, load_resume_state,
    require_resume_dir, restore_rng_state, save_resume_dir,
)

# Arguments that must match between an interrupted run and its resumption.
RESUME_INVARIANT_ARGS = (
    "base_model", "model_type", "train_path", "val_path", "salience_train_path",
    "image_dir", "num_epochs", "batch_size", "grad_accum", "lr", "max_len",
    "max_examples", "lora_rank", "lora_alpha", "seed",
)

print(">> All imports done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chain-of-Belief Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_chain_of_belief(example: dict) -> str:
    """Convert belief_prediction example into chain-of-belief target text.

    Format:
        <perception>scene objects and visual salience</perception>
        <belief_1st>speaker's 1st-order beliefs about partner</belief_1st>
        <belief_2nd>speaker's 2nd-order beliefs</belief_2nd>
        <response>the actual utterance</response>
    """
    # Perception: scene objects with per-agent salience
    scene_objects = example.get("target_scene_objects", [])
    perception_parts = []
    for obj in scene_objects[:7]:  # cap at 7 objects
        line = f"- {obj['object']}: {obj.get('description', '')}"
        if obj.get("salience_speaker"):
            line += f" [speaker_salience={obj['salience_speaker']}]"
        if obj.get("salience_partner"):
            line += f" [partner_salience={obj['salience_partner']}]"
        perception_parts.append(line)
    perception = "\n".join(perception_parts)

    # 1st-order beliefs
    belief_1st_data = example.get("target_speaker_belief", {})
    belief_1st_parts = []
    for key in ["partner_visual_focus", "partner_intent", "partner_knowledge", "partner_emotion"]:
        val = belief_1st_data.get(key, "")
        if val:
            label = key.replace("partner_", "").replace("_", " ").title()
            belief_1st_parts.append(f"- {label}: {val}")
    belief_1st = "\n".join(belief_1st_parts)

    # 2nd-order beliefs
    belief_2nd_data = example.get("target_speaker_2nd_order", {})
    belief_2nd_parts = []
    for key in ["partner_thinks_i_see", "partner_thinks_i_want", "partner_thinks_i_know"]:
        val = belief_2nd_data.get(key, "")
        if val:
            label = key.replace("partner_thinks_i_", "Partner thinks I ").replace("_", " ")
            belief_2nd_parts.append(f"- {label}: {val}")
    belief_2nd = "\n".join(belief_2nd_parts)

    # Response: the current utterance
    response = example.get("current_utterance", "")

    target = (
        f"<perception>\n{perception}\n</perception>\n"
        f"<belief_1st>\n{belief_1st}\n</belief_1st>\n"
        f"<belief_2nd>\n{belief_2nd}\n</belief_2nd>\n"
        f"<response>\n{response}\n</response>"
    )
    return target


def build_sft_prompt(example: dict) -> str:
    """Build the input prompt for SFT (without the target)."""
    speaker = example.get("speaker_name", "Speaker")
    partner = example.get("partner_name", "Partner")
    speaker_profile = example.get("speaker_profile", "")[:1500]
    partner_profile = example.get("partner_profile", "")[:1500]

    history_lines = []
    for t in example.get("dialogue_history", []):
        history_lines.append(f"[{t['speaker']}]: {t['utterance']}")
    history_text = "\n".join(history_lines[-6:])  # last 6 turns

    prompt = (
        f"You are {speaker}, engaging in a conversation with {partner} about an image.\n\n"
        f"## {speaker}'s Profile\n{speaker_profile}\n\n"
        f"## {partner}'s Profile\n{partner_profile}\n\n"
    )
    if history_text:
        prompt += f"## Dialogue History\n{history_text}\n\n"

    prompt += (
        f"Analyze the image from {speaker}'s perspective and generate a structured "
        f"chain-of-belief response. First describe what you perceive in the scene and "
        f"each character's visual salience. Then reason about your 1st-order beliefs "
        f"(what you think {partner} sees, intends, knows, and feels) and 2nd-order "
        f"beliefs (what you think {partner} thinks about you). Finally, provide your "
        f"in-character response.\n\n"
        f"Use the format:\n"
        f"<perception>...</perception>\n"
        f"<belief_1st>...</belief_1st>\n"
        f"<belief_2nd>...</belief_2nd>\n"
        f"<response>...</response>"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class VisualToMSFTDataset(Dataset):
    """Dataset for Visual ToM SFT training with a multimodal chat backbone."""

    def __init__(self, belief_path: str, image_dir: str,
                 salience_path: Optional[str] = None,
                 max_examples: int = -1):
        self.image_dir = image_dir
        self.examples = []

        # Load belief prediction data
        print(f"Loading belief data from {belief_path}...", flush=True)
        with open(belief_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        if max_examples > 0:
            self.examples = self.examples[:max_examples]

        # Optionally load salience data to augment (10% of examples get salience-only task)
        self.salience_examples = []
        if salience_path and os.path.exists(salience_path):
            print(f"Loading salience data from {salience_path}...", flush=True)
            with open(salience_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.salience_examples.append(json.loads(line))
            # Sample 10% of salience examples as auxiliary task
            n_aux = max(1, len(self.salience_examples) // 10)
            random.shuffle(self.salience_examples)
            self.salience_examples = self.salience_examples[:n_aux]

        print(f"  Belief examples: {len(self.examples)}", flush=True)
        print(f"  Salience auxiliary: {len(self.salience_examples)}", flush=True)

    def __len__(self):
        return len(self.examples) + len(self.salience_examples)

    def __getitem__(self, idx):
        if idx < len(self.examples):
            return self._get_belief_item(idx)
        else:
            return self._get_salience_item(idx - len(self.examples))

    def _get_belief_item(self, idx):
        example = self.examples[idx]
        prompt = build_sft_prompt(example)
        target = format_chain_of_belief(example)
        image_path = resolve_image(example, self.image_dir)

        return {
            "task": "belief",
            "prompt": prompt,
            "target": target,
            "image_path": image_path,
            "example_id": example.get("example_id", ""),
        }

    def _get_salience_item(self, idx):
        example = self.salience_examples[idx]
        agent = example.get("agent_name", "Agent")
        agent_profile = example.get("agent_profile", "")[:1000]

        prompt = (
            f"You are {agent}. Look at the image and describe what objects/elements "
            f"you notice and how salient each is to you given your background.\n\n"
            f"## {agent}'s Profile\n{agent_profile}\n\n"
            f"List each object with its salience level (high/medium/low/none) and why."
        )

        salience_list = example.get("target_salience", [])
        target_parts = []
        for obj in salience_list:
            line = (f"- {obj['object']} [{obj.get('salience', 'medium')}]: "
                    f"{obj.get('reason', '')}")
            target_parts.append(line)
        target = "\n".join(target_parts)

        image_path = resolve_image(example, self.image_dir)

        return {
            "task": "salience",
            "prompt": prompt,
            "target": target,
            "image_path": image_path,
            "example_id": example.get("example_id", ""),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (model-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def create_collate_fn(processor_or_tokenizer, model_type, max_len=2048):
    """Create a collate function for multimodal chat backbones."""

    def collate_fn(batch):
        texts = []
        images = []
        prompt_lengths = []

        for item in batch:
            prompt = item["prompt"]
            target = item["target"]
            img_path = item["image_path"]

            # Load image
            img = None
            if img_path and os.path.exists(img_path):
                img = load_and_resize_image(img_path)
            images.append(img)

            # Build full chat text (prompt + response)
            # For qwen-vl-chat, pass image_path so <img> tags are embedded
            # For qwen2.5-vl, image is passed separately via processor
            effective_img = img_path if (img is not None) else None
            text = build_full_chat_text(
                prompt, target, effective_img,
                processor_or_tokenizer, model_type
            )
            texts.append(text)

            # Compute prompt length for label masking
            plen = compute_prompt_length(
                prompt, processor_or_tokenizer, model_type, max_len, image=img,
            )
            prompt_lengths.append(plen)

        # Tokenize batch
        inputs = tokenize_batch(texts, images, processor_or_tokenizer, model_type, max_len)

        # Create labels (mask prompt tokens + padding)
        inputs["labels"] = create_labels(
            inputs["input_ids"], inputs["attention_mask"], prompt_lengths
        )

        return inputs

    return collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # Setup device
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Device: {device}, GPUs: {torch.cuda.device_count()}", flush=True)

    # Detect model type
    model_type = args.model_type or detect_model_type(args.base_model)
    print(f"Model type: {model_type}", flush=True)

    # Load model + processor/tokenizer
    model, processor, model_type = load_base_model(args.base_model, model_type)
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus > 1:
        print(
            f"Visible GPUs: {visible_gpus}. This trainer runs in a single process, "
            f"so the full backbone will stay on {device} and will not use "
            "device_map='auto'.",
            flush=True,
        )
        print(
            "  Avoiding auto-sharding because multimodal forwards can mix tensors "
            "across devices in this SFT path.",
            flush=True,
        )
    model = model.to(device)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing: enabled with use_reentrant=False", flush=True)
    except TypeError:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing: enabled with default settings", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Apply LoRA
    print("Applying LoRA...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = VisualToMSFTDataset(
        args.train_path, args.image_dir,
        salience_path=args.salience_train_path,
        max_examples=args.max_examples,
    )
    val_dataset = None
    if args.val_path and os.path.exists(args.val_path):
        val_dataset = VisualToMSFTDataset(
            args.val_path, args.image_dir,
            salience_path=args.salience_val_path,
            max_examples=min(500, args.max_examples) if args.max_examples > 0 else 500,
        )

    collate_fn = create_collate_fn(processor, model_type, max_len=args.max_len)

    def make_train_loader(indices):
        # Deterministic per-epoch order (epoch_order) makes mid-epoch resume exact.
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=indices,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator(),
        )

    batches_per_epoch = len(train_dataset) // args.batch_size
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
            generator=torch.Generator(),
        )

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = (batches_per_epoch * args.num_epochs) // args.grad_accum
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 1: Visual ToM SFT", flush=True)
    print(f"  Base model: {args.base_model}", flush=True)
    print(f"  Train examples: {len(train_dataset)}", flush=True)
    print(f"  Val examples: {len(val_dataset) if val_dataset else 0}", flush=True)
    print(f"  Epochs: {args.num_epochs}", flush=True)
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} accum", flush=True)
    print(f"  LR: {args.lr}, LoRA r={args.lora_rank}", flush=True)
    print(f"  Total steps: {total_steps}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["total_steps"] = total_steps
    config["train_size"] = len(train_dataset)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training
    best_val_loss = float("inf")
    global_step = 0
    start_epoch, start_batch, resumed_epoch_loss = 0, 0, (0.0, 0)
    if args.resume:
        resume_dir = require_resume_dir(args.output_dir)
        state = load_resume_state(resume_dir)
        check_resume_args(state["args"], args, RESUME_INVARIANT_ARGS)
        load_lora_weights(model, os.path.join(resume_dir, "adapter"))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step, best_val_loss = state["global_step"], state["best_val_loss"]
        start_epoch, start_batch = state["epoch"], state["next_batch"]
        resumed_epoch_loss = tuple(state["epoch_loss"])
        restore_rng_state(state["rng"])
        print(f"Resumed from {resume_dir}: epoch={start_epoch} batch={start_batch} "
              f"step={global_step}", flush=True)

    def save_resume(epoch, next_batch, epoch_loss_state):
        save_resume_dir(args.output_dir, lambda d: model.save_pretrained(os.path.join(d, "adapter")), {
            "args": config, "epoch": epoch, "next_batch": next_batch,
            "global_step": global_step, "epoch_loss": epoch_loss_state,
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        })

    start_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        first_batch = start_batch if epoch == start_epoch else 0
        epoch_loss, epoch_steps = resumed_epoch_loss if epoch == start_epoch else (0.0, 0)
        train_loader = make_train_loader(
            epoch_order(len(train_dataset), args.seed, epoch)[first_batch * args.batch_size:]
        )

        for batch_idx, batch in enumerate(train_loader, start=first_batch):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            loss.backward()

            stepped = (batch_idx + 1) % args.grad_accum == 0
            if stepped:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            epoch_steps += 1

            if (batch_idx + 1) % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{args.num_epochs} "
                    f"Step {batch_idx+1}/{batches_per_epoch} "
                    f"(global {global_step}/{total_steps}) "
                    f"loss={avg_loss:.4f} lr={lr_now:.2e} "
                    f"time={elapsed:.0f}s",
                    flush=True,
                )

            # Save checkpoint periodically
            if args.save_every > 0 and global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                print(f"  Saving checkpoint at step {global_step}...", flush=True)
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
            if stepped and args.resume_every > 0 and global_step % args.resume_every == 0:
                save_resume(epoch, batch_idx + 1, (epoch_loss, epoch_steps))

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch+1} done: avg_loss={avg_epoch_loss:.4f}", flush=True)

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            print(f"  Validation loss: {avg_val_loss:.4f}", flush=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_dir = os.path.join(args.output_dir, "best")
                print(f"  New best! Saving to {best_dir}", flush=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)

        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)
        save_resume(epoch + 1, 0, (0.0, 0))

    # Final save
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 1 SFT Complete!", flush=True)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nNext: Stage 2 GRPO", flush=True)
    print(f"  python stage2_grpo_tom_reward.py \\", flush=True)
    print(f"    --sft_checkpoint {os.path.join(args.output_dir, 'best')} \\", flush=True)
    print(f"    --base_model {args.base_model}", flush=True)

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Visual ToM SFT")
    # Model
    parser.add_argument("--base_model", type=str,
                        default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat", "llava-next"],
                        help="Model type (auto-detected from base_model if empty)")
    # Data
    parser.add_argument("--train_path", type=str,
                        default="projects/mmrole/training_data/train/belief_prediction.jsonl")
    parser.add_argument("--val_path", type=str,
                        default="projects/mmrole/training_data/val/belief_prediction.jsonl")
    parser.add_argument("--salience_train_path", type=str,
                        default="projects/mmrole/training_data/train/salience_prediction.jsonl")
    parser.add_argument("--salience_val_path", type=str,
                        default="projects/mmrole/training_data/val/salience_prediction.jsonl")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_examples", type=int, default=-1)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Output
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/checkpoints/stage1_sft_llava_next_mistral")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save checkpoint every N global steps (0=disable)")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <output_dir>/last after an interruption.")
    parser.add_argument("--resume_every", type=int, default=50,
                        help="Refresh <output_dir>/last every N optimizer steps (and every epoch).")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
