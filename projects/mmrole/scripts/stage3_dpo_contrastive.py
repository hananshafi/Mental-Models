#!/usr/bin/env python3
"""
Stage 3: Contrastive DPO on ToM Preference Pairs
===================================================
Direct Preference Optimization on ~50K contrastive preference pairs.
Each pair has (preferred, rejected) where rejected violates ToM in one of:
  - visual: wrong visual grounding
  - belief: incorrect 1st-order beliefs
  - order2: incorrect 2nd-order beliefs
  - no_tom: generic response without any ToM reasoning

Starts from Stage 2 GRPO checkpoint (LoRA).

Supported models:
  - Qwen/Qwen2.5-VL-7B-Instruct  (model_type: qwen2.5-vl, default)
  - Qwen/Qwen-VL-Chat             (model_type: qwen-vl-chat)

Usage:
    # Qwen2.5-VL
    CUDA_VISIBLE_DEVICES=0,1 python stage3_dpo_contrastive.py \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --grpo_checkpoint projects/mmrole/checkpoints/stage2_grpo/best

    # Qwen-VL-Chat
    CUDA_VISIBLE_DEVICES=0 python stage3_dpo_contrastive.py \
        --base_model Qwen/Qwen-VL-Chat \
        --grpo_checkpoint projects/mmrole/checkpoints/stage2_grpo_qwenvl/best
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

print(">> stage3_dpo_contrastive.py starting...", flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model
from PIL import Image

# Shared model utilities (supports Qwen2.5-VL + Qwen-VL-Chat)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    detect_model_type, load_base_model, get_tokenizer,
    resolve_image, load_and_resize_image,
    build_full_chat_text, compute_prompt_length,
    tokenize_batch, create_labels, default_lora_target_modules,
)
from resume_state import (
    check_resume_args, epoch_order, load_lora_weights, load_resume_state,
    require_resume_dir, restore_rng_state, save_resume_dir,
)

# Arguments that must match between an interrupted run and its resumption.
RESUME_INVARIANT_ARGS = (
    "base_model", "model_type", "grpo_checkpoint", "sft_checkpoint", "train_path",
    "val_path", "image_dir", "violation_types", "max_examples", "beta",
    "label_smoothing", "num_epochs", "batch_size", "grad_accum", "lr", "max_len",
    "lora_rank", "lora_alpha", "seed",
)

print(">> All imports done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# DPO Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DPOPreferenceDataset(Dataset):
    """Dataset for DPO training on ToM preference pairs."""

    def __init__(self, data_path: str, image_dir: str,
                 violation_types: Optional[List[str]] = None,
                 max_examples: int = -1):
        self.image_dir = image_dir
        self.examples = []

        print(f"Loading DPO data from {data_path}...", flush=True)
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                # Filter by violation type if specified
                if violation_types:
                    if example.get("violation_type", "") not in violation_types:
                        continue
                self.examples.append(example)

        if max_examples > 0:
            random.shuffle(self.examples)
            self.examples = self.examples[:max_examples]

        # Count violation types
        vtype_counts = {}
        for ex in self.examples:
            vt = ex.get("violation_type", "unknown")
            vtype_counts[vt] = vtype_counts.get(vt, 0) + 1

        print(f"  DPO examples: {len(self.examples)}", flush=True)
        print(f"  Violation types: {vtype_counts}", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_dpo_prompt(example: dict) -> str:
    """Build the role-play prompt for DPO."""
    speaker = example.get("speaker_name", "Speaker")
    partner = example.get("partner_name", "Partner")
    speaker_profile = example.get("speaker_profile", "")[:1500]
    partner_profile = example.get("partner_profile", "")[:1500]

    history_lines = []
    for t in example.get("dialogue_history", []):
        history_lines.append(f"[{t['speaker']}]: {t['utterance']}")
    history_text = "\n".join(history_lines[-6:])

    prompt = (
        f"You are {speaker}, talking with {partner} about an image.\n\n"
        f"## {speaker}'s Profile\n{speaker_profile}\n\n"
        f"## {partner}'s Profile\n{partner_profile}\n\n"
    )
    if history_text:
        prompt += f"## Dialogue History\n{history_text}\n\n"

    prompt += (
        f"Respond as {speaker} to {partner}'s most recent message about the image, "
        f"staying in character and demonstrating understanding of {partner}'s "
        f"perspective and beliefs."
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# DPO Collate + Loss
# ──────────────────────────────────────────────────────────────────────────────

def create_dpo_collate_fn(processor_or_tokenizer, model_type: str,
                          image_dir: str, max_len: int = 1536):
    """Create collate function for DPO: processes (prompt, chosen, rejected) triples."""

    def collate_fn(batch):
        prompts = []
        chosen_texts = []
        rejected_texts = []
        images = []
        image_paths = []

        for item in batch:
            prompt = build_dpo_prompt(item)
            chosen = item.get("preferred_response", "")
            rejected = item.get("rejected_response", "")
            img_path = resolve_image(item, image_dir)

            prompts.append(prompt)
            chosen_texts.append(chosen)
            rejected_texts.append(rejected)
            image_paths.append(img_path)

            if img_path and os.path.exists(img_path):
                images.append(load_and_resize_image(img_path))
            else:
                images.append(None)

        # Process chosen and rejected separately
        chosen_inputs = _process_pairs(
            processor_or_tokenizer, model_type,
            prompts, chosen_texts, images, image_paths, max_len
        )
        rejected_inputs = _process_pairs(
            processor_or_tokenizer, model_type,
            prompts, rejected_texts, images, image_paths, max_len
        )

        # Store violation types for per-type logging
        violation_types = [item.get("violation_type", "unknown") for item in batch]

        return {
            "chosen": chosen_inputs,
            "rejected": rejected_inputs,
            "violation_types": violation_types,
        }

    return collate_fn


def _process_pairs(processor_or_tokenizer, model_type, prompts, responses,
                   images, image_paths, max_len):
    """Process prompt+response pairs into model inputs with labels."""
    texts = []
    prompt_lengths = []

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # For qwen-vl-chat, need image_path for <img> tags; for qwen2.5-vl, need image object
        effective_img = image_paths[i] if images[i] is not None else None
        text = build_full_chat_text(
            prompt, response, effective_img,
            processor_or_tokenizer, model_type
        )
        texts.append(text)
        prompt_lengths.append(
            compute_prompt_length(
                prompt, processor_or_tokenizer, model_type, max_len,
                image=images[i],
            )
        )

    inputs = tokenize_batch(texts, images, processor_or_tokenizer, model_type, max_len)
    inputs["labels"] = create_labels(
        inputs["input_ids"], inputs["attention_mask"], prompt_lengths
    )

    return inputs


def compute_dpo_log_probs(model, inputs, device):
    """Compute per-sequence log probs of the response tokens."""
    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in inputs:
            model_inputs[key] = inputs[key].to(device)
    labels = inputs["labels"].to(device)

    with torch.amp.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**model_inputs)
        logits = outputs.logits

    # Only score response-token positions. Prompt/padding tokens are masked with
    # -100 in labels, so building a full [B, L, V] log_softmax tensor wastes a
    # large amount of memory on rows that never contribute to DPO.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    seq_log_probs = []
    for i in range(shift_logits.shape[0]):
        valid_mask = shift_labels[i] != -100
        if not valid_mask.any():
            seq_log_probs.append(torch.zeros((), device=device, dtype=torch.float32))
            continue

        selected_logits = shift_logits[i][valid_mask].float()
        selected_labels = shift_labels[i][valid_mask]
        vocab_size = selected_logits.size(-1)
        if selected_labels.numel() > 0:
            min_id = int(selected_labels.min().item())
            max_id = int(selected_labels.max().item())
            if min_id < 0 or max_id >= vocab_size:
                raise RuntimeError(
                    f"DPO labels out of range for vocab_size={vocab_size}: "
                    f"min_id={min_id}, max_id={max_id}"
                )
        selected_logits = torch.nan_to_num(
            selected_logits, nan=0.0, posinf=50.0, neginf=-50.0
        ).clamp_(-50.0, 50.0)
        token_log_probs = F.log_softmax(selected_logits, dim=-1).gather(
            1, selected_labels.unsqueeze(1)
        ).squeeze(1)
        token_log_probs = torch.nan_to_num(
            token_log_probs, nan=-50.0, posinf=0.0, neginf=-50.0
        )
        seq_log_probs.append(token_log_probs.mean())

    del outputs, logits, shift_logits
    return torch.stack(seq_log_probs, dim=0)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta: float = 0.1, label_smoothing: float = 0.0):
    """
    Compute DPO loss.

    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    """
    chosen_logratios = torch.nan_to_num(
        policy_chosen_logps - ref_chosen_logps,
        nan=0.0, posinf=50.0, neginf=-50.0,
    )
    rejected_logratios = torch.nan_to_num(
        policy_rejected_logps - ref_rejected_logps,
        nan=0.0, posinf=50.0, neginf=-50.0,
    )

    logits = (beta * (chosen_logratios - rejected_logratios)).clamp(-50.0, 50.0)

    if label_smoothing > 0:
        # Label-smoothed DPO
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    # Reward metrics
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    reward_margins = (chosen_rewards - rejected_rewards).mean()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()

    loss = torch.nan_to_num(losses.mean(), nan=0.0, posinf=50.0, neginf=50.0)

    return loss, {
        "reward_margin": reward_margins.item(),
        "reward_accuracy": reward_accuracies.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {n_gpus}", flush=True)

    ref_device = f"cuda:{n_gpus - 1}" if n_gpus > 1 else "cuda:0"

    # Detect model type
    model_type = args.model_type or detect_model_type(args.base_model)
    print(f"Model type: {model_type}", flush=True)

    # Load model + processor/tokenizer
    base_model, processor, model_type = load_base_model(args.base_model, model_type)

    if args.grpo_checkpoint and os.path.exists(args.grpo_checkpoint):
        print(f"Loading GRPO LoRA from {args.grpo_checkpoint}...", flush=True)
        policy_model = PeftModel.from_pretrained(
            base_model, args.grpo_checkpoint, is_trainable=True,
        )
        print("  GRPO LoRA loaded (trainable).", flush=True)
    elif args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        print(f"Loading SFT LoRA from {args.sft_checkpoint}...", flush=True)
        policy_model = PeftModel.from_pretrained(
            base_model, args.sft_checkpoint, is_trainable=True,
        )
        print("  SFT LoRA loaded (trainable).", flush=True)
    else:
        print("WARNING: No checkpoint provided, using base model with fresh LoRA.", flush=True)
        target_modules = default_lora_target_modules(model_type)
        print(f"LoRA target modules: {target_modules}", flush=True)
        lora_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_model, lora_config)

    policy_model = policy_model.to(device)
    try:
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  Policy gradient checkpointing: enabled with use_reentrant=False",
              flush=True)
    except TypeError:
        policy_model.gradient_checkpointing_enable()
        print("  Policy gradient checkpointing: enabled with default settings",
              flush=True)
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
    if hasattr(policy_model, "config"):
        policy_model.config.use_cache = False
    policy_model.print_trainable_parameters()

    # Load reference model (frozen, from same checkpoint as policy started from)
    #
    # Keep the reference as a frozen PeftModel instead of merging the LoRA into
    # the base weights. In Stage 2, merging the Qwen2.5-VL LoRA introduced
    # measurable bf16 logit drift versus the live PEFT path; keeping policy/ref
    # on the same PEFT execution path avoids that mismatch.
    print(f"Loading reference model...", flush=True)
    ref_base, _, _ = load_base_model(args.base_model, model_type)
    ckpt_for_ref = args.grpo_checkpoint or args.sft_checkpoint
    if ckpt_for_ref and os.path.exists(ckpt_for_ref):
        ref_model = PeftModel.from_pretrained(ref_base, ckpt_for_ref)
    else:
        ref_model = ref_base
    ref_model = ref_model.to(ref_device)
    ref_model.eval()
    if hasattr(ref_model, "config"):
        ref_model.config.use_cache = False
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  Reference model on {ref_device}", flush=True)

    # Datasets
    violation_types = args.violation_types.split(",") if args.violation_types else None
    train_dataset = DPOPreferenceDataset(
        args.train_path, args.image_dir,
        violation_types=violation_types,
        max_examples=args.max_examples,
    )
    val_dataset = None
    if args.val_path and os.path.exists(args.val_path):
        val_dataset = DPOPreferenceDataset(
            args.val_path, args.image_dir,
            violation_types=violation_types,
            max_examples=min(1000, args.max_examples) if args.max_examples > 0 else 1000,
        )

    collate_fn = create_dpo_collate_fn(
        processor, model_type, args.image_dir, max_len=args.max_len
    )

    def make_train_loader(indices):
        # Deterministic per-epoch order (epoch_order) makes mid-epoch resume exact.
        return DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=indices,
            collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
            generator=torch.Generator(),
        )

    batches_per_epoch = len(train_dataset) // args.batch_size
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=1,
            generator=torch.Generator(),
        )

    # Optimizer + scheduler
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (batches_per_epoch * args.num_epochs) // args.grad_accum
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 3: Contrastive DPO", flush=True)
    print(f"  Base model: {args.base_model}", flush=True)
    print(f"  GRPO checkpoint: {args.grpo_checkpoint}", flush=True)
    print(f"  Train pairs: {len(train_dataset)}", flush=True)
    print(f"  Val pairs: {len(val_dataset) if val_dataset else 0}", flush=True)
    print(f"  Epochs: {args.num_epochs}", flush=True)
    print(f"  Batch: {args.batch_size} x {args.grad_accum} accum", flush=True)
    print(f"  LR: {args.lr}, Beta: {args.beta}", flush=True)
    print(f"  Total steps: {total_steps}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args)
    config["total_steps"] = total_steps
    config["train_size"] = len(train_dataset)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training
    best_val_accuracy = 0.0
    global_step = 0
    start_epoch, start_batch, resumed_sums = 0, 0, None
    if args.resume:
        resume_dir = require_resume_dir(args.output_dir)
        state = load_resume_state(resume_dir)
        check_resume_args(state["args"], args, RESUME_INVARIANT_ARGS)
        load_lora_weights(policy_model, os.path.join(resume_dir, "policy"))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step, best_val_accuracy = state["global_step"], state["best_val_accuracy"]
        start_epoch, start_batch, resumed_sums = state["epoch"], state["next_batch"], state["sums"]
        restore_rng_state(state["rng"])
        print(f"Resumed from {resume_dir}: epoch={start_epoch} batch={start_batch} "
              f"step={global_step}", flush=True)

    def save_resume(epoch, next_batch, sums):
        save_resume_dir(args.output_dir, lambda d: policy_model.save_pretrained(os.path.join(d, "policy")), {
            "args": config, "epoch": epoch, "next_batch": next_batch,
            "global_step": global_step, "best_val_accuracy": best_val_accuracy, "sums": sums,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        })

    start_time = time.time()
    log_file = open(os.path.join(args.output_dir, "training_log.jsonl"), "a")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.num_epochs):
        policy_model.train()
        first_batch = start_batch if epoch == start_epoch else 0
        sums = resumed_sums if epoch == start_epoch and resumed_sums else None
        epoch_loss = sums["loss"] if sums else 0.0
        epoch_accuracy = sums["accuracy"] if sums else 0.0
        epoch_margin = sums["margin"] if sums else 0.0
        epoch_steps = sums["steps"] if sums else 0
        skipped_nonfinite = sums["skipped_nonfinite"] if sums else 0

        # Per-violation-type tracking
        vtype_metrics = sums["vtype_metrics"] if sums else {}
        train_loader = make_train_loader(
            epoch_order(len(train_dataset), args.seed, epoch)[first_batch * args.batch_size:]
        )

        for batch_idx, batch in enumerate(train_loader, start=first_batch):
            chosen_inputs = batch["chosen"]
            rejected_inputs = batch["rejected"]
            vtypes = batch["violation_types"]

            # Policy log probs
            policy_chosen_logps = compute_dpo_log_probs(policy_model, chosen_inputs, device)
            policy_rejected_logps = compute_dpo_log_probs(policy_model, rejected_inputs, device)

            # Reference log probs
            with torch.no_grad():
                ref_chosen_logps = compute_dpo_log_probs(ref_model, chosen_inputs, ref_device)
                ref_rejected_logps = compute_dpo_log_probs(ref_model, rejected_inputs, ref_device)

            # Move ref log probs to policy device
            ref_chosen_logps = ref_chosen_logps.to(device)
            ref_rejected_logps = ref_rejected_logps.to(device)

            finite_tensors = [
                ("policy_chosen_logps", policy_chosen_logps),
                ("policy_rejected_logps", policy_rejected_logps),
                ("ref_chosen_logps", ref_chosen_logps),
                ("ref_rejected_logps", ref_rejected_logps),
            ]
            bad_name = next(
                (name for name, tensor in finite_tensors if not torch.isfinite(tensor).all()),
                None,
            )
            if bad_name is not None:
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"  Warning: skipping non-finite DPO batch at "
                    f"epoch {epoch+1} step {batch_idx+1} ({bad_name})",
                    flush=True,
                )
                continue

            # DPO loss
            loss, metrics = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta, label_smoothing=args.label_smoothing,
            )
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"  Warning: skipping non-finite DPO loss at "
                    f"epoch {epoch+1} step {batch_idx+1}",
                    flush=True,
                )
                continue
            loss = loss / args.grad_accum
            loss.backward()

            stepped = (batch_idx + 1) % args.grad_accum == 0
            if stepped:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            epoch_accuracy += metrics["reward_accuracy"]
            epoch_margin += metrics["reward_margin"]
            epoch_steps += 1

            # Track per-violation-type
            for vt in vtypes:
                if vt not in vtype_metrics:
                    vtype_metrics[vt] = {"count": 0, "accuracy": 0.0}
                vtype_metrics[vt]["count"] += 1
                vtype_metrics[vt]["accuracy"] += metrics["reward_accuracy"]

            if (batch_idx + 1) % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_acc = epoch_accuracy / epoch_steps
                avg_margin = epoch_margin / epoch_steps
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{args.num_epochs} "
                    f"Step {batch_idx+1}/{batches_per_epoch} "
                    f"(global {global_step}/{total_steps}) "
                    f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                    f"margin={avg_margin:.3f} lr={lr_now:.2e} "
                    f"time={elapsed:.0f}s",
                    flush=True,
                )

            # Save checkpoint
            if args.save_every > 0 and global_step > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                print(f"  Saving checkpoint at step {global_step}...", flush=True)
                policy_model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
            if stepped and args.resume_every > 0 and global_step % args.resume_every == 0:
                save_resume(epoch, batch_idx + 1, {
                    "loss": epoch_loss, "accuracy": epoch_accuracy, "margin": epoch_margin,
                    "steps": epoch_steps, "skipped_nonfinite": skipped_nonfinite,
                    "vtype_metrics": vtype_metrics,
                })

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        avg_epoch_acc = epoch_accuracy / max(epoch_steps, 1)
        avg_epoch_margin = epoch_margin / max(epoch_steps, 1)

        print(f"\n  Epoch {epoch+1} done: loss={avg_epoch_loss:.4f} "
              f"acc={avg_epoch_acc:.3f} margin={avg_epoch_margin:.3f}", flush=True)
        if skipped_nonfinite:
            print(f"  Skipped non-finite batches: {skipped_nonfinite}", flush=True)

        # Per-violation-type breakdown
        print(f"  Per-violation accuracy:", flush=True)
        for vt, vm in sorted(vtype_metrics.items()):
            vt_acc = vm["accuracy"] / max(vm["count"], 1)
            print(f"    {vt}: {vt_acc:.3f} (n={vm['count']})", flush=True)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "train_accuracy": avg_epoch_acc,
            "train_margin": avg_epoch_margin,
            "per_vtype": {vt: vm["accuracy"] / max(vm["count"], 1)
                          for vt, vm in vtype_metrics.items()},
        }

        # Validation
        if val_loader:
            policy_model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    chosen_inputs = batch["chosen"]
                    rejected_inputs = batch["rejected"]

                    policy_chosen_logps = compute_dpo_log_probs(
                        policy_model, chosen_inputs, device
                    )
                    policy_rejected_logps = compute_dpo_log_probs(
                        policy_model, rejected_inputs, device
                    )
                    ref_chosen_logps = compute_dpo_log_probs(
                        ref_model, chosen_inputs, ref_device
                    ).to(device)
                    ref_rejected_logps = compute_dpo_log_probs(
                        ref_model, rejected_inputs, ref_device
                    ).to(device)

                    loss, metrics = dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps,
                        beta=args.beta,
                    )
                    val_loss += loss.item()
                    val_accuracy += metrics["reward_accuracy"]
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            avg_val_acc = val_accuracy / max(val_steps, 1)
            print(f"  Validation: loss={avg_val_loss:.4f} acc={avg_val_acc:.3f}", flush=True)

            log_entry["val_loss"] = avg_val_loss
            log_entry["val_accuracy"] = avg_val_acc

            if avg_val_acc > best_val_accuracy:
                best_val_accuracy = avg_val_acc
                best_dir = os.path.join(args.output_dir, "best")
                print(f"  New best val accuracy! Saving to {best_dir}", flush=True)
                policy_model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)

        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        policy_model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)
        save_resume(epoch + 1, 0, None)

    log_file.close()

    # Final save
    final_dir = os.path.join(args.output_dir, "final")
    policy_model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 3 DPO Complete!", flush=True)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Best val accuracy: {best_val_accuracy:.3f}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nNext: Evaluate on official test set:", flush=True)
    print(f"  python step6_evaluate_tom.py response \\", flush=True)
    print(f"    --responses_path <generate_responses_from_stage3> \\", flush=True)
    print(f"    --annotations_path projects/mmrole/mmrole_official_test_annotated_clean.jsonl",
          flush=True)

    del policy_model, ref_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Contrastive DPO")
    # Model
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_type", type=str, default="",
                        choices=["", "qwen2.5-vl", "qwen-vl-chat"],
                        help="Model type (auto-detected from base_model if empty)")
    parser.add_argument("--grpo_checkpoint", type=str,
                        default="projects/mmrole/checkpoints/stage2_grpo/best")
    parser.add_argument("--sft_checkpoint", type=str, default="",
                        help="Fallback: SFT checkpoint if GRPO not available")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    # Data
    parser.add_argument("--train_path", type=str,
                        default="projects/mmrole/training_data/train/preference_pairs.jsonl")
    parser.add_argument("--val_path", type=str,
                        default="projects/mmrole/training_data/val/preference_pairs.jsonl")
    parser.add_argument("--image_dir", type=str,
                        default="projects/mmrole/images")
    parser.add_argument("--violation_types", type=str, default="",
                        help="Comma-separated violation types to train on (empty=all)")
    parser.add_argument("--max_examples", type=int, default=-1)
    # DPO
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (inverse temperature)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=1536)
    # Output
    parser.add_argument("--output_dir", type=str,
                        default="projects/mmrole/checkpoints/stage3_dpo")
    parser.add_argument("--save_every", type=int, default=200)
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
