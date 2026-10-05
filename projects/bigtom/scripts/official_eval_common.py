import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage1_train_mental_reward import build_encoder_context
from stage3_policy_sft import MENTAL_PREFIX_LEN, MentalPrefixProjector, load_stage1_encoder


DEFAULT_SYSTEM_PROMPT = (
    "You are answering a theory-of-mind benchmark question. "
    "Return only the answer, with no explanation unless the benchmark explicitly asks for one."
)

FANTOM_HEADER = (
    "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, "
    "based on the following in-person conversation between individuals who have just met.\n\n"
)


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token, count in ref_counts.items():
        overlap += min(count, pred_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def build_benchmark_prompt(
    story: str,
    question: str,
    *,
    dataset_name: str,
    choices: Optional[List[str]] = None,
    extra_instruction: str = "",
) -> str:
    prompt = (
        f"System: {DEFAULT_SYSTEM_PROMPT}\n"
        f"Dataset: {dataset_name}\n"
        f"Story: {story.strip()}\n"
        f"Question: {question.strip()}\n"
    )
    if choices:
        prompt += "Choices:\n"
        for idx, choice in enumerate(choices):
            prompt += f"{chr(ord('A') + idx)}. {choice}\n"
        prompt += "Respond with the best answer from the choices.\n"
    if extra_instruction:
        prompt += f"{extra_instruction.strip()}\n"
    prompt += "Answer:"
    return prompt


@dataclass
class PolicyBundle:
    mode: str
    tokenizer: AutoTokenizer
    policy: torch.nn.Module
    policy_device: torch.device
    encoder: Optional[torch.nn.Module] = None
    encoder_device: Optional[torch.device] = None
    projector: Optional[MentalPrefixProjector] = None
    z_dim: int = 128

    def build_mental_prefix(self, story: str, question: str) -> Optional[torch.Tensor]:
        if self.encoder is None or self.projector is None or self.encoder_device is None:
            return None
        context = build_encoder_context(story, question)
        encoded = self.tokenizer(
            context, return_tensors="pt", truncation=True, max_length=768,
        )
        with torch.no_grad():
            mu1, mu2 = self.encoder.encode_z1_z2_deterministic(
                encoded["input_ids"].to(self.encoder_device),
                encoded["attention_mask"].to(self.encoder_device),
            )
        z1 = mu1.to(self.policy_device, dtype=torch.float32)
        z2 = mu2.to(self.policy_device, dtype=torch.float32)
        return self.projector(z1, z2).to(torch.bfloat16)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        story: str,
        question: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        )["input_ids"].to(self.policy_device)
        mental = self.build_mental_prefix(story, question)

        if mental is None:
            output = self.policy.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
            new_tokens = output[0, prompt_ids.size(1):]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        embed = self.policy.get_input_embeddings()
        tok_embeds = embed(prompt_ids)
        full_embeds = torch.cat([mental, tok_embeds], dim=1)
        attention_mask = torch.ones(full_embeds.size()[:2], dtype=torch.long, device=self.policy_device)
        output = self.policy.generate(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _sequence_logprob(
        self,
        prompt: str,
        completion: str,
        *,
        story: str,
        question: str,
    ) -> float:
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=True,
        )["input_ids"][0].to(self.policy_device)
        completion_ids = self.tokenizer(
            " " + completion,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(self.policy_device)
        mental = self.build_mental_prefix(story, question)

        if mental is None:
            full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
            with torch.no_grad():
                out = self.policy(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=False,
                )
            logits = out.logits[0]
            start = prompt_ids.size(0) - 1
        else:
            embed = self.policy.get_input_embeddings()
            full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
            tok_embeds = embed(full_ids)
            full_embeds = torch.cat([mental.to(tok_embeds.dtype), tok_embeds], dim=1)
            full_mask = torch.ones(full_embeds.size()[:2], dtype=torch.long, device=self.policy_device)
            with torch.no_grad():
                out = self.policy(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
            logits = out.logits[0]
            start = mental.size(1) + prompt_ids.size(0) - 1

        end = start + completion_ids.size(0)
        completion_logits = logits[start:end]
        log_probs = F.log_softmax(completion_logits.float(), dim=-1)
        selected = log_probs.gather(1, completion_ids.unsqueeze(-1)).squeeze(-1)
        return selected.mean().item()

    def score_choices(
        self,
        prompt: str,
        choices: List[str],
        *,
        story: str,
        question: str,
    ) -> List[float]:
        return [
            self._sequence_logprob(prompt, choice, story=story, question=question)
            for choice in choices
        ]

    def pick_choice(
        self,
        prompt: str,
        choices: List[str],
        *,
        story: str,
        question: str,
    ) -> Dict[str, object]:
        scores = self.score_choices(prompt, choices, story=story, question=question)
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        return {
            "choice_index": best_idx,
            "choice_text": choices[best_idx],
            "scores": scores,
        }


def load_policy_bundle(
    *,
    mode: str,
    base_model: str,
    stage1_ckpt: Optional[str],
    policy_ckpt: Optional[str],
    z_dim: int = 128,
) -> PolicyBundle:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ngpus = torch.cuda.device_count()
    encoder_device = torch.device("cuda:0")
    policy_device = torch.device(f"cuda:{1 if ngpus > 1 else 0}")

    if mode == "base":
        policy = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map={"": policy_device},
        )
        policy.eval()
        return PolicyBundle(
            mode=mode,
            tokenizer=tokenizer,
            policy=policy,
            policy_device=policy_device,
            z_dim=z_dim,
        )

    if not stage1_ckpt or not policy_ckpt:
        raise ValueError("stage1_ckpt and policy_ckpt are required for non-base modes.")

    encoder = load_stage1_encoder(
        base_model, Path(stage1_ckpt), z_dim, encoder_device,
    )
    policy_base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": policy_device},
    )
    policy_lora_dir = Path(policy_ckpt) / "policy_lora"
    if not policy_lora_dir.exists():
        raise FileNotFoundError(f"{policy_lora_dir} not found")
    policy = PeftModel.from_pretrained(policy_base, str(policy_lora_dir), is_trainable=False)
    policy.eval()

    projector = MentalPrefixProjector(
        z_dim=z_dim,
        hidden_size=policy.config.hidden_size,
        num_prefix=MENTAL_PREFIX_LEN,
    ).to(policy_device).float()
    projector_pt = Path(policy_ckpt) / "projector.pt"
    if not projector_pt.exists():
        raise FileNotFoundError(f"{projector_pt} not found")
    projector.load_state_dict(torch.load(projector_pt, map_location=policy_device)["projector"])
    projector.eval()

    return PolicyBundle(
        mode=mode,
        tokenizer=tokenizer,
        policy=policy,
        policy_device=policy_device,
        encoder=encoder,
        encoder_device=encoder_device,
        projector=projector,
        z_dim=z_dim,
    )
