# agents/grpo_policy_agent.py
"""
GRPO-trained policy agent for SOTOPIA benchmark evaluation.
Uses a locally-loaded Qwen2.5-7B-Instruct + LoRA adapter to generate actions.
"""
from __future__ import annotations
import json
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sotopia.agents.llm_agent import LLMAgent
from sotopia.database import AgentProfile
from sotopia.messages import AgentAction, Observation


# Shared singleton to avoid loading model per agent instance
_POLICY_MODEL = None
_POLICY_TOKENIZER = None
_POLICY_DEVICE = None


def load_policy_model(
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: str | None = None,
    device: str = "cuda:0",
    merge: bool = True,
) -> None:
    """Load the policy model once as a global singleton."""
    global _POLICY_MODEL, _POLICY_TOKENIZER, _POLICY_DEVICE
    if _POLICY_MODEL is not None:
        return  # already loaded

    _POLICY_DEVICE = torch.device(device)
    print(f"[GRPOPolicyAgent] Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        print(f"[GRPOPolicyAgent] Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            model, adapter_path, torch_dtype=torch.bfloat16,
        )
        if merge:
            model = model.merge_and_unload()
            print("[GRPOPolicyAgent] LoRA merged.")
    model = model.to(_POLICY_DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _POLICY_MODEL = model
    _POLICY_TOKENIZER = tokenizer
    print(f"[GRPOPolicyAgent] Model loaded on {_POLICY_DEVICE}")


class GRPOPolicyAgent(LLMAgent):
    """
    A SOTOPIA LLMAgent that generates actions using a locally-loaded
    GRPO-trained policy model instead of calling an external LLM API.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "grpo-policy",
        max_gen_len: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ):
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            model_name=model_name,
        )
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

    def _build_prompt(self, obs: Observation) -> str:
        """Build the action prompt matching SOTOPIA's agenerate_action template."""
        history = "\n".join(
            f"{y.to_natural_language()}" for _, y in self.inbox
        )
        action_list = " ".join(obs.available_actions)

        prompt = (
            f"Imagine you are {self.agent_name}, your task is to act/speak as "
            f"{self.agent_name} would, keeping in mind {self.agent_name}'s social goal.\n"
            f"You can find {self.agent_name}'s goal (or background) in the "
            f"'Here is the context of the interaction' field.\n"
            f"Note that {self.agent_name}'s goal is only visible to you.\n"
            f"You should try your best to achieve {self.agent_name}'s goal in a way "
            f"that align with their character traits.\n"
            f"Additionally, maintaining the conversation's naturalness and realism is "
            f"essential (e.g., do not repeat what other people has already said before).\n"
            f"{history}.\n"
            f"You are at Turn #{obs.turn_number}. Your available action types are\n"
            f"{action_list}.\n"
            f'Note: You can "leave" this conversation if 1. you have achieved your '
            f"social goals, 2. this conversation makes you uncomfortable, 3. you find "
            f"it uninteresting/you lose your patience, 4. or for other reasons you "
            f"want to leave.\n\n"
            f"Please only generate a JSON string including the action type and the argument.\n"
            f"Your action should follow the given format:\n"
            f'{{"action_type": "action_type_here", "argument": "your utterance here"}}'
        )
        return prompt

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Generate a response using the local policy model."""
        assert _POLICY_MODEL is not None, (
            "Policy model not loaded. Call load_policy_model() first."
        )
        enc = _POLICY_TOKENIZER(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        enc = {k: v.to(_POLICY_DEVICE) for k, v in enc.items()}

        outputs = _POLICY_MODEL.generate(
            **enc,
            max_new_tokens=self.max_gen_len,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=1,
            pad_token_id=_POLICY_TOKENIZER.pad_token_id,
        )
        response_ids = outputs[0][enc["input_ids"].shape[1]:]
        return _POLICY_TOKENIZER.decode(response_ids, skip_special_tokens=True).strip()

    def _parse_action(self, text: str, available_actions: list[str]) -> AgentAction:
        """Parse model output into an AgentAction."""
        text = text.strip()

        # Try JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                action_type = data.get("action_type", "speak")
                argument = data.get("argument", "")
                if action_type in available_actions:
                    return AgentAction(action_type=action_type, argument=argument)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        if '{"action_type"' in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                action_type = data.get("action_type", "speak")
                argument = data.get("argument", "")
                if action_type in available_actions:
                    return AgentAction(action_type=action_type, argument=argument)
            except (ValueError, json.JSONDecodeError):
                pass

        # Truncate at turn boundaries
        for stop in ["\nTurn", "\nImagine you", "\nHere is"]:
            if stop in text:
                text = text[:text.index(stop)].strip()

        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        text = text[:500]

        if not text or text.lower() in ("none", "did nothing", "..."):
            return AgentAction(action_type="none", argument="")

        if "leave" in text.lower()[:20]:
            return AgentAction(action_type="leave", argument=text)

        return AgentAction(action_type="speak", argument=text)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        # Set goal from background (same as LLMAgent)
        if self._goal is None:
            # Extract goal from the first observation's background
            bg_text = self.inbox[0][1].to_natural_language()
            # Simple extraction: use the full background as goal context
            self._goal = bg_text

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")

        prompt = self._build_prompt(obs)
        raw_output = self._generate(prompt)
        action = self._parse_action(raw_output, obs.available_actions)
        return action
