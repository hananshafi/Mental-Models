"""
Stage 3: SOTOPIA Evaluation (Official Framework Integration)
=============================================================
Evaluates the GRPO-trained agent on official SOTOPIA episodes using the
official SOTOPIA framework (ParallelSotopiaEnv, EpisodeLLMEvaluator, etc.)
with local storage backend (no Redis server needed).

The trained policy model is wrapped in a custom GRPOAgent that extends
SOTOPIA's BaseAgent, ensuring full compatibility with the official pipeline.

Evaluation uses GPT-4o as judge with SotopiaDimensions (7 dimensions):
  believability, relationship, knowledge, secret, social_rules,
  financial_and_material_benefits, goal
"""

# OPENAI_API_KEY="sk-..." CUDA_VISIBLE_DEVICES=0 python projects/sotopia/scripts/stage3_evaluate_sotopia.py \
#   --policy_model_name Qwen/Qwen2.5-7B-Instruct \
#   --policy_adapter_path projects/sotopia/checkpoints/grpo_agent_qwen_v3/best \
#   --merge_adapter \
#   --use_hf \
#   --deduplicate_envs \
#   --output_path projects/sotopia/runs/evaluation/qwen_grpo.jsonl \
#   --max_turns 10 \
#   --policy_agent_index 0 \
#   --partner_model gpt-4o-mini \
#   --judge_model gpt-4o \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --seed 42
#   --task hard

# task - all, hard, cooperative, competitive


# OPENAI_API_KEY="sk-..." CUDA_VISIBLE_DEVICES=0 python projects/sotopia/scripts/stage3_evaluate_sotopia.py \
#   --policy_model_name Qwen/Qwen2.5-7B-Instruct \
#   --policy_adapter_path projects/sotopia/checkpoints/grpo_agent_qwen_v3/best \
#   --merge_adapter \
#   --use_hf \
#   --deduplicate_envs \
#   --task hard \
#   --output_path projects/sotopia/runs/evaluation/qwen_grpo_hard.jsonl \
#   --max_turns 10 \
#   --seed 42



# === Qwen SFT ===
# CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY="sk-..." python projects/sotopia/scripts/stage3_evaluate_sotopia.py \
#   --policy_model_name Qwen/Qwen2.5-7B-Instruct \
#   --policy_adapter_path projects/sotopia/checkpoints/grpo_agent_qwen_v3/sft_warmup \
#   --merge_adapter \
#   --use_hf \
#   --deduplicate_envs \
#   --task all \
#   --output_path projects/sotopia/runs/evaluation/qwen_sft.jsonl \
#   --max_turns 10 \
#   --partner_model gpt-4o-mini \
#   --judge_model gpt-4o \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --seed 42 --gpu 3


import os
import sys
import json
import argparse
import asyncio
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Set storage backend BEFORE importing sotopia
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

# Load .env from sotopia subdir if present
_env_path = Path(__file__).parent / "sotopia" / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                # Don't override env vars already set (e.g. OPENAI_API_KEY from CLI)
                # Skip OPENAI_BASE_URL from .env — it may point to a non-OpenAI
                # endpoint (e.g. Gemini) which conflicts with the actual OpenAI key.
                if _key == "OPENAI_BASE_URL":
                    continue
                os.environ.setdefault(_key, _val)

from sotopia.agents.base_agent import BaseAgent
from sotopia.agents.llm_agent import LLMAgent, Agents
from sotopia.database import (
    AgentProfile,
    EnvironmentProfile,
    EpisodeLog,
    SotopiaDimensions,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    RuleBasedTerminatedEvaluator,
)
from sotopia.generation_utils import agenerate, PydanticOutputParser
from sotopia.messages import AgentAction, Observation, SimpleMessage

# Ensure OPENAI_BASE_URL is not set AFTER all imports.
# litellm calls dotenv.load_dotenv() on import, which picks up the .env file
# containing a Gemini base URL. We must remove it after litellm has loaded.
os.environ.pop("OPENAI_BASE_URL", None)

SOTOPIA_DIMENSIONS = [
    "believability", "relationship", "knowledge", "secret",
    "social_rules", "financial_and_material_benefits", "goal",
]

# Hard environment IDs from the official SOTOPIA benchmark
# (EnvironmentList ID: 01HAK34YPB1H1RWXQDASDKHSNS)
HARD_ENV_IDS = {
    "01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R",
    "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY",
    "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T",
    "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM",
    "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0",
    "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK",
    "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4",
}

VALID_ACTION_TYPES = ("speak", "non-verbal communication", "action", "leave", "none")


def _message_to_text(message) -> str:
    return message.to_natural_language() if hasattr(message, "to_natural_language") else str(message)


def _profile_background_text(agent_profile: AgentProfile | None) -> str:
    if agent_profile is None:
        return "Not provided."

    parts = []
    age = getattr(agent_profile, "age", None)
    gender = getattr(agent_profile, "gender", "") or "person"
    occupation = getattr(agent_profile, "occupation", "") or ""
    intro_bits = [str(bit) for bit in [age, gender.lower(), occupation] if bit]
    if intro_bits:
        parts.append(" ".join(intro_bits))

    public_info = getattr(agent_profile, "public_info", "") or ""
    if public_info:
        parts.append(public_info)

    personality = getattr(agent_profile, "personality_and_values", "") or ""
    if personality:
        parts.append(f"Personality and values: {personality}")

    return ". ".join(part.strip().rstrip(".") for part in parts if part).strip() or "Not provided."


def _build_stage2_style_prompt(agent_name: str, agent_profile: AgentProfile | None,
                               goal: str, inbox, turn_number: int) -> str:
    context_lines = []
    history_lines = []

    for sender, message in inbox:
        text = _message_to_text(message).strip()
        if not text:
            continue
        if sender == "Environment" and not history_lines:
            context_lines.append(text)
            continue
        if sender != "Environment":
            history_lines.append(text)

    context_block = "\n".join(context_lines).strip() or "No environment context provided."
    history_block = "\n".join(history_lines).strip() or "No prior dialogue."
    background = _profile_background_text(agent_profile)
    secret = getattr(agent_profile, "secret", "") if agent_profile is not None else ""
    secret_text = secret if secret else "None"

    return (
        f"Imagine you are {agent_name}, your task is to act/speak as {agent_name} would, "
        f"keeping in mind {agent_name}'s social goal.\n"
        f"Here is the context of the interaction:\n"
        f"{context_block}\n"
        f"Background: {background}\n"
        f"Goal: {goal}\n"
        f"Secret: {secret_text}\n"
        f"Dialogue History:\n{history_block}\n"
        f"You are at Turn #{turn_number}.\n"
        f"Generate the next natural response for {agent_name}. "
        f"Stay in character and work towards your goal.\n"
        f"{agent_name}:"
    )


def _build_structured_action_prompt(agent_name: str, inbox, turn_number: int,
                                    available_actions) -> str:
    history = "\n".join(_message_to_text(message) for _, message in inbox)
    action_list = " ".join(available_actions)
    return (
        f"Imagine you are {agent_name}, your task is to act/speak as {agent_name} would, "
        f"keeping in mind {agent_name}'s social goal.\n"
        f"You can find {agent_name}'s goal (or background) in the "
        "'Here is the context of the interaction' field.\n"
        f"Note that {agent_name}'s goal is only visible to you.\n"
        f"You should try your best to achieve {agent_name}'s goal in a way that "
        "align with their character traits.\n"
        "Additionally, maintaining the conversation's naturalness and realism "
        "is essential (e.g., do not repeat what other people has already said before).\n"
        f"{history}.\n"
        f"You are at Turn #{turn_number}. Your available action types are\n"
        f"{action_list}.\n"
        'Note: You can "leave" this conversation if 1. you have achieved your '
        "social goals, 2. this conversation makes you uncomfortable, 3. you find "
        "it uninteresting/you lose your patience, 4. or for other reasons you want to leave.\n\n"
        "Please only generate a JSON string including the action type and the argument.\n"
        "Your action should follow the given format:\n"
        '{"action_type": "speak", "argument": "your utterance here"}'
    )


def _parse_model_text_to_action(text: str) -> tuple[str, str]:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            at = data.get("action_type", "speak")
            arg = data.get("argument", "")
            if at in VALID_ACTION_TYPES:
                return at, arg
    except json.JSONDecodeError:
        pass

    if '{"action_type"' in text:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            at = data.get("action_type", "speak")
            arg = data.get("argument", "")
            if at in VALID_ACTION_TYPES:
                return at, arg
        except (ValueError, json.JSONDecodeError):
            pass

    for stop in ["\nTurn", "\nImagine you", "\nHere is"]:
        if stop in text:
            text = text[:text.index(stop)].strip()

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    if not text or text.strip() in ("", "...", "none", "did nothing"):
        return "none", ""

    return "speak", text[:500]


def _extract_speak_argument(text: str) -> str:
    action_type, argument = _parse_model_text_to_action(text)
    if action_type == "speak":
        return argument[:500]
    return ""


# ------------------------------------------------------------------------------
# GRPO Policy Agent (wraps local model into SOTOPIA's BaseAgent interface)
# ------------------------------------------------------------------------------
class GRPOAgent(BaseAgent[Observation, AgentAction]):
    """SOTOPIA-compatible agent that uses the GRPO-trained policy model."""

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "grpo-policy",
        # These will be set via class-level shared state
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name

    # -- Class-level shared model (loaded once, shared across all instances) --
    _shared_model = None
    _shared_tokenizer = None
    _shared_device = "cuda"
    _max_gen_len = 256
    _temperature = 0.7
    _top_p = 0.9
    _model_tag = "policy"  # for distinguishing policy vs partner in logs

    @classmethod
    def load_model(cls, model_name: str, adapter_path: str | None = None,
                   merge_adapter: bool = True, device: str = "cuda",
                   max_gen_len: int = 256, temperature: float = 0.7,
                   top_p: float = 0.9):
        """Load the policy model once and share across all GRPOAgent instances."""
        print(f"Loading policy model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        )

        if adapter_path:
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(
                    f"Policy adapter {adapter_path} does not exist. "
                    "Pass --no_adapter to evaluate the base model."
                )
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(
                model, adapter_path, torch_dtype=torch.bfloat16,
            )
            if merge_adapter:
                model = model.merge_and_unload()
                print("LoRA adapter merged.")

        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        cls._shared_model = model
        cls._shared_tokenizer = tokenizer
        cls._shared_device = device
        cls._max_gen_len = max_gen_len
        cls._temperature = temperature
        cls._top_p = top_p
        print("Policy model loaded and ready.")

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        raise NotImplementedError("Use aact() instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        """Generate action using the local GRPO policy model."""
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])

        # Match Stage 2 training whenever "speak" is available: generate a plain
        # next utterance and wrap it as a speak action.
        if "speak" in obs.available_actions:
            prompt = _build_stage2_style_prompt(
                agent_name=self.agent_name,
                agent_profile=getattr(self, "profile", None),
                goal=self.goal,
                inbox=self.inbox,
                turn_number=obs.turn_number,
            )
            response_text = self._generate(prompt)
            argument = _extract_speak_argument(response_text)
            if argument:
                return AgentAction(action_type="speak", argument=argument, to=[])
            if "none" in obs.available_actions:
                return AgentAction(action_type="none", argument="", to=[])

        prompt = _build_structured_action_prompt(
            agent_name=self.agent_name,
            inbox=self.inbox,
            turn_number=obs.turn_number,
            available_actions=obs.available_actions,
        )
        response_text = self._generate(prompt)
        action_type, argument = self._parse_to_action(response_text)
        return AgentAction(action_type=action_type, argument=argument, to=[])

    def _generate(self, prompt: str) -> str:
        """Generate text from the shared policy model."""
        model = self.__class__._shared_model
        tokenizer = self.__class__._shared_tokenizer
        device = self.__class__._shared_device

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=self.__class__._max_gen_len,
                do_sample=True,
                temperature=self.__class__._temperature,
                top_p=self.__class__._top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def _parse_to_action(self, text: str) -> tuple[str, str]:
        """Parse model output into (action_type, argument)."""
        return _parse_model_text_to_action(text)


# ------------------------------------------------------------------------------
# Local Partner Agent (zero-shot local model as partner)
# ------------------------------------------------------------------------------
class LocalPartnerAgent(BaseAgent[Observation, AgentAction]):
    """SOTOPIA-compatible agent using a local model (zero-shot, no LoRA).

    Uses a SEPARATE class-level model from GRPOAgent so both can coexist
    on different GPUs (policy on cuda:0, partner on cuda:1).
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "local-partner",
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name

    _shared_model = None
    _shared_tokenizer = None
    _shared_device = "cuda"
    _max_gen_len = 256
    _temperature = 0.7
    _top_p = 0.9

    @classmethod
    def load_model(cls, model_name: str, device: str = "cuda",
                   max_gen_len: int = 256, temperature: float = 0.7,
                   top_p: float = 0.9):
        """Load a zero-shot local model as partner."""
        print(f"Loading local partner model: {model_name} on {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        cls._shared_model = model
        cls._shared_tokenizer = tokenizer
        cls._shared_device = device
        cls._max_gen_len = max_gen_len
        cls._temperature = temperature
        cls._top_p = top_p
        print(f"Local partner model loaded on {device}.")

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        raise NotImplementedError("Use aact() instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])

        if "speak" in obs.available_actions:
            prompt = _build_stage2_style_prompt(
                agent_name=self.agent_name,
                agent_profile=getattr(self, "profile", None),
                goal=self.goal,
                inbox=self.inbox,
                turn_number=obs.turn_number,
            )
            response_text = self._generate(prompt)
            argument = _extract_speak_argument(response_text)
            if argument:
                return AgentAction(action_type="speak", argument=argument, to=[])
            if "none" in obs.available_actions:
                return AgentAction(action_type="none", argument="", to=[])

        prompt = _build_structured_action_prompt(
            agent_name=self.agent_name,
            inbox=self.inbox,
            turn_number=obs.turn_number,
            available_actions=obs.available_actions,
        )
        response_text = self._generate(prompt)
        action_type, argument = self._parse_to_action(response_text)
        return AgentAction(action_type=action_type, argument=argument, to=[])

    def _generate(self, prompt: str) -> str:
        model = self.__class__._shared_model
        tokenizer = self.__class__._shared_tokenizer
        device = self.__class__._shared_device

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=self.__class__._max_gen_len,
                do_sample=True,
                temperature=self.__class__._temperature,
                top_p=self.__class__._top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def _parse_to_action(self, text: str) -> tuple[str, str]:
        """Parse model output into (action_type, argument). Same logic as GRPOAgent."""
        return _parse_model_text_to_action(text)


# ------------------------------------------------------------------------------
# Data Loading: HuggingFace episodes ? SOTOPIA profiles
# ------------------------------------------------------------------------------
def parse_agent_background(name: str, bg_text: str) -> AgentProfile:
    """Parse an agent background string into an AgentProfile object."""
    parts = name.split()
    first_name = parts[0] if parts else name
    last_name = parts[1] if len(parts) > 1 else ""

    # Extract age
    age_match = re.search(r"(\d+)-year-old", bg_text)
    age = int(age_match.group(1)) if age_match else 30

    # Extract gender
    gender = ""
    if " male " in bg_text.lower():
        gender = "Man"
    elif " female " in bg_text.lower():
        gender = "Woman"

    # Extract pronouns
    pronoun_match = re.search(r"((?:He|She|They)/\w+) pronouns", bg_text)
    gender_pronoun = pronoun_match.group(1) if pronoun_match else ""

    # Extract occupation (word before the first period)
    occ_match = re.search(r"(?:year-old (?:male|female|nonbinary) )(\w[\w\s]*?)\.", bg_text)
    occupation = occ_match.group(1).strip() if occ_match else ""

    # Extract public info (sentence after pronouns, before "Personality")
    public_info = ""
    pi_match = re.search(r"pronouns\.\s*(.*?)Personality and values", bg_text, re.DOTALL)
    if pi_match:
        public_info = pi_match.group(1).strip()

    # Extract personality
    personality = ""
    pv_match = re.search(r"Personality and values description:\s*(.*?)(?:\w+'s secrets:|$)", bg_text, re.DOTALL)
    if pv_match:
        personality = pv_match.group(1).strip()

    # Extract secret
    secret = ""
    sec_match = re.search(r"secrets?:\s*(.*?)$", bg_text, re.DOTALL)
    if sec_match:
        secret = sec_match.group(1).strip()

    return AgentProfile(
        first_name=first_name,
        last_name=last_name,
        age=age,
        gender=gender,
        gender_pronoun=gender_pronoun,
        occupation=occupation,
        public_info=public_info,
        personality_and_values=personality,
        secret=secret,
        pk=f"eval_{first_name}_{last_name}",
    )


def load_hf_episodes(deduplicate_envs: bool = True) -> list[dict]:
    """Download official SOTOPIA episodes from HuggingFace."""
    from huggingface_hub import hf_hub_download
    data_path = hf_hub_download(
        repo_id="cmu-lti/sotopia",
        filename="sotopia_episodes_v1.jsonl",
        repo_type="dataset",
    )
    print(f"Downloaded SOTOPIA episodes to: {data_path}")

    episodes = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    if deduplicate_envs:
        seen = set()
        deduped = []
        for ep in episodes:
            env_id = ep.get("environment_id", ep.get("episode_id", ""))
            if env_id not in seen:
                seen.add(env_id)
                deduped.append(ep)
        print(f"Deduplicated: {len(episodes)} -> {len(deduped)} unique environments")
        episodes = deduped

    return episodes


def episode_to_profiles(episode: dict):
    """Convert a HF episode dict into (EnvironmentProfile, [AgentProfile, AgentProfile])."""
    agent_names = list(episode["agents_background"].keys())
    goals = episode["social_goals"]

    # Parse agent profiles from background strings
    agent_profiles = [
        parse_agent_background(name, episode["agents_background"][name])
        for name in agent_names
    ]

    # Build environment profile
    env_profile = EnvironmentProfile(
        pk=episode.get("environment_id", ""),
        codename=episode.get("codename", ""),
        scenario=episode["scenario"],
        agent_goals=[goals[name] for name in agent_names],
        relationship=0,  # stranger (default, since HF data doesn't always include this)
    )

    return env_profile, agent_profiles


# ------------------------------------------------------------------------------
# Main Evaluation Pipeline
# ------------------------------------------------------------------------------
async def evaluate_episodes(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load Policy Model --
    if args.llm_policy:
        print(f"*** LLM POLICY MODE: Both agents use API model ***")
        print(f"  Policy: {args.policy_model_name}, Partner: {args.partner_model}")
    else:
        adapter_path = None if args.no_adapter else args.policy_adapter_path
        if args.no_adapter:
            print("*** ZERO-SHOT MODE: Using base model without LoRA adapter ***")
        GRPOAgent.load_model(
            model_name=args.policy_model_name,
            adapter_path=adapter_path,
            merge_adapter=args.merge_adapter,
            device=str(device),
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # -- Load Local Partner Model (optional) --
    if args.local_partner:
        partner_model_name = args.local_partner_model or args.policy_model_name
        if args.partner_device:
            partner_dev = args.partner_device
        else:
            num_gpus = torch.cuda.device_count()
            partner_dev = f"cuda:{num_gpus - 1}" if num_gpus > 1 else str(device)
        LocalPartnerAgent.load_model(
            model_name=partner_model_name,
            device=partner_dev,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"  Local partner: {partner_model_name} on {partner_dev}")

    # -- Load Episodes --
    if args.eval_data and not args.use_hf:
        episodes_raw = []
        with open(args.eval_data) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes_raw.append(json.loads(line))
        print(f"Loaded {len(episodes_raw)} episodes from {args.eval_data}")
    else:
        episodes_raw = load_hf_episodes(deduplicate_envs=args.deduplicate_envs)

    print(f"Total episodes (before task filter): {len(episodes_raw)}")

    # -- Task Filter --
    if args.task == "hard":
        episodes_raw = [
            ep for ep in episodes_raw
            if ep.get("environment_id", "") in HARD_ENV_IDS
        ]
        print(f"Filtered to {len(episodes_raw)} hard episodes")
    elif args.task == "cooperative":
        episodes_raw = [
            ep for ep in episodes_raw
            if "mutual" in ep.get("codename", "").lower()
        ]
        print(f"Filtered to {len(episodes_raw)} cooperative episodes")
    elif args.task == "competitive":
        episodes_raw = [
            ep for ep in episodes_raw
            if "craigslist" in ep.get("codename", "").lower()
        ]
        print(f"Filtered to {len(episodes_raw)} competitive episodes")

    if args.max_episodes > 0:
        episodes_raw = episodes_raw[:args.max_episodes]
        print(f"Evaluating first {len(episodes_raw)} episodes.")

    # -- Run Episodes (incremental saving) --
    results = []
    all_scores = defaultdict(lambda: defaultdict(list))
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    # Truncate output file at start so we don't mix with old results
    with open(args.output_path, 'w') as f:
        pass

    for ep_idx, episode in enumerate(episodes_raw):
        ep_id = episode.get("environment_id", episode.get("episode_id", "N/A"))
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx+1}/{len(episodes_raw)} (id={ep_id})")
        print(f"{'='*60}")

        try:
            env_profile, agent_profiles = episode_to_profiles(episode)
        except Exception as e:
            print(f"  [Skip] Failed to parse episode: {e}")
            continue

        # Create SOTOPIA environment with official evaluators
        env = ParallelSotopiaEnv(
            env_profile=env_profile,
            action_order="round-robin",
            model_name=args.judge_model,
            evaluators=[
                RuleBasedTerminatedEvaluator(
                    max_turn_number=args.max_turns,
                    max_stale_turn=2,
                ),
            ],
            terminal_evaluators=[
                EpisodeLLMEvaluator(
                    args.judge_model,
                    EvaluationForAgents[SotopiaDimensions],
                ),
            ],
        )

        # Create agents: GRPO policy (or LLM) at policy_agent_index, LLM partner as the other
        agent_list = []
        for i, ap in enumerate(agent_profiles):
            if i == args.policy_agent_index:
                if args.llm_policy:
                    agent = LLMAgent(agent_profile=ap, model_name=args.policy_model_name)
                else:
                    agent = GRPOAgent(agent_profile=ap, model_name="grpo-policy")
                tag_str = "[POLICY]"
            else:
                if args.local_partner:
                    agent = LocalPartnerAgent(agent_profile=ap, model_name="local-partner")
                    tag_str = "[PARTNER-LOCAL]"
                else:
                    agent = LLMAgent(agent_profile=ap, model_name=args.partner_model)
                    tag_str = "[PARTNER]"
            agent_list.append(agent)
            print(f"  Agent {i+1}: {ap.first_name} {ap.last_name} {tag_str}")

        print(f"  Scenario: {env_profile.scenario[:100]}...")

        # Run episode using official SOTOPIA components with custom loop
        # (arun_one_episode has a bug where complete_rating is always 0,
        #  so we run the loop ourselves to capture p1_rate/p2_rate properly)
        max_episode_retries = 3
        episode_success = False
        for _ep_attempt in range(max_episode_retries):
            try:
                # Re-create env and agents on retry to get a clean state
                if _ep_attempt > 0:
                    print(f"  [Retry] Attempt {_ep_attempt + 1}/{max_episode_retries}")
                    env = ParallelSotopiaEnv(
                        env_profile=env_profile,
                        action_order="round-robin",
                        model_name=args.judge_model,
                        evaluators=[
                            RuleBasedTerminatedEvaluator(
                                max_turn_number=args.max_turns,
                                max_stale_turn=2,
                            ),
                        ],
                        terminal_evaluators=[
                            EpisodeLLMEvaluator(
                                args.judge_model,
                                EvaluationForAgents[SotopiaDimensions],
                            ),
                        ],
                    )
                    agent_list = []
                    for i, ap in enumerate(agent_profiles):
                        if i == args.policy_agent_index:
                            if args.llm_policy:
                                agent_list.append(LLMAgent(agent_profile=ap, model_name=args.policy_model_name))
                            else:
                                agent_list.append(GRPOAgent(agent_profile=ap, model_name="grpo-policy"))
                        else:
                            if args.local_partner:
                                agent_list.append(LocalPartnerAgent(agent_profile=ap, model_name="local-partner"))
                            else:
                                agent_list.append(LLMAgent(agent_profile=ap, model_name=args.partner_model))

                agents = Agents({agent.agent_name: agent for agent in agent_list})
                environment_messages = env.reset(agents=agents, omniscient=False)
                agents.reset()

                # Set goals for agents
                for index, agent_name in enumerate(env.agents):
                    agents[agent_name].goal = env.profile.agent_goals[index]

                messages = []
                messages.append([
                    ("Environment", agent_name, environment_messages[agent_name])
                    for agent_name in env.agents
                ])

                done = False
                info = {}
                while not done:
                    agent_messages = {}
                    actions = await asyncio.gather(*[
                        agents[agent_name].aact(environment_messages[agent_name])
                        for agent_name in env.agents
                    ])

                    for idx, agent_name in enumerate(env.agents):
                        action = actions[idx]
                        for _retry in range(3):
                            # Retry if argument is None
                            if action.argument is None:
                                print(f"    [Retry {_retry+1}/3] {agent_name} returned None argument, regenerating...")
                                agents[agent_name].recv_message(
                                    "Environment",
                                    SimpleMessage(message="Your response was empty. Please provide a valid action."),
                                )
                                action = await agents[agent_name].aact(
                                    environment_messages[agent_name]
                                )
                                continue
                            try:
                                AgentAction.model_validate(
                                    action.model_dump(),
                                    context={"agent_names": env.agents, "sender": agent_name},
                                )
                                break
                            except ValueError as e:
                                print(f"    [Retry {_retry+1}/3] {agent_name} invalid action: {e}")
                                agents[agent_name].recv_message(
                                    "Environment",
                                    SimpleMessage(message=f"Invalid action: {e}. Regenerate."),
                                )
                                action = await agents[agent_name].aact(
                                    environment_messages[agent_name]
                                )
                        # Last resort: if still None after retries, fallback to empty string
                        if action.argument is None:
                            action.argument = ""
                        agent_messages[agent_name] = action
                        messages[-1].append((agent_name, "Environment", action))

                    (
                        environment_messages,
                        rewards_in_turn,
                        terminated,
                        ___,
                        info,
                    ) = await env.astep(agent_messages)

                    messages.append([
                        ("Environment", agent_name, environment_messages[agent_name])
                        for agent_name in env.agents
                    ])
                    done = all(terminated.values())

                episode_success = True
                break  # Episode completed successfully

            except Exception as e:
                print(f"  [Error] Episode attempt {_ep_attempt + 1} failed: {e}")
                import traceback
                traceback.print_exc()

        if not episode_success:
            print(f"  [Skip] Episode failed after {max_episode_retries} attempts")
            continue

        # Count turns
        n_turns = sum(
            1 for turn in messages for sender, _, msg in turn
            if sender != "Environment"
        )
        print(f"  Completed in {n_turns} agent actions")

        # Build result
        agent_names_list = [f"{ap.first_name} {ap.last_name}" for ap in agent_profiles]
        flat_messages = [
            (sender, receiver, msg.to_natural_language() if hasattr(msg, 'to_natural_language') else str(msg))
            for turn in messages for sender, receiver, msg in turn
        ]
        ep_result = {
            "episode_id": ep_id,
            "scenario": env_profile.scenario,
            "agent_names": agent_names_list,
            "messages": flat_messages,
        }

        # Extract scores from the terminal evaluation response
        # The env's astep computes p1_rate/p2_rate via terminal_evaluators
        # but stores them in the ScriptEnvironmentResponse, not in info dict.
        # We re-extract from info["comments"] which contains the evaluation.
        # More reliably, the env.inbox contains all messages including evaluator output.
        # The actual scores come from the last astep's response object.
        # Since astep merges terminal_response into response, and response has
        # p1_rate and p2_rate, but these aren't passed through info...
        # We need to access them from the env's internal state.
        # The cleanest way: run the terminal evaluator directly on the conversation.

        # Run terminal evaluation using agenerate directly
        # (EpisodeLLMEvaluator has a bug: it indexes model_dump()[dim][1]
        #  but SotopiaDimensions fields are dicts with 'reasoning'/'score' keys,
        #  not tuples, causing KeyError that gets silently swallowed)
        try:
            os.environ.pop("OPENAI_BASE_URL", None)

            # Build history string from conversation messages
            history_lines = []
            for turn in messages:
                for sender, receiver, msg in turn:
                    msg_text = msg.to_natural_language() if hasattr(msg, 'to_natural_language') else str(msg)
                    if "did nothing" in msg_text:
                        continue
                    if sender != "Environment":
                        history_lines.append(f"{sender} {msg_text}")
                    else:
                        history_lines.append(msg_text)
            history_str = "\n".join(history_lines)

            agent_instruction = (
                "There are exactly 2 agents. Under the 'evaluations' field, "
                'use exactly these keys: ["agent_1", "agent_2"] (no other keys).\n'
            )

            response_format_class = EvaluationForAgents[SotopiaDimensions]

            # Call agenerate directly (same as evaluator but without buggy parsing)
            response = await agenerate(
                model_name=args.judge_model,
                template="""{history}
                    Based on previous interactions, evaluate how well participants achieve their goals.
                    {agent_instruction}
                    Please follow the format:
                    {format_instructions}
                """,
                input_values=dict(history=history_str, agent_instruction=agent_instruction),
                output_parser=PydanticOutputParser[response_format_class](
                    pydantic_object=response_format_class
                ),
                temperature=0.0,
                structured_output=True,
            )

            # Parse response.evaluations properly (fields have .reasoning and .score)
            agent_dim_scores = defaultdict(dict)
            raw_eval_items = list(response.evaluations.items())
            if all(key in {"agent_1", "agent_2"} for key, _ in raw_eval_items):
                parsed_items = raw_eval_items
            else:
                print(f"  [Judge Warning] Unexpected evaluation keys: {[key for key, _ in raw_eval_items]}")
                parsed_items = [(f"agent_{i+1}", evaluation) for i, (_, evaluation) in enumerate(raw_eval_items)]

            for agent_key, evaluation in parsed_items:
                dumped = evaluation.model_dump()
                for dim_name, dim_val in dumped.items():
                    # dim_val is {"reasoning": "...", "score": N}
                    agent_dim_scores[agent_key][dim_name] = dim_val["score"]

            for i, agent_key in enumerate(["agent_1", "agent_2"]):
                is_policy = (i == args.policy_agent_index)
                tag_label = " [POLICY]" if is_policy else " [PARTNER]"
                agent_name = agent_names_list[i]

                if agent_key in agent_dim_scores:
                    dim_scores = agent_dim_scores[agent_key]
                    print(f"  {agent_name}{tag_label}:")
                    score_values = []
                    for dim in SOTOPIA_DIMENSIONS:
                        score = dim_scores.get(dim, 0)
                        print(f"    {dim}: {score}")
                        all_scores[agent_key][dim].append(score)
                        score_values.append(score)
                    overall = sum(score_values) / len(score_values) if score_values else 0
                    print(f"    overall: {overall:.2f}")
                    all_scores[agent_key]["overall_score"].append(overall)
                    ep_result[f"{agent_key}_scores"] = dim_scores
                    ep_result[f"{agent_key}_overall"] = overall
                else:
                    print(f"  [Judge Warning] No scores for {agent_key}")

        except Exception as e:
            print(f"  [Judge Error] {e}")
            import traceback
            traceback.print_exc()

        results.append(ep_result)

        # Append this episode's result to JSONL immediately
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(ep_result, ensure_ascii=False, default=str) + "\n")
        print(f"  [Saved] Episode result appended to {args.output_path}")

    # -- Aggregate Results --
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    policy_agent_key = f"agent_{args.policy_agent_index + 1}"
    partner_agent_key = f"agent_{2 - args.policy_agent_index}"

    summary = {"policy_agent": {}, "partner_agent": {}, "config": vars(args)}

    for label, agent_key in [("POLICY AGENT", policy_agent_key),
                              ("PARTNER AGENT", partner_agent_key)]:
        print(f"\n{label}:")
        agent_summary = {}
        for dim in SOTOPIA_DIMENSIONS + ["overall_score"]:
            scores = all_scores[agent_key].get(dim, [])
            if scores:
                mean_s = sum(scores) / len(scores)
                print(f"  {dim}: {mean_s:.3f} (n={len(scores)})")
                agent_summary[dim] = {"mean": mean_s, "n": len(scores), "values": scores}
            else:
                print(f"  {dim}: N/A")
                agent_summary[dim] = {"mean": None, "n": 0}

        if label == "POLICY AGENT":
            summary["policy_agent"] = agent_summary
        else:
            summary["partner_agent"] = agent_summary

    # -- Save Summary --
    print(f"\nDetailed results saved to: {args.output_path} ({len(results)} episodes)")

    summary_path = args.output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 3: SOTOPIA Evaluation (Official Framework)")

    # Model config
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy_adapter_path", type=str,
                        default="projects/sotopia/checkpoints/grpo_agent_qwen_v3/best",
                        help="Path to GRPO LoRA adapter (use --no_adapter for the base model)")
    parser.add_argument("--merge_adapter", action="store_true", default=True,
                        help="Merge LoRA adapter into base model for faster inference")
    parser.add_argument("--no_adapter", action="store_true", default=False,
                        help="Zero-shot evaluation: use base model without LoRA adapter")

    # Evaluation config
    parser.add_argument("--eval_data", type=str, default=None,
                        help="JSONL file with episodes (ignored if --use_hf is set)")
    parser.add_argument("--use_hf", action="store_true", default=True,
                        help="Download official SOTOPIA episodes from HuggingFace")
    parser.add_argument("--no_use_hf", dest="use_hf", action="store_false")
    parser.add_argument("--deduplicate_envs", action="store_true", default=True,
                        help="Keep one episode per unique environment (90 for official SOTOPIA)")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "hard", "cooperative", "competitive"],
                        help="Filter episodes by task type (hard=14 envs, cooperative=mutual, competitive=craigslist)")
    parser.add_argument("--output_path", type=str,
                        default="projects/sotopia/runs/evaluation/grpo_eval_results.jsonl")
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="Max episodes to evaluate (-1 for all)")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--policy_agent_index", type=int, default=0,
                        help="Which agent slot the policy model plays (0 or 1)")

    # Partner and Judge config
    parser.add_argument("--partner_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--llm_policy", action="store_true", default=False,
                        help="Use LLMAgent (API) for policy instead of local model. "
                             "Model name is taken from --policy_model_name.")
    parser.add_argument("--local_partner", action="store_true", default=False,
                        help="Use a local HF model as partner (zero-shot, no LoRA). "
                             "Model loaded from --local_partner_model.")
    parser.add_argument("--local_partner_model", type=str, default=None,
                        help="HF model name for local partner. Defaults to --policy_model_name if not set.")
    parser.add_argument("--partner_device", type=str, default=None,
                        help="Device for local partner model (e.g. 'cuda:1'). Auto-detected if not set.")

    # Generation config
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Misc
    parser.add_argument("--tag", type=str, default="grpo_eval",
                        help="Tag for the evaluation run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    asyncio.run(evaluate_episodes(args))


if __name__ == "__main__":
    main()
