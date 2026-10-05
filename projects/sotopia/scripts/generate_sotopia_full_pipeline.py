#!/usr/bin/env python3
"""
SOTOPIA Full Data Pipeline
==========================
Combines per-turn reward/mental-state annotation (Stage A) and second-order
Theory of Mind annotation (Stage B) into a single script.

Stage A: For each episode, calls GPT-4o with structured JSON output to produce:
  - 7-dim reward scores + reasoning per turn (believability, relationship, ...)
  - 1st-order mental states: partner_belief, strategic_intent, thought_process,
    hard_negative_response

Stage B: For each turn's mental state, calls GPT-4o to produce:
  - second_order_belief, second_order_intent, second_order_thought

Output format: same as sotopia_turn_rewards_v3.jsonl
  - Saved immediately after each episode (flush on every write)
  - Resume support: skips complete episodes, reprocesses incomplete ones

Usage:
  OPENAI_API_KEY="sk-..." python generate_sotopia_full_pipeline.py \\
    --output projects/sotopia/data/sotopia_turn_rewards_v3.jsonl \\
    --limit 1500 \\
    --turn_workers 6 \\
    --concurrency 15

Cost estimate (gpt-4o, 1500 episodes ~10 turns avg):
  Stage A: ~15,000 turns x ~800 tokens ≈ 12M tokens  → ~$60-80
  Stage B: ~30,000 agent-turns x ~400 tokens ≈ 12M tokens → ~$80-100
  Total: ~$140-180
"""

import os
import sys
import json
import asyncio
import time
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

# ── Load .env if present ──────────────────────────────────────────────────────
_env_path = Path(__file__).parent / "sotopia" / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key == "OPENAI_BASE_URL":
                    continue
                os.environ.setdefault(_key, _val)
os.environ.pop("OPENAI_BASE_URL", None)

# ─────────────────────────────────────────────────────────────────────────────
# Stage A: Per-turn evaluation schema (structured output)
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_1_evaluation": {"$ref": "#/$defs/EvaluationBySocialDimensions"},
        "agent_2_evaluation": {"$ref": "#/$defs/EvaluationBySocialDimensions"}
    },
    "required": ["agent_1_evaluation", "agent_2_evaluation"],
    "additionalProperties": False,
    "$defs": {
        "DimensionScore": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {"type": "integer"}
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False
        },
        "Believability": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 10}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "Relationship": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": -5, "maximum": 5}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "Knowledge": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 10}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "Secret": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": -10, "maximum": 0}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "SocialRules": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": -10, "maximum": 0}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "FinancialBenefits": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": -5, "maximum": 5}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "Goal": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 10}},
            "required": ["reasoning", "score"], "additionalProperties": False
        },
        "MentalState": {
            "type": "object",
            "properties": {
                "partner_belief": {"type": "string"},
                "strategic_intent": {"type": "string"},
                "thought_process": {"type": "string"},
                "hard_negative_response": {"type": "string"}
            },
            "required": ["partner_belief", "strategic_intent", "thought_process", "hard_negative_response"],
            "additionalProperties": False
        },
        "EvaluationBySocialDimensions": {
            "type": "object",
            "properties": {
                "believability": {"$ref": "#/$defs/Believability"},
                "relationship": {"$ref": "#/$defs/Relationship"},
                "knowledge": {"$ref": "#/$defs/Knowledge"},
                "secret": {"$ref": "#/$defs/Secret"},
                "social_rules": {"$ref": "#/$defs/SocialRules"},
                "financial_and_material_benefits": {"$ref": "#/$defs/FinancialBenefits"},
                "goal": {"$ref": "#/$defs/Goal"},
                "mental_state": {"$ref": "#/$defs/MentalState"}
            },
            "required": ["believability", "relationship", "knowledge", "secret",
                         "social_rules", "financial_and_material_benefits", "goal", "mental_state"],
            "additionalProperties": False
        }
    }
}

STAGE_A_SYSTEM_PROMPT = """\
You are evaluating social interactions in a multi-agent conversation system.
Assess both agents based on the conversation up to the current turn.

BELIEVABILITY (0-10):
How natural and consistent is the agent's behavior?
1. Analyze if actions make sense in the context of the conversation.
2. Analyze if actions align with character traits (personality, values, etc.).
Higher score = more believable.

RELATIONSHIP (-5 to 5):
Analyze the relationship before and after the interaction.
Evaluate if interactions preserve or enhance personal relations.
Positive score = improved relationship, Negative = damaged, 0 = no change.

KNOWLEDGE (0-10):
Assess what information the agent gained.
Is this information new to them? Is it important?
Higher score = more new and important knowledge gained.

SECRET (-10 to 0):
Analyze what secrets or secretive intentions the agent wants to keep.
-10 = critical secrets/intentions leaked, 0 = no secrets revealed.

SOCIAL RULES (-10 to 0):
Evaluate if the agent violates any moral rules or laws.
0 = no violations, Negative = violations occurred.

FINANCIAL/MATERIAL BENEFITS (-5 to 5):
Consider short-term and long-term benefits.
Positive = gain, Negative = loss, 0 = no change.

GOAL (0-10):
Reiterate the agent's social goals. Analyze extent of achievement.
0 = minimal achievement, 10 = complete achievement.

MENTAL STATE:
For each agent, infer their internal mental state at this turn:

PARTNER_BELIEF: What does this agent believe about their partner's hidden intentions?
STRATEGIC_INTENT: What is the agent's immediate sub-goal for THIS specific turn?
THOUGHT_PROCESS: The latent reasoning connecting their belief to their action.
HARD_NEGATIVE_RESPONSE: A fluent but strategically poor or socially inappropriate alternative response."""


# ─────────────────────────────────────────────────────────────────────────────
# Stage B: Second-order ToM prompt
# ─────────────────────────────────────────────────────────────────────────────
SECOND_ORDER_PROMPT = """\
You are analyzing a social interaction to identify second-order Theory of Mind states.

=== SCENARIO ===
{scenario}

=== AGENT BACKGROUNDS ===
Agent A ({agent_name}): {agent_background}
Agent B ({partner_name}): {partner_background}

=== PRIVATE GOALS ===
Agent A's goal: {agent_goal}
Agent B's goal: {partner_goal}

=== DIALOGUE HISTORY (up to current turn) ===
{history}

=== CURRENT TURN ===
Agent A ({agent_name}) just said/did: "{current_utterance}"

=== FIRST-ORDER MENTAL STATES (Agent A's model of Agent B) ===
Agent A believes about Agent B:
  - Partner belief: {partner_belief}
  - Strategic intent: {strategic_intent}
  - Thought process: {thought_process}

=== YOUR TASK ===
Generate SECOND-ORDER mental states: what does Agent B ({partner_name}) think Agent A ({agent_name}) believes/intends/thinks?

These are Agent B's model of Agent A's mental states — the recursive layer.
Be specific, grounded in the dialogue, and consistent with both agents' goals.
Keep each field to 1-2 concise sentences.

Return ONLY valid JSON with exactly these three fields:
{{
  "second_order_belief": "Agent B thinks Agent A believes [...]",
  "second_order_intent": "Agent B thinks Agent A intends to [...]",
  "second_order_thought": "Agent B thinks Agent A is thinking [...]"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Data loading / saving helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return episodes


def is_complete(ep: dict) -> bool:
    """True if episode has turns AND all agent-turns have both 1st and 2nd order fields."""
    trs = ep.get("turn_rewards", [])
    if not trs:
        return False  # empty episodes are NOT complete
    for tr in trs:
        for agent_key in ["agent_1_rewards", "agent_2_rewards"]:
            ms = tr.get(agent_key, {}).get("mental_state", {})
            if ms:
                required = {"partner_belief", "strategic_intent", "thought_process",
                            "second_order_belief", "second_order_intent", "second_order_thought"}
                if not required.issubset(ms.keys()):
                    return False
    return True


def load_complete_episodes(output_path: str) -> tuple[dict, set]:
    """Returns (complete_eps dict by id, incomplete_ids set)."""
    if not os.path.exists(output_path):
        return {}, set()
    complete, incomplete_ids = {}, set()
    for ep in load_jsonl(output_path):
        ep_id = ep.get("episode_id", "")
        if is_complete(ep):
            complete[ep_id] = ep
        else:
            incomplete_ids.add(ep_id)
    return complete, incomplete_ids


# ─────────────────────────────────────────────────────────────────────────────
# Episode parsing (from HF sotopia-pi format)
# ─────────────────────────────────────────────────────────────────────────────
def parse_episode(episode_data: dict) -> dict:
    """Parse raw HF episode into structured format used by this pipeline.

    Handles two HF formats:
      Format A (older): 'messages' field with [sender, receiver, content] tuples,
                        agent names in 'participants', backgrounds in context text.
      Format B (sotopia-pi): 'raw_messages' field with nested lists,
                             'agents_background' dict, 'social_goals' dict,
                             'scenario' as top-level string.
    """
    # ── Determine agent names ──
    # Format B: agents_background is a dict {"Name1": "...", "Name2": "..."}
    agents_bg = episode_data.get('agents_background', {})
    if isinstance(agents_bg, str):
        try:
            agents_bg = json.loads(agents_bg)
        except json.JSONDecodeError:
            agents_bg = {}

    social_goals = episode_data.get('social_goals', {})
    if isinstance(social_goals, str):
        try:
            social_goals = json.loads(social_goals)
        except json.JSONDecodeError:
            social_goals = {}

    agent_names = list(agents_bg.keys()) if agents_bg else episode_data.get('participants', [])
    agent_1_name = agent_names[0] if len(agent_names) > 0 else "Agent 1"
    agent_2_name = agent_names[1] if len(agent_names) > 1 else "Agent 2"

    # ── Extract backgrounds and goals ──
    a1_bg = agents_bg.get(agent_1_name, episode_data.get("agent_1_background", ""))
    a2_bg = agents_bg.get(agent_2_name, episode_data.get("agent_2_background", ""))
    a1_goal = social_goals.get(agent_1_name, episode_data.get("agent_1_goal", ""))
    a2_goal = social_goals.get(agent_2_name, episode_data.get("agent_2_goal", ""))
    scenario = episode_data.get("scenario", "")
    a1_secret = episode_data.get("agent_1_secret", "")
    a2_secret = episode_data.get("agent_2_secret", "")

    # If backgrounds/goals are still empty, try to extract from context message
    messages = episode_data.get('messages', []) or episode_data.get('raw_messages', [])
    if not a1_bg or not a1_goal:
        context_message = ""
        for msg_list in messages:
            if isinstance(msg_list, list):
                for item in msg_list:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        sender, receiver, content = item
                        if sender == 'Environment' and receiver != 'Environment':
                            context_message = content
                            break
            if context_message:
                break

        def _extract_field(text, field_marker, stop_markers):
            if field_marker not in text:
                return ""
            start = text.find(field_marker) + len(field_marker)
            end = len(text)
            for m in stop_markers:
                pos = text.find(m, start)
                if pos != -1 and pos < end:
                    end = pos
            return text[start:end].strip()

        if not scenario:
            scenario = _extract_field(context_message, "Scenario: ",
                [f"{agent_1_name}'s background:", "Participants:"])
        if not a1_bg:
            a1_bg = _extract_field(context_message,
                f"{agent_1_name}'s background:", [f"{agent_2_name}'s background:", "Conversation Starts:"])
        if not a2_bg:
            a2_bg = _extract_field(context_message,
                f"{agent_2_name}'s background:", ["Conversation Starts:"])
        if not a1_goal:
            a1_goal = _extract_field(context_message,
                f"{agent_1_name}'s goal:", [f"{agent_2_name}'s goal:", "Conversation Starts:"])
        if not a2_goal:
            a2_goal = _extract_field(context_message,
                f"{agent_2_name}'s goal:", ["Conversation Starts:"])

    # ── Extract turns from raw_messages or messages ──
    # raw_messages format: list of lists, each inner list has [sender, receiver, content]
    # content looks like: "Turn #0: AgentName said: \"text\""
    turns = []
    turn_num = 0
    turn_pattern = re.compile(r'Turn #\d+:\s*(.+?)\s+(said|did nothing|left the conversation)[:\s]*(.*)', re.DOTALL)

    for msg_list in messages:
        if not isinstance(msg_list, list):
            continue
        for item in msg_list:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            sender, receiver, content = item

            # Format A: non-Environment -> Environment (agent utterances)
            if sender != 'Environment' and receiver == 'Environment':
                if 'said:' in content:
                    action = 'said'
                    text = content.split('said:', 1)[1].strip().strip('"')
                elif 'did nothing' in content:
                    continue
                elif 'left the conversation' in content:
                    action = 'left'
                    text = ''
                else:
                    action = 'unknown'
                    text = content
                turns.append({
                    'turn': turn_num, 'agent': sender,
                    'action': action, 'content': text
                })
                turn_num += 1

            # Format B: Environment -> Agent, content has "Turn #N: Agent said: ..."
            elif sender == 'Environment' and receiver != 'Environment':
                m = turn_pattern.search(content)
                if m:
                    agent_name_in_msg = m.group(1).strip()
                    action = m.group(2).strip()
                    text = m.group(3).strip().strip('"').strip("'")
                    if action == 'did nothing':
                        continue
                    if action == 'left the conversation':
                        action = 'left'
                        text = ''
                    turns.append({
                        'turn': turn_num, 'agent': agent_name_in_msg,
                        'action': action, 'content': text
                    })
                    turn_num += 1

    # Deduplicate turns: Format B sends the same turn to both agents, so we see
    # the same "Turn #N: X said: ..." twice. Keep only unique (turn content, agent).
    seen = set()
    deduped = []
    new_idx = 0
    for t in turns:
        key = (t['agent'], t['content'])
        if key not in seen:
            seen.add(key)
            t['turn'] = new_idx
            deduped.append(t)
            new_idx += 1
    turns = deduped

    pk = episode_data.get('pk') or episode_data.get('_id') or episode_data.get('episode_id')

    return {
        'pk': pk,
        'scenario': scenario,
        'agent_1_name': agent_1_name,
        'agent_2_name': agent_2_name,
        'agent_1_background': a1_bg,
        'agent_2_background': a2_bg,
        'agent_1_secret': a1_secret,
        'agent_2_secret': a2_secret,
        'agent_1_goal': a1_goal,
        'agent_2_goal': a2_goal,
        'turns': turns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage A: Per-turn reward + 1st-order mental state annotation
# ─────────────────────────────────────────────────────────────────────────────
def build_turn_prompt(parsed: dict, up_to_turn: int) -> str:
    prompt = (
        f"Here is the context of this interaction:\n"
        f"Scenario: {parsed['scenario']}\n"
        f"Participants: {parsed['agent_1_name']} and {parsed['agent_2_name']}\n"
        f"{parsed['agent_1_name']}'s background: {parsed['agent_1_background']}\n"
        f"{parsed['agent_2_name']}'s background: {parsed['agent_2_background']}\n"
        f"{parsed['agent_1_name']}'s goal: {parsed['agent_1_goal']}\n"
        f"{parsed['agent_2_name']}'s goal: {parsed['agent_2_goal']}\n"
    )
    for turn in parsed['turns'][:up_to_turn + 1]:
        prompt += f"\nTurn #{turn['turn'] + 1}\n"
        prompt += f"{turn['agent']} {turn['action']}"
        if turn['content']:
            prompt += f': "{turn["content"]}"'
        prompt += "\n"
    prompt += "\nBased on the interaction so far, evaluate how well each participant is achieving their goals at this point in the conversation.\n\nPlease follow the evaluation schema provided and give detailed reasoning for each dimension."
    return prompt


def evaluate_turn_sync(client: OpenAI, model: str, parsed: dict, turn_idx: int,
                       retries: int = 3) -> dict:
    prompt = build_turn_prompt(parsed, turn_idx)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": STAGE_A_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sotopia_evaluation",
                        "strict": True,
                        "schema": EVALUATION_SCHEMA
                    }
                }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"evaluate_turn failed after {retries} attempts: {e}")


def format_rewards(evaluation: dict) -> dict:
    rewards = {}
    for dimension, data in evaluation.items():
        if dimension == 'mental_state':
            rewards[dimension] = {
                'partner_belief': data.get('partner_belief', ''),
                'strategic_intent': data.get('strategic_intent', ''),
                'thought_process': data.get('thought_process', ''),
                'hard_negative_response': data.get('hard_negative_response', '')
            }
        else:
            rewards[dimension] = {
                'reasoning': data.get('reasoning', ''),
                'score': data.get('score', 0)
            }
    scores = [v['score'] for k, v in rewards.items() if isinstance(v, dict) and 'score' in v]
    rewards['overall_score'] = sum(scores) / len(scores) if scores else 0
    return rewards


def annotate_stage_a(client: OpenAI, model: str, episode_data: dict,
                     turn_workers: int = 5) -> dict:
    """Run Stage A on raw HF episode. Returns structured episode with turn_rewards."""
    parsed = parse_episode(episode_data)
    num_turns = len(parsed['turns'])
    turn_rewards = [None] * num_turns

    with ThreadPoolExecutor(max_workers=turn_workers) as executor:
        future_to_turn = {
            executor.submit(evaluate_turn_sync, client, model, parsed, i): i
            for i in range(num_turns)
        }
        for future in as_completed(future_to_turn):
            turn_idx = future_to_turn[future]
            evaluation = future.result()  # raises on failure
            turn_rewards[turn_idx] = {
                'turn': turn_idx,
                'agent': parsed['turns'][turn_idx]['agent'],
                'agent_1_rewards': format_rewards(evaluation['agent_1_evaluation']),
                'agent_2_rewards': format_rewards(evaluation['agent_2_evaluation']),
            }

    pk = episode_data.get('pk') or episode_data.get('_id') or episode_data.get('episode_id')
    return {
        'episode_id': pk or f"ep_{abs(hash(str(episode_data)[:100]))}",
        'parsed_episode': parsed,
        'turn_rewards': turn_rewards,
        'original_rewards': episode_data.get('rewards', []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage B: Second-order ToM annotation (async)
# ─────────────────────────────────────────────────────────────────────────────
def build_history_str(turns: list[dict], up_to_turn: int) -> str:
    lines = []
    for t in turns:
        if t["turn"] >= up_to_turn:
            break
        lines.append(f"  Turn {t['turn']} | {t['agent']} {t['action']}: \"{t['content']}\"")
    return "\n".join(lines) if lines else "  (No prior dialogue)"


async def generate_second_order_async(
    model: str, scenario: str,
    agent_name: str, partner_name: str,
    agent_background: str, partner_background: str,
    agent_goal: str, partner_goal: str,
    history: str, current_utterance: str,
    partner_belief: str, strategic_intent: str, thought_process: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict | None:
    import litellm
    os.environ.pop("OPENAI_BASE_URL", None)

    prompt = SECOND_ORDER_PROMPT.format(
        scenario=scenario,
        agent_name=agent_name, partner_name=partner_name,
        agent_background=agent_background[:400],
        partner_background=partner_background[:400],
        agent_goal=agent_goal[:300], partner_goal=partner_goal[:300],
        history=history,
        current_utterance=current_utterance[:300],
        partner_belief=partner_belief,
        strategic_intent=strategic_intent,
        thought_process=thought_process,
    )

    async with semaphore:
        for attempt in range(retries):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content.strip())
                required = {"second_order_belief", "second_order_intent", "second_order_thought"}
                if not required.issubset(data.keys()):
                    raise ValueError(f"Missing keys: {required - set(data.keys())}")
                return {k: str(data[k]) for k in required}
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"    [2nd-order failed after {retries} attempts]: {e}", flush=True)
                    return None


async def annotate_stage_b(ep: dict, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Add second-order fields to all turns in-place (deep copy first)."""
    pe = ep["parsed_episode"]
    scenario = pe["scenario"]
    a1, a2 = pe["agent_1_name"], pe["agent_2_name"]
    a1_bg, a2_bg = pe.get("agent_1_background", ""), pe.get("agent_2_background", "")
    a1_goal, a2_goal = pe.get("agent_1_goal", ""), pe.get("agent_2_goal", "")
    turns = pe.get("turns", [])

    tasks, keys = [], []
    for turn_idx, tr in enumerate(ep["turn_rewards"]):
        turn_num = tr["turn"]
        history = build_history_str(turns, up_to_turn=turn_num)
        current_utterance = next((t["content"] for t in turns if t["turn"] == turn_num), "")

        for agent_key, agent_name, partner_name, agent_bg, partner_bg, agent_goal, partner_goal in [
            ("agent_1_rewards", a1, a2, a1_bg, a2_bg, a1_goal, a2_goal),
            ("agent_2_rewards", a2, a1, a2_bg, a1_bg, a2_goal, a1_goal),
        ]:
            ms = tr.get(agent_key, {}).get("mental_state", {})
            if ms and "second_order_belief" not in ms:
                tasks.append(generate_second_order_async(
                    model=model, scenario=scenario,
                    agent_name=agent_name, partner_name=partner_name,
                    agent_background=agent_bg, partner_background=partner_bg,
                    agent_goal=agent_goal, partner_goal=partner_goal,
                    history=history, current_utterance=current_utterance,
                    partner_belief=ms.get("partner_belief", ""),
                    strategic_intent=ms.get("strategic_intent", ""),
                    thought_process=ms.get("thought_process", ""),
                    semaphore=semaphore,
                ))
                keys.append((turn_idx, agent_key))

    if not tasks:
        return ep

    results = await asyncio.gather(*tasks)
    ep_copy = json.loads(json.dumps(ep))
    for (turn_idx, agent_key), result in zip(keys, results):
        if result is not None:
            ep_copy["turn_rewards"][turn_idx][agent_key]["mental_state"].update(result)
    return ep_copy


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
async def run_pipeline(args):
    os.environ.pop("OPENAI_BASE_URL", None)

    # ── Load raw HF episodes ──
    print(f"Downloading episodes from HuggingFace ({args.repo_id} / {args.filename})...", flush=True)
    from huggingface_hub import hf_hub_download
    hf_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, repo_type="dataset")
    raw_episodes = load_jsonl(hf_path)
    print(f"Total episodes available: {len(raw_episodes)}", flush=True)

    if args.limit > 0:
        raw_episodes = raw_episodes[:args.limit]
        print(f"Limiting to {len(raw_episodes)} episodes.", flush=True)

    # ── Resume: find already complete episodes ──
    complete_eps, incomplete_ids = load_complete_episodes(args.output)
    print(f"Already complete in output: {len(complete_eps)}", flush=True)
    if incomplete_ids:
        print(f"Incomplete (will reprocess): {len(incomplete_ids)}", flush=True)
        # Rewrite output with only complete episodes
        with open(args.output, 'w', encoding='utf-8') as f:
            for ep in complete_eps.values():
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # Filter raw episodes to process
    to_process = [ep for ep in raw_episodes
                  if (ep.get('pk') or ep.get('_id') or ep.get('episode_id')) not in complete_eps]
    print(f"Episodes to process: {len(to_process)}", flush=True)

    if not to_process:
        print("Nothing to process. Done.", flush=True)
        return

    client = OpenAI()
    semaphore_b = asyncio.Semaphore(args.concurrency)
    write_lock = threading.Lock()
    loop = asyncio.get_event_loop()
    total = len(to_process)
    processed = 0

    output_f = open(args.output, 'a', encoding='utf-8')

    async def process_one(raw_ep):
        nonlocal processed
        ep_id = raw_ep.get('pk') or raw_ep.get('_id') or raw_ep.get('episode_id', '?')
        t0 = time.time()

        # Stage A: per-turn annotation (sync, runs in thread pool)
        try:
            ep = await loop.run_in_executor(
                None, annotate_stage_a, client, args.model, raw_ep, args.turn_workers
            )
        except Exception as e:
            print(f"  [SKIP] Stage A failed for {ep_id}: {e}", flush=True)
            return

        # Stage B: second-order annotation (async)
        ep = await annotate_stage_b(ep, args.model, semaphore_b)

        # Save immediately
        with write_lock:
            output_f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            output_f.flush()
            processed += 1

        elapsed = time.time() - t0
        complete = is_complete(ep)
        print(f"  [{processed}/{total}] {ep_id}  {elapsed:.1f}s  complete={complete}", flush=True)

    try:
        # Process in batches to avoid overwhelming the event loop with thousands of coroutines
        batch_size = args.episode_batch
        for i in range(0, total, batch_size):
            batch = to_process[i:i + batch_size]
            await asyncio.gather(*[process_one(ep) for ep in batch])
            print(f"  --- Batch {i//batch_size + 1} done ({min(i+batch_size, total)}/{total}) ---",
                  flush=True)
    finally:
        output_f.close()

    print(f"\nDone. {processed}/{total} episodes written to {args.output}", flush=True)

    # Final completeness check
    final_eps = load_jsonl(args.output)
    n_complete = sum(1 for ep in final_eps if is_complete(ep))
    print(f"Final: {len(final_eps)} episodes in output, {n_complete} fully complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="SOTOPIA full pipeline: per-turn annotation + 2nd-order ToM in one pass"
    )
    parser.add_argument("--output", type=str,
                        default="projects/sotopia/data/sotopia_turn_rewards_v3.jsonl",
                        help="Output JSONL file (also used for resume)")
    parser.add_argument("--repo_id", type=str, default="cmu-lti/sotopia-pi",
                        help="HuggingFace repo to download from")
    parser.add_argument("--filename", type=str, default="sotopia_pi_episodes.jsonl",
                        help="Filename within the HF repo")
    parser.add_argument("--limit", type=int, default=1500,
                        help="Max episodes to process (-1 = all available)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model for both stages")
    parser.add_argument("--turn_workers", type=int, default=6,
                        help="Concurrent turn evaluations per episode (Stage A threads)")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Max concurrent async API calls for Stage B")
    parser.add_argument("--episode_batch", type=int, default=20,
                        help="Episodes processed concurrently per batch")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
