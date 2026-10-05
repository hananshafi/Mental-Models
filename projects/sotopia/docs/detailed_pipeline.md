# Training Pipeline Overview: Mental-Model-Guided GRPO for Social Agents

## High-Level Architecture

```
Stage 1: Train Coupled Mental+Reward Model
    ↓ (frozen)
Stage 2: GRPO Agent Policy Training (guided by Stage 1 model)
    ↓ (checkpoint)
Stage 3: SOTOPIA Benchmark Evaluation (GPT-4o judge)
```

Base Model: **Qwen2.5-7B-Instruct** (used across all stages)
Data: **SOTOPIA episodes** (social interaction scenarios with 2 agents)

---

## Stage 1: Coupled Mental + Reward Model

### Purpose
Train a model that jointly learns:
1. **Mental state inference** (Theory of Mind) — what is the other agent thinking/intending?
2. **Reward prediction** — how socially appropriate is a given response?

The coupling forces the reward signal to be grounded in mental state understanding, not just surface-level pattern matching.

### Architecture

```
                    Context (scenario + background + goal + history)
                                    |
                            [Qwen2.5-7B + LoRA]
                                    |
                              hidden states
                                    |
                        ┌───────────┴───────────┐
                        ▼                       ▼
                  context_mu              context_logvar
                  (Linear)                (Linear)
                        │                       │
                        └───────┬───────────────┘
                                ▼
                        z ~ N(mu, sigma²)        ← latent mental state (128-dim)
                                |
                ┌───────────────┼───────────────────┐
                ▼               ▼                   ▼
        z_to_hidden      Mental Decoder       z_only_reward_head
        (Linear)         (cross-attn +           (MLP → R^d)
                │         FFN over response)
                │               │
                ▼               ▼
        Response Encoding   Mental text
        [Qwen2.5-7B]       generation loss
        (shared backbone)
                │
                ▼
        ┌───────┴────────┐
        ▼                ▼
  joint_outcome_head   expl_reward_head
  [z || resp_hidden]   (cross-attn z × resp)
    → R^d rewards        → R^d rewards
```

Where d = 7 (v1) or 3 (v2) reward dimensions.

### Mental State Components
- **z (latent)**: 128-dimensional latent variable encoding the agent's inferred mental state (beliefs, intentions, thoughts)
- **Mental decoder**: Cross-attention + FFN that generates mental state text from z, conditioned on the response. Trained with teacher-forced generation loss.
- **3 prefix projections**: z_belief_to_prefix, z_intent_to_prefix, z_thought_to_prefix — project z into prefix tokens for different mental facets

### Reward Dimensions

**v1 (7 dimensions):**
| Dimension | Range | Type |
|-----------|-------|------|
| believability | 0 to 10 | Positive |
| relationship | -5 to 5 | Bidirectional |
| knowledge | 0 to 10 | Positive |
| secret | -10 to 0 | Negative |
| social_rules | -10 to 0 | Negative |
| financial_and_material_benefits | -5 to 5 | Bidirectional |
| goal | 0 to 10 | Positive |

**v2 (3 dimensions):** goal, relationship, knowledge only

All scores are normalized to [0, 1] for training.

### Training Data Format
Each sample is a (context, positive_response, negative_response) triple from SOTOPIA episodes, where positive has higher human/GPT scores.

### Loss Function (7 components)

```
L_total = L_pref + L_reward_reg + λ_z · L_z_only
        + λ_kl · L_kl + λ_future · L_future
        + λ_mental · L_mental_gen + λ_expl · L_expl
```

1. **L_pref (Preference Loss)**: Bradley-Terry pairwise preference
   ```
   L_pref = -log(σ(r_positive - r_negative))
   ```
   The joint_outcome_head must score the positive response higher than negative.

2. **L_reward_reg (Reward Regularization)**: MSE between predicted reward and ground-truth normalized scores
   ```
   L_reward_reg = MSE(r_pred, r_target)
   ```

3. **L_z_only (z-only Reward Regularization)**: Ensures z alone (without seeing the response) can roughly predict rewards. Forces z to encode useful information.
   ```
   L_z_only = MSE(z_only_reward_head(z), r_target)
   ```

4. **L_kl (KL Divergence)**: Regularizes the latent z toward N(0,I)
   ```
   L_kl = KL(N(mu, sigma²) || N(0, I))
   ```
   Annealed over first 200 steps (starts at 0, ramps to full weight).

5. **L_future (Future Prediction)**: Predicts next-turn reward from current z, encouraging z to capture forward-looking state.
   ```
   L_future = MSE(future_head(z_t), r_{t+1})
   ```

6. **L_mental_gen (Mental Generation)**: Cross-entropy loss for generating mental state text (beliefs/intentions/thoughts) from z.
   ```
   L_mental = CE(mental_decoder(z, response), mental_text_target)
   ```

7. **L_expl (Explainable Reward)**: A secondary reward head using cross-attention between z and response, ensuring reward is interpretable through mental state.
   ```
   L_expl = MSE(expl_reward_head(cross_attn(z, resp)), r_target)
   ```

### What Gets Trained
- LoRA adapters on Qwen2.5-7B (v1: top 8 layers, v2: top 16 layers, r=16, alpha=32)
- All custom heads (context_mu, context_logvar, joint_outcome_head, z_only_reward_head, expl_reward_head, mental_cross_attn, mental_decode_ffn, z_to_hidden, z_belief/intent/thought_to_prefix)

### v2 Additions
- **Mental Pre-warmup Phase**: Before main training, pre-trains only the mental decoding pathway (LoRA + mental heads) on persona-based mental reasoning data for 3 epochs. This gives the mental state encoder a head start before coupling with reward learning.

### Key Hyperparameters
| Parameter | v1 | v2 |
|-----------|----|----|
| LoRA rank | 16 | 16 |
| LoRA layers | top 8 | top 16 |
| z_dim | 128 | 128 |
| Reward dims | 7 | 3 |
| Epochs | 10 | 10 |
| LR | 2e-5 | 2e-5 |
| Head LR multiplier | 10x | 10x |
| KL weight | 0.1 | 0.1 |
| Future weight | 0.5 | 0.5 |
| Mental weight | 0.3 | 0.3 |
| Batch size | 2 (v1) / 4 (v2) | - |

---

## Stage 2: GRPO Agent Policy Training

### Purpose
Train the dialogue policy (agent) to generate socially appropriate responses, guided by the frozen Stage 1 mental+reward model as the reward signal.

### Algorithm: Group Relative Policy Optimization (GRPO)

GRPO is a variant of PPO adapted for language generation. Instead of requiring a value function baseline, it uses **group-relative normalization** across multiple candidate completions.

### Training Flow

```
For each prompt (scenario + history up to turn t):

1. GENERATE: Sample G=8 candidate responses from current policy
   π_θ(response | prompt)  [temperature=0.8, top_p=0.95]

2. SCORE: Frozen Stage 1 model scores each candidate
   r_i = mean(reward_model(context, response_i))  [scalar, mean of d dims]

3. NORMALIZE: Group-relative advantage
   A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)

4. UPDATE: Clipped surrogate policy gradient
   ratio = π_θ_new(resp) / π_θ_old(resp)
   L_clip = -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
   L_kl = β · KL(π_ref || π_θ)    [KL penalty vs frozen base model]
   L = L_clip + L_kl
```

### Why GRPO over PPO?
- No value network needed (saves memory and complexity)
- Group normalization provides a natural baseline
- Well-suited for language generation where absolute reward scale is arbitrary

### Two-Phase Training

**Phase 1: SFT Warm-up** (1 epoch)
- Standard supervised fine-tuning on reference utterances from SOTOPIA episodes
- Initializes the policy to produce reasonable dialogue before RL
- LR: 2e-5

**Phase 2: GRPO** (3 epochs)
- RL fine-tuning with the frozen reward model
- LR: 5e-6 (10x lower than SFT — small updates to avoid catastrophic forgetting)

### Two Prompt Formats
The script maintains two separate prompt formats:
1. **Policy prompt** (rich instruction format): Used for generation — includes "Imagine you are {agent}..." framing
2. **Reward context** (Stage 1 format): Used for reward scoring — matches the exact format Stage 1 was trained on: "Scenario: ... Background: ... Goal: ... Dialogue History: ..."

This ensures the reward model sees inputs in the format it was trained on, while the policy sees a more natural instruction format.

### What Gets Trained
- LoRA adapters on Qwen2.5-7B-Instruct (top 16 layers, r=8, alpha=16)
- ~0.5% of total parameters

### What's Frozen
- Stage 1 mental+reward model (scores candidates, never updated)
- Reference model (base Qwen2.5-7B, for KL penalty computation)

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Group size (G) | 8 candidates per prompt |
| Prompts per step | 4 |
| Clip epsilon | 0.2 |
| KL coefficient (β) | 0.04 |
| GRPO LR | 5e-6 |
| SFT LR | 2e-5 |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Max generation length | 256 tokens |
| GRPO epochs | 3 |
| LoRA rank | 8 |
| LoRA layers | top 16 (of 28) |
| Save every | 50 steps |

### Memory Layout (2 GPUs)
- GPU 0: Policy model (needs gradient memory)
- GPU 1: Frozen reward model + frozen reference model

---

## Stage 3: SOTOPIA Evaluation

### Purpose
Evaluate the trained agent on the official SOTOPIA benchmark using GPT-4o as an independent judge.

### Setup
- **90 unique social scenarios** from the official SOTOPIA dataset (deduplicated by environment)
- **Policy agent**: GRPO-trained Qwen2.5-7B-Instruct + LoRA adapter (runs locally on GPU)
- **Partner agent**: GPT-4o-mini (API call, simulates the other person)
- **Judge**: GPT-4o (API call, evaluates both agents after conversation ends)
- **Max turns**: 10 (with max_stale_turn=2 — ends early if 2 consecutive "did nothing")

### Evaluation Flow

```
1. Load scenario + agent profiles from SOTOPIA dataset
2. Initialize ParallelSotopiaEnv (official SOTOPIA environment)
3. Set agent goals (each agent only sees their own goal, not partner's)
4. Run conversation loop:
   - Round-robin: agents take turns
   - Policy agent generates locally (Qwen on GPU)
   - Partner agent generates via GPT-4o-mini API
   - Environment checks termination conditions
5. After conversation ends, GPT-4o judges both agents on 7 dimensions
6. Record per-dimension and overall scores
```

### Information Isolation (No Leakage)
- `omniscient=False`: Each agent only sees their own goal and background
- Partner's goal/secret shown as "Unknown"
- Goals wrapped in XML viewer tags: `<root viewer='agent_0'>goal</root>`

### Scoring (7 SOTOPIA Dimensions)
| Dimension | Range | What it measures |
|-----------|-------|------------------|
| believability | 0-10 | Natural, realistic behavior |
| relationship | -5 to 5 | Did the relationship improve or deteriorate? |
| knowledge | 0-10 | New important information gained |
| secret | -10 to 0 | Were secrets/intentions leaked? |
| social_rules | -10 to 0 | Were moral rules or laws violated? |
| financial_and_material_benefits | -5 to 5 | Financial/material gain or loss |
| goal | 0-10 | How well were social goals achieved? |

**Overall score** = mean of all 7 dimensions.

Single GPT-4o call evaluates BOTH agents simultaneously (not two separate calls). Only the policy agent's scores are reported.

### Ablation Conditions
Three evaluation conditions for comparison:

1. **GRPO + Mental+Reward** (full model): `--policy_adapter_path .../step_950`
2. **Zero-shot baseline**: `--no_adapter` (base Qwen2.5-7B-Instruct, no training)
3. **GRPO without reward model**: Trained with random rewards instead of Stage 1 model — tests whether the learned reward signal matters

---

## Data Flow Summary

```
SOTOPIA Episodes (HuggingFace)
        │
        ├──→ Per-turn decomposition → sotopia_turn_rewards.jsonl
        │         │
        │         ├──→ Stage 1: Train mental+reward model
        │         │         │
        │         │         ▼
        │         ├──→ Stage 2: GRPO training (reward model frozen)
        │         │         │
        │         │         ▼
        │         │    Policy LoRA checkpoint
        │         │
        ▼         ▼
   90 unique scenarios → Stage 3: SOTOPIA Evaluation
                              │
                              ▼
                     Per-dimension scores (GPT-4o judge)
```

---

## Key Design Decisions

1. **Why coupled mental+reward?**
   Reward prediction alone can learn spurious correlations. By coupling with Theory of Mind (mental state inference), the reward signal is grounded in understanding *why* a response is good — not just pattern matching surface features.

2. **Why VAE-style latent z?**
   The latent z compresses the social context into a compact representation that captures beliefs, intentions, and goals. This enables:
   - Efficient reward computation (encode context once, score many candidates)
   - Future prediction (z should encode forward-looking state)
   - Mental state generation (z should be interpretable)

3. **Why GRPO over PPO?**
   No value network needed. Group-relative normalization provides a natural baseline from the candidate pool. Simpler, more memory-efficient for LLM training.

4. **Why separate LoRA configs for Stage 1 vs Stage 2?**
   - Stage 1 (reward model): Needs top layers for representation extraction (r=16, 8-16 layers)
   - Stage 2 (policy): Needs mid-layer reasoning + upper-layer realization (r=8, 16 layers)
   - Lower rank in Stage 2 because policy changes should be conservative (KL-regularized)

5. **Why SFT warm-up before GRPO?**
   Without SFT, the base model generates off-distribution responses that the reward model can't meaningfully differentiate. SFT provides a reasonable starting point for RL exploration.
