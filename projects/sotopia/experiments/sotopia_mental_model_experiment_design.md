# SOTOPIA Mental-Model Ablation Design

This is the SOTOPIA-first experiment plan for answering the advisor questions:

1. Is the mental model just compression?
2. Why decompose into belief, intent, and thought/planning?
3. What is the intrinsic value of the recursive `z1 -> z2` mental state?

The current SOTOPIA pipeline already has the core stages:

- Stage 1 mental/reward model: `stage1_train_coupled_mental_reward_v3.py`
- Stage 2 mental-reward GRPO policy: `stage2_grpo_agent_training_v3.py`
- Stage 2 no-mental reward baseline: `stage2_grpo_ablation_no_mental.py`
- Stage 3 official evaluation: `stage3_evaluate_sotopia.py`
- Latent analysis: `analyse_tom_latents.py`

All commands should keep model paths flexible. Use either CLI flags or environment variables:

```bash
export SOTOPIA_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export SOTOPIA_POLICY_MODEL="$SOTOPIA_BASE_MODEL"
export SOTOPIA_REWARD_MODEL="$SOTOPIA_BASE_MODEL"
export SOTOPIA_RUN_ROOT="projects/sotopia/experiments/runs"
```

For a local model/checkpoint, set `SOTOPIA_BASE_MODEL` or pass `--base-model /path/to/model`.

## Core Claim To Test

A generic bottleneck can summarize the dialogue. A mental model must preserve hidden partner-state variables that change the correct action under similar surface contexts. Therefore the key SOTOPIA evaluations should emphasize hard episodes, partner generalization, counterfactual action ranking, and latent interventions rather than only average in-domain score.

## P0 Experiment Matrix

These are the first runs to prioritize.

| Variant | Purpose | Runnable now? | Stage 1 settings | Stage 2 settings |
|---|---|---:|---|---|
| `full_recursive_bit` | Main model: recursive z1/z2, mental text, reward coupling | yes | default v3 | mental GRPO |
| `reward_only_bottleneck` | Tests compression/reward-only latent without mental text | yes | `mental1_weight=0`, `mental2_weight=0`, `expl_weight=0`, `future_weight=0` | mental GRPO with this reward |
| `first_order_only` | Tests value of second-order supervision | yes | `mental2_weight=0` | mental GRPO with this reward |
| `no_antibypass` | Tests whether z-only reward heads matter | yes | `z_only_weight=0` | mental GRPO with this reward |
| `no_explanation` | Tests rationale/explanation reward alignment | yes | `expl_weight=0` | mental GRPO with this reward |
| `simple_reward_no_mental` | Tests no mental model at policy-training time | yes | none | `stage2_grpo_ablation_no_mental.py` |
| `zero_shot_base` | Base model floor | yes | none | none, eval with `--no_adapter` |

Important distinction:

- `simple_reward_no_mental` is a no-latent reward baseline. It does not test pure compression.
- `reward_only_bottleneck` is a reward-trained bottleneck baseline. It tests whether mental supervision adds value beyond a latent reward bottleneck, but it is still not pure compression.
- A true compression-only baseline should train the same-size latent only to preserve input information, then test whether that representation supports the same counterfactual flips.

Architecture variants below require code changes after P0:

| Variant | Purpose | Required implementation |
|---|---|---|
| `context_autoencoder_bottleneck` | Pure compression null: same-size z trained to reconstruct context or summary, no reward/mental labels | Train z with reconstruction only; freeze z encoder; attach/evaluate downstream scorer |
| `summary_autoencoder_bottleneck` | Stronger semantic compression null: z reconstructs LLM summary of context | Generate/freeze summaries; train same-size z to reconstruct summary only |
| `parallel_z1_z2` | Tests whether recursive `z1 -> z2` matters | Add `--latent_arch parallel`; make z2 condition only on context hidden |
| `reverse_z2_z1` | Negative control for recursive direction | Add reverse latent ordering |
| `single_latent_256` | Capacity-matched nonrecursive latent | Add single latent with total dim equal to `z1+z2` |
| `merged_subspace` | Tests belief/intent/thought partition vs merged z | Remove fixed 48/40/40 decoder split or decode from whole z |
| `flat_mental_summary` | Tests schema without BIT tags | Add dataset transform for `mental1_text`/`mental2_text` |
| `shuffled_mental_labels` | Negative control for mental supervision | Add deterministic label shuffling by seed |

## SOTOPIA Evaluation Protocol

Run every P0 variant on:

1. `task=all`, partner `gpt-4o-mini`, judge `gpt-4o`.
2. `task=hard`, partner `gpt-4o-mini`, judge `gpt-4o`.
3. `task=all`, local/self partner if compute allows.

Report:

- Overall average.
- Goal.
- Knowledge.
- Believability.
- Relationship.
- Social rules and secret penalties.
- Hard subset average.
- Partner-averaged score when both partner types are available.

Use three seeds if compute allows: `42, 43, 44`. If full retraining is too expensive, run three evaluation seeds for trained checkpoints and bootstrap episode-level confidence intervals.

## Compression Null Tests

### A. Reward-Only Bottleneck

Use `reward_only_bottleneck` as the fastest compression baseline. It keeps the same recursive latent architecture but removes mental/rationale/future supervision. If this baseline matches full model, then the current evidence is mostly "reward bottleneck helps." If full model wins on hard/OOD/counterfactual tests, the mental supervision adds value beyond compression.

Strictly, this is not pure compression because the latent is still trained by reward losses. It is the cheapest runnable approximation of the compression null, not the final compression-only baseline.

### A0. Pure Compression-Only Bottleneck

Add a stricter baseline:

```text
context -> encoder -> z_compress
z_compress -> decoder -> original context or generated context summary
```

Training losses:

- Context/summary reconstruction loss.
- Optional next-utterance reconstruction loss if we want a stronger sequence-compression baseline.
- No mental-state loss.
- No reward regression.
- No preference loss.

Evaluation modes:

1. Frozen compression representation + same-capacity reward head trained afterward.
2. Test-time scoring with z held fixed from the observable context only.
3. Counterfactual-flip visual where the compressed z should not flip unless the visible text changes.

This is the cleanest answer to "is the mental model just compression?" because it gives reviewers a same-dimensional representation whose only job was to compress information.

Implemented script:

```bash
python projects/sotopia/experiments/stage1_train_compression_vae.py \
  --model_name "$SOTOPIA_BASE_MODEL" \
  --data_path projects/sotopia/data/sotopia_turn_rewards_v3.jsonl \
  --output_dir projects/sotopia/experiments/runs/stage1/compression_vae_summary_seed42 \
  --target_mode summary \
  --z_dim 256 \
  --num_epochs 5 \
  --probe_epochs 3 \
  --seed 42 \
  --gpu 0
```

The VAE phase trains only:

```text
context -> z_compress -> observed context/summary reconstruction
```

The optional probe phase freezes the compression encoder and trains only a small candidate-scoring head. This keeps the representation compression-only while letting the visual script score candidate A/B responses.

### B. Context-Hidden Reward Baseline

Use `simple_reward_no_mental` as the no-z, no-mental policy baseline. It trains a simple reward head and then GRPO. This tests whether SOTOPIA gains come from reward optimization alone.

### C. Counterfactual Action Ranking

Add a held-out candidate-ranking set from `sotopia_turn_rewards_v3.jsonl`:

- Context.
- Positive response.
- Hard negative response.
- Optional minimally edited/counterfactual response.

For each reward model, score candidates and report:

- Pairwise positive-vs-negative accuracy.
- Mean reward margin.
- AUC if multiple negatives are available.
- Per-dimension ranking for goal, knowledge, relationship, social_rules.

This can be run before GRPO, making it cheaper than full Stage 2.

For the paper-facing visual version of this test, see
`counterfactual_flip_experiment_design.md` and run
`run_counterfactual_flip_experiment.py` after curating the counterfactual JSONL.

## Decomposition Tests

### B1. First-Order vs Second-Order

Compare `full_recursive_bit` and `first_order_only`.

Expected:

- `first_order_only` may preserve basic goal/knowledge performance.
- Full model should improve harder negotiation turns where anticipating the partner's view of the speaker matters.

### B2. Factor Dropout

Requires a small dataset/code change, but the desired variants are:

- `belief_only`
- `intent_only`
- `planning_only`
- `no_belief`
- `no_intent`
- `no_planning`

Expected SOTOPIA mapping:

- Belief: knowledge, secret handling, privacy-sensitive responses.
- Intent: goal completion, financial/material benefits.
- Planning/rationale: relationship, believability, social rules.

### B3. Flat vs Structured Mental Supervision

Generate a flat mental summary:

```text
The partner likely knows ..., wants ..., and may respond by ...
```

without explicit `Partner Belief`, `Strategic Intent`, `Thought Process` tags. Compare against current structured labels. If flat is close, frame BIT as an interpretable interface. If structured wins, claim decomposition improves supervision and sample efficiency.

## Recursive z1 -> z2 Tests

### C1. Runnable Now

Use existing analyses:

- `analyse_tom_latents.py` for probes and traversals.
- Compare latent artifacts for `full_recursive_bit`, `reward_only_bottleneck`, and `first_order_only`.

Report:

- Probe F1 for intent/knowledge/strategy labels.
- kNN local purity.
- Traversal monotonicity for `strategy=offer_proposal`.
- Reward-head response along traversal direction.

### C2. Requires Implementation

Add `--latent_arch {recursive,parallel,single,reverse}` to Stage 1 and Stage 2 reward loading. Then run:

- Recursive full.
- Parallel z1/z2.
- Single latent with equal total capacity.
- Reverse latent order.

The key metric should be not only SOTOPIA average, but intervention coherence:

- Does moving along a concept direction change the relevant reward dimension?
- Does it leave unrelated dimensions mostly stable?
- Does random matched-norm traversal stay flat?

## Suggested Table For The Paper

| Variant | Mental schema | Latent arch | Reward AUC | SOTOPIA All Avg | Hard Avg | Goal | Know. | Bel. | Traversal mono. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot base | none | none | - | | | | | | - |
| simple reward GRPO | none | none | | | | | | | - |
| reward-only bottleneck | none | recursive z | | | | | | | |
| first-order only | partial BIT | recursive z | | | | | | | |
| no anti-bypass | BIT | recursive z | | | | | | | |
| full recursive BIT | BIT | recursive z1->z2 | | | | | | | |

## Command Rendering

Use:

```bash
python projects/sotopia/experiments/render_sotopia_experiment_commands.py \
  --variant full_recursive_bit \
  --stage all \
  --base-model "$SOTOPIA_BASE_MODEL" \
  --run-root "$SOTOPIA_RUN_ROOT"
```

The renderer prints commands only. It does not launch training.

## Immediate Run Order

1. Stage 1 for `full_recursive_bit`, `reward_only_bottleneck`, `first_order_only`, `no_antibypass`.
2. Latent analysis for those four reward checkpoints.
3. Stage 2 for the same four plus `simple_reward_no_mental`.
4. Stage 3 `task=hard` first, because it is cheaper and advisor-relevant.
5. Stage 3 `task=all` for the strongest variants.
6. Add the code-level architecture ablations once P0 results identify the strongest baseline gap.
