# Counterfactual Flip Experiment: Mental Model vs Compression

This experiment is designed to visually answer:

> Is the mental model just a compressed summary of context?

The central trick is to keep the visible dialogue context fixed while changing only the latent mental-state branch used by the mental reward model. A pure simple-reward GRPO baseline has no latent mental-state channel, so its score should remain flat across the two hidden-state branches. If the mental model is more than compression, the same two candidate responses should swap preference when the latent partner state is swapped.

## Core Visual Claim

For each matched example:

- Same observable SOTOPIA context.
- Two hidden partner-state branches:
  - State A: response A should be better.
  - State B: response B should be better.
- Two candidate responses, both plausible on the surface.

Plot the signed margin:

```text
margin = score(candidate_a) - score(candidate_b)
```

Expected pattern:

```text
Mental model:
  State A margin > 0
  State B margin < 0
  clear crossover

Compression-only VAE / simple reward / pure GRPO baseline reward:
  State A margin ~= State B margin
  no systematic crossover, because it sees the same observable context
```

## Why This Addresses Compression

A generic compressed context representation can summarize what was said. It does not, by itself, provide a controllable state variable whose intervention flips which action is preferred under the same surface dialogue.

The decisive evidence is:

1. The full mental reward model flips candidate preference under latent mental-state swap.
2. The pure compression VAE baseline does not flip because its z is computed from the unchanged observable context.
3. The simple reward baseline does not flip because the observable input is unchanged.
4. A shuffled/wrong mental state damages flip accuracy.
5. Random or unrelated state directions do not produce the same coherent flip.

This makes the paper claim sharper:

> The learned mental state is not merely compact; it is decision-relevant and intervention-sensitive.

## Dataset Schema

Create a JSONL file where each row is one counterfactual pair:

```json
{
  "pair_id": "budget_vs_bluff_001",
  "observable_context": "Scenario: ...\nBackground: ...\nGoal: ...\nSecret: ...\nDialogue History:\n...\nTurn 5 | Speaker:",
  "candidate_a": "I can offer a modest discount if that helps meet your budget.",
  "candidate_b": "Before lowering the price, could you clarify what constraint you are working with?",
  "state_a": {
    "name": "genuine_constraint",
    "z_context": "same SOTOPIA context plus a counterfactual partner-state note saying the partner truly has a budget constraint",
    "correct": "a"
  },
  "state_b": {
    "name": "strategic_bluff",
    "z_context": "same SOTOPIA context plus a counterfactual partner-state note saying the partner is bluffing and can pay more",
    "correct": "b"
  },
  "notes": "Optional short explanation for the figure caption."
}
```

The `observable_context` is used by the simple reward baseline and compression-only VAE. The `z_context` fields are used only to produce the mental model's branch-specific latent state. For a stricter latent intervention, `z_context` can be replaced later by stored `z_a`/`z_b` arrays or concept-direction traversal, but this schema is the easiest version to run now.

## How To Build The Evaluation Set

Target size:

- Minimum for a paper figure: 20 carefully curated pairs.
- Stronger quantitative appendix: 100-200 pairs.

Recommended categories:

- Negotiation constraint: genuine budget constraint vs bluff.
- Privacy/secret: partner is unaware vs suspicious.
- Relationship repair: partner wants reconciliation vs wants boundaries.
- Information seeking: partner lacks key information vs already knows it.
- Cooperation: partner intends to coordinate vs defect/avoid.

Each pair should satisfy:

- Candidate A and B are both fluent and plausible.
- The correct candidate changes only because the inferred partner state changes.
- Surface lexical cues in `observable_context` do not directly reveal the branch.
- The `z_context` notes are short and structurally identical except for the hidden-state content.

## Conditions To Plot

Primary conditions:

1. `mental_correct_z`: full mental reward model, state A uses `z_context_a`, state B uses `z_context_b`.
2. `simple_reward_observed`: simple reward model, both states use the same `observable_context`.
3. `compression_observed`: compression-only VAE, both states use the same `observable_context`.
4. `mental_swapped_z`: full mental reward model with state A/B z-contexts swapped.

Optional controls:

5. `mental_shuffled_z`: each example uses z-context from another random example.
6. `mental_observed_only`: full model uses `observable_context` for both branches, showing how much flip comes from the branch latent.

## Metrics

Per example:

- `signed_margin_a = score_a(candidate_a) - score_a(candidate_b)`
- `signed_margin_b = score_b(candidate_a) - score_b(candidate_b)`
- `flip_success = signed_margin_a > 0 and signed_margin_b < 0`, assuming A is correct under state A and B under state B.
- `state_sensitivity = abs(signed_margin_a - signed_margin_b)`

Aggregate:

- Flip accuracy.
- Mean signed margin by state.
- Mean state sensitivity.
- Paired bootstrap confidence interval over examples.

## Figures

### Figure 1: Counterfactual Crossover

X-axis: State A, State B.

Y-axis: `score(candidate_a) - score(candidate_b)`.

Lines:

- Mental model: should cross zero.
- Compression-only VAE: should remain flat because visible context is fixed.
- Simple reward: should remain flat or weak.
- Swapped/shuffled z: should reverse or collapse.

### Figure 2: Flip Accuracy Bar

Bars:

- Simple reward baseline.
- Compression-only VAE.
- Mental observed-only.
- Mental correct z.
- Mental swapped/shuffled z.

### Figure 3: Example Heatmap

Rows:

- State A latent.
- State B latent.

Columns:

- Candidate A.
- Candidate B.

Expected heatmap:

```text
             Candidate A   Candidate B
State A z       high           low
State B z       low            high
```

## Run Command

```bash
python projects/sotopia/experiments/run_counterfactual_flip_experiment.py \
  --counterfactual-jsonl /path/to/counterfactual_pairs.jsonl \
  --output-dir projects/sotopia/experiments/counterfactual_flip_results \
  --mental-model-name Qwen/Qwen2.5-7B-Instruct \
  --mental-checkpoint projects/sotopia/checkpoints/coupled_mental_reward_qwen_v3/best \
  --compression-model-name Qwen/Qwen2.5-7B-Instruct \
  --compression-checkpoint projects/sotopia/experiments/runs/stage1/compression_vae_summary_seed42/best \
  --simple-model-name Qwen/Qwen2.5-7B-Instruct \
  --simple-reward-head /path/to/simple_reward_head/reward_head.pth \
  --scoring-dims goal,relationship,knowledge \
  --include-shuffled-z
```

If the simple reward head is not available, omit `--simple-reward-head`; the script will still produce mental-model, compression, and swapped-z plots. For the compression line, train the compression VAE with its default frozen-latent probe phase, because the probe head is what scores candidate A/B responses.

## Reviewer-Facing Interpretation

The clean caption:

> The visible dialogue is fixed, but the latent partner state is changed. The mental model reverses its action preference in the predicted direction, while the simple reward baseline remains largely invariant. This shows that the learned state is used as an intervention-sensitive decision variable, not just as a compressed text summary.

Stronger version with the new baseline:

> The visible dialogue is fixed, but the latent partner state is changed. The mental model reverses its action preference in the predicted direction, while both the compression-only VAE and the simple reward baseline remain largely invariant. This shows that the learned state is an intervention-sensitive decision variable, not merely a compact summary of the observed dialogue.
