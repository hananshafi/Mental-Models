# FANToM Rebuttal Results

Official FANToM scores are percentages. The primary task contains
information asymmetry; the control task does not.

| Model | Source | All* | All | Belief Choice | First Order | Second Order | Control First | Control Second |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bigtom_grpo_step300` | prior full run | 0.0 | 0.2 | 26.9 | 28.7 | 43.0 | 81.1 | 82.4 |

## Partial Runs

- `bigtom_grpo_step300`: local-only or partial run at `projects/fantom/runs/smoke_bigtom_grpo_step300/summary.json`
- `sotopia_qwen_grpo_best`: local-only or partial run at `projects/fantom/runs/smoke_sotopia_qwen_grpo_best/summary.json`

The SOTOPIA policy-only control must not be described as latent-conditioned
at inference. Only the BigToM SFT/GRPO entries use the learned `z1`/`z2`
mental prefix when answering FANToM questions.
