# SOTOPIA Stage-1 aux-loss decomposition (matched epoch_0)

Single well-defined metric: mean Pearson correlation between predicted and gold 7-dim
SOTOPIA reward vector, on held-out val (n=1508). Everything fixed (data, z_dim, LoRA config,
10-epoch LR schedule); only ONE component removed per cell; all evaluated at the identical
epoch_0 checkpoint.

| Configuration | Reward-regression corr (mean) | Δ vs Full | Hard-neg pref acc |
|---|---:|---:|---:|
| **Full (all aux)** | **0.285** | — | 100.0% |
| − mental-state recon | 0.202 | **−0.083 (−29%)** | 100.0% |
| − rationale/explanation | 0.211 | **−0.074 (−26%)** | 100.0% |
| − hard negatives | 0.266 | −0.019 (−7%) | **81.0% (−19pp)** |

## Per-dimension correlation

| dim | Full | −mental | −hardneg | −rationale |
|---|---:|---:|---:|---:|
| believability | 0.155 | 0.003 | 0.085 | 0.018 |
| relationship | 0.503 | 0.385 | 0.564 | 0.388 |
| knowledge | 0.512 | 0.367 | 0.479 | 0.391 |
| secret | 0.084 | 0.067 | 0.045 | 0.071 |
| social_rules | 0.041 | 0.024 | 0.021 | 0.004 |
| financial | 0.124 | 0.176 | 0.176 | 0.093 |
| goal | 0.575 | 0.508 | 0.488 | 0.514 |

## Reading

- **Mental-state reconstruction and rationale/explanation are the two components that most
  improve reward-prediction quality itself.** Removing either drops mean correlation by
  ~26-29%, on nearly every dimension (largest drops: believability, knowledge — the most
  ToM-flavored dimensions). This is a genuine, non-circular effect: the metric is reward
  regression, not decode quality, so this shows the mental/rationale supervision makes the
  reward model a *better predictor of the 7-dim SOTOPIA reward*, not merely more interpretable.
- **Hard negatives have a smaller effect on regression correlation (−7%) but a large, clean
  effect on the metric they specifically train: preference discrimination.** Hard-negative
  preference accuracy collapses 100% → 81% once the preference loss is removed — the model
  is no longer explicitly taught to rank the hard negative below the good response, and this
  shows up directly at eval time.
- Together, all three ablated components move the same single metric in the expected direction,
  with distinguishable effect sizes: mental-recon ≈ rationale > hard-negatives (on regression
  correlation), while hard-negatives dominates on preference-discrimination.

Note: this is a matched 1-epoch (epoch_0) snapshot for early signal; the `−all-aux` combined
cell and later epochs are still training.
