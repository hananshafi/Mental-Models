# Stage-1 auxiliary-loss decomposition (BigToM reward+mental model)

Everything fixed (data, z_dim, budget = 1 epoch, seed); only ONE auxiliary component removed per
run. Held-out scenario-level split (n=5,976). Full = all aux losses.

| Stage-1 configuration | pair-acc | Preference margin | belief-probe | Mental-decode NLL (1st / 2nd) |
|---|---:|---:|---:|---:|
| Full (all aux) | 96.6 | 1.05 | 99.8 | 1.47 / 1.38 |
| − hard negatives | 96.9 | **0.68** (−35%) | 99.7 | 1.47 / 1.38 |
| − mental-state recon | 97.2 | 1.05 | 99.8 | **12.1 / 12.5** |
| − future prediction | 98.3 | 1.08 | 99.8 | 1.47 / 1.38 |
| − anti-bypass heads | 97.5 | 1.03 | 99.9 | 1.47 / 1.37 |

## Confirmed findings (used in rebuttal)
- **Hard negatives → confidence/calibration of preferences.** Removing the hard-negative preference
  term collapses the reward margin 1.05 → 0.68 (−35%) with ranking accuracy unchanged: hard negatives
  make the reward model separate good from bad responses *confidently*, not just correctly. (Consistent
  with the paper's Fig. 8 preference-margin analysis.)
- **Mental-state reconstruction → interpretability.** Removing it blows up the mental-decode NLL
  (1.4 → ~12) — the latent is no longer decodable into belief/intent/thought text — at **zero** cost to
  reward ranking or belief-probe. This loss buys the interpretable belief representation essentially
  for free.

## Honest null results (not claimed)
- **Future prediction** and **anti-bypass heads** show no measurable effect on these Stage-1 metrics
  (margin/NLL/probe within noise). Their intended roles — policy-stage predictive sufficiency and
  reward-routing-through-z — are not probed by reward ranking, so we do not claim a Stage-1 effect.

## Note
Auxiliary losses do not improve pure *ranking accuracy* (saturated, even slightly higher without them);
their value is in **margin, calibration, and interpretability**. The complementary downstream GRPO run
(Full vs − all-aux) tests the aggregate effect on the BigToM TB∧FB benchmark.
