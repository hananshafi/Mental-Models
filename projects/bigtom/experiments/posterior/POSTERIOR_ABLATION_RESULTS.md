# Posterior-family ablation (BigToM, Stage-1 mental+reward model)

Everything fixed (data = 9,964-row BigToM 5k set → ~54k samples, z_dim=128, reward coupling,
1 epoch, seed, effective batch 32); **only the posterior family/structure is swapped**.
Held-out **scenario-level** split (val n=5,976). KL for low-rank validated vs. brute-force
full-covariance (diff ~5e-7).

| Posterior variant | Pairwise reward acc. | Mean margin | Belief-probe acc. |
|---|---:|---:|---:|
| Diagonal Gaussian (ours) | 96.6 | 1.05 | 99.8 |
| Low-rank / full covariance ($\Sigma=\mathrm{diag}(\sigma^2)+UU^\top$) | 96.8 | 1.05 | 99.9 |
| Deterministic (z=μ, no KL) | 99.2 | 1.33 | 99.9 |
| Parallel (non-recursive z²) | 97.1 | 1.01 | 99.7 |

## What is clean (use in rebuttal)
- **A full/low-rank-covariance posterior gives no measurable benefit over diagonal**: 96.8 vs 96.6
  pairwise (within ±0.65% sampling noise for n=5,976), identical mean margin (1.05), tied belief
  probe. Modeling within-level correlations directly does not help — **the diagonal posterior is
  sufficient**, which is exactly the reviewer's question. This is the headline result.

## What is nuanced (do NOT overclaim)
- **Deterministic is not worse on this in-distribution metric** (99.2 vs 96.6) — expected: with no KL
  pressure the latent can sharpen the reward margin on held-out scenarios drawn from the same
  distribution. This metric does not probe the property the variational posterior is *for* —
  calibrated uncertainty over partner state, which matters for (i) policy-stage GRPO guidance and
  (ii) OOD transfer. To show the deterministic downside we need a calibration (ECE/NLL) or OOD
  metric, not in-distribution ranking. So this run does **not** by itself justify the KL term.
- **Parallel ≈ diagonal here** because the eval metric (belief probe on z¹, first-order reward
  ranking) does not isolate second-order reasoning, which is where the recursion is designed to help.
  A second-order-specific probe (or ToMi/BigToM 2nd-order split) is needed to show the recursion's
  value; this metric can't.

## Takeaway for the reviewer response
The direct answer — *"does an alternative to the diagonal Gaussian help?"* — is **no**: a structured
full-covariance posterior matches diagonal on held-out reward ranking, margin, and belief recovery,
at higher cost. The variational (vs deterministic) and recursive (vs parallel) choices should be
defended on calibration/second-order grounds with the additional metrics above, not on this
in-distribution ranking number.
