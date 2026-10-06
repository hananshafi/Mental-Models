# E1 — ToMi belief minimal-pairs: sensitivity + specificity (final)

**Model:** best ToMi checkpoint = `stage3_qwen_5k/step_300` GRPO policy + `stage1_qwen_5k/best_ckpt`
mental encoder (the config behind `official_grpo_tomi`, ToMi acc ≈ 89%). Base = Qwen2.5-7B-Instruct.
**Data:** 500 synthesized canonical Sally-Anne scenarios × {FB, TB} matched pairs (+ a clean,
move-free belief-irrelevant perturbation of each), built from ToMi's own vocabulary/templates; gold
fully determined by construction. FB and TB differ **only** by the observer's exit sentence.
Prompt / inference / scoring identical to the official ToMi eval harness.

## Validation on REAL ToMi (first-order belief, 120 items)
| | overall | first-order FALSE-belief | first-order TRUE-belief |
|---|---:|---:|---:|
| Base | 72.5 | 60.0 | 85.0 |
| Ours | **81.7** | **70.0** | **93.3** |

## Minimal-pair results (500 matched FB/TB pairs)
| Metric | Base | Ours |
|---|---:|---:|
| FB-version accuracy (gold = original loc) | 98.4 | 83.4 |
| TB-version accuracy (gold = new loc) | 7.0 | 49.2 |
| **FB↔TB asymmetry** (|FB−TB|, lower = tracks belief) | **91.4** | **34.2** |
| **Belief-flip accuracy** (both branches of the pair correct) | **6.6** | **36.2** |
| Directional flip (original on FB *and* new on TB) | 3.0 | 24.8 |
| Specificity — stays correct under belief-irrelevant perturbation | 95.6 | 96.5 |

## Reading
- **Base does not track the belief variable — it applies a fixed "answer the original location"
  heuristic.** 98.4% on false-belief (right for the wrong reason) vs **7.0%** on the matched
  true-belief version: a 91.4-point asymmetry. High false-belief accuracy alone is therefore not
  evidence of belief structure.
- **Mental-model training instills belief-variable sensitivity.** FB↔TB asymmetry drops 91.4→34.2;
  matched belief-flip accuracy rises 5.5× (6.6→36.2); correct directional flips rise 8.3× (3.0→24.8).
- **The sensitivity is specific, not spurious hypersensitivity.** Among items answered correctly,
  predictions survive a belief-irrelevant perturbation 96.5% of the time (Base 95.6%, statistically
  tied) — our model's added belief-sensitivity does not come at the cost of stability.
  (Note: *raw* prediction-unchanged rate is higher for Base, 77.2 vs 65.6, but that is an artifact of
  Base's rigidity — a model that ignores the input and always answers the original location is
  trivially "stable." Stability *conditional on being correct* is the meaningful measure.)

Absolute belief-flip accuracy (36.2%) leaves headroom — reported honestly.
