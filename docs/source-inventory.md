# Consolidation inventory

This repository contains source code only. The consolidation imported the
current custom scripts from the research workspace and removed machine-specific
paths, caches, generated data, checkpoints, and transient logs.

| Destination | Imported custom material | Deliberately excluded |
|---|---|---|
| `projects/sotopia/` | canonical Stage 1/2/3 scripts, latent analyses, causal/auxiliary ablations, supervision-fraction experiment, legacy v2 scripts | installed SOTOPIA checkout, Redis state, generated annotations, checkpoints, run directories |
| `projects/bigtom/` | data generation, annotation, Stage 1/3/4 training, official and transfer evaluators, posterior and E1 analyses, unit fixtures | upstream BigToM checkout, generated CSV/JSONL, model artifacts, full evaluation outputs |
| `projects/mmrole/` | download/restructure/annotation pipeline, validation, Stage 0/1/2/3 training, response generation and official evaluation | raw MMRole/COCO images, annotations, training data, visual checkpoints |
| `projects/fantom/` | compatibility audit, inference, partial/full official scoring, launchers, concise historical reports | released FANToM checkout, prediction JSONL, local checkpoint registry |
| `projects/tom_sb/` | generation, structured mental/reward training, SFT/GRPO, pairwise evaluation | AIDA checkout, generated security-game data and model outputs |
| `projects/mindpower/` | installable preprocessing package, collection/annotation builders, training/evaluation scaffolds | simulator binaries, exported episodes, rendered frames, model outputs |

Transfer-only project folders (`tomi`, `opentom`, `hitom`, and
`craigslist_bargain`) contain focused runbooks and point to the shared canonical
evaluator instead of duplicating code.

## Upstream code

The following repositories are referenced by exact commit in
`third_party/sources.lock.json`:

- SOTOPIA
- BigToM
- FANToM
- Hi-ToM
- OpenToM
- ToMi
- AIDA (Double Agent Defenders)

Local changes required by SOTOPIA and AIDA live as explicit patches or overlays
under `third_party/`; no hidden edits are made during bootstrap.

## Historical material

Files under `legacy/` preserve older runnable variants for provenance but are
not canonical entrypoints. Markdown files under `reports/` and selected
experiment directories retain compact result summaries; raw predictions and
large generated outputs are excluded.
