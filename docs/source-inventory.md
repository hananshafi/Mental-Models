# Consolidation inventory

This repository contains source code for the datasets and evaluations reported
in *Mental Models for Multi-Agent Systems*. Machine-specific paths, caches,
generated data, checkpoints, and transient logs are excluded.

| Destination | Retained custom material | Deliberately excluded |
|---|---|---|
| `projects/sotopia/` | canonical Stage 1/2/3 scripts, latent analyses, causal and auxiliary ablations, supervision-fraction experiments | installed SOTOPIA checkout, Redis state, generated annotations, checkpoints, run directories |
| `projects/bigtom/` | data generation, annotation, mental/reward training, latent-prefix SFT, GRPO, official BigToM evaluation, and BigToM-to-ToMi/FANToM transfer | upstream BigToM checkout, generated CSV/JSONL, model artifacts, full evaluation outputs |
| `projects/mmrole/` | download/restructure/annotation pipeline, validation, reward/SFT/GRPO/DPO training, response generation, and official evaluation | raw MMRole and COCO images, generated annotations, training data, visual checkpoints |
| `projects/fantom/` | compatibility audit, inference, official scoring, launchers, tests, and compact result reports | released FANToM checkout, raw prediction JSONL, local checkpoint registry |

`projects/tomi/` and `projects/craigslist_bargain/` contain focused transfer
runbooks and reuse the canonical BigToM and SOTOPIA evaluators rather than
duplicating training code.

## Pinned upstream code

`third_party/sources.lock.json` records exact revisions for:

- SOTOPIA;
- BigToM;
- FANToM;
- ToMi.

The SOTOPIA integration is represented explicitly by the patch and overlays
under `third_party/`. No hidden source edits are made during bootstrap.

## Historical material

Files under `legacy/` preserve older variants needed to interpret paper
experiments but are not canonical entrypoints. Compact reports are retained
where useful; raw predictions and large generated outputs are excluded.
