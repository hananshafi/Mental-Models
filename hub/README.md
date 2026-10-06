---
pretty_name: Mental Models for Multi-Agent Systems
language:
- en
- zh
license: other
license_name: mixed-cc-by-4.0-mit-apache-2.0
task_categories:
- text-generation
- question-answering
- visual-question-answering
tags:
- theory-of-mind
- multi-agent
- mental-modeling
- social-reasoning
- role-playing
configs:
- config_name: sotopia-turn-rewards
  default: true
  data_files:
  - split: train
    path: data/sotopia/sotopia_turn_rewards_v3.jsonl
- config_name: sotopia-mental-personas
  data_files:
  - split: train
    path: data/sotopia/mental_model_persona_dataset.jsonl
- config_name: bigtom
  data_files:
  - split: train
    path: data/bigtom/bigtom_qwen_5k_annotated.jsonl
- config_name: mmrole-belief
  data_files:
  - split: train
    path: data/mmrole/train/belief_prediction.jsonl
  - split: validation
    path: data/mmrole/val/belief_prediction.jsonl
  - split: test_in
    path: data/mmrole/test_in/belief_prediction.jsonl
  - split: test_out
    path: data/mmrole/test_out/belief_prediction.jsonl
  - split: test_official
    path: data/mmrole/official_test/belief_prediction.jsonl
- config_name: mmrole-preference
  data_files:
  - split: train
    path: data/mmrole/train/preference_pairs.jsonl
  - split: validation
    path: data/mmrole/val/preference_pairs.jsonl
  - split: test_in
    path: data/mmrole/test_in/preference_pairs.jsonl
  - split: test_out
    path: data/mmrole/test_out/preference_pairs.jsonl
  - split: test_official
    path: data/mmrole/official_test/preference_pairs.jsonl
- config_name: mmrole-salience
  data_files:
  - split: train
    path: data/mmrole/train/salience_prediction.jsonl
  - split: validation
    path: data/mmrole/val/salience_prediction.jsonl
  - split: test_in
    path: data/mmrole/test_in/salience_prediction.jsonl
  - split: test_out
    path: data/mmrole/test_out/salience_prediction.jsonl
  - split: test_official
    path: data/mmrole/official_test/salience_prediction.jsonl
- config_name: mmrole-probe-qa
  data_files:
  - split: train
    path: data/mmrole/train/probe_qa.jsonl
  - split: validation
    path: data/mmrole/val/probe_qa.jsonl
  - split: test_in
    path: data/mmrole/test_in/probe_qa.jsonl
  - split: test_out
    path: data/mmrole/test_out/probe_qa.jsonl
  - split: test_official
    path: data/mmrole/official_test/probe_qa.jsonl
- config_name: mmrole-raw-annotations
  data_files:
  - split: train
    path: data/mmrole/train/raw_annotated.jsonl
  - split: validation
    path: data/mmrole/val/raw_annotated.jsonl
  - split: test_in
    path: data/mmrole/test_in/raw_annotated.jsonl
  - split: test_out
    path: data/mmrole/test_out/raw_annotated.jsonl
- config_name: mmrole-raw-official
  data_files:
  - split: test
    path: data/mmrole/official_test/raw_annotated.jsonl
- config_name: mmrole-reward-belief
  data_files:
  - split: train
    path: data/mmrole/train/belief_prediction_mmrole_reward_openai.jsonl
  - split: validation
    path: data/mmrole/val/belief_prediction_mmrole_reward_openai.jsonl
  - split: test_in
    path: data/mmrole/test_in/belief_prediction_mmrole_reward_openai.jsonl
- config_name: mmrole-reward-preference
  data_files:
  - split: train
    path: data/mmrole/train/preference_pairs_mmrole_reward_openai.jsonl
  - split: validation
    path: data/mmrole/val/preference_pairs_mmrole_reward_openai.jsonl
  - split: test_in
    path: data/mmrole/test_in/preference_pairs_mmrole_reward_openai.jsonl
---

# Mental Models for Multi-Agent Systems

Prepared training and evaluation data for **Mental Models for Multi-Agent
Systems** (NeurIPS 2026) by Hanan Gani, Lulu Shao, and Manmohan Chandraker.

The repository contains the paper-ready SOTOPIA, BigToM, and MMRole data in one
Hugging Face dataset repository. Independent configurations keep their distinct
schemas compatible with the Dataset Viewer.

## Contents

| Benchmark | Configuration | Contents |
|---|---|---|
| SOTOPIA | `sotopia-turn-rewards` | 1,647 interaction episodes with 16,166 turn-level reward records and mental-state supervision |
| SOTOPIA | `sotopia-mental-personas` | 500 mental-model persona examples |
| BigToM | `bigtom` | 9,964 annotated conditions from 4,982 paired scenarios |
| MMRole | `mmrole-belief` | First- and second-order belief targets |
| MMRole | `mmrole-preference` | Preferred responses and hard negatives |
| MMRole | `mmrole-salience` | Visual perspective and salience targets |
| MMRole | `mmrole-probe-qa` | Theory-of-Mind diagnostic questions |
| MMRole | `mmrole-raw-annotations` | Full validated mental-state annotations |
| MMRole | `mmrole-raw-official` | Official test annotations and test metadata |
| MMRole | `mmrole-reward-belief` | Belief examples with eight-dimensional reward labels |
| MMRole | `mmrole-reward-preference` | Preference pairs with eight-dimensional reward labels |

## Loading

```python
from datasets import load_dataset

sotopia = load_dataset(
    "hanangani/Mental-Model-Annotation-Dataset",
    "sotopia-turn-rewards",
)
bigtom = load_dataset("hanangani/Mental-Model-Annotation-Dataset", "bigtom")
mmrole = load_dataset("hanangani/Mental-Model-Annotation-Dataset", "mmrole-belief")
```

Download the original files without schema conversion when using the released
training scripts:

```bash
hf download hanangani/Mental-Model-Annotation-Dataset \
  --repo-type dataset \
  --local-dir Mental-Models-data
```

## MMRole images

This repository does not duplicate MMRole's 11,032 source images. MMRole rows
retain the upstream `image` and `image_local` references. Download the images
from [`YanqiDai/MMRole_dataset`](https://huggingface.co/datasets/YanqiDai/MMRole_dataset)
and follow its instructions for any referenced MS-COCO files.

## Validation

All released JSONL records were parsed before upload. `MANIFEST.json` records
the row count, byte size, and SHA-256 checksum of every staged file. Temporary
files, failed annotation attempts, smoke tests, model checkpoints, and API
credentials are excluded.

## Sources and licenses

This is a derived, mixed-source research dataset. See
[`DATA_SOURCES.md`](DATA_SOURCES.md) for provenance, upstream licenses, and
redistribution notes. No additional rights are granted for upstream content.

## Citation

```bibtex
@inproceedings{gani2026mentalmodels,
  title     = {Mental Models for Multi-Agent Systems},
  author    = {Gani, Hanan and Shao, Lulu and Chandraker, Manmohan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
