# Data Sources and Licensing

This repository contains derived annotations and task-specific training views
used by *Mental Models for Multi-Agent Systems*. It combines sources with
different licenses, so the dataset card uses `license: other` rather than
assigning one license to all records.

## SOTOPIA

- Upstream framework: [sotopia-lab/sotopia](https://github.com/sotopia-lab/sotopia)
- Upstream interactive-learning data: [sotopia-lab/sotopia-pi](https://github.com/sotopia-lab/sotopia-pi)
- Upstream licenses: MIT for SOTOPIA and Apache-2.0 for SOTOPIA-pi
- Released additions: turn-level utility targets, recursive mental-state
  supervision, rationales, and hard-negative information used by the paper

## BigToM

- Upstream repository: [cicl-stanford/procedural-evals-tom](https://github.com/cicl-stanford/procedural-evals-tom)
- Upstream license: MIT
- Released additions: first- and second-order belief descriptions, rationales,
  observer metadata, and fields used by the coupled mental/reward objectives

## MMRole

- Upstream repository: [YanqiDai/MMRole](https://github.com/YanqiDai/MMRole)
- Upstream dataset: [YanqiDai/MMRole_dataset](https://huggingface.co/datasets/YanqiDai/MMRole_dataset)
- Upstream dataset metadata: MIT
- Released additions: belief, preference, salience, probe, and reward-training
  views produced from the validated annotations used by the paper
- Media: upstream MMRole and MS-COCO images are not redistributed here

## Generated supervision

The mental-state, utility, rationale, hard-negative, and reward fields include
machine-generated supervision. The paper describes the generators, automated
validation, filtering, and blinded human audit. These labels should be treated
as model-generated annotations rather than direct measurements of a person's
private mental state.

Users are responsible for complying with each upstream license and the terms
that apply to any separately downloaded media.
