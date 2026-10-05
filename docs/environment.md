# Environment strategy

## Shared environment

`environment.yml` is the primary environment for every training and evaluation
pipeline in the paper:

- SOTOPIA and Craigslist-Bargain;
- BigToM, ToMi, and the FANToM transfer harness;
- MMRole visual training and evaluation.

It standardizes on Python 3.10, PyTorch 2.5.1, Transformers 4.53.3, PEFT
0.18.1, and CUDA 12.1. This reconciles the language and vision-language
pipelines while retaining the APIs used by Qwen2.5 and Qwen2.5-VL.

```bash
conda env create -f environment.yml
conda activate mental-models
./tools/bootstrap_third_party.sh
pip install -e third_party/src/sotopia
```

For CPU-only source inspection, create a Python 3.10 environment, install the
CPU PyTorch wheel, and then install each file under `requirements/`. Training
and local 7B-model evaluation require CUDA.

## FANToM release environment

The released FANToM environment pins Python 3.9, OpenAI 0.27, Transformers
4.34, and Torch 2.1, which conflict with the shared training stack. The normal
BigToM-to-FANToM evaluator invokes the official scoring bridge from the shared
environment. Use the isolated environment only for strict release-level
reproduction:

```bash
conda env create -f environments/fantom-official.yml
conda activate fantom-official
```

## Updating an existing environment

Conda does not always reconcile pip packages cleanly across major upgrades. A
fresh environment is preferred. To update an existing development environment:

```bash
conda env update -n mental-models -f environment.yml --prune
python tools/doctor.py --strict
```
