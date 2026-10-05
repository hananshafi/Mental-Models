# Environment strategy

## Shared environment

`environment.yml` is the primary environment for:

- SOTOPIA training and evaluation;
- BigToM training and the ToMi, OpenToM, Hi-ToM, and FANToM transfer harness;
- MMRole visual training and evaluation;
- MindPower preprocessing, annotation, and non-simulator code;
- most ToM-SB data generation and local model training.

It standardizes on Python 3.10, PyTorch 2.5.1, Transformers 4.53.3, PEFT
0.18.1, and CUDA 12.1. This reconciles the previously separate text and visual
environments while retaining the APIs used by Qwen2.5 and Qwen2.5-VL.

```bash
conda env create -f environment.yml
conda activate mental-models
./tools/bootstrap_third_party.sh
pip install -e third_party/src/sotopia
pip install -e projects/mindpower
```

For CPU-only inspection, create a Python 3.10 environment, install the CPU
PyTorch wheel, then install each file in `requirements/`. Training and local
7B-model evaluation require CUDA.

## Why two compatibility environments remain

### FANToM official release

The released FANToM environment pins Python 3.9, OpenAI 0.27, Transformers
4.34, and Torch 2.1. These conflict with the training stack. The shared
BigToM/FANToM bridge reproduces official scoring without switching environments
for normal use. For strict release-level reproduction:

```bash
conda env create -f environments/fantom-official.yml
conda activate fantom-official
```

### AIDA

AIDA pins Transformers 5, Torch 2.8, TRL 1.0, and vLLM 0.10.2. Keep these
packages isolated from the primary Transformers 4 stack:

```bash
conda env create -f environments/aida.yml
conda activate mental-models-aida
```

## Optional simulators

MindPower's `--dry-run` and exported-dataset paths work in the shared
environment. Live collection requires simulator-specific installation:

- VirtualHome: install the Unity executable and Python tooling documented by
  the VirtualHome project.
- TDW: install `tdw` in a separate environment if its version conflicts with
  your simulator build.

These packages are excluded from the default environment because they are not
needed for the paper's language and multimodal pipelines.

## Updating an existing environment

Conda does not always reconcile pip packages cleanly across major upgrades. For
reproducible runs, create a fresh environment. For a development environment:

```bash
conda env update -n mental-models -f environment.yml --prune
```

Run `python tools/doctor.py --strict` afterward.
