# Mental Models for Multi-Agent Reasoning

Research code for learning explicit first- and second-order mental-state
representations and using them to train language and multimodal policies. The
repository consolidates the previously separate SOTOPIA, BigToM, MMRole,
transfer-evaluation, and exploratory pipelines into one reproducible layout.

Large datasets, checkpoints, model caches, and run outputs are intentionally
excluded from Git. Every project uses the same `data/`, `checkpoints/`, and
`runs/` convention, while external benchmark repositories are pinned under
`third_party/`.

## Projects

| Project | Purpose | Status |
|---|---|---|
| [SOTOPIA](projects/sotopia/README.md) | Coupled recursive mental/reward model, GRPO policy training, official social-agent evaluation | Main pipeline |
| [Craigslist-Bargain](projects/craigslist_bargain/README.md) | Zero-shot negotiation transfer through the SOTOPIA competitive split | Transfer evaluation |
| [BigToM](projects/bigtom/README.md) | Text ToM data generation, mental/reward training, latent-prefix SFT and GRPO | Main pipeline |
| [ToMi](projects/tomi/README.md) | Zero-shot synthetic false-belief transfer and latent analysis | Transfer evaluation |
| [FANToM](projects/fantom/README.md) | Multi-party ToM transfer with the official FANToM scorer | Transfer evaluation |
| [OpenToM](projects/opentom/README.md) | Official benchmark transfer through the shared BigToM harness | Transfer evaluation |
| [Hi-ToM](projects/hitom/README.md) | Higher-order ToM transfer through the shared BigToM harness | Transfer evaluation |
| [MMRole](projects/mmrole/README.md) | Multimodal role-playing annotation, mental/reward training, SFT/GRPO/DPO, evaluation | Main pipeline |
| [ToM-SB / AIDA](projects/tom_sb/README.md) | Security-game extension and AIDA evaluation | Research extension |
| [MindPower](projects/mindpower/README.md) | Embodied mental-modeling pipeline scaffold | Research extension |

## Repository layout

```text
Mental-Models/
├── projects/                 # One self-contained folder per dataset/task
│   └── <project>/
│       ├── scripts/          # Canonical runnable entrypoints
│       ├── experiments/      # Ablations and analyses, when applicable
│       ├── data/             # Generated/downloaded data (ignored)
│       ├── checkpoints/      # Model artifacts (ignored)
│       └── runs/             # Evaluation outputs and logs (ignored)
├── third_party/
│   ├── sources.lock.json     # Exact upstream revisions
│   ├── overlays/             # Local source additions
│   ├── patches/              # Minimal upstream modifications
│   └── src/                  # Bootstrapped upstream repositories (ignored)
├── requirements/             # Shared dependency groups
├── environments/             # Compatibility environments
├── tools/                    # Bootstrap, diagnostics, and validation
└── artifacts/                # Hugging Face/model caches (ignored)
```

Run commands from the repository root unless a project README explicitly says
otherwise.

## Installation

### 1. Create the shared environment

The consolidated environment covers SOTOPIA, BigToM and its transfer suites,
MMRole, and the local MindPower pipeline:

```bash
conda env create -f environment.yml
conda activate mental-models
```

The default file targets CUDA 12.1. For another CUDA release, install a
compatible PyTorch build first and then install the five files under
`requirements/`. See [the environment guide](docs/environment.md).

### 2. Fetch pinned benchmark repositories

```bash
./tools/bootstrap_third_party.sh
pip install -e third_party/src/sotopia
pip install -e projects/mindpower
```

To fetch only selected repositories:

```bash
./tools/bootstrap_third_party.sh sotopia bigtom tomi
```

### 3. Configure credentials and caches

```bash
cp .env.example .env
# Edit .env, then export its values into the current shell.
set -a
source .env
set +a
mkdir -p artifacts/huggingface
```

Never commit `.env`, raw key files, model weights, or generated annotations.

### 4. Check the installation

```bash
python tools/doctor.py
python tools/validate_repository.py
```

Use `python tools/doctor.py --strict` after downloading all upstream sources and
installing the complete environment.

## Pipeline overview

The main text pipelines follow the same conceptual sequence:

1. **Prepare data**: download or generate scenarios and normalize them to
   per-turn examples.
2. **Add supervision**: annotate belief, intent, thought, recursive mental
   state, rewards, rationales, and hard negatives as required by the dataset.
3. **Train the mental/reward model**: jointly optimize recursive latent mental
   variables and outcome prediction.
4. **Train the policy**: warm-start with SFT, then optimize with the frozen
   mental/reward model using GRPO; MMRole also contains a DPO stage.
5. **Evaluate**: use the released benchmark split and official scorer whenever
   available. Transfer datasets are evaluation-only.

The exact commands, expected inputs, and checkpoint formats are documented in
each project README. The most common starting points are:

```bash
# SOTOPIA data generation
python projects/sotopia/scripts/generate_sotopia_full_pipeline.py

# BigToM data generation and annotation
bash projects/bigtom/scripts/run_generate_and_annotate.sh

# MMRole data preparation
bash projects/mmrole/scripts/run_pipeline.sh --pilot

# BigToM transfer-suite configuration check (no model loading)
python projects/bigtom/scripts/evaluate_official_benchmarks.py \
  --datasets all \
  --out_dir projects/bigtom/runs/dry_run \
  --dry_run
```

## Data and checkpoints

No large artifact was copied into this repository. Restore or regenerate files
under the project-local ignored directories:

```text
projects/<name>/data/
projects/<name>/checkpoints/
projects/<name>/runs/
```

FANToM's model registry provides the canonical checkpoint names expected by the
transfer scripts. See [data and checkpoint conventions](docs/data-and-checkpoints.md)
for the complete mapping.

## Environments

One Python 3.10 environment is used wherever dependency constraints are
compatible. Two isolated environments remain because their released tooling
pins incompatible core stacks:

- `environments/fantom-official.yml`: the legacy FANToM release environment.
  The primary evaluator normally calls the official scorer bridge from the
  shared environment; use this file only if strict release reproduction is
  required.
- `environments/aida.yml`: AIDA/ToM-SB with Transformers 5 and vLLM.

The VirtualHome or TDW simulators needed for full MindPower collection are also
optional and are not installed by the shared environment.

## Validation

Fast source-only checks:

```bash
python tools/validate_repository.py
python -m compileall -q projects tools
find projects tools -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

After installing the shared environment:

```bash
pytest -q projects/bigtom/tests
pytest -q projects/fantom/tests
bash projects/mindpower/scripts/run_pipeline.sh --dry-run
```

The FANToM asset tests skip large checkpoint checks unless
`MENTAL_MODELS_RUN_ASSET_TESTS=1` is set.

## Provenance

Custom scripts were consolidated without copying local datasets, caches,
checkpoints, or transient run directories. [The source inventory](docs/source-inventory.md)
records where each component came from and which materials were intentionally
excluded. Upstream benchmark revisions are locked in
`third_party/sources.lock.json`.
