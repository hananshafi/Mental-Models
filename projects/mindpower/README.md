# MindPower embodied extension

Installable scaffold for collecting embodied episodes, selecting intervention
points, annotating belief/desire/intention state, building training data, and
connecting that data to mental/reward, SFT, GRPO, and evaluation stages.

This is a research extension rather than a main-paper benchmark result.

## Install

```bash
pip install -e projects/mindpower
```

Copy `configs/paths.example.yaml` when using custom simulator or export paths.

## Dry run

The dry run exercises the file and schema pipeline without launching a
simulator:

```bash
bash projects/mindpower/scripts/run_pipeline.sh --dry-run
```

## Exported VirtualHome data

```bash
python projects/mindpower/scripts/inspect_virtualhome_export.py --help
python projects/mindpower/scripts/step1_collect_episodes.py \
  --simulator virtualhome \
  --virtualhome_dataset_root /path/to/virtualhome/export
```

Then run the intervention, annotation, and training-data builders individually
or use `run_pipeline.sh`. `step3_annotate_tom_openai.py` provides API-based ToM
annotation; export `OPENAI_API_KEY` rather than storing it in a config file.

## Simulator dependencies

Live VirtualHome and TDW collection are intentionally excluded from the shared
environment. Install the simulator version matching your Unity build in an
isolated environment. Pre-exported JSON episodes can be processed entirely in
the shared `mental-models` environment.

See `docs/original_runbook.md` for schema details and the current scope of the
training scaffolds.
