# MindPower Scaffold

Scaffolding for a MindPower-style embodied mental-modeling pipeline under `projects/mindpower`.

This project is shaped to match the staged workflow you already use in `sotopia/` and `mmrole/`, but adapted to simulator-backed embodied data:

1. collect simulator episodes
2. build intervention points
3. annotate BDI-style mental state targets
3b. annotate explicit ToM targets
3c. optionally replace heuristic ToM with OpenAI-compatible LLM belief tracking
4. build training splits and formats
5. stage reward / SFT / GRPO / evaluation scripts

The first pass is intentionally simulator-safe:

- `VirtualHome` and `TDW` adapters are included as scaffolds.
- `VirtualHome` is the default collection target.
- VirtualHome exported data ingestion is supported now.
- `--dry-run` mode produces schema-valid mock episodes so we can develop the rest of the pipeline before the real data stack is wired in.

## Project Layout

```text
mindpower/
├── configs/
├── data/
│   ├── raw/
│   └── intermediate/
├── scripts/
├── src/mindpower/
│   └── simulators/
├── training_data/
├── checkpoints/
└── logs/
```

## Intended Stages

- `scripts/step1_collect_episodes.py`
  - collect episodes from VirtualHome or TDW
- `scripts/step2_build_intervention_points.py`
  - convert episodes into robot-assistance decision points
- `scripts/step3_annotate_bdi.py`
  - create `perception -> belief -> desire -> intention -> decision -> action` targets
  - also emits explicit ToM annotations:
    - visual perspective asymmetry
    - first-order belief
    - second-order belief
    - false-belief risk
    - intervention criticality
- `scripts/step3_annotate_tom_openai.py`
  - OpenAI-compatible LLM annotator for belief tracking
  - defaults to the NRP endpoint and `Qwen/Qwen3.5-397B-A17B-FP8`
- `scripts/render_virtualhome_from_script.py`
  - render camera views from a VirtualHome scene or executable program
  - useful once public MindPower-style VirtualHome scripts are available
- `scripts/step4_build_training_data.py`
  - produce `hierarchy_prediction`, `preference_pairs`, `probe_qa`, and `action_targets`
- `scripts/stage0_reward_mindpower.py`
  - reward-model scaffold
- `scripts/stage1_sft_mindpower.py`
  - SFT scaffold
- `scripts/stage2_grpo_mindpower.py`
  - GRPO scaffold
- `scripts/eval_mindpower.py`
  - evaluator scaffold

## Quickstart

Dry-run end-to-end:

```bash
cd projects/mindpower
bash scripts/run_pipeline.sh --dry-run
```

That will:

- create mock VirtualHome-style episodes from seed tasks
- build intervention examples
- create heuristic BDI annotations
- create heuristic ToM annotations
- write train/val JSONL splits

Import real VirtualHome exported programs:

```bash
python projects/mindpower/scripts/step1_collect_episodes.py \
  --simulator virtualhome \
  --virtualhome_dataset_root /path/to/VirtualHome/programs_processed_precond_nograb_morepreconds \
  --output_path projects/mindpower/data/raw/virtualhome_episodes.jsonl
```

The importer looks for the standard VirtualHome export layout:

```text
<export_root>/
├── executable_programs/
├── state_list/
├── initstate/
└── withoutconds/
```

If `state_list/` JSON exists, it will be converted into object and agent observations.
If it doesn't, the collector falls back to action-derived observations so the rest of the pipeline can still run.

LLM-based ToM annotation:

```bash
export MINDPOWER_NRP_API_KEY="..."
python projects/mindpower/scripts/step3_annotate_tom_openai.py \
  --input_path projects/mindpower/data/intermediate/intervention_points.jsonl \
  --output_path projects/mindpower/data/intermediate/tom_annotations_llm.jsonl \
  --provider_config_path projects/mindpower/configs/nrp_provider.json \
  --model Qwen/Qwen3.5-397B-A17B-FP8
```

Then build training data from the LLM annotations:

```bash
python projects/mindpower/scripts/step4_build_training_data.py \
  --tom_input_path projects/mindpower/data/intermediate/tom_annotations_llm.jsonl
```

Render VirtualHome camera views once executable scripts are available:

```bash
python projects/mindpower/scripts/render_virtualhome_from_script.py \
  --dataset_root /path/to/VirtualHome/programs_processed_precond_nograb_morepreconds \
  --script_path TrimmedTestScene1_graph/results_intentions_march-13-18/file123_0.txt \
  --unity_binary /path/to/VirtualHome.exe \
  --camera_ids 0,1,2 \
  --dump_before_execution
```

## Public MindPower HF Release

As of April 23, 2026, the public Hugging Face dataset config is a plain text dataset with a single
`text` field and `90` test rows. The published files currently resolve to annotation `.txt` files
rather than directly downloadable videos or VirtualHome scripts. That means:

- you can run text-only transfer evaluation now
- you cannot reconstruct the original images from the HF release alone
- you can render VirtualHome images later if executable scripts or exported programs are released

## Simulator Notes

- `VirtualHome` is usually installed from source rather than PyPI.
- `TDW` has a Python package and a simulator build.
- This scaffold keeps those integrations behind adapter classes so the rest of the pipeline can mature independently.

## Next Implementation Pass

The most useful next step is to replace the dry-run `VirtualHomeAdapter.collect_episode()` path with a real collector that:

- loads executable programs / goals
- steps the simulator
- records egocentric or allocentric frames
- exports symbolic state and action traces
- writes them into the episode schema already defined here
