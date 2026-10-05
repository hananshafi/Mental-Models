# OpenToM transfer

OpenToM is evaluated zero-shot with BigToM-trained base, SFT, or GRPO policies.
The shared harness writes predictions and invokes the released scorer.

```bash
./tools/bootstrap_third_party.sh opentom
python projects/bigtom/scripts/evaluate_official_benchmarks.py \
  --datasets opentom \
  --mode grpo \
  --stage1_ckpt projects/bigtom/checkpoints/stage1_qwen_5k/best_ckpt \
  --policy_ckpt projects/bigtom/checkpoints/stage4_qwen_5k/step_300 \
  --out_dir projects/opentom/runs/bigtom_grpo
```

Use `--opentom_root`, `--opentom_test_path`, or `--opentom_scorer_path` only
when overriding the pinned release checkout.
