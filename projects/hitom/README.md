# Hi-ToM transfer

Hi-ToM tests higher-order belief reasoning without target-dataset training. It
uses the same BigToM mental encoder and policy checkpoints as the other transfer
benchmarks.

```bash
./tools/bootstrap_third_party.sh hitom
python projects/bigtom/scripts/evaluate_official_benchmarks.py \
  --datasets hitom \
  --mode grpo \
  --stage1_ckpt projects/bigtom/checkpoints/stage1_qwen_5k/best_ckpt \
  --policy_ckpt projects/bigtom/checkpoints/stage4_qwen_5k/step_300 \
  --out_dir projects/hitom/runs/bigtom_grpo
```

The default `choice` inference mode scores released alternatives locally. Use
`--hitom_inference official` to reproduce the release prompt style.
