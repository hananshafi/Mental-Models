# ToMi transfer

ToMi is an evaluation-only transfer target for BigToM-trained mental models and
policies. The released repository is pinned at `third_party/src/tomi`.

```bash
./tools/bootstrap_third_party.sh tomi
python projects/bigtom/scripts/evaluate_official_benchmarks.py \
  --datasets tomi \
  --mode grpo \
  --stage1_ckpt projects/bigtom/checkpoints/stage1_qwen_5k/best_ckpt \
  --policy_ckpt projects/bigtom/checkpoints/stage4_qwen_5k/step_300 \
  --out_dir projects/tomi/runs/bigtom_grpo
```

The harness aligns the released story and trace files and reports exact-match
metrics through the ToMi protocol bridge. Latent transfer diagnostics are in
`projects/bigtom/scripts/analyse_tomi_latents.py`; the E1 belief minimal-pair
study is under `projects/bigtom/experiments/e1/`.
