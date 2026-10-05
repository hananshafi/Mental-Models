# Craigslist-Bargain transfer

Craigslist-Bargain is the competitive negotiation subset of the official
SOTOPIA evaluation. A policy trained on SOTOPIA-derived mental supervision is
evaluated zero-shot; there is no separate training pipeline or duplicated
evaluator in this folder.

```bash
export OPENAI_API_KEY=...
python projects/sotopia/scripts/stage3_evaluate_sotopia.py \
  --policy_model_name Qwen/Qwen2.5-7B-Instruct \
  --policy_adapter_path projects/sotopia/checkpoints/grpo_agent_qwen_v3/best \
  --use_hf \
  --deduplicate_envs \
  --task competitive \
  --partner_model gpt-4o-mini \
  --judge_model gpt-4o \
  --output_path projects/craigslist_bargain/runs/qwen_grpo.jsonl \
  --gpu 0
```

The evaluator identifies competitive episodes by their Craigslist environment
codename and uses the standard SOTOPIA partner and judge protocol.
