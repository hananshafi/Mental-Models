# FANToM First-5,000 Results

These are preliminary official-scorer results on the first 5,000 flattened probes. Full evaluations continue independently.

| Model | Status | N | All* | All | Belief Choice | Belief Dist. | First Order | Second Order | Control First | Control Second |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen25_7b_base` | completed | 5000 | 0.0 | 0.0 | 25.4 | 35.2 | 26.8 | 49.7 | 79.5 | 83.0 |
| `bigtom_sft_epoch1` | completed | 5000 | 0.0 | 0.0 | 26.5 | 33.8 | 29.4 | 41.4 | 80.8 | 80.4 |
| `bigtom_grpo_step300` | completed | 5000 | 0.0 | 0.0 | 27.9 | 38.0 | 29.7 | 52.2 | 82.1 | 85.7 |
| `sotopia_qwen_grpo_best` | completed | 5000 | 0.0 | 0.0 | 22.8 | 41.1 | 30.5 | 59.2 | 76.9 | 75.9 |

Caveats: this is the first contiguous prefix rather than a random or
stratified sample. The prefix boundary can cut through a question set, so
`All*` and `All` are interim diagnostics rather than final leaderboard
numbers. The immutable snapshots and detailed JSON summaries are stored
under `runs/<model>/partial_5000/`.
