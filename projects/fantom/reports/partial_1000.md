# FANToM First-1,000 Results

These are preliminary official-scorer results on the first 1,000 flattened probes. Full evaluations continue independently.

| Model | Status | N | All* | All | Belief Choice | Belief Dist. | First Order | Second Order | Control First | Control Second |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen25_7b_base` | completed | 1000 | 0.0 | 0.0 | 20.7 | 24.1 | 22.4 | 26.3 | 90.0 | 100.0 |
| `bigtom_sft_epoch1` | completed | 1000 | 0.0 | 0.0 | 31.0 | 24.1 | 22.4 | 26.3 | 80.0 | 100.0 |
| `bigtom_grpo_step300` | completed | 1000 | 0.0 | 0.0 | 31.0 | 27.6 | 22.4 | 34.2 | 90.0 | 90.0 |
| `sotopia_qwen_grpo_best` | completed | 1000 | 0.0 | 0.0 | 17.2 | 40.2 | 30.6 | 52.6 | 60.0 | 95.0 |

Caveats: this is the first contiguous prefix rather than a random or
stratified sample. The 1,000th probe can cut through a question set, so
`All*` and `All` are interim diagnostics rather than final leaderboard
numbers. The immutable snapshots and detailed JSON summaries are stored
under `runs/<model>/partial_1000/`.
