# Baseline SLM Comparison (Zero-Shot, No Fine-Tuning)

Evaluated on 60 held-out examples from `dataset/processed/eval.jsonl`.

| Model | Diagnosis F1 | Medication F1 | P50 Latency | P95 Latency | Tokens/sec | Peak Memory | JSON Parse Failures |
|---|---|---|---|---|---|---|---|
| Llama 3.2 3B | 0.15 | 0.47 | 2681 ms | 5835 ms | 40.8 | 2.17 GB | 0.0% |
| Phi-4-mini | 0.18 | 0.49 | 3726 ms | 11756 ms | 27.3 | 2.65 GB | 0.0% |
| Ministral 8B (Q4) | 0.15 | 0.49 | 7046 ms | 19635 ms | 16.6 | 4.68 GB | 0.0% |

## Precision / Recall detail

| Model | Diag P | Diag R | Med P | Med R |
|---|---|---|---|---|
| Llama 3.2 3B | 0.29 | 0.11 | 0.45 | 0.62 |
| Phi-4-mini | 0.32 | 0.14 | 0.47 | 0.62 |
| Ministral 8B (Q4) | 0.31 | 0.12 | 0.47 | 0.62 |
