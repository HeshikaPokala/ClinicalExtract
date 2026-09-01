# Adapters

> 🤗 Hosted on Hugging Face: [HeshikaPokala/ClinicalExtract-Models](https://huggingface.co/HeshikaPokala/ClinicalExtract-Models) → `adapters/colab_lora/`

LoRA adapter checkpoints from fine-tuning Phi-4-mini on 8,000 synthetic clinical notes (Google Colab T4 GPU, 1 epoch).

## Contents

- `colab_lora/checkpoint-{100,200,...,1000}/` — adapter weights saved every 100 steps
  - `adapter_config.json` — LoRA config (rank, target modules, etc.)
  - `adapter_model.safetensors` — the trained LoRA delta weights
  - `optimizer.pt` — optimizer state (needed only to resume training)
  - `trainer_state.json` — loss curve and training metadata

Checkpoint 500 was selected as the best based on held-out task-level F1 (not training loss).
