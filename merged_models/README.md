# Merged Models

> 🤗 Hosted on Hugging Face: [HeshikaPokala/ClinicalExtract-Models](https://huggingface.co/HeshikaPokala/ClinicalExtract-Models) → `merged_model/`

The LoRA adapter (checkpoint 500) merged into the Phi-4-mini base model weights, producing a standalone fine-tuned model in standard Hugging Face format.

## Contents (`phi4mini_lora_ckpt500/`)

| File | Description |
|---|---|
| `model.safetensors` | Full merged model weights (~7.1 GB) |
| `config.json` | Model architecture config |
| `tokenizer.json` / `tokenizer_config.json` | Tokenizer |
| `generation_config.json` | Default generation parameters |
| `chat_template.jinja` | Chat template for inference |

## How it was created

```bash
python src/merge_lora_adapter.py
```

The merged model was then converted to GGUF and quantized — see `gguf_models/`.
