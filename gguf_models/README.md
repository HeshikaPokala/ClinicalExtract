# GGUF Models

> 🤗 Hosted on Hugging Face: [HeshikaPokala/ClinicalExtract-Models](https://huggingface.co/HeshikaPokala/ClinicalExtract-Models) → `gguf/`

GGUF-format versions of the fine-tuned Phi-4-mini model, ready to run with [llama.cpp](https://github.com/ggerganov/llama.cpp) or [Ollama](https://ollama.com/).

## Files

| File | Size | Description |
|---|---|---|
| `phi4mini_lora_ckpt500_fp16.gguf` | ~7.2 GB | Full precision (FP16) — best quality, high memory |
| `phi4mini_lora_ckpt500_q4km.gguf` | ~2.3 GB | Q4_K_M quantized — recommended for local inference |
| `Modelfile` | — | Ollama Modelfile to serve the quantized model |

## Usage with Ollama

```bash
ollama create clinicalextract -f Modelfile
ollama run clinicalextract
```
