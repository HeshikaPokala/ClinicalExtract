"""
Merges a LoRA adapter into its base model, producing a standalone fine-tuned
model (no PEFT needed at inference time). One-shot operation: load, merge, save, exit.

Usage:
    .venv/bin/python src/merge_lora_adapter.py --adapter adapters/colab_lora/checkpoint-500 --out merged_models/phi4mini_lora_ckpt500
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "microsoft/Phi-4-mini-instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adapter_path = ROOT / args.adapter
    out_path = ROOT / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {BASE_MODEL} (fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)

    actual_dtype = next(base_model.parameters()).dtype
    print(f"Base model loaded in dtype: {actual_dtype}")
    if actual_dtype != torch.float16:
        raise RuntimeError(f"Expected float16, got {actual_dtype}")

    print(f"Loading adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {out_path}...")
    merged_model.save_pretrained(out_path, safe_serialization=True)
    tokenizer.save_pretrained(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
