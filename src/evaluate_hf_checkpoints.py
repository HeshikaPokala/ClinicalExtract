"""
Evaluates HF/PEFT LoRA checkpoints (from the Colab training runs) on held-out
eval.jsonl. Loads the base model once, then swaps in each checkpoint's adapter
weights via PEFT's multi-adapter support (load_adapter/set_adapter) rather than
reloading the full base model each time.

Usage:
    .venv/bin/python src/evaluate_hf_checkpoints.py \
        --adapter-dir adapters/colab_lora \
        --checkpoints 100 300 500 700 900 1000 \
        --n 20
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import normalize_condition, normalize_medication, score_set

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "dataset" / "processed" / "eval.jsonl"
BASE_MODEL = "microsoft/Phi-4-mini-instruct"
RANDOM_SEED = 42

PROMPT_TEMPLATE = """You are a clinical information extraction system. Read the clinical note below \
and extract ONLY the diagnoses and medications explicitly mentioned in it.

Return strict JSON with exactly this shape, and nothing else:
{{"diagnoses": ["...", "..."], "medications": ["...", "..."]}}

If none are mentioned, return empty lists. Do not invent conditions or drugs not in the text.

Clinical note:
\"\"\"
{note_text}
\"\"\"

JSON:"""


def load_eval_sample(n: int):
    records = [json.loads(line) for line in EVAL_PATH.open()]
    import random

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(records)
    return records[:n]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_json(model, tokenizer, device, note_text, max_new_tokens=300):
    prompt = PROMPT_TEMPLATE.format(note_text=note_text)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_checkpoint(model, tokenizer, device, samples):
    diag_scores, med_scores = [], []
    parse_failures = 0

    for i, record in enumerate(samples):
        t0 = time.time()
        raw_text = generate_json(model, tokenizer, device, record["note_text"])
        print(f"    example {i+1}/{len(samples)} took {time.time()-t0:.1f}s", flush=True)
        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            parsed = json.loads(raw_text[start:end])
            predicted_diagnoses = parsed.get("diagnoses", [])
            predicted_medications = parsed.get("medications", [])
        except (json.JSONDecodeError, ValueError, AttributeError):
            parse_failures += 1
            predicted_diagnoses, predicted_medications = [], []

        diag_scores.append(score_set(predicted_diagnoses, record["diagnoses"], normalize_condition))
        med_scores.append(score_set(predicted_medications, record["medications"], normalize_medication))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    def avg(key, scores):
        vals = [s[key] for s in scores]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n_samples": len(samples),
        "diagnosis_f1": avg("f1", diag_scores),
        "diagnosis_recall": avg("recall", diag_scores),
        "diagnosis_precision": avg("precision", diag_scores),
        "medication_f1": avg("f1", med_scores),
        "medication_recall": avg("recall", med_scores),
        "medication_precision": avg("precision", med_scores),
        "parse_failure_rate": parse_failures / len(samples) if samples else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    adapter_dir = ROOT / args.adapter_dir
    device = get_device()
    print(f"Using device: {device}")

    samples = load_eval_sample(args.n)
    print(f"Loaded {len(samples)} eval samples")

    print(f"\nLoading base model {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to(device)
    base_model.eval()

    results = []
    peft_model = None
    for idx, ckpt in enumerate(args.checkpoints):
        ckpt_path = adapter_dir / f"checkpoint-{ckpt}"
        if not ckpt_path.exists():
            print(f"  Skipping checkpoint {ckpt} (not found at {ckpt_path})")
            continue

        print(f"\n=== Evaluating checkpoint {ckpt} ===")
        adapter_name = f"ckpt_{ckpt}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base_model, str(ckpt_path), adapter_name=adapter_name)
        else:
            peft_model.load_adapter(str(ckpt_path), adapter_name=adapter_name)
        peft_model.set_adapter(adapter_name)
        peft_model.eval()

        metrics = evaluate_checkpoint(peft_model, tokenizer, device, samples)
        metrics["checkpoint"] = ckpt
        results.append(metrics)

        gc.collect()

    print("\n=== Checkpoint comparison ===")
    print(f"{'Ckpt':>6} {'DiagF1':>8} {'DiagRec':>8} {'MedF1':>8} {'MedRec':>8} {'ParseFail':>10}")
    for r in results:
        print(
            f"{r['checkpoint']:>6} {r['diagnosis_f1']:>8.3f} {r['diagnosis_recall']:>8.3f} "
            f"{r['medication_f1']:>8.3f} {r['medication_recall']:>8.3f} {r['parse_failure_rate']*100:>9.1f}%"
        )

    out_path = ROOT / "results" / f"{adapter_dir.name}_checkpoint_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
