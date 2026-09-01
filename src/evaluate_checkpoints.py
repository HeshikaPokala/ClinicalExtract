"""
Evaluates multiple LoRA checkpoints on held-out eval.jsonl to pick the one that
actually generalizes best (not just lowest val loss), since val loss is a proxy
and the training run showed signs of overfitting after iteration ~100.

Usage:
    .venv/bin/python src/evaluate_checkpoints.py \
        --adapter-dir adapters/lora_phi4mini \
        --checkpoints 100 200 300 400 500 \
        --n 30
"""

import argparse
import json
import shutil
from pathlib import Path

from mlx_lm import generate, load

from eval_utils import normalize_condition, normalize_medication, score_set

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "dataset" / "processed" / "eval.jsonl"
BASE_MODEL = "mlx-community/Phi-4-mini-instruct-mlx-fp16"
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


def stage_checkpoint(adapter_dir: Path, iteration: int, staging_dir: Path):
    staging_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = adapter_dir / f"{iteration:07d}_adapters.safetensors"
    if not ckpt_file.exists():
        raise FileNotFoundError(ckpt_file)
    shutil.copy(ckpt_file, staging_dir / "adapters.safetensors")
    shutil.copy(adapter_dir / "adapter_config.json", staging_dir / "adapter_config.json")
    return staging_dir


def evaluate_checkpoint(iteration: int, adapter_dir: Path, samples: list, staging_root: Path):
    staging_dir = stage_checkpoint(adapter_dir, iteration, staging_root / f"ckpt_{iteration}")
    print(f"\n=== Evaluating checkpoint iter {iteration} ===")
    model, tokenizer = load(BASE_MODEL, adapter_path=str(staging_dir))

    diag_scores, med_scores = [], []
    parse_failures = 0

    for i, record in enumerate(samples):
        prompt = PROMPT_TEMPLATE.format(note_text=record["note_text"])
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw_text = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

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

    del model
    shutil.rmtree(staging_dir, ignore_errors=True)

    def avg(key, scores):
        vals = [s[key] for s in scores]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "checkpoint_iter": iteration,
        "n_samples": len(samples),
        "diagnosis_f1": avg("f1", diag_scores),
        "medication_f1": avg("f1", med_scores),
        "diagnosis_recall": avg("recall", diag_scores),
        "medication_recall": avg("recall", med_scores),
        "parse_failure_rate": parse_failures / len(samples) if samples else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    adapter_dir = ROOT / args.adapter_dir
    staging_root = ROOT / "adapters" / "_staging"
    samples = load_eval_sample(args.n)
    print(f"Loaded {len(samples)} eval samples")

    results = []
    for it in args.checkpoints:
        results.append(evaluate_checkpoint(it, adapter_dir, samples, staging_root))

    shutil.rmtree(staging_root, ignore_errors=True)

    print("\n=== Checkpoint comparison ===")
    print(f"{'Iter':>6} {'DiagF1':>8} {'DiagRec':>8} {'MedF1':>8} {'MedRec':>8} {'ParseFail':>10}")
    for r in results:
        print(
            f"{r['checkpoint_iter']:>6} {r['diagnosis_f1']:>8.3f} {r['diagnosis_recall']:>8.3f} "
            f"{r['medication_f1']:>8.3f} {r['medication_recall']:>8.3f} {r['parse_failure_rate']*100:>9.1f}%"
        )

    out_path = adapter_dir.parent.parent / "results" / f"{adapter_dir.name}_checkpoint_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
