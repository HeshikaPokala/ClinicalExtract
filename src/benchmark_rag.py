"""
Arm 2 (RAG) and Arm 4 (RAG + fine-tuned) evaluation: retrieval-augmented
few-shot prompting, scored against held-out eval.jsonl.

By default targets the zero-shot base model via Ollama (arm 2). Pass
--model-tag pointing at a fine-tuned model already loaded into Ollama for arm 4.

Usage:
    .venv/bin/python src/benchmark_rag.py --n 60 --k 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import normalize_condition, normalize_medication, percentile, score_set
from rag_utils import RetrievalIndex, build_few_shot_prompt

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "dataset" / "processed" / "eval.jsonl"
RESULTS_DIR = ROOT / "results"

OLLAMA_URL = "http://localhost:11434"
RANDOM_SEED = 42


def load_eval_sample(n: int):
    records = [json.loads(line) for line in EVAL_PATH.open()]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(records)
    return records[:n]


def ollama_generate(model: str, prompt: str, timeout=180):
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            # Ollama defaults num_ctx to 2048 regardless of the model's max supported
            # context -- few-shot RAG prompts (target note + k retrieved examples) can
            # exceed that and get silently truncated, so this needs to be raised.
            # IMPORTANT: num_ctx=8192 causes generation to hang indefinitely on this
            # Ollama/Metal setup (reproduced repeatedly, isolated via direct testing --
            # not prompt-length related, since the identical prompt completes in ~22s
            # at num_ctx=4096). Confirmed 4096 comfortably covers k=3 few-shot prompts.
            "options": {"temperature": 0, "num_ctx": 4096},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--k", type=int, default=3, help="number of retrieved few-shot examples")
    parser.add_argument("--model-tag", default="phi4-mini", help="Ollama model tag to evaluate")
    parser.add_argument("--label", default="RAG (zero-shot base)", help="label for the results file")
    args = parser.parse_args()

    print("Loading retrieval index...")
    index = RetrievalIndex()
    print(f"Index loaded: {len(index.metadata)} notes")

    samples = load_eval_sample(args.n)
    print(f"Loaded {len(samples)} eval samples")

    print("Warming up model...")
    ollama_generate(args.model_tag, "Say OK.", timeout=180)

    diag_scores, med_scores = [], []
    parse_failures = 0
    latencies_ms = []

    for i, record in enumerate(samples):
        retrieved = index.retrieve(record["note_text"], k=args.k)
        prompt = build_few_shot_prompt(record["note_text"], retrieved)

        try:
            result = ollama_generate(args.model_tag, prompt)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] request failed: {e}")
            parse_failures += 1
            continue

        latencies_ms.append(result.get("total_duration", 0) / 1e6)
        raw_text = result.get("response", "")
        try:
            parsed = json.loads(raw_text)
            predicted_diagnoses = parsed.get("diagnoses", [])
            predicted_medications = parsed.get("medications", [])
        except (json.JSONDecodeError, AttributeError):
            parse_failures += 1
            predicted_diagnoses, predicted_medications = [], []

        diag_scores.append(score_set(predicted_diagnoses, record["diagnoses"], normalize_condition))
        med_scores.append(score_set(predicted_medications, record["medications"], normalize_medication))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    def avg(key, scores):
        vals = [s[key] for s in scores]
        return sum(vals) / len(vals) if vals else 0.0

    result_summary = {
        "label": args.label,
        "model_tag": args.model_tag,
        "k": args.k,
        "n_samples": len(samples),
        "diagnosis_f1": avg("f1", diag_scores),
        "diagnosis_recall": avg("recall", diag_scores),
        "diagnosis_precision": avg("precision", diag_scores),
        "medication_f1": avg("f1", med_scores),
        "medication_recall": avg("recall", med_scores),
        "medication_precision": avg("precision", med_scores),
        "parse_failure_rate": parse_failures / len(samples) if samples else 0.0,
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
    }

    print("\n=== Results ===")
    for key, val in result_summary.items():
        print(f"  {key}: {val}")

    out_name = args.label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = RESULTS_DIR / f"rag_{out_name}.json"
    out_path.write_text(json.dumps(result_summary, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
