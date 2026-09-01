"""
Benchmarks 3 baseline SLMs (zero-shot, no fine-tuning) on the diagnosis/medication
extraction task, using dataset/processed/eval.jsonl.

For each model, measures:
  - Accuracy: set-based F1 on extracted diagnoses and medications (fuzzy string match)
  - Latency: P50/P95 total request time (ms)
  - Throughput: output tokens/sec
  - Memory: peak resident model size while loaded (from `ollama ps`)
  - JSON parse failure rate (how often the model didn't return valid JSON)

Requires Ollama running locally (`brew services start ollama`) with the models pulled:
  ollama pull llama3.2:3b
  ollama pull phi4-mini
  ollama pull hf.co/bartowski/Ministral-8B-Instruct-2410-GGUF:Q4_K_M

Usage:
    .venv/bin/python src/benchmark_baseline.py [--n 60]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

from eval_utils import normalize_condition, normalize_medication, percentile, score_set

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "dataset" / "processed" / "eval.jsonl"
RESULTS_DIR = ROOT / "results"

OLLAMA_URL = "http://localhost:11434"
RANDOM_SEED = 42

MODELS = {
    "Llama 3.2 3B": "llama3.2:3b",
    "Phi-4-mini": "phi4-mini",
    "Ministral 8B (Q4)": "hf.co/bartowski/Ministral-8B-Instruct-2410-GGUF:Q4_K_M",
}

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


def ollama_generate(model: str, prompt: str, timeout=120):
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_loaded_model_memory_gb(model: str):
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json()
        for m in resp.get("models", []):
            loaded_name = m.get("model") or m.get("name") or ""
            if loaded_name == model or loaded_name.split(":")[0] == model.split(":")[0]:
                return m.get("size_vram", m.get("size", 0)) / (1024**3)
    except Exception:
        pass
    return None


def unload_model(model: str):
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        subprocess.run(["ollama", "stop", model], capture_output=True)
    time.sleep(2)


def benchmark_model(display_name: str, model_tag: str, samples: list):
    print(f"\n=== Benchmarking {display_name} ({model_tag}) ===")

    print("  Warming up (loading model)...")
    ollama_generate(model_tag, "Say OK.", timeout=180)

    latencies_ms = []
    tokens_per_sec = []
    diag_scores = []
    med_scores = []
    parse_failures = 0
    peak_memory_gb = 0.0

    for i, record in enumerate(samples):
        prompt = PROMPT_TEMPLATE.format(note_text=record["note_text"])
        try:
            result = ollama_generate(model_tag, prompt)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] request failed: {e}")
            parse_failures += 1
            continue

        total_ms = result.get("total_duration", 0) / 1e6
        latencies_ms.append(total_ms)

        eval_count = result.get("eval_count", 0)
        eval_duration_s = result.get("eval_duration", 1) / 1e9
        if eval_duration_s > 0:
            tokens_per_sec.append(eval_count / eval_duration_s)

        mem = get_loaded_model_memory_gb(model_tag)
        if mem:
            peak_memory_gb = max(peak_memory_gb, mem)

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
            print(f"  [{i+1}/{len(samples)}] avg latency so far: {sum(latencies_ms)/len(latencies_ms):.0f} ms")

    unload_model(model_tag)

    def avg(key, scores):
        vals = [s[key] for s in scores]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "model": display_name,
        "ollama_tag": model_tag,
        "n_samples": len(samples),
        "parse_failure_rate": parse_failures / len(samples) if samples else 0.0,
        "diagnosis_f1": avg("f1", diag_scores),
        "diagnosis_precision": avg("precision", diag_scores),
        "diagnosis_recall": avg("recall", diag_scores),
        "medication_f1": avg("f1", med_scores),
        "medication_precision": avg("precision", med_scores),
        "medication_recall": avg("recall", med_scores),
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0.0,
        "peak_memory_gb": peak_memory_gb,
    }


def write_report(results: list):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "baseline_comparison.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Baseline SLM Comparison (Zero-Shot, No Fine-Tuning)\n",
        f"Evaluated on {results[0]['n_samples']} held-out examples from `dataset/processed/eval.jsonl`.\n",
        "| Model | Diagnosis F1 | Medication F1 | P50 Latency | P95 Latency | Tokens/sec | Peak Memory | JSON Parse Failures |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['diagnosis_f1']:.2f} | {r['medication_f1']:.2f} | "
            f"{r['latency_p50_ms']:.0f} ms | {r['latency_p95_ms']:.0f} ms | "
            f"{r['avg_tokens_per_sec']:.1f} | {r['peak_memory_gb']:.2f} GB | "
            f"{r['parse_failure_rate']*100:.1f}% |"
        )

    lines.append("\n## Precision / Recall detail\n")
    lines.append("| Model | Diag P | Diag R | Med P | Med R |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['model']} | {r['diagnosis_precision']:.2f} | {r['diagnosis_recall']:.2f} | "
            f"{r['medication_precision']:.2f} | {r['medication_recall']:.2f} |"
        )

    (RESULTS_DIR / "baseline_comparison.md").write_text("\n".join(lines) + "\n")
    print("\nResults written to results/baseline_comparison.md and .json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="number of eval examples to sample")
    args = parser.parse_args()

    samples = load_eval_sample(args.n)
    print(f"Loaded {len(samples)} eval samples")

    results = []
    for display_name, model_tag in MODELS.items():
        results.append(benchmark_model(display_name, model_tag, samples))

    write_report(results)


if __name__ == "__main__":
    main()
