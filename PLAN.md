# ClinicalExtract-SLM: Comparative Small Language Model Benchmarking & Fine-Tuning for Structured Clinical Information Extraction

## One-line resume description
Benchmarked three small language models (Llama 3.2 3B, Phi-4-mini, Ministral 8B) on latency, memory, and accuracy for clinical diagnosis/medication extraction; fine-tuned the top performer using LoRA and QLoRA on synthetic EHR data, quantized it for on-prem deployment, and measured accuracy/efficiency gains end-to-end.

## Project goal
Given a raw clinical note (e.g., a discharge summary or physician note), extract structured output:
```json
{
  "diagnoses": ["Type 2 Diabetes Mellitus", "Hypertension"],
  "medications": [
    {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
    {"name": "Lisinopril", "dosage": "10mg", "frequency": "once daily"}
  ]
}
```
This is a real, narrow, measurable task used in production healthcare NLP (e.g., populating structured EHR fields from free-text notes) — not a generic chatbot demo.

## Why this project (interview narrative)
- Demonstrates the full SLM lifecycle: baseline comparison → fine-tuning → quantization → local/on-prem deployment.
- Directly addresses a real enterprise concern: running specialized models on-prem for data privacy/security instead of calling external LLM APIs with sensitive data.
- Produces defensible, reproducible metrics at every stage (not just "it works").

## Data source: Synthea (synthetic patient data, no approval needed)
- Site: https://synthetichealth.github.io/synthea/
- GitHub: https://github.com/synthetichealth/synthea
- Generates fully synthetic (fake but realistic) patient records: encounters, conditions, medications, and can output clinical notes in FHIR/CCDA format.
- No PHI, no IRB/PhysioNet credentialing needed — safe to use and publish publicly on GitHub.
- Optional stretch goal (later, not required for v1): apply for PhysioNet access to MIMIC-III/IV for real de-identified notes, to strengthen the "real-world data" story. Not needed to start.

### Getting clinical note text out of Synthea
Synthea's raw output is structured FHIR/CCDA (JSON/XML), not free-text notes. Two options:
1. Synthea can generate a "Clinical Note" / CCDA narrative text export per encounter — use this as your source text.
2. If narrative text is too clean/structured, use an LLM (one-time, not part of your pipeline) to lightly "denoise" structured Synthea records into realistic messy free-text notes — document this as a data augmentation step in your README so it's transparent, not hidden.

Either way: you will always have ground truth, because Synthea's structured records ARE the labels (the diagnoses/medications lists come directly from the source data before you turn them into prose). This is what lets you compute real accuracy metrics.

## Phase-by-phase plan

### Phase 1 — Setup & Data Prep (Week 1)
- [ ] Install Synthea, generate ~500-1000 synthetic patients with realistic conditions/medications
- [ ] Extract encounter notes + corresponding ground-truth diagnosis/medication lists
- [ ] Build eval set: ~150-300 note/label pairs, held out and untouched until evaluation
- [ ] Build training set: remaining note/label pairs, formatted as instruction-tuning examples (note -> structured JSON)
- [ ] Write a data loading + preprocessing script (`src/data_prep.py`)

### Phase 2 — Baseline SLM Comparison (Week 1-2)
- [ ] Set up local inference for 3 models: Llama 3.2 3B, Phi-4-mini, Ministral 8B (via Ollama or Hugging Face transformers)
- [ ] Run all 3 on the eval set with the same extraction prompt (zero-shot or few-shot)
- [ ] Metrics to compute:
  - Accuracy: F1 score on extracted diagnoses/medications (exact + fuzzy match against ground truth)
  - Latency: P50/P95 per-query inference time
  - Memory: peak VRAM/RAM usage during inference
  - Throughput: tokens/sec
- [ ] Produce a comparison table/chart, pick a winner with a written justification (best accuracy-per-GB, not just raw accuracy)
- [ ] Script: `src/benchmark_baseline.py`, results in `results/baseline_comparison.md`

### Phase 3 — Fine-Tuning the Winner (Week 2-3)
- [ ] Fine-tune winner with **LoRA** (full/half precision base) on the training set
- [ ] Fine-tune winner with **QLoRA** (4-bit base) on the same training set, same hyperparameters
- [ ] Compare: training time, peak GPU memory, trainable parameter %, and eval-set accuracy for both
- [ ] Check for catastrophic forgetting: test both fine-tuned models on a few general (non-clinical) prompts vs the base model
- [ ] Pick the better adapter (document why)
- [ ] Script: `src/finetune_lora.py`, `src/finetune_qlora.py`, results in `results/finetune_comparison.md`

### Phase 4 — Quantization for Deployment (Week 3)
- [ ] Merge the winning LoRA adapter into the base model
- [ ] Quantize the merged model (GGUF for llama.cpp, or AWQ) for efficient inference
- [ ] Re-benchmark: memory footprint, latency, and accuracy of quantized vs unquantized fine-tuned model
- [ ] Script: `src/quantize.py`, results in `results/quantization_comparison.md`

### Phase 5 — Local Deployment (Week 3-4)
- [ ] Serve the quantized fine-tuned model locally via llama.cpp or vLLM
- [ ] Wrap in a simple FastAPI endpoint (`/extract` takes raw note text, returns structured JSON)
- [ ] No external API calls anywhere — this is the "on-prem/security" story
- [ ] Script: `src/serve.py`

### Phase 6 — Final Report
- [ ] Write up full README with: problem statement, methodology, all comparison tables/charts, final before/after numbers (base SLM vs fine-tuned+quantized model), and architecture diagram
- [ ] Push to GitHub with clean commit history

## Repo structure (to be created)
```
project-1-clinical-extract-slm/
├── PLAN.md
├── README.md
├── data/
│   ├── raw/              # Synthea output
│   ├── processed/        # note/label pairs
│   └── eval_set.jsonl
├── src/
│   ├── data_prep.py
│   ├── benchmark_baseline.py
│   ├── finetune_lora.py
│   ├── finetune_qlora.py
│   ├── quantize.py
│   ├── serve.py
│   └── eval_utils.py
├── results/
│   ├── baseline_comparison.md
│   ├── finetune_comparison.md
│   └── quantization_comparison.md
└── requirements.txt
```

## Resume bullet (after completion, fill in real numbers)
"Built ClinicalExtract-SLM, an end-to-end pipeline benchmarking 3 SLMs (Llama 3.2 3B, Phi-4-mini, Ministral 8B) on synthetic EHR data for diagnosis/medication extraction; fine-tuned the top performer via LoRA/QLoRA improving F1 by X%, quantized for on-prem deployment reducing memory by Y% with <Z% accuracy loss, and served via FastAPI with zero external API dependency."
