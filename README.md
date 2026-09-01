# ClinicalExtract

Extracting diagnoses and medications from clinical notes using small language models (SLMs) run entirely on-prem — no hosted API calls, no patient data leaving the local environment. This project compares three approaches to closing the accuracy gap in a weak zero-shot baseline: **fine-tuning**, **retrieval-augmented prompting (RAG)**, and the two **combined**.

## Why on-prem SLMs

Clinical text carries privacy and regulatory constraints that make routing patient notes through a third-party API a non-starter in many real deployments. The question this project asks: with only small, locally-servable models (2-8B parameters) and no external API, how far can accuracy be pushed on a real structured-extraction task, and which technique — fine-tuning or retrieval — actually moves the needle?

## Task

Given a clinical note (synthetic, Synthea-generated), extract every **diagnosis** and **medication** mentioned as structured JSON:

```json
{"diagnoses": ["Essential hypertension (disorder)", "..."], "medications": ["lisinopril 10 MG Oral Tablet", "..."]}
```

Notes follow a realistic structure: a dense "History of Present Illness" narrative (often listing 10+ cumulative conditions in prose) plus a clean, delimited medications section — an intentional difficulty asymmetry that shapes most of the findings below.

## Results

### Step 1 — Baseline: which small model to build on

Three off-the-shelf models compared zero-shot (n=60, held-out eval set):

| Model | Diagnosis F1 | Medication F1 | P50 Latency | Peak Memory |
|---|---|---|---|---|
| Llama 3.2 3B | 0.15 | 0.47 | 2.68s | 2.17 GB |
| **Phi-4-mini** ✓ | **0.18** | **0.49** | 3.73s | 2.65 GB |
| Ministral 8B (Q4) | 0.15 | 0.49 | 7.05s | 4.68 GB |

**Phi-4-mini won** on an accuracy-per-GB basis — Ministral 8B used 2.5x the memory for no accuracy gain. But the more important finding was the *shape* of the error: diagnosis recall averaged **~0.12** across all three models — the models were finding barely 1 in 8 real diagnoses, because they're buried in narrative prose rather than a clean list. That gap became the target for everything that followed.

### Step 2 — Four ways to close the gap

| Approach | n | Diagnosis F1 | Medication F1 | Latency (P50) |
|---|---|---|---|---|
| Zero-shot baseline | 60 | 0.18 | 0.49 | 3.7s |
| + RAG (retrieved few-shot examples) | 60 | 0.356 | 0.465 | 13.8s |
| **Fine-tuned (LoRA)** | 30 | **0.540** | **0.679** | — |
| Fine-tuned + RAG combined | 60 | 0.524 | 0.483 | 11.9s |

**Fine-tuning wins outright** — nearly 3x the baseline's diagnosis F1, and the best medication F1 of any approach.

### The interesting finding: RAG helps and hurts, depending on the subtask

RAG nearly doubled diagnosis F1 (0.18 → 0.356) — retrieved similar notes clearly help a model recognize the pattern of diagnoses scattered through dense prose. But RAG **consistently hurt medication extraction**, both on the base model (0.49 → 0.465) and on the fine-tuned model (0.679 → 0.483, a 29% relative drop). This wasn't noise — it replicated in both conditions.

The likely explanation: medications were already easy to extract from a clean, delimited section of the note. Retrieved few-shot examples added little signal there and instead introduced *other patients'* medication lists as confusable content, while for the genuinely hard subtask (diagnoses in prose), the same retrieved context was a net positive. **RAG isn't a uniform accuracy lever — its value depends on whether the base task is already easy or hard**, and combining it with fine-tuning didn't add anything the fine-tuning hadn't already captured.

## Approach

```
Baseline comparison (3 models, zero-shot, via Ollama)
        │
        ▼
Fine-tuning (LoRA, Phi-4-mini, Google Colab T4 GPU)
        │  8,000 training notes, 1 epoch, checkpoint selection via held-out eval
        ▼
RAG comparison (4 arms: baseline / RAG / fine-tuned / RAG+fine-tuned)
        │  retrieval corpus: 29,029 notes, embedded via nomic-embed-text
        ▼
Deployment (merge LoRA → GGUF → quantize → Ollama)
```

Full methodology, debugging log, and interview-ready technical talking points: [PROJECT_NOTES.md](PROJECT_NOTES.md).

## Technical highlights

- **Fine-tuning**: LoRA on Phi-4-mini (8,000 notes, 1 epoch), with checkpoint selection driven by held-out task-level F1 rather than trusting training loss alone — loss and downstream accuracy didn't always track together.
- **Retrieval**: 29,029 note embeddings (nomic-embed-text via Ollama) indexed in a persistent **ChromaDB** collection with cosine-space HNSW search.
- **Root-cause debugging**: traced a Colab CUDA OOM and an unrelated local memory-thrashing issue to the same underlying cause (a `torch_dtype`/`dtype` argument rename silently defaulting to fp32 instead of fp16); isolated a reproducible Ollama hang to a specific `num_ctx` value through systematic single-variable testing rather than trial and error.
- **Deployment**: merged the LoRA adapter into the base model, converted to GGUF via `llama.cpp`, quantized to Q4_K_M, and deployed through Ollama — the fine-tuned model is servable identically to the baseline models it's compared against.
- **Statistical discipline**: every evaluation uses a strictly held-out set never touched during training, retrieval-index construction, or checkpoint selection; sample-size limitations are explicitly acknowledged rather than overclaiming precision from small evaluations.

## Repository structure

```
src/                    Core scripts: baseline benchmarking, fine-tune data prep,
                         checkpoint evaluation, RAG retrieval + prompting, model merging
colab/                   Fine-tuning training scripts (Colab GPU)
dataset/processed/       train.jsonl (29,029) / eval.jsonl (7,404, held out)
adapters/                LoRA checkpoints
retrieval_index/         Precomputed note embeddings (source data)
chroma_db/               ChromaDB persistent vector store (RAG retrieval index)
merged_models/           LoRA merged into the base model
gguf_models/              Quantized model + Ollama Modelfile
results/                 All evaluation outputs (JSON + logs)
```

## Limitations

- **Sample sizes (n=30-60)** are directional, not statistically precise — a deliberate scoping choice for compute/time budget, documented rather than hidden.
- **Synthetic data (Synthea)**: avoids PHI/regulatory issues but is cleaner than real EHR text (no OCR artifacts, non-standard abbreviations, or negation like "patient denies chest pain").
- **Scoring is fuzzy string matching** (normalized, substring-based) against ground truth, not clinician-validated or UMLS/SNOMED concept-normalized — a reasonable engineering proxy, not a clinical-grade evaluation.
- **No frontier-model reference point** (e.g., GPT-4-class) — this project is scoped to what's servable on-prem on modest hardware, so it doesn't answer "how much would a much larger hosted model close this gap."
- A QLoRA (4-bit quantized-base) comparison was built and debugged but deliberately not run, in favor of the RAG comparison — see [PROJECT_NOTES.md](PROJECT_NOTES.md) for the reasoning.

## Stack

Ollama · Phi-4-mini · LoRA / PEFT · Hugging Face `transformers`/`trl` · Apple MLX · Google Colab (T4 GPU) · `llama.cpp` (GGUF conversion + quantization) · `nomic-embed-text` · ChromaDB
# ClinicalExtract
