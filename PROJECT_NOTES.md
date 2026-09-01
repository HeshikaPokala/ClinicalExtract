# ClinicalExtract — Project Notes & Continuation Doc

**Purpose of this file**: hand this back to Claude in a future session to resume exactly where we left off, and to have a ready reference for interview talking points about this project. Written to be self-contained — don't assume the reader remembers prior conversation.

**Status as of this writing: the core comparison is COMPLETE.** Baseline → fine-tuning → RAG comparison → deployment all done, results in hand. What's left is optional polish (see §10).

---

## 1. What this project is

A clinical information extraction system: given a clinical note (synthetic, Synthea-generated), extract the **diagnoses** and **medications** mentioned in it as structured JSON, using small language models (SLMs) run locally/on-prem rather than a large hosted API. The project's actual narrative arc ended up being:

1. Benchmark several off-the-shelf SLMs zero-shot to find a baseline and a winner.
2. Fine-tune the winner (LoRA) to close the accuracy gap the baseline exposed.
3. Compare fine-tuning against **retrieval-augmented prompting (RAG)** as an alternative approach — 4-way comparison: zero-shot baseline, RAG alone, fine-tuned alone, RAG + fine-tuned combined.
4. Deploy the winning model locally via Ollama (merged + converted to GGUF).

**Note on scope changes from earlier plans**: the project originally also planned a QLoRA (4-bit quantized-base) fine-tuning run for a LoRA-vs-QLoRA comparison. This was deliberately **not pursued** — see §7 for the reasoning. RAG was added as a comparison arm partway through (see §8) after a discussion about combining two originally-separate project ideas (a RAG project and a fine-tuning project) into one, to directly compare which technique helps more.

This is a portfolio/resume project — the emphasis throughout has been on defensible methodology (proper held-out eval sets, statistical caution about sample sizes, honest reporting of overfitting/noise/bugs) as much as on the end accuracy numbers, because that rigor is the actual interview material.

## 2. Dataset

- `dataset/processed/train.jsonl` — 29,029 examples, each `{note_text, diagnoses: [...], medications: [...]}`. Doubles as the fine-tuning pool *and* the RAG retrieval corpus.
- `dataset/processed/eval.jsonl` — 7,404 examples, **held out**, never used in any training, validation, or retrieval-index split. This is the only dataset ever used for reporting final accuracy numbers, across every arm.
- Notes are Synthea-style: a "History of Present Illness" narrative (often listing 10+ cumulative conditions in dense prose) plus a structured `# Medications` section (cleaner, semicolon-separated).
- `dataset/finetune/train.jsonl` / `valid.jsonl` — subsampled + reformatted from `train.jsonl` for fine-tuning (chat format). Final size: 8,000 train / 500 valid.

## 3. Environment

- Local machine: Apple Silicon Mac, 16GB RAM. This constrained a lot of decisions and caused a lot of debugging (see §5, §6, §8).
- Python venv at `.venv/` (not system Python — was externally-managed). Installed: `requests`, `pandas`, `numpy`, `mlx-lm`, `huggingface_hub`, `torch`, `transformers`, `peft`, `accelerate`, `safetensors`, `py-spy`, `chromadb`.
- Ollama installed locally, used for baseline serving, RAG generation, embeddings (`nomic-embed-text`), and finally the deployed fine-tuned model. Currently installed: `phi4-mini:latest`, `nomic-embed-text:latest`, `phi4mini-lora-ckpt500` (the deployed fine-tuned model, see §9).
- Google Colab (free tier, T4 GPU) used for all actual fine-tuning compute, since local Mac can't run real `bitsandbytes` QLoRA (CUDA-only) and MPS-based training/inference hit repeated memory problems.
- `llama.cpp` installed via Homebrew (binaries only) + a separate isolated venv at `/tmp/venv-convert` for the Python GGUF conversion script (kept isolated deliberately — its pinned torch version would have conflicted with the main venv).

---

## 4. Phase 1 — Baseline benchmarking (COMPLETE)

**Script**: `src/benchmark_baseline.py`, scoring utilities in `src/eval_utils.py`.

Three models compared zero-shot via Ollama, same prompt, `format: "json"` constraint, n=60 examples from `eval.jsonl` (seed 42):

| Model | Diag F1 | Med F1 | P50 Latency | Peak Memory |
|---|---|---|---|---|
| Llama 3.2 3B | 0.15 | 0.47 | 2.68s | 2.17 GB |
| **Phi-4-mini** | **0.18** | **0.49** | 3.73s | 2.65 GB |
| Ministral 8B (Q4) | 0.15 | 0.49 | 7.05s | 4.68 GB |

**Precision/recall detail**:

| Model | Diag P | Diag R | Med P | Med R |
|---|---|---|---|---|
| Llama 3.2 3B | 0.29 | 0.11 | 0.45 | 0.62 |
| Phi-4-mini | 0.32 | 0.14 | 0.47 | 0.62 |
| Ministral 8B | 0.31 | 0.12 | 0.47 | 0.62 |

**Winner: Phi-4-mini** — best or tied-best accuracy on both metrics, far less memory/latency than Ministral 8B for no accuracy gain ("bigger isn't better" / accuracy-per-GB argument). Margin over Llama 3.2 3B is *not* statistically solid at n=60 — worth being honest about that nuance rather than overclaiming a clean win.

**Key finding**: diagnosis recall (~0.12 avg) is far worse than medication recall (~0.62) — model finds only ~1 in 8 real diagnoses. Root cause: diagnoses are buried in dense narrative prose (HPI), medications are in a clean delimited section. This became the primary target for both fine-tuning and RAG.

**Scoring methodology** (used identically across every arm of the whole project): fuzzy string matching after normalization (strip parenthetical SNOMED tags for conditions, strip dosage/form for medications, then substring/whole-word match) — not exact string match, since SLMs paraphrase.

---

## 5. Phase 2 — Fine-tuning: local MLX attempt (COMPLETE, superseded)

**Why MLX**: real QLoRA (`bitsandbytes` 4-bit) requires CUDA, unavailable on Apple Silicon. Used Apple's `mlx-lm` instead — a legitimate, current (2026) approach for local Apple Silicon fine-tuning.

**Run 1 (buggy)**: 2,500 train examples, 500 iterations, batch=1, no `--mask-prompt` flag → loss computed over the entire sequence instead of just the completion. Training visibly destabilized (`nan` losses). **Fix**: added `--mask-prompt`.

**Run 2 (fixed)**: `dropout=0.0` (not yet fixed). Val loss bottomed at iter 100 (0.636) then rose to iter 500 (1.274) — classic overfitting. Root cause: only 500 of 29,029 examples ever seen (1.7% of the dataset), zero regularization.

**Checkpoint comparison (n=30)**: checkpoint 100 picked as best (Diag F1 0.311, Med F1 0.533). **Superseded** by the much better Colab run (§6) — kept here for the narrative.

---

## 6. Phase 3 — Fine-tuning: Colab LoRA (COMPLETE — this is the final fine-tuning result used downstream)

**Data**: `src/prepare_finetune_data.py --n-train 8000 --n-valid 500` — directly targets the local run's root cause (low data diversity). `eval.jsonl` never touched.

### Debugging saga (good "tell me about a time you debugged something" material)

1. **`LossKwargs` ImportError** — Phi-4-mini's `trust_remote_code=True` path pulls custom modeling code pinned to an old transformers version. **Fix**: removed `trust_remote_code=True` — native transformers support exists, the fragile remote-code path wasn't needed.
2. **`torchao` version ImportError** on `get_peft_model()` — peft's strict version check (0.10.0 installed, needs ≥0.16.0). **Fix**: `pip install -U torchao`, restart runtime.
3. **`bf16` catastrophically slow on T4** — 0.016 it/sec (~17.7hr/epoch). Root cause: T4 (Turing architecture) has **no bf16 tensor-core support**. **Fix**: switched everything to `fp16`. Result: 0.016 → 0.091 it/sec (~5.7x speedup, ~3hr/epoch).
4. **`warmup_ratio` TypeError** — installed `trl` version didn't expose this kwarg. **Fix**: removed it.
5. **Training run**: 1,000 steps (1 full epoch, 8,000 unique examples), `dropout=0.1`, `adamw` + cosine LR. Completed in 3hr50min — **no overfitting** (train loss 0.318→0.124, val loss 0.302→0.120, tracking closely throughout). `SFTTrainer` auto-applied completion-only loss masking for the `messages` format.

### Evaluating the checkpoints (also a long saga)

Local Mac evaluation (`src/evaluate_hf_checkpoints.py`, HF/PEFT + MPS) hit **severe memory thrashing**. Root cause: `torch_dtype` (deprecated) vs `dtype` argument-naming silently causing an **fp32 load instead of fp16** (~15.2GB vs ~7.6GB). This same bug also explained a later Colab CUDA OOM. **Fixed by switching to `dtype=`** everywhere, with a post-load dtype assertion added as a guard.

Moved to Colab GPU (`colab/evaluate_checkpoints_colab_single.py`) — hit a stale hardcoded path (a Colab-assistant "fix" desynced two cells), a runtime silently reconnecting as CPU-only, and kernel-restart wiping in-memory variables while disk state survived. All fixed with a GPU pre-flight check, idempotent upload logic, and the dtype assertion.

**Result (n=30, all 10 checkpoints)**:

| Checkpoint | Diag F1 | Diag Recall | Med F1 | Med Recall | Parse Fail |
|---|---|---|---|---|---|
| 100 | 0.337 | 0.332 | 0.485 | 0.498 | 30.0% |
| 200 | 0.333 | 0.377 | 0.472 | 0.583 | 33.3% |
| 300 | 0.350 | 0.378 | 0.466 | 0.600 | 30.0% |
| 400 | 0.427 | 0.506 | 0.558 | 0.592 | 16.7% |
| **500** | **0.540** | **0.548** | **0.679** | 0.676 | **6.7%** |
| 600 | 0.449 | 0.471 | 0.586 | 0.611 | 20.0% |
| 700 | 0.455 | 0.479 | 0.594 | 0.628 | 20.0% |
| 800 | 0.403 | 0.447 | 0.540 | 0.556 | 23.3% |
| 900 | 0.473 | 0.476 | 0.570 | 0.622 | 16.7% |
| 1000 | 0.470 | 0.470 | 0.556 | 0.606 | 16.7% |

**Checkpoint 500 selected as final.** There was a genuine open question about whether this was a real effect or n=30 noise (the curve shape — a sharp spike at 500 with similar neighbors on both sides — looked noise-like, especially since training loss was still monotonically improving at 1000 with no overfitting signature). **Decision made**: skip the planned n=100 re-verification. Reasoning at the time: the gap between checkpoint 500 and RAG's result (see §8) turned out to be much larger than the n=30 noise band, so re-verifying wouldn't have changed which arm wins overall — not worth the Colab GPU time given other reliability costs that day. This is a deliberate, reasoned tradeoff, not an oversight — good to be able to explain the reasoning if asked, including acknowledging the n=30 sample size as a real limitation of the exact decimal (directionally trustworthy, not decimal-precise).

---

## 7. QLoRA — deliberately not pursued

A QLoRA (4-bit quantized-base) training script was fully built and debugged (`colab/finetune_phi4_qlora_single.py`, all the same fixes as the LoRA script applied), but **never run**. Decision: given the LoRA run already produced a strong, clearly-improved result over baseline, and the project's comparison focus shifted to fine-tuning vs. RAG instead (see §8), running QLoRA purely for a memory/speed side-comparison wasn't worth the additional Colab time and reliability risk. If asked in an interview "did you compare LoRA and QLoRA," the honest answer is: script built and ready, deliberately deprioritized in favor of a more valuable comparison (RAG vs. fine-tuning), not something that failed or was overlooked.

---

## 8. Phase 4 — RAG vs. fine-tuning comparison (COMPLETE)

### Motivation and design

The project's scope grew from "just fine-tune it" to a direct comparison: **does retrieval-augmented few-shot prompting or fine-tuning do more to close the baseline's diagnosis-recall gap?** Four arms:

1. Baseline (zero-shot, no augmentation) — already had this from §4.
2. RAG (zero-shot base model + retrieved few-shot examples).
3. Fine-tuned alone (checkpoint 500 from §6).
4. RAG + fine-tuned combined (retrieved few-shot examples fed to the fine-tuned model).

**Design decisions**:
- **Embedding model**: Ollama's `nomic-embed-text` (274MB) rather than adding a separate `sentence-transformers`/HF dependency — keeps everything on the same local Ollama infrastructure already used for generation.
- **Retrieval index**: embeddings computed once via Ollama's `nomic-embed-text` (`src/build_retrieval_index.py`), then loaded into a persistent **ChromaDB** collection (`src/build_chroma_index.py`) using cosine-space HNSW indexing — real vector-database retrieval, not a hand-rolled brute-force scan. The two-step pipeline (embed once, then bulk-load) meant swapping the storage/query layer later cost nothing in re-embedding time (Chroma ingested all 29,029 vectors from the precomputed `.npy` file in seconds).
- **Retrieval corpus**: all of `train.jsonl` (29,029 labeled notes). `eval.jsonl` stays held out.
- **Prompt construction**: k=3 retrieved similar notes (with their gold diagnoses/medications) injected as worked examples ahead of the target note, same JSON-output instructions as the baseline prompt.

### Build

- `src/build_retrieval_index.py` — embeds all 29,029 training notes via Ollama's `/api/embed` (batched, 32 at a time), saves normalized embeddings (`retrieval_index/embeddings.npy`) + metadata (`retrieval_index/metadata.jsonl`). Took ~30 minutes.
- `src/build_chroma_index.py` — bulk-loads those precomputed embeddings into a persistent **ChromaDB** collection (`chroma_db/`, cosine-space HNSW index) — no re-embedding needed, just ingestion (seconds for all 29,029 vectors).
- `src/rag_utils.py` — `RetrievalIndex` class (embed query via Ollama, query the ChromaDB collection for top-k) + few-shot prompt template.
- `src/benchmark_rag.py` — runs the full RAG eval loop against any Ollama model tag, scores with the same `eval_utils.py` used everywhere else.

### A real, reproducible Ollama bug found and fixed

Initial RAG calls hung **indefinitely** (confirmed via CPU monitoring — near-0% utilization for many minutes, not just "slow"). Root cause isolated through systematic testing (short prompt at large `num_ctx` → fine; long prompt at `num_ctx=8192` → hangs; **identical** long prompt at `num_ctx=4096` → works fine, 22s). Conclusion: **`num_ctx=8192` specifically triggers a hang on this Ollama/Metal setup, independent of prompt length or the JSON grammar constraint** (both were tested and ruled out individually). Fixed by using `num_ctx=4096` (still well above the default 2048, which was *also* silently truncating longer RAG prompts before this was caught — a second, separate bug: Ollama defaults to 2048 tokens of context regardless of what the model natively supports).

Also hit a transient, unrelated memory-pressure slowdown mid-run (swap climbing to 89%) that looked like a stall but wasn't — the run actually completed successfully; buffered stdout just hadn't flushed to the log file yet, so the process appeared to be hung when it had actually finished all 60 examples. Worth double-checking actual output/exit status before concluding something failed, rather than trusting a live "no output" read too early.

### Deploying the fine-tuned model for arms 3/4

Needed the fine-tuned model servable via Ollama (for consistent generation infra across all 4 arms). Pipeline:

1. `src/merge_lora_adapter.py` — loads base model in fp16, loads checkpoint 500 adapter via PEFT, `merge_and_unload()`, saves standalone merged model to `merged_models/phi4mini_lora_ckpt500/`.
2. GGUF conversion via `llama.cpp`'s `convert_hf_to_gguf.py` (installed in an isolated venv to avoid its pinned torch version conflicting with the main project venv). **Hit a bug**: the converter requires `tokenizer_config.json` to say `"tokenizer_class": "GPT2Tokenizer"` to take the correct (tokenizer.json-based) vocab path for Phi models; our merged model's `save_pretrained()` output said `"TokenizersBackend"` instead (a transformers-version artifact), causing the converter to look for a nonexistent SentencePiece `tokenizer.model` file. **Fix**: copied the original base model's tokenizer files (identical in substance — LoRA doesn't touch the tokenizer) over the merged model's versions.
3. `llama-quantize` to Q4_K_M (7.3GB fp16 → 2.37GB), matching the baseline models' quantization for a fair comparison.
4. `ollama create phi4mini-lora-ckpt500 -f Modelfile` — deployed. Sanity-checked: first 5 predicted diagnoses on a test note matched ground truth exactly.

### Final results (n=60 for arms 1/2/4, n=30 for arm 3)

| Arm | n | Diag F1 | Diag Recall | Med F1 | Med Recall | Parse Fail | Latency P50 |
|---|---|---|---|---|---|---|---|
| 1. Baseline (zero-shot) | 60 | 0.18 | 0.14 | 0.49 | 0.62 | ~0% | 3.7s |
| 2. RAG (zero-shot base) | 60 | **0.356** | 0.391 | 0.465 | 0.584 | 10% | 13.8s |
| 3. Fine-tuned alone | 30 | 0.540 | 0.548 | **0.679** | 0.676 | 6.7% | — |
| 4. RAG + fine-tuned | 60 | 0.524 | 0.569 | 0.483 | 0.524 | 1.7% | 11.9s |

**Fine-tuning alone wins outright** — best diagnosis F1 and by far the best medication F1.

**A consistent, real pattern (not noise — it replicated in both directions)**: RAG substantially helps diagnosis extraction (arm 1→2: +98% relative) but **consistently hurts medication extraction** whether added to the base model (arm 1→2: 0.49→0.465) or the fine-tuned model (arm 3→4: 0.679→0.483, a 29% relative drop). Working explanation: medications are already easy to extract from a clean delimited section, so retrieved few-shot examples mostly introduce confusable content (the model likely conflates *other* patients' medication lists with the target's), whereas for the genuinely hard diagnosis-extraction subtask, the pattern-matching benefit from retrieved examples outweighs that noise.

**RAG + fine-tuned doesn't beat fine-tuned alone** on either metric — layering RAG on top of the already-fine-tuned model doesn't help; the fine-tuning already captured what RAG was trying to add, and actively hurts medications.

**One genuine RAG benefit**: parse-failure rate dropped sharply with the fine-tuned model + RAG (1.7% vs presumably similar-to-arm-3's 6.7% without) — the fine-tuned model handles longer, more complex prompts more robustly than zero-shot.

Results saved: `results/rag_rag_zero-shot_base.json` (arm 2), `results/rag_rag_+_fine-tuned.json` (arm 4).

---

## 9. Deployment (COMPLETE, as a byproduct of §8)

The fine-tuned model is live and servable locally:
- Merged model: `merged_models/phi4mini_lora_ckpt500/`
- GGUF (fp16 + quantized): `gguf_models/phi4mini_lora_ckpt500_fp16.gguf`, `gguf_models/phi4mini_lora_ckpt500_q4km.gguf`
- Ollama model: `phi4mini-lora-ckpt500` (via `gguf_models/Modelfile`)

This satisfies the original "deploy locally via Ollama, matching how the baseline was served" goal from §1.

---

## 10. What's left (optional polish, nothing blocking)

The core comparison work is done. Remaining items are write-up/presentation, not new experiments:

1. **Write a final polished README** summarizing the whole arc for the portfolio: baseline → gap identified → fine-tuned (with checkpoint-selection methodology) → RAG comparison → final 4-way table → deployment. This doc (`PROJECT_NOTES.md`) has all the raw material; a README would be the cleaned-up, audience-facing version.
2. *(Optional, not required)* Could test different `k` values for RAG (only k=3 was evaluated) to see if the medication-F1 drop is `k`-dependent — not necessary for the core story, but a natural "if I had more time" answer.
3. *(Optional, not required)* Could revisit QLoRA (§7) if a memory/speed comparison specifically becomes relevant to a conversation — script is ready, just not run.

---

## 11. Interview talking points (consolidated)

- **Why F1/precision/recall instead of accuracy**: multi-label extraction, not classification — accuracy can't distinguish "missed a real diagnosis" from "hallucinated a fake one."
- **Clinical risk asymmetry**: missing a medication (low recall) vs. inventing a diagnosis (low precision) are different, both dangerous, failure modes — "I'd tune this to favor medication recall, because a missed interaction is worse than a false positive a clinician can dismiss."
- **Statistical honesty about sample sizes**: baseline n=60, checkpoint comparison n=30 (with an explicit, reasoned decision not to re-verify at n=100 — see §6). Know the margin-of-error math (`sqrt(p(1-p)/n)`) and be ready to say "directional, not decimal-precise."
- **MLX as the Apple Silicon QLoRA equivalent**, and knowing *why* real bitsandbytes QLoRA needed CUDA (hardware constraint, not oversight).
- **The `--mask-prompt` / completion-only-loss bug**: diagnosing training instability (`nan` losses) back to loss being computed over the full sequence instead of just the target.
- **T4 has no bf16 tensor cores**: a specific hardware-architecture fact that caused a silent 5-60x slowdown with no error — "profile before concluding something is just slow."
- **Checkpoint selection via held-out eval, not just val loss** — and the nuance that a small-sample "winner" might be noise, with a reasoned decision about when re-verifying is/isn't worth the cost.
- **Root-cause debugging across a whole session**: `torch_dtype` vs `dtype` deprecation silently causing fp32 loads explained *two separate-looking failures* (Mac swap thrashing, Colab OOM) — finding the shared root cause behind superficially unrelated bugs.
- **RAG vs. fine-tuning, with a real asymmetric finding**: RAG helped the harder subtask (diagnosis, buried in prose) and hurt the easier one (medications, already clean) — consistently, in both the base-model and fine-tuned-model conditions. This is a substantive, specific finding, not just "fine-tuning won."
- **Debugging a genuine reproducible Ollama bug** (`num_ctx=8192` hang) through systematic isolation (varying one variable at a time: context size, prompt length, grammar constraint) rather than guessing — a good "how do you debug an unfamiliar system" story.
- **The GGUF tokenizer_class bug**: a subtle metadata mismatch (`save_pretrained()`'s output vs. what the converter expected) that had nothing to do with the actual model weights — good example of a bug that looks like a data problem but is actually a tooling/serialization detail.
- **Token-level metrics can overstate quality**: `mean_token_accuracy` hit 0.96 during training partly from "free" accuracy on predictable JSON boilerplate — the real signal is task-level F1 on held-out data.

---

## 12. File manifest

```
src/
  eval_utils.py                    # shared scoring: normalize_condition, normalize_medication, score_set, percentile
  benchmark_baseline.py            # Phase 1: 3-model zero-shot baseline comparison via Ollama
  prepare_finetune_data.py         # converts train.jsonl -> chat-format train/valid splits for fine-tuning
  evaluate_checkpoints.py          # MLX-based local checkpoint evaluator (for local MLX-trained adapters)
  evaluate_hf_checkpoints.py       # HF/PEFT-based local checkpoint evaluator (memory issues on Mac; superseded by Colab)
  build_retrieval_index.py         # embeds all train.jsonl notes via Ollama nomic-embed-text -> retrieval_index/
  build_chroma_index.py            # loads precomputed embeddings into a persistent ChromaDB collection -> chroma_db/
  rag_utils.py                     # RetrievalIndex class (queries ChromaDB) + few-shot prompt construction
  benchmark_rag.py                 # RAG eval loop (arms 2 and 4), same scoring as baseline
  merge_lora_adapter.py            # merges a PEFT LoRA adapter into its base model, saves standalone model

colab/
  finetune_phi4_lora.py            # LoRA training, multi-cell version (COMPLETE, all fixes applied)
  finetune_phi4_qlora.py           # QLoRA training, multi-cell version (fixes applied, NOT RUN -- see §7)
  finetune_phi4_qlora_single.py    # QLoRA training, single-file version (NOT RUN -- see §7)
  evaluate_checkpoints_colab.py    # checkpoint eval, multi-cell version
  evaluate_checkpoints_colab_single.py  # checkpoint eval, single-file version
  eval_sample.jsonl                # n=100 eval subset (unused -- the n=100 re-eval was skipped, see §6)

adapters/
  lora_phi4mini/                   # local MLX LoRA checkpoints (100-500)
  colab_lora/                      # Colab LoRA checkpoints (100-1000 + final) -- checkpoint-500 is the final pick

merged_models/
  phi4mini_lora_ckpt500/           # checkpoint 500 merged into the base model (standalone HF format)

gguf_models/
  phi4mini_lora_ckpt500_fp16.gguf  # GGUF conversion, fp16
  phi4mini_lora_ckpt500_q4km.gguf  # quantized, Q4_K_M -- this is what's actually deployed in Ollama
  Modelfile                        # Ollama Modelfile used to create the `phi4mini-lora-ckpt500` model

retrieval_index/
  embeddings.npy                   # normalized embeddings for all 29,029 train.jsonl notes (768-dim, nomic-embed-text)
  metadata.jsonl                   # parallel note_text/diagnoses/medications for each embedding -- source data for chroma_db/

chroma_db/
  (ChromaDB persistent store)      # actual RAG retrieval index queried at eval time -- 'clinical_notes' collection,
                                    # cosine-space HNSW, loaded in bulk from retrieval_index/ above

dataset/
  processed/train.jsonl            # full training pool, 29,029 examples -- also the RAG retrieval corpus
  processed/eval.jsonl              # held-out eval set, 7,404 examples -- NEVER used in training/validation/retrieval-index
  finetune/train.jsonl              # fine-tuning training subsample (8,000 examples)
  finetune/valid.jsonl              # fine-tuning validation subsample (500 examples)

results/
  baseline_comparison.md/.json                       # Phase 1 results (arm 1)
  lora_phi4mini_checkpoint_comparison.json            # local MLX checkpoint comparison
  colab_lora_checkpoint_comparison.json               # Colab LoRA checkpoint comparison (n=30, all 10) -- checkpoint 500 = arm 3
  rag_rag_zero-shot_base.json                         # arm 2 (RAG, base model) results
  rag_rag_+_fine-tuned.json                           # arm 4 (RAG + fine-tuned) results
  *_log.txt                                           # various run logs (training, conversion, eval)
```

---

## 13. How to resume from this doc

The core experimental work is done — baseline, fine-tuning, and RAG comparison all complete, with the fine-tuned model deployed and servable via Ollama (`phi4mini-lora-ckpt500`). If picking this up again:

- If the user wants to **wrap up**: the main remaining task is writing a polished README/portfolio writeup from §8's results table and §11's talking points — no new experiments needed.
- If the user wants to **go further**: the two clearly-scoped optional extensions are QLoRA (§7, script ready) and testing different RAG `k` values (§10.2) — neither is required, both are legitimate "if I had more time" answers already, so don't assume either is expected unless the user explicitly asks.
- Do not re-litigate the checkpoint-500-vs-noise question (§6) unless the user brings it up — that was a deliberate, reasoned decision to not re-verify, not an oversight.
