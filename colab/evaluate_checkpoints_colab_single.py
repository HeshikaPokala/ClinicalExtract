# ClinicalExtract: evaluate all 10 LoRA checkpoints on Colab's GPU -- single cell.
#
# HOW TO USE:
# 1. Paste this ENTIRE file into ONE fresh Colab cell (Runtime > Restart runtime
#    first if you've been fighting version errors, to start clean).
# 2. Run it. It will pop up TWO upload prompts in sequence:
#      - eval_sample.jsonl  (60KB, from your Mac's colab/ folder)
#      - your adapter zip (e.g. lora_phi4mini_lora.zip) -- ONLY if
#        /content/adapters/lora_phi4mini_lora/checkpoint-100 isn't already there
# 3. It evaluates all 10 checkpoints and downloads colab_lora_checkpoint_comparison.json
#    back to your Mac at the end.
#
# Takes a while (GPU-bound generation across 10 checkpoints x 30 examples) --
# just let it run, progress prints as it goes.

# ---- Cell 1 equivalent: install/upgrade deps ----
!pip install -q -U torchao

import gc
import glob
import json
import os 
import re
import shutil
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import files

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    _free, _total = torch.cuda.mem_get_info()
    print(f"GPU memory: {(_total - _free) / 1e9:.2f} GB already in use / {_total / 1e9:.2f} GB total")
    if (_total - _free) / 1e9 > 2.0:
        print(
            "WARNING: significant GPU memory is already in use before this script has loaded "
            "anything. This is very likely leftover state from a previous run (e.g. the training "
            "notebook's trainer/model still resident) in THIS SAME runtime. Loading the base model "
            "on top of this will likely OOM. Go to Runtime > Restart runtime ON THIS SPECIFIC TAB/"
            "NOTEBOOK (not a different one), then re-run this script from scratch."
        )

# ---- Cell 2 equivalent: get eval_sample.jsonl onto disk ----
_eval_sample_path = None
for _candidate in ["eval_sample.jsonl", "/content/eval_sample.jsonl"]:
    if os.path.isfile(_candidate):
        _eval_sample_path = _candidate
        break

if _eval_sample_path:
    print(f"\nFound existing {_eval_sample_path} -- skipping upload.")
else:
    print("\nUpload eval_sample.jsonl:")
    uploaded = files.upload()
    assert "eval_sample.jsonl" in uploaded, "You must upload eval_sample.jsonl"
    _eval_sample_path = "eval_sample.jsonl"

samples = [json.loads(l) for l in open(_eval_sample_path)]
print(f"Loaded {len(samples)} eval samples")

# ---- Cell 2b equivalent: get adapter checkpoints onto disk ----
ADAPTER_DIR = "/content/adapters/lora_phi4mini_lora"

if os.path.isdir(f"{ADAPTER_DIR}/checkpoint-100"):
    print(f"\nFound existing checkpoints at {ADAPTER_DIR} -- skipping upload.")
else:
    existing_zips = glob.glob("/content/*.zip")
    if existing_zips:
        zip_name = existing_zips[0]
        print(f"\nFound existing zip at {zip_name} -- extracting that instead of re-uploading.")
    else:
        print("\nUpload your downloaded adapter zip (e.g. lora_phi4mini_lora.zip):")
        uploaded_zip = files.upload()
        zip_name = next(iter(uploaded_zip))

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    shutil.unpack_archive(zip_name, ADAPTER_DIR)

    # flatten if the zip had an extra nested top-level folder
    if not os.path.isdir(f"{ADAPTER_DIR}/checkpoint-100"):
        subdirs = [d for d in os.listdir(ADAPTER_DIR) if os.path.isdir(f"{ADAPTER_DIR}/{d}")]
        nested = [d for d in subdirs if os.path.isdir(f"{ADAPTER_DIR}/{d}/checkpoint-100")]
        if nested:
            nested_path = f"{ADAPTER_DIR}/{nested[0]}"
            for item in os.listdir(nested_path):
                shutil.move(f"{nested_path}/{item}", f"{ADAPTER_DIR}/{item}")
            shutil.rmtree(nested_path)

    found = sorted(d for d in os.listdir(ADAPTER_DIR) if d.startswith("checkpoint-"))
    if found:
        print(f"Extracted to {ADAPTER_DIR}. Checkpoints found: {found}")
    else:
        raise RuntimeError(
            f"Extracted to {ADAPTER_DIR} but no checkpoint-* folders found. "
            f"Contents: {os.listdir(ADAPTER_DIR)}"
        )

# ---- Cell 3 equivalent: scoring utilities ----
_PAREN_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_LEADING_DRUG_NAME_RE = re.compile(r"^([a-z][a-z\s\-]*?)(?=\s+[\d.]|\s*$)")


def normalize_condition(text):
    text = text.lower().strip()
    text = _PAREN_TAG_RE.sub("", text)
    text = _NON_ALNUM_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_medication(text):
    text = text.lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    m = _LEADING_DRUG_NAME_RE.match(text)
    name = m.group(1).strip() if m else text
    return name if name else text


def fuzzy_match(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    return (a in b or b in a) and min(len(a), len(b)) >= 4


def score_set(predicted, gold, normalize_fn):
    pred_norm = [normalize_fn(p) for p in predicted if isinstance(p, str) and p.strip()]
    gold_norm = [normalize_fn(g) for g in gold if isinstance(g, str) and g.strip()]

    matched_gold = set()
    true_positives = 0
    for p in pred_norm:
        for i, g in enumerate(gold_norm):
            if i in matched_gold:
                continue
            if fuzzy_match(p, g):
                matched_gold.add(i)
                true_positives += 1
                break

    precision = true_positives / len(pred_norm) if pred_norm else (1.0 if not gold_norm else 0.0)
    recall = true_positives / len(gold_norm) if gold_norm else (1.0 if not pred_norm else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---- Cell 4 equivalent: load base model once (GPU) ----
MODEL_ID = "microsoft/Phi-4-mini-instruct"
print(f"\nLoading base model {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to("cuda")
base_model.eval()

_actual_dtype = next(base_model.parameters()).dtype
print(f"Base model loaded in dtype: {_actual_dtype}")
if _actual_dtype != torch.float16:
    raise RuntimeError(
        f"Model loaded as {_actual_dtype}, not float16 -- this will likely OOM on a T4's "
        f"~15GB VRAM (fp32 for a 3.8B model is ~15.2GB alone). Fix the dtype argument before continuing."
    )

# ---- Cell 5 equivalent: prompt template + generation ----
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


def generate_json(model, note_text, max_new_tokens=300):
    prompt = PROMPT_TEMPLATE.format(note_text=note_text)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---- Cell 6 equivalent: evaluate shortlisted checkpoints at a larger sample size ----
# n=30 showed checkpoint 500 as a sharp outlier vs its neighbors (400, 600) -- a
# shape consistent with sampling noise rather than a genuine peak. Re-checking the
# top candidates plus 100 (as a sanity check that "early checkpoints are worse"
# holds at a larger n too) with eval_sample.jsonl now at n=100 instead of n=30.
CHECKPOINTS = [100, 500, 700, 900, 1000]
results = []
peft_model = None

for ckpt in CHECKPOINTS:
    ckpt_path = f"{ADAPTER_DIR}/checkpoint-{ckpt}"
    print(f"\n=== Evaluating checkpoint {ckpt} ===")
    adapter_name = f"ckpt_{ckpt}"
    if peft_model is None:
        peft_model = PeftModel.from_pretrained(base_model, ckpt_path, adapter_name=adapter_name)
    else:
        peft_model.load_adapter(ckpt_path, adapter_name=adapter_name)
    peft_model.set_adapter(adapter_name)
    peft_model.eval()

    diag_scores, med_scores = [], []
    parse_failures = 0
    t_ckpt_start = time.time()

    for i, record in enumerate(samples):
        raw_text = generate_json(peft_model, record["note_text"])
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

    results.append({
        "checkpoint": ckpt,
        "diagnosis_f1": avg("f1", diag_scores),
        "diagnosis_recall": avg("recall", diag_scores),
        "diagnosis_precision": avg("precision", diag_scores),
        "medication_f1": avg("f1", med_scores),
        "medication_recall": avg("recall", med_scores),
        "medication_precision": avg("precision", med_scores),
        "parse_failure_rate": parse_failures / len(samples),
    })
    print(f"  checkpoint {ckpt} took {time.time()-t_ckpt_start:.1f}s total")
    gc.collect()
    torch.cuda.empty_cache()

# ---- Cell 7 equivalent: print comparison table + download results ----
print("\n=== Checkpoint comparison ===")
print(f"{'Ckpt':>6} {'DiagF1':>8} {'DiagRec':>8} {'MedF1':>8} {'MedRec':>8} {'ParseFail':>10}")
for r in results:
    print(
        f"{r['checkpoint']:>6} {r['diagnosis_f1']:>8.3f} {r['diagnosis_recall']:>8.3f} "
        f"{r['medication_f1']:>8.3f} {r['medication_recall']:>8.3f} {r['parse_failure_rate']*100:>9.1f}%"
    )

with open("colab_lora_checkpoint_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
files.download("colab_lora_checkpoint_comparison.json")
