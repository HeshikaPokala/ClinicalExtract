# ClinicalExtract: evaluate LoRA checkpoints on Colab's GPU (fixes local Mac
# memory-thrashing problem -- T4 has 16GB dedicated VRAM, not shared with the OS).
#
# HOW TO USE:
# If your training notebook's runtime is STILL CONNECTED, just add these cells
# to the END of that same notebook -- the checkpoints are already sitting at
# /content/adapters/lora_phi4mini_lora/checkpoint-* and nothing needs re-uploading.
#
# If the runtime disconnected, you'll need to re-upload the adapter zip you
# already downloaded to your Mac (colab/lora_phi4mini_lora.zip equivalent) in
# Cell 2b, and unzip it to /content/adapters/lora_phi4mini_lora/.
#
# Either way, upload colab/eval_sample.jsonl (60KB, same 30 examples used in
# the local checkpoint-100 run, same seed=42) when Cell 2 prompts you.

# ===================== CELL 1: Install/upgrade deps + imports =====================
# torchao is a peft dependency for a version check -- Colab's preinstalled version
# (0.10.0) is too old and will raise ImportError on PeftModel.from_pretrained().
# !pip install -q -U torchao

import gc
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print("CUDA available:", torch.cuda.is_available())

# ===================== CELL 2: Upload eval sample =====================
from google.colab import files
uploaded = files.upload()  # upload eval_sample.jsonl
assert "eval_sample.jsonl" in uploaded, "Upload eval_sample.jsonl"

samples = [json.loads(l) for l in open("eval_sample.jsonl")]
print(f"Loaded {len(samples)} eval samples")

# ===================== CELL 2b: re-upload adapter zip (run this if the runtime was refreshed) =====================
# Skip this cell only if /content/adapters/lora_phi4mini_lora/checkpoint-100 already exists
# (i.e. your training notebook's runtime is still the same one that trained it).
#
# This is the ONE place ADAPTER_DIR is defined -- Cell 4/6 below reuse this same
# variable rather than hardcoding the path again, so they can't silently drift
# out of sync with wherever the checkpoints actually ended up.
import glob
import os
import shutil

ADAPTER_DIR = "/content/adapters/lora_phi4mini_lora"

if os.path.isdir(f"{ADAPTER_DIR}/checkpoint-100"):
    print(f"Found existing checkpoints at {ADAPTER_DIR} -- skipping upload.")
else:
    # Check for a zip already sitting in /content (e.g. uploaded in a previous
    # attempt) before prompting for a fresh upload.
    existing_zips = glob.glob("/content/*.zip")
    if existing_zips:
        zip_name = existing_zips[0]
        print(f"Found existing zip at {zip_name} -- extracting that instead of re-uploading.")
    else:
        print("Upload your downloaded adapter zip (e.g. lora_phi4mini_lora.zip):")
        uploaded_zip = files.upload()
        zip_name = next(iter(uploaded_zip))  # works whatever the uploaded filename actually is

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    shutil.unpack_archive(zip_name, ADAPTER_DIR)

    # If the zip contained a single top-level folder (e.g. lora_phi4mini_lora/checkpoint-100/...)
    # instead of the checkpoint-* folders directly at the root, flatten it up one level.
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
        print(f"WARNING: extracted to {ADAPTER_DIR} but no checkpoint-* folders found. "
              f"Contents: {os.listdir(ADAPTER_DIR)}")

# ===================== CELL 3: Scoring utilities (inlined from local eval_utils.py) =====================
import re

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


# ===================== CELL 4: Load base model once (GPU) =====================
# ADAPTER_DIR is already set by Cell 2b -- reused here, not redefined, so this
# can't silently point somewhere different than where the checkpoints actually are.
MODEL_ID = "microsoft/Phi-4-mini-instruct"
assert "ADAPTER_DIR" in dir(), "Run Cell 2b first (it defines ADAPTER_DIR)."
print(f"Using ADAPTER_DIR = {ADAPTER_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
base_model.eval()

# ===================== CELL 5: Prompt template (same as local scripts) =====================
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


# ===================== CELL 6: Evaluate all 10 checkpoints =====================
CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
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

# ===================== CELL 7: Print comparison table + download results =====================
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
