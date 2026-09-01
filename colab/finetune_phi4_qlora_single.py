# ClinicalExtract: Phi-4-mini QLoRA fine-tuning on Colab -- single cell.
#
# HOW TO USE:
# 1. Runtime > Change runtime type > T4 GPU (must be a FRESH runtime -- check
#    the "CUDA available" print below actually says True before trusting anything
#    that follows; a stale/CPU runtime silently wrecked an earlier attempt at this).
# 2. Paste this ENTIRE file into ONE cell, run it.
# 3. It uploads dataset/finetune/train.jsonl and valid.jsonl (from your Mac) if
#    they're not already sitting in this session, runs a 20-step calibration to
#    print a time estimate, then proceeds straight into the full 1-epoch training
#    run using the same settings that worked for the LoRA run (fp16, dropout=0.1,
#    cosine LR). Watch the calibration printout as it goes -- if the estimated
#    time looks unreasonable, use Colab's stop button before it commits further.
# 4. Ends by zipping and downloading the adapter + all 10 checkpoints.

# ---- install/upgrade deps ----
!pip install -q -U transformers accelerate peft bitsandbytes trl datasets torchao

import glob
import os
import shutil
import time

import torch
from datasets import load_dataset
from google.colab import files
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- fix Runtime type before continuing")

if torch.cuda.is_available():
    _free, _total = torch.cuda.mem_get_info()
    print(f"GPU memory: {(_total - _free) / 1e9:.2f} GB already in use / {_total / 1e9:.2f} GB total")
    if (_total - _free) / 1e9 > 2.0:
        print(
            "WARNING: significant GPU memory already in use before anything has loaded here. "
            "Likely leftover state from a previous run in THIS SAME runtime. Restart the runtime "
            "on this specific tab before continuing, or training will likely OOM."
        )

# ---- upload training data (skips if already present) ----
_have_train = os.path.isfile("train.jsonl") or os.path.isfile("/content/train.jsonl")
_have_valid = os.path.isfile("valid.jsonl") or os.path.isfile("/content/valid.jsonl")

if _have_train and _have_valid:
    print("\nFound existing train.jsonl and valid.jsonl -- skipping upload.")
else:
    print("\nUpload dataset/finetune/train.jsonl and dataset/finetune/valid.jsonl (select both together):")
    uploaded = files.upload()
    assert "train.jsonl" in uploaded and "valid.jsonl" in uploaded, "Upload both train.jsonl and valid.jsonl"

# ---- load base model in 4-bit (QLoRA) ----
MODEL_ID = "microsoft/Phi-4-mini-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 (Turing) has no bf16 tensor cores -- fp16 runs far faster on this GPU
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\nLoading base model {MODEL_ID} in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)  # trust_remote_code intentionally omitted -- uses transformers' built-in Phi3
   # implementation instead of the repo's custom remote code (avoids a LossKwargs
   # ImportError from a version mismatch between the remote code and transformers)
model.config.use_cache = False

# ---- LoRA config (dropout for regularization -- the local run used 0.0 and overfit) ----
lora_config = LoraConfig(
    r=8,
    lora_alpha=20,
    lora_dropout=0.1,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- load dataset ----
train_path = "train.jsonl" if os.path.isfile("train.jsonl") else "/content/train.jsonl"
valid_path = "valid.jsonl" if os.path.isfile("valid.jsonl") else "/content/valid.jsonl"
dataset = load_dataset("json", data_files={"train": train_path, "validation": valid_path})
print(dataset)

# ---- calibration: 20 steps, prints a time estimate before the real run ----
calib_args = SFTConfig(
    output_dir="/content/calib",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_steps=20,
    logging_steps=5,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=1024,
    dataset_text_field=None,
)
calib_trainer = SFTTrainer(model=model, args=calib_args, train_dataset=dataset["train"])

print("\n=== Calibration run (20 steps) ===")
t0 = time.time()
calib_trainer.train()
elapsed = time.time() - t0
it_per_sec = 20 / elapsed
print(f"\nCalibration: {elapsed:.1f}s for 20 steps -> {it_per_sec:.3f} it/sec")

n_train_examples = len(dataset["train"])
effective_batch = 4 * 2
steps_per_epoch = n_train_examples // effective_batch
for epochs in [1, 2, 3]:
    total_steps = steps_per_epoch * epochs
    est_seconds = total_steps / it_per_sec
    print(f"  {epochs} epoch(s) = {total_steps} steps -> est. {est_seconds/60:.1f} min ({est_seconds/3600:.2f} hr)")

print("\nProceeding to full training (1 epoch) in 10 seconds -- use the stop button now if the estimate above looks wrong.")
time.sleep(10)

# ---- full training run (1 epoch, same settings that worked for LoRA) ----
train_args = SFTConfig(
    output_dir="/content/adapters/lora_phi4mini_qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=10,
    logging_steps=20,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=1024,
    dataset_text_field=None,
)

trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

print("\n=== Full training run ===")
trainer.train()
trainer.save_model("/content/adapters/lora_phi4mini_qlora/final")

print(f"\nPeak GPU memory (QLoRA, 4-bit base): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# ---- zip and download all checkpoints ----
shutil.make_archive("/content/lora_phi4mini_qlora", "zip", "/content/adapters/lora_phi4mini_qlora")
files.download("/content/lora_phi4mini_qlora.zip")
