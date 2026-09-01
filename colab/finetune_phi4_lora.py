# ClinicalExtract: Phi-4-mini plain LoRA fine-tuning on Google Colab (free T4 GPU)
#
# This is the LoRA counterpart to finetune_phi4_qlora.py -- same model, same data,
# same LoRA config, run on the same GPU/framework, but the BASE MODEL IS NOT
# QUANTIZED (bf16 instead of 4-bit). Running both on identical Colab hardware
# isolates the effect of quantization alone (memory, speed, accuracy), rather
# than comparing across different machines/frameworks.
#
# HOW TO USE: same as finetune_phi4_qlora.py -- copy each "# CELL n" block into
# its own notebook cell, run top to bottom, read Cell 7's calibration output
# before running Cell 8. You can run this in the SAME notebook as the QLoRA
# script (just restart the runtime between them to clear GPU memory) or in a
# separate notebook.

# ===================== CELL 1: Install dependencies =====================
# !pip install -q -U transformers accelerate peft bitsandbytes trl datasets torchao

# ===================== CELL 2: Imports & GPU check =====================
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- set Runtime > Change runtime type > T4 GPU")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ===================== CELL 3: Upload training data =====================
# Upload dataset/finetune/train.jsonl and dataset/finetune/valid.jsonl from your Mac.
from google.colab import files
uploaded = files.upload()  # select train.jsonl and valid.jsonl together
assert "train.jsonl" in uploaded and "valid.jsonl" in uploaded, "Upload both train.jsonl and valid.jsonl"

# ===================== CELL 4: Load base model in fp16 (no quantization) =====================
# NOTE: using float16, not bfloat16 -- the T4 (Turing architecture) has no bf16
# tensor-core support, so bf16 ops run drastically slower (measured ~0.016 it/sec,
# i.e. ~17.7hr for 1 epoch) than the same ops in fp16, which T4 accelerates natively.
MODEL_ID = "microsoft/Phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()  # fp16 full weights use ~7.6GB on a 16GB T4 -- trade compute for memory headroom

# ===================== CELL 5: LoRA config (identical to the QLoRA run, for a fair comparison) =====================
lora_config = LoraConfig(
    r=8,
    lora_alpha=20,
    lora_dropout=0.1,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== CELL 6: Load dataset =====================
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})
print(dataset)

# ===================== CELL 7: CALIBRATION RUN -- read this before Cell 8 =====================
# Smaller batch than the QLoRA script (2 vs 4) since the bf16 base model uses
# more memory than the 4-bit one -- keeping effective batch size the same (8)
# via a higher gradient_accumulation_steps.
import time

calib_args = SFTConfig(
    output_dir="/content/calib",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=20,
    logging_steps=5,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    max_length=1024,
    dataset_text_field=None,
)

calib_trainer = SFTTrainer(
    model=model,
    args=calib_args,
    train_dataset=dataset["train"],
)

t0 = time.time()
calib_trainer.train()
elapsed = time.time() - t0
it_per_sec = 20 / elapsed
print(f"\nCalibration: {elapsed:.1f}s for 20 steps -> {it_per_sec:.3f} it/sec")
print(f"Peak GPU memory during calibration: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

n_train_examples = len(dataset["train"])
effective_batch = 2 * 4  # per_device_train_batch_size * grad_accum
steps_per_epoch = n_train_examples // effective_batch
for epochs in [1, 2, 3]:
    total_steps = steps_per_epoch * epochs
    est_seconds = total_steps / it_per_sec
    print(f"  {epochs} epoch(s) = {total_steps} steps -> est. {est_seconds/60:.1f} min ({est_seconds/3600:.2f} hr)")

# If you see an out-of-memory error here, drop per_device_train_batch_size to 1
# and raise gradient_accumulation_steps to 8 in both this cell and Cell 8.

# ===================== CELL 8: Full training run =====================
train_args = SFTConfig(
    output_dir="/content/adapters/lora_phi4mini_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=10,
    logging_steps=20,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="adamw_torch",
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

trainer.train()
trainer.save_model("/content/adapters/lora_phi4mini_lora/final")

print(f"\nPeak GPU memory (LoRA, bf16 base): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# ===================== CELL 9: Zip and download all checkpoints =====================
import shutil
shutil.make_archive("/content/lora_phi4mini_lora", "zip", "/content/adapters/lora_phi4mini_lora")
files.download("/content/lora_phi4mini_lora.zip")
