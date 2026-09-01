# ClinicalExtract: Phi-4-mini QLoRA fine-tuning on Google Colab (free T4 GPU)
#
# HOW TO USE:
# 1. Open a new Colab notebook, set Runtime -> Change runtime type -> T4 GPU.
# 2. Copy each "# CELL n" block below into its own notebook cell, in order.
# 3. Run cells top to bottom. Cell 3 will prompt you to upload files -- upload
#    train.jsonl and valid.jsonl from dataset/finetune/ on your local machine.
# 4. Cell 7 is a short calibration run (20 steps) that prints an estimated
#    total training time BEFORE you commit to the full run -- read its output
#    before running Cell 8.
# 5. Cell 9 zips and downloads the trained adapter + checkpoints back to you.

# ===================== CELL 1: Install dependencies =====================
# !pip install -q -U transformers accelerate peft bitsandbytes trl datasets torchao

# ===================== CELL 2: Imports & GPU check =====================
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- set Runtime > Change runtime type > T4 GPU")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ===================== CELL 3: Upload training data =====================
# Upload dataset/finetune/train.jsonl and dataset/finetune/valid.jsonl from your Mac.
from google.colab import files
uploaded = files.upload()  # select train.jsonl and valid.jsonl together
assert "train.jsonl" in uploaded and "valid.jsonl" in uploaded, "Upload both train.jsonl and valid.jsonl"

# ===================== CELL 4: Load base model in 4-bit (QLoRA) =====================
MODEL_ID = "microsoft/Phi-4-mini-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 (Turing) has no bf16 tensor cores -- fp16 runs ~10-60x faster on this GPU
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)  # trust_remote_code removed -- uses transformers' built-in Phi3 implementation instead
   # of the repo's custom remote code, which was causing a LossKwargs ImportError
model.config.use_cache = False

# ===================== CELL 5: LoRA config (with dropout, fixing the overfitting issue) =====================
lora_config = LoraConfig(
    r=8,
    lora_alpha=20,
    lora_dropout=0.1,          # regularization -- the local MLX run used 0.0 and overfit by iter ~200
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== CELL 6: Load dataset =====================
# train.jsonl / valid.jsonl are already in {"messages": [{"role":"user",...},{"role":"assistant",...}]}
# chat format -- SFTTrainer applies the model's chat template automatically.
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})
print(dataset)

# ===================== CELL 7: CALIBRATION RUN -- read this before Cell 8 =====================
# Runs 20 steps only, to measure actual it/sec on your Colab GPU allocation
# (Colab free tier gives T4 usually, but this varies) before committing to a long run.
import time

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

n_train_examples = len(dataset["train"])
effective_batch = 4 * 2  # per_device_train_batch_size * grad_accum
steps_per_epoch = n_train_examples // effective_batch
for epochs in [1, 2, 3]:
    total_steps = steps_per_epoch * epochs
    est_seconds = total_steps / it_per_sec
    print(f"  {epochs} epoch(s) = {total_steps} steps -> est. {est_seconds/60:.1f} min ({est_seconds/3600:.2f} hr)")

# ===================== CELL 8: Full training run =====================
# Adjust max_steps/num_train_epochs based on Cell 7's estimate.
# Colab free sessions can disconnect after ~90 min idle or ~12hr hard cap --
# keep this comfortably under a few hours, and eval_steps/save_steps let you
# recover the best checkpoint even if the session drops before the end.
train_args = SFTConfig(
    output_dir="/content/adapters/lora_phi4mini_qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,          # start with 1 epoch (~8000 examples); increase once you trust the timing
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=10,         # keep last 10 checkpoints so you can pick the best later
    logging_steps=20,
    learning_rate=1e-4,          # standard QLoRA LR, higher than the local MLX run's 1e-5
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

trainer.train()
trainer.save_model("/content/adapters/lora_phi4mini_qlora/final")

print(f"\nPeak GPU memory (QLoRA, 4-bit base): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# ===================== CELL 9: Zip and download all checkpoints =====================
import shutil
shutil.make_archive("/content/lora_phi4mini_qlora", "zip", "/content/adapters/lora_phi4mini_qlora")
files.download("/content/lora_phi4mini_qlora.zip")
