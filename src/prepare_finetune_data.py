"""
Converts dataset/processed/train.jsonl into MLX chat-format train/valid JSONL files
for `mlx_lm.lora`. Uses the same extraction prompt as the baseline benchmark so that
fine-tuned results are directly comparable to the zero-shot baseline.

IMPORTANT: only reads from train.jsonl. dataset/processed/eval.jsonl is never touched
here -- it stays fully held out for scoring the fine-tuned model later.

Usage:
    .venv/bin/python src/prepare_finetune_data.py [--n-train 2500] [--n-valid 300]
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SRC = ROOT / "dataset" / "processed" / "train.jsonl"
OUT_DIR = ROOT / "dataset" / "finetune"

RANDOM_SEED = 42

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


def to_chat_example(record: dict) -> dict:
    user_content = PROMPT_TEMPLATE.format(note_text=record["note_text"])
    assistant_content = json.dumps(
        {"diagnoses": record["diagnoses"], "medications": record["medications"]}
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=2500)
    parser.add_argument("--n-valid", type=int, default=300)
    args = parser.parse_args()

    records = [json.loads(line) for line in TRAIN_SRC.open()]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(records)

    total_needed = args.n_train + args.n_valid
    if len(records) < total_needed:
        raise ValueError(f"Only {len(records)} train records available, need {total_needed}")

    valid_records = records[: args.n_valid]
    train_records = records[args.n_valid : args.n_valid + args.n_train]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_chat_example(r)) + "\n")

    write_jsonl(OUT_DIR / "train.jsonl", train_records)
    write_jsonl(OUT_DIR / "valid.jsonl", valid_records)

    print(f"Wrote {len(train_records)} train examples -> {OUT_DIR / 'train.jsonl'}")
    print(f"Wrote {len(valid_records)} valid examples -> {OUT_DIR / 'valid.jsonl'}")
    print("Note: dataset/processed/eval.jsonl was NOT used here -- stays held out for final scoring.")


if __name__ == "__main__":
    main()
