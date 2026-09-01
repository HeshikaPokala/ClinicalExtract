"""
Builds a retrieval index over dataset/processed/train.jsonl for RAG-augmented
few-shot prompting. Embeds every training note via Ollama's nomic-embed-text,
saves embeddings + metadata to disk for reuse.

Usage:
    .venv/bin/python src/build_retrieval_index.py [--batch-size 32]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "dataset" / "processed" / "train.jsonl"
INDEX_DIR = ROOT / "retrieval_index"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def embed_batch(texts: list) -> list:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = [json.loads(line) for line in TRAIN_PATH.open()]
    print(f"Loaded {len(records)} training records")

    all_embeddings = []
    metadata = []

    for i in range(0, len(records), args.batch_size):
        batch = records[i : i + args.batch_size]
        texts = [r["note_text"] for r in batch]
        embeddings = embed_batch(texts)
        all_embeddings.extend(embeddings)
        for r in batch:
            metadata.append({
                "note_text": r["note_text"],
                "diagnoses": r["diagnoses"],
                "medications": r["medications"],
            })

        if (i // args.batch_size + 1) % 20 == 0 or i + args.batch_size >= len(records):
            print(f"  embedded {min(i + args.batch_size, len(records))}/{len(records)}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    # normalize so cosine similarity = dot product
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings_array)
    with open(INDEX_DIR / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"\nSaved {len(metadata)} embeddings ({embeddings_array.shape}) to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
