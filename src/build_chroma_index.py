"""
Loads the precomputed embeddings (retrieval_index/embeddings.npy + metadata.jsonl,
built by build_retrieval_index.py) into a persistent ChromaDB collection, replacing
the raw-numpy brute-force index with a real vector database.

Doesn't re-embed via Ollama (already have the vectors) -- just bulk-inserts them.

Usage:
    .venv/bin/python src/build_chroma_index.py
"""

import json
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "retrieval_index"
CHROMA_DIR = ROOT / "chroma_db"

BATCH_SIZE = 5000


def main():
    print("Loading precomputed embeddings + metadata...")
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    metadata = [json.loads(l) for l in open(INDEX_DIR / "metadata.jsonl")]
    assert len(metadata) == embeddings.shape[0]
    print(f"Loaded {len(metadata)} embeddings, dim {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.delete_collection("clinical_notes") if "clinical_notes" in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(
        name="clinical_notes",
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(metadata), BATCH_SIZE):
        batch_meta = metadata[i : i + BATCH_SIZE]
        batch_emb = embeddings[i : i + BATCH_SIZE]

        collection.add(
            ids=[str(i + j) for j in range(len(batch_meta))],
            embeddings=batch_emb.tolist(),
            documents=[m["note_text"] for m in batch_meta],
            metadatas=[
                {
                    "diagnoses": json.dumps(m["diagnoses"]),
                    "medications": json.dumps(m["medications"]),
                }
                for m in batch_meta
            ],
        )
        print(f"  inserted {min(i + BATCH_SIZE, len(metadata))}/{len(metadata)}")

    print(f"\nChromaDB collection 'clinical_notes' built at {CHROMA_DIR} ({collection.count()} documents)")


if __name__ == "__main__":
    main()
