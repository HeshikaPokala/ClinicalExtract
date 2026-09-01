"""
Retrieval + few-shot prompt construction for RAG-augmented extraction.
Requires a ChromaDB collection at chroma_db/ (see build_chroma_index.py), which is
built from precomputed embeddings in retrieval_index/ (see build_retrieval_index.py).
"""

import json
from pathlib import Path

import chromadb
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


class RetrievalIndex:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection("clinical_notes")

    def embed_query(self, text: str) -> np.ndarray:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def retrieve(self, query_text: str, k: int = 3) -> list:
        query_vec = self.embed_query(query_text)
        results = self.collection.query(query_embeddings=[query_vec.tolist()], n_results=k)
        return [
            {
                "note_text": doc,
                "diagnoses": json.loads(meta["diagnoses"]),
                "medications": json.loads(meta["medications"]),
            }
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]


FEW_SHOT_PROMPT_TEMPLATE = """You are a clinical information extraction system. Read the clinical note below \
and extract ONLY the diagnoses and medications explicitly mentioned in it.

Return strict JSON with exactly this shape, and nothing else:
{{"diagnoses": ["...", "..."], "medications": ["...", "..."]}}

If none are mentioned, return empty lists. Do not invent conditions or drugs not in the text.

Here are {n_examples} worked examples of correctly extracted notes, retrieved for their similarity \
to the note you need to extract from:

{examples_block}

Now extract from this clinical note:
\"\"\"
{note_text}
\"\"\"

JSON:"""

EXAMPLE_BLOCK_TEMPLATE = """Example {i}:
Clinical note:
\"\"\"
{note_text}
\"\"\"
Extracted JSON: {{"diagnoses": {diagnoses}, "medications": {medications}}}
"""


def build_few_shot_prompt(note_text: str, retrieved_examples: list) -> str:
    blocks = []
    for i, ex in enumerate(retrieved_examples, 1):
        blocks.append(
            EXAMPLE_BLOCK_TEMPLATE.format(
                i=i,
                note_text=ex["note_text"],
                diagnoses=json.dumps(ex["diagnoses"]),
                medications=json.dumps(ex["medications"]),
            )
        )
    return FEW_SHOT_PROMPT_TEMPLATE.format(
        n_examples=len(retrieved_examples),
        examples_block="\n".join(blocks),
        note_text=note_text,
    )
