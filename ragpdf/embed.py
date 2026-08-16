"""Chunks -> a searchable index on disk.

Stored as `chunks.json` + `embeddings.npy` (float32, L2-normalised).

The original saved a DataFrame to CSV, which wrote each 768-dim vector as its
`str()` repr and parsed it back with `np.fromstring`. That round-trip is
*correct* -- measured max abs error 1.3e-7, cosine 0.9999999 -- but it costs
22 MB on disk for 5 MB of floats and ~20 s to parse. .npy is the boring fix.
"""

import json
import os

import numpy as np

MODEL_NAME = "all-mpnet-base-v2"


def _model(name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer  # slow import, keep it local
    return SentenceTransformer(name, device="cpu")


def embed_chunks(chunks: list[dict], model=None, batch_size: int = 32) -> np.ndarray:
    """(n_chunks, dim) float32, unit norm. Row i belongs to chunks[i]."""
    model = model or _model()
    vecs = model.encode(
        [c["text"] for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(query: str, model=None) -> np.ndarray:
    model = model or _model()
    return np.asarray(model.encode(query, normalize_embeddings=True), dtype=np.float32)


def save(index_dir: str, chunks: list[dict], embeddings: np.ndarray) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)


def load(index_dir: str) -> tuple[list[dict], np.ndarray]:
    with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
    if len(chunks) != len(embeddings):
        raise ValueError(f"index at {index_dir} is inconsistent: "
                         f"{len(chunks)} chunks, {len(embeddings)} embeddings")
    return chunks, embeddings
