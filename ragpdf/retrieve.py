"""Two retrievers over the same chunks, so they can be compared on one number.

  dense  -- cosine similarity over all-mpnet-base-v2 embeddings (the model path)
  bm25   -- Okapi BM25 over whitespace tokens (the boring path)

Both take data and return data: (indices, scores). Neither reads a file.
"""

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def dense(query_vec: np.ndarray, embeddings: np.ndarray, k: int = 5):
    """Top-k by cosine. Both sides are unit norm, so cosine == dot product."""
    if k > len(embeddings):
        raise ValueError(f"asked for {k} results from {len(embeddings)} chunks")
    scores = embeddings @ query_vec
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist(), scores[idx].tolist()


class BM25:
    """Okapi BM25. ~30 lines beats adding rank_bm25 as a dependency."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in corpus]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0
        n = len(self.docs)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        # BM25+ idf floor: keeps a term appearing in >half the corpus from
        # scoring negative, which is what makes plain Okapi idf misbehave.
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_len or 1.0))
        for t in terms:
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top(self, query: str, k: int = 5):
        if k > len(self.docs):
            raise ValueError(f"asked for {k} results from {len(self.docs)} chunks")
        s = self.scores(query)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx.tolist(), s[idx].tolist()
