"""Reproduce the RESULTS.md sections 1 and 2 numbers about the ORIGINAL artifact.

    python eval/measure_original.py

Needs two files that are gitignored because they are large and derived:

    text_chunks_and_embeddings_df.csv   the original pipeline's output (22 MB)
    human-nutrition-text.pdf            the source document (26 MB)

Both are produced by notebooks/original/rag-pipeline.ipynb; if you do not have
them, this script says so and exits rather than inventing a number.

Requires pandas, which the package itself does not -- the original's index is a
CSV and this is the only thing that reads one.
"""

import os
import re
import sys

import fitz
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragpdf.chunk import PAGE_OFFSET  # noqa: E402

CSV = "text_chunks_and_embeddings_df.csv"
PDF = "human-nutrition-text.pdf"

collapse = lambda s: re.sub(r"\s+", " ", s).strip()   # noqa: E731
strip_ws = lambda s: re.sub(r"\s+", "", s)            # noqa: E731


def main(csv=CSV, pdf=PDF):
    for f in (csv, pdf):
        if not os.path.exists(f):
            print(f"missing {f} -- see this file's docstring. Nothing measured.")
            return 1
    import pandas as pd

    df = pd.read_csv(csv)
    doc = fitz.open(pdf)
    print(f"original artifact: {len(df)} chunks | PDF: {doc.page_count} pages\n")

    # --- RESULTS.md section 1: is each chunk a quote of the page it cites? ---
    pages_c, pages_n = {}, {}
    for p in range(doc.page_count):
        t = doc.load_page(p).get_text()
        pages_c[p - PAGE_OFFSET], pages_n[p - PAGE_OFFSET] = collapse(t), strip_ws(t)

    verbatim = ws_only = neither = 0
    for _, r in df.iterrows():
        pn, c = int(r.page_number), str(r.sentence_chunk)
        if pn not in pages_c:
            neither += 1
        elif collapse(c) in pages_c[pn]:
            verbatim += 1
        elif strip_ws(c) in pages_n[pn]:
            ws_only += 1          # same characters, whitespace mangled
        else:
            neither += 1

    n = len(df)
    print("SECTION 1 -- verbatim citation rate of the original")
    print(f"  verbatim quote of cited page   : {verbatim:5d}  {verbatim/n:.3f}")
    print(f"  same chars, whitespace mangled : {ws_only:5d}  {ws_only/n:.3f}")
    print(f"  altered further, or wrong page : {neither:5d}  {neither/n:.3f}")

    # --- RESULTS.md section 2: was the CSV round-trip actually lossy? ---
    print("\nSECTION 2 -- CSV round-trip of the 768-float vectors")
    saved = df.embedding.astype(str)
    print(f"  vectors truncated by numpy's '...' repr : {saved.str.contains(r'\\.\\.\\.').sum()} / {n}")
    arrs = saved.apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    dims = arrs.apply(len).value_counts().to_dict()
    E = np.stack(arrs.values)
    norms = np.linalg.norm(E, axis=1)
    print(f"  dimensions                              : {dims}")
    print(f"  NaN / zero-norm rows                    : {int(np.isnan(E).sum())} / {int((norms == 0).sum())}")
    print(f"  duplicate chunk texts                   : {int(df.sentence_chunk.duplicated().sum())}")

    rng = np.random.default_rng(42)
    idx = rng.choice(n, 12, replace=False)
    from sentence_transformers import SentenceTransformer
    fresh = SentenceTransformer("all-mpnet-base-v2", device="cpu").encode(
        [df.sentence_chunk.iloc[i] for i in idx])
    ref = E[idx]
    cos = np.sum(fresh * ref, 1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"  max abs error vs a fresh encode         : {np.abs(fresh - ref).max():.3e}")
    print(f"  min cosine vs a fresh encode            : {cos.min():.9f}")

    # --- RESULTS.md section 2: did the min-token filter lose real content? ---
    have = set(df.page_number)
    lost = sum(1 for p in range(doc.page_count)
               if (p - PAGE_OFFSET) not in have
               and len(doc.load_page(p).get_text().split()) > 60)
    missing = doc.page_count - len(have)
    print(f"\n  pages with no surviving chunk           : {missing}")
    print(f"  ...of those, pages with >60 words        : {lost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
