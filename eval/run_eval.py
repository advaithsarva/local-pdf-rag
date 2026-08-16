"""Every number in RESULTS.md comes from here.  `python -m ragpdf.cli eval`

Three things get measured, all on the same index and the same seed:

  A  titles.json      104 section titles from the PDF's own TOC (nobody authored
                      them). Keyword search's home ground.
  B  questions.json   34 questions written in lay wording that avoids the
                      section title. Authored -- see the caveat in RESULTS.md.
  C  out_of_corpus    12 questions the textbook cannot answer. Measures whether
                      the system shuts up.

Dense (all-mpnet-base-v2) and BM25 run over the identical chunk list, so the
comparison is of retrievers, not of pipelines.
"""

import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragpdf import answer as ans  # noqa: E402
from ragpdf import chunk as ch  # noqa: E402
from ragpdf import embed as emb  # noqa: E402
from ragpdf import retrieve as ret  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5


def _load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def _load_questions():
    """questions.json stores gold as [first_page, last_page]; expand it."""
    rows = _load("questions.json")
    for r in rows:
        lo, hi = r["gold_pages"]
        r["gold_pages"] = list(range(lo, hi + 1))
    return rows


def _hit(pages, gold):
    return any(p in gold for p in pages)


def score_set(rows, chunks, vecs, bm25, model, name):
    """recall@5 for both retrievers on one labelled set."""
    dense_hits = bm25_hits = 0
    dense_ms, bm25_ms = [], []
    dense_fail, both_fail = [], []
    for r in rows:
        gold = set(r["gold_pages"])

        t = time.perf_counter()
        di, _ = ret.dense(emb.embed_query(r["query"], model), vecs, K)
        dense_ms.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        bi, _ = bm25.top(r["query"], K)
        bm25_ms.append((time.perf_counter() - t) * 1000)

        d = _hit([chunks[i]["page"] for i in di], gold)
        b = _hit([chunks[i]["page"] for i in bi], gold)
        dense_hits += d
        bm25_hits += b
        if not d:
            dense_fail.append(r["query"])
        if not d and not b:
            both_fail.append(r["query"])

    n = len(rows)
    print(f"\n--- Set {name}: {n} queries, recall@{K} ---")
    print(f"  dense (all-mpnet-base-v2) : {dense_hits}/{n} = {dense_hits/n:.3f}   "
          f"median {statistics.median(dense_ms):.1f} ms")
    print(f"  bm25  (keyword baseline)  : {bm25_hits}/{n} = {bm25_hits/n:.3f}   "
          f"median {statistics.median(bm25_ms):.2f} ms")
    if both_fail:
        print(f"  missed by BOTH ({len(both_fail)}) -- suspect the label, not the retriever:")
        for q in both_fail[:6]:
            print(f"      {q}")
    return {"n": n, "dense": dense_hits / n, "bm25": bm25_hits / n,
            "dense_ms": statistics.median(dense_ms), "bm25_ms": statistics.median(bm25_ms),
            "dense_fail": dense_fail, "both_fail": both_fail}


def score_abstention(queries, in_corpus, vecs, model):
    """Top-1 dense score distribution, in-corpus vs out-of-corpus.

    This is what MIN_SCORE is calibrated against; it is not tuned per query.
    """
    def tops(qs):
        return [ret.dense(emb.embed_query(q, model), vecs, 1)[1][0] for q in qs]

    out = tops(queries)
    inn = tops(in_corpus)
    print(f"\n--- Set C: abstention, MIN_SCORE = {ans.MIN_SCORE} ---")
    print(f"  in-corpus  top-1 score : min {min(inn):.3f}  median {statistics.median(inn):.3f}")
    print(f"  out-corpus top-1 score : max {max(out):.3f}  median {statistics.median(out):.3f}")
    refused = sum(s < ans.MIN_SCORE for s in out)
    kept = sum(s >= ans.MIN_SCORE for s in inn)
    print(f"  refuses out-of-corpus  : {refused}/{len(out)} = {refused/len(out):.3f}")
    print(f"  answers in-corpus      : {kept}/{len(inn)} = {kept/len(inn):.3f}")
    print(f"  extractive baseline refuses: 0/{len(out)} = 0.000  (it always returns 5 passages)")
    return {"refused": refused, "n_out": len(out), "kept": kept, "n_in": len(inn),
            "in_min": min(inn), "out_max": max(out)}


def main(index_dir="index", pdf="human-nutrition-text.pdf"):
    chunks, vecs = emb.load(index_dir)
    ok, total = ch.verify_verbatim(chunks, pdf)
    print(f"=== index: {total} chunks | INVARIANT verbatim {ok}/{total} = {ok/total:.4f} ===")

    model = emb._model()
    t = time.perf_counter()
    bm25 = ret.BM25([c["text"] for c in chunks])
    print(f"    bm25 index built in {time.perf_counter()-t:.1f}s")

    titles = _load("titles.json")
    questions = _load_questions()
    a = score_set(titles, chunks, vecs, bm25, model, "A (TOC titles, unauthored)")
    b = score_set(questions, chunks, vecs, bm25, model, "B (lay questions, authored)")
    c = score_abstention(_load("out_of_corpus.json"), [q["query"] for q in questions], vecs, model)

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": total, "verbatim": ok / total, "A": a, "B": b, "C": c}, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
