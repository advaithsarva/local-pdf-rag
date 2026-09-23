"""Base vs fine-tuned retriever, on data the fine-tune never saw.

    python -m eval.compare_finetuned

Two sets, both untouched by ragpdf/finetune_retriever.py:

  test_A  the 1-in-5 titles `split_titles()` held out of training   (in-domain, unseen)
  B       eval/questions.json -- never used for training at all     (out-of-domain, unseen)

Reuses `score_set` from run_eval.py unmodified: same recall@5 definition, same K, same
hit rule, just pointed at a second (model, embeddings) pair. If this script and
run_eval.py ever disagreed about what "a hit" means, the two tables would not be
comparable -- importing rather than re-implementing is what rules that out.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragpdf import embed as emb  # noqa: E402
from ragpdf import retrieve as ret  # noqa: E402
from ragpdf.finetune_retriever import split_titles  # noqa: E402
from eval.run_eval import score_set, _load, _load_questions  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(index_dir="index", finetuned_dir="models/finetuned-retriever"):
    finetuned_dir = os.path.join(ROOT, finetuned_dir) if not os.path.isabs(finetuned_dir) else finetuned_dir
    if not os.path.isdir(finetuned_dir):
        raise SystemExit(f"no fine-tuned model at {finetuned_dir} -- run "
                         f"`python -m ragpdf.finetune_retriever` first")

    chunks, base_vecs = emb.load(index_dir)
    titles = _load("titles.json")
    _, test_a = split_titles(titles)
    questions = _load_questions()
    print(f"test_A: {len(test_a)} held-out titles | Set B: {len(questions)} questions")

    base_model = emb._model()
    bm25 = ret.BM25([c["text"] for c in chunks])

    print("\n=== base retriever (all-mpnet-base-v2, off the shelf) ===")
    base_a = score_set(test_a, chunks, base_vecs, bm25, base_model, "A_test (held out, base)")
    base_b = score_set(questions, chunks, base_vecs, bm25, base_model, "B (base)")

    print("\nembedding all chunks with the fine-tuned model (one-time cost)...")
    from sentence_transformers import SentenceTransformer
    ft_model = SentenceTransformer(finetuned_dir, device="cpu")
    ft_vecs_path = os.path.join(ROOT, "index", "embeddings_finetuned.npy")
    import numpy as np
    if os.path.exists(ft_vecs_path):
        ft_vecs = np.load(ft_vecs_path)
        if len(ft_vecs) != len(chunks):
            ft_vecs = emb.embed_chunks(chunks, model=ft_model)
            np.save(ft_vecs_path, ft_vecs)
    else:
        ft_vecs = emb.embed_chunks(chunks, model=ft_model)
        np.save(ft_vecs_path, ft_vecs)

    print("\n=== fine-tuned retriever (same architecture, trained on Set A's train split) ===")
    ft_a = score_set(test_a, chunks, ft_vecs, bm25, ft_model, "A_test (held out, fine-tuned)")
    ft_b = score_set(questions, chunks, ft_vecs, bm25, ft_model, "B (fine-tuned)")

    print("\n--- summary: recall@5, dense column only ---")
    print(f"{'set':<10}{'base':>10}{'fine-tuned':>12}{'delta':>10}")
    for name, b, f in (("test_A", base_a, ft_a), ("B", base_b, ft_b)):
        print(f"{name:<10}{b['dense']:>10.3f}{f['dense']:>12.3f}{f['dense']-b['dense']:>+10.3f}")

    out = {
        "test_A": {"n": len(test_a), "base_dense": base_a["dense"], "finetuned_dense": ft_a["dense"],
                   "base_bm25": base_a["bm25"], "finetuned_bm25": ft_a["bm25"]},
        "B": {"n": len(questions), "base_dense": base_b["dense"], "finetuned_dense": ft_b["dense"],
             "base_bm25": base_b["bm25"], "finetuned_bm25": ft_b["bm25"]},
    }
    out_path = os.path.join(HERE, "finetune_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
