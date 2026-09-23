"""BLEU, ROUGE-L and token-F1 for the generated answers already recorded in
eval/generation_results.json, scored against the retrieved context.

Reuses the cached generated answers rather than re-running the model (that
takes ~30 min and needs the 1 GB download -- see run_generation.py). Only
retrieval is re-run here, to reconstruct the context each generated answer
was actually produced from; retrieval is deterministic and the index has
not changed, so this reconstructs the exact same context, not an
approximation of it.

    python eval/measure_text_metrics.py
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragpdf import embed as emb          # noqa: E402
from ragpdf import retrieve as ret       # noqa: E402
from ragpdf import answer as ans         # noqa: E402
from ragpdf import text_metrics as tm    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5


def main(index_dir="index"):
    with open(os.path.join(HERE, "generation_results.json"), encoding="utf-8") as f:
        cached = json.load(f)

    chunks, vecs = emb.load(index_dir)
    model = emb._model()

    rows = []
    for sample in cached["samples"]:
        idx, scores = ret.dense(emb.embed_query(sample["query"], model), vecs, K)
        context = ans.format_context(chunks, idx)
        candidate = sample["answer"]

        bleu = tm.bleu_score(candidate, context)
        rouge = tm.rouge_l(candidate, context)
        f1 = tm.token_f1(candidate, context)

        rows.append({
            "query": sample["query"], "pages": sample["pages"],
            "grounding": sample["grounding"],  # already-computed precision, for comparison
            "bleu": round(bleu, 4), "rouge_l": round(rouge, 4),
            "f1_precision": f1["precision"], "f1_recall": f1["recall"], "f1": f1["f1"],
        })
        print(f"  BLEU {bleu:.3f}  ROUGE-L {rouge:.3f}  F1 {f1['f1']:.3f}  {sample['query'][:50]}")

    print(f"\n--- {len(rows)} questions, generated answers vs. retrieved context ---")
    print(f"  BLEU     mean {statistics.mean(r['bleu'] for r in rows):.3f}   "
         f"median {statistics.median(r['bleu'] for r in rows):.3f}")
    print(f"  ROUGE-L  mean {statistics.mean(r['rouge_l'] for r in rows):.3f}   "
         f"median {statistics.median(r['rouge_l'] for r in rows):.3f}")
    print(f"  F1       mean {statistics.mean(r['f1'] for r in rows):.3f}   "
         f"median {statistics.median(r['f1'] for r in rows):.3f}   "
         f"(precision {statistics.mean(r['f1_precision'] for r in rows):.3f}, "
         f"recall {statistics.mean(r['f1_recall'] for r in rows):.3f})")

    worst = sorted(rows, key=lambda r: r["f1"])[:3]
    print("\n  lowest-F1 answers:")
    for r in worst:
        print(f"    F1 {r['f1']:.3f} (precision {r['f1_precision']:.3f}, "
             f"recall {r['f1_recall']:.3f})  {r['query']}")

    out = {
        "n": len(rows),
        "bleu_mean": round(statistics.mean(r["bleu"] for r in rows), 4),
        "rouge_l_mean": round(statistics.mean(r["rouge_l"] for r in rows), 4),
        "f1_mean": round(statistics.mean(r["f1"] for r in rows), 4),
        "f1_precision_mean": round(statistics.mean(r["f1_precision"] for r in rows), 4),
        "f1_recall_mean": round(statistics.mean(r["f1_recall"] for r in rows), 4),
        "rows": rows,
    }
    out_path = os.path.join(HERE, "text_metrics_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
