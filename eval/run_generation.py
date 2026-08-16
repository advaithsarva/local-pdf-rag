"""Phase 3.5: does the generator beat the extractive baseline it sits on?

    python eval/run_generation.py

Measured on the 34 Set B questions, same index, same retrieved context:

  grounding  fraction of the answer's content words present in the retrieved
             context. The extractive baseline scores 1.0 by construction --
             its text IS the context. Anything the generator scores below 1.0
             is content it introduced from its own weights, i.e. the risk you
             take on in exchange for a readable answer.
  length     characters. The baseline hands back ~5 passages; the point of the
             generator is that a person reads one paragraph instead.
  latency    seconds per answer on CPU, and the cost of the trade.

Run separately from run_eval.py because it downloads ~1 GB and takes minutes.
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragpdf import answer as ans  # noqa: E402
from ragpdf import embed as emb  # noqa: E402
from ragpdf import retrieve as ret  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = 5


def main(index_dir="index", model_id=ans.MODEL_ID, limit=None):
    with open(os.path.join(HERE, "questions.json"), encoding="utf-8") as f:
        rows = json.load(f)
    if limit:
        rows = rows[:limit]

    chunks, vecs = emb.load(index_dir)
    st = emb._model()

    generator, reason = ans.load_generator(model_id)
    if generator is None:
        print(f"generation unavailable: {reason}")
        print("Nothing to measure. The extractive path is unaffected.")
        return

    gen_ground, gen_len, gen_secs = [], [], []
    base_ground, base_len = [], []
    samples = []

    for r in rows:
        idx, scores = ret.dense(emb.embed_query(r["query"], st), vecs, K)
        ctx = ans.format_context(chunks, idx)

        base = ans.extractive(chunks, idx, scores)
        base_ground.append(ans.grounding(base["answer"], ctx))
        base_len.append(len(base["answer"]))

        t = time.perf_counter()
        out = ans.answer(r["query"], chunks, idx, scores, generator=generator)
        gen_secs.append(time.perf_counter() - t)
        gen_ground.append(out["grounding"])
        gen_len.append(len(out["answer"]))
        samples.append({"query": r["query"], "pages": out["pages"],
                        "grounding": out["grounding"], "answer": out["answer"]})
        print(f"  {out['grounding']:.3f}  {gen_secs[-1]:6.1f}s  {r['query'][:58]}")

    n = len(rows)
    print(f"\n--- generation vs extractive, {n} questions, {model_id} ---")
    print(f"  extractive  grounding {statistics.mean(base_ground):.3f}   "
          f"median {statistics.median(base_len):5.0f} chars   ~0.0 s")
    print(f"  generated   grounding {statistics.mean(gen_ground):.3f}   "
          f"median {statistics.median(gen_len):5.0f} chars   "
          f"{statistics.median(gen_secs):.1f} s median, {max(gen_secs):.1f} s max")
    worst = min(samples, key=lambda s: s["grounding"])
    print(f"\n  least-grounded answer ({worst['grounding']:.3f}) -- {worst['query']}")
    print(f"    {worst['answer'][:300]}")

    out_path = os.path.join(HERE, "generation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_id, "n": n,
            "extractive": {"grounding": statistics.mean(base_ground),
                           "median_chars": statistics.median(base_len)},
            "generated": {"grounding": statistics.mean(gen_ground),
                          "median_chars": statistics.median(gen_len),
                          "median_secs": statistics.median(gen_secs),
                          "max_secs": max(gen_secs)},
            "samples": samples,
        }, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
