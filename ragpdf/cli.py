"""build | ask | eval | verify -- and `ask --json` is the agent-callable surface.

    python -m ragpdf.cli build --pdf human-nutrition-text.pdf
    python -m ragpdf.cli ask "what are the fat soluble vitamins" --json
    python -m ragpdf.cli eval
    python -m ragpdf.cli verify
"""

import argparse
import json
import os
import sys
import time

from . import answer as ans
from . import chunk as ch
from . import embed as emb
from . import retrieve as ret

DEFAULT_PDF = "human-nutrition-text.pdf"
DEFAULT_INDEX = "index"


def cmd_build(a):
    t0 = time.perf_counter()
    chunks = ch.build_chunks(a.pdf, a.sentences, a.min_chars)
    ok, total = ch.verify_verbatim(chunks, a.pdf)
    print(f"[build] {total} chunks | verbatim {ok}/{total} = {ok/total:.4f}")
    if ok != total:
        raise SystemExit("[build] INVARIANT VIOLATED: a chunk is not a slice of its page")
    vecs = emb.embed_chunks(chunks)
    emb.save(a.index, chunks, vecs)
    print(f"[build] wrote {a.index}/ ({vecs.shape[0]}x{vecs.shape[1]} float32) "
          f"in {time.perf_counter()-t0:.1f}s")


def cmd_ask(a):
    chunks, vecs = emb.load(a.index)
    model = emb._model()
    t0 = time.perf_counter()
    idx, scores = ret.dense(emb.embed_query(a.query, model), vecs, a.k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    generator = None
    reason = "generation not requested"
    if a.generate:
        generator, reason = ans.load_generator(a.model)
        if generator is None:
            print(f"[warn] generation off: {reason}", file=sys.stderr)

    result = ans.answer(a.query, chunks, idx, scores, generator)
    result["query"] = a.query
    result["retrieval_ms"] = round(retrieval_ms, 2)
    if a.generate and generator is None:
        result["reason"] = reason

    if a.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"Query: {a.query}")
        print(f"Pages: {result['pages']}  ({result['retrieval_ms']} ms)\n")
        print(result["answer"])


def cmd_eval(a):
    # eval/ is a sibling of the package, not part of it -- reach it from
    # __file__ so this works regardless of the caller's working directory.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from eval.run_eval import main as run
    run(a.index, a.pdf)


def cmd_verify(a):
    chunks, _ = emb.load(a.index)
    ok, total = ch.verify_verbatim(chunks, a.pdf)
    print(f"verbatim {ok}/{total} = {ok/total:.4f}")
    raise SystemExit(0 if ok == total else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragpdf")
    p.add_argument("--pdf", default=DEFAULT_PDF)
    p.add_argument("--index", default=DEFAULT_INDEX)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--sentences", type=int, default=10)
    b.add_argument("--min-chars", dest="min_chars", type=int, default=120)

    q = sub.add_parser("ask"); q.set_defaults(fn=cmd_ask)
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--generate", action="store_true", help="synthesise an answer from the passages")
    q.add_argument("--model", default=ans.MODEL_ID,
                   help="generator model id; gated ones (google/gemma-2b-it) need HF_TOKEN")
    q.add_argument("--json", action="store_true")

    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
