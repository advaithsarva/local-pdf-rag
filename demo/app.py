"""Gradio front end for the PBL "live demo" requirement.

Deliberately outside `ragpdf/` and touches nothing in it: CLAUDE.md rule 10 says a web
UI was not built on purpose, because `ask --json` is already the whole machine-callable
API. This wraps that same API for a human -- `emb.load`, `emb._model`, `ret.dense`,
`ans.answer`, `ans.load_generator` -- and adds no retrieval, scoring or generation logic
of its own; every number a reader sees here is the same number `ragpdf.cli ask` prints.

    python demo/app.py                 # http://127.0.0.1:7860
"""

import os
import sys
import time

import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ragpdf import answer as ans  # noqa: E402
from ragpdf import embed as emb  # noqa: E402
from ragpdf import retrieve as ret  # noqa: E402

INDEX_DIR = os.path.join(ROOT, "index")
FINETUNED_DIR = os.path.join(ROOT, "models", "finetuned-retriever")

_chunks, _base_vecs = emb.load(INDEX_DIR)
_base_model = emb._model()
_retrievers = {"base (all-mpnet-base-v2)": (_base_model, _base_vecs)}

if os.path.isdir(FINETUNED_DIR):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ft_vecs_path = os.path.join(INDEX_DIR, "embeddings_finetuned.npy")
    if os.path.exists(_ft_vecs_path):
        _ft_model = SentenceTransformer(FINETUNED_DIR, device="cpu")
        _ft_vecs = np.load(_ft_vecs_path)
        _retrievers["fine-tuned (Set A train split)"] = (_ft_model, _ft_vecs)

_generator_cache = {}  # model_id -> (tok, model) or (None, reason), loaded once


def _get_generator(model_id):
    if model_id not in _generator_cache:
        _generator_cache[model_id] = ans.load_generator(model_id)
    return _generator_cache[model_id]


def run(query, retriever_name, k, generate, model_id):
    if not query.strip():
        return "Type a question first.", "", ""

    model, vecs = _retrievers[retriever_name]
    t0 = time.perf_counter()
    idx, scores = ret.dense(emb.embed_query(query, model), vecs, int(k))
    retrieval_ms = (time.perf_counter() - t0) * 1000

    generator, reason = (None, "generation not requested")
    if generate:
        generator, reason = _get_generator(model_id)

    result = ans.answer(query, _chunks, idx, scores, generator)

    status = (f"retrieval {retrieval_ms:.0f} ms | retriever: {retriever_name} | "
             f"{'abstained' if result['abstained'] else result['reason']}")
    if generate and generator is None:
        status += f" | generation skipped: {reason}"
    if result.get("generated"):
        status += f" | grounding {result['grounding']}"

    pages = ", ".join(str(p) for p in result["pages"]) or "-"
    return result["answer"], pages, status


def build():
    with gr.Blocks(title="local-pdf-rag demo") as demo:
        gr.Markdown(
            "## Human Nutrition RAG\n"
            "Question answering over *Human Nutrition: 2020 Edition* (1,208 pages). "
            "Every citation is a verbatim slice of the page it names -- see the "
            "project's `chunk.py` invariant. Out-of-book questions are refused, not "
            "guessed at."
        )
        with gr.Row():
            query = gr.Textbox(label="Question", scale=4,
                               placeholder="which vitamins are stored in body fat")
            k = gr.Slider(1, 10, value=5, step=1, label="passages (k)", scale=1)
        with gr.Row():
            retriever = gr.Dropdown(list(_retrievers.keys()),
                                    value=next(iter(_retrievers)), label="retriever")
            generate = gr.Checkbox(value=False, label="generate (opt-in; extractive is the default)")
            model_id = gr.Textbox(value=ans.MODEL_ID, label="generator model id")
        ask_btn = gr.Button("Ask", variant="primary")
        answer_box = gr.Textbox(label="Answer", lines=10)
        pages_box = gr.Textbox(label="Cited pages")
        status_box = gr.Textbox(label="Status")

        ask_btn.click(run, [query, retriever, k, generate, model_id],
                     [answer_box, pages_box, status_box])
        query.submit(run, [query, retriever, k, generate, model_id],
                     [answer_box, pages_box, status_box])
    return demo


if __name__ == "__main__":
    build().launch()
