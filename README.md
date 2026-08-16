# local-pdf-rag

Retrieval-augmented question answering over a PDF, running entirely on CPU, with
**page citations you can check against the source document.**

The corpus is [*Human Nutrition: 2020 Edition*](https://pressbooks.oer.hawaii.edu/humannutrition2/)
(1,208 pages, CC BY-NC-SA) → 1,715 chunks → 768-dim embeddings → dense retrieval
→ an optional generated answer that must cite the pages it used, and abstains
when the book does not cover the question.

No GPU, no API key, no vector database, three runtime dependencies.

---

## The invariant

> **A chunk's text is a verbatim slice of the page it cites.**
>
> `chunk["text"] == page_text(pdf, chunk["page"])[chunk["start"]:chunk["end"]]`

Stated at the top of `ragpdf/chunk.py`, enforced at build time (the build aborts
if it does not hold), and checkable at any time with
`python -m ragpdf.cli verify`, which exits non-zero if a single chunk drifts.

It exists because the original violated it, silently, on 21% of its chunks. See
[RESULTS.md §1](RESULTS.md).

---

## Run it

```bash
pip install -r requirements.txt

# 1. get the source document (26 MB, not redistributed here)
curl -L "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf" \
     -o human-nutrition-text.pdf

# 2. chunk + embed  (~7 min on CPU, writes index/)
python -m ragpdf.cli build

# 3. ask
python -m ragpdf.cli ask "which vitamins are stored in body fat"
python -m ragpdf.cli ask "how does the body break down starch" --generate
python -m ragpdf.cli ask "who won the 2010 World Cup"       # -> abstains

# 4. reproduce every number in RESULTS.md
python -m ragpdf.cli eval          # retrieval + abstention   (~2 min)
python eval/run_generation.py      # generation vs baseline   (~30 min)
python tests/test_ragpdf.py        # 14 tests                 (<1 s)
python tests/verify_tests.py       # the same tests vs the original code
python -m ragpdf.cli verify        # re-check the invariant, exit 1 on drift

# the "before" numbers, if you still have the original CSV
python eval/measure_original.py
```

### For an agent, not a person

`ask --json` is the machine surface. It prints one object and nothing else:

```bash
$ python -m ragpdf.cli ask "which mineral is added to water to protect teeth" --json
{
  "answer": "Fluoride ... Fluoride is known mostly as the mineral that combats
              tooth decay. ... Fluoride was first added to drinking water in 1945
              in Grand Rapids, Michigan; ...",
  "pages": [695, 696, 1083, 1083, 702],
  "scores": [0.7042, 0.5819, 0.5256, 0.5209, 0.5173],
  "generated": false,
  "abstained": false,
  "reason": "extractive baseline",
  "query": "which mineral is added to water to protect teeth",
  "retrieval_ms": 116.4
}

$ python -m ragpdf.cli ask "who won the 2010 FIFA World Cup" --json
{
  "answer": "The textbook does not cover this.",
  "pages": [],
  "scores": [],
  "generated": false,
  "abstained": true,
  "reason": "best retrieval score 0.168 < MIN_SCORE 0.35",
  "query": "who won the 2010 FIFA World Cup",
  "retrieval_ms": 125.2
}
```

Real output, `answer` elided for length — everything else verbatim.
`abstained: true` with `pages: []` is the "not in this document" answer, and the
`reason` always names the number that produced it. `pages` is one entry per
retrieved chunk and **may repeat** (1083 twice above) when two chunks come from
the same page. Because of the invariant, every page listed can be opened and the
quoted text found on it.

---

## Results

Full detail and commands in [RESULTS.md](RESULTS.md).

| | dense (all-mpnet-base-v2) | BM25 (keyword baseline) |
|---|---|---|
| **Set A** — 104 section titles from the PDF's own table of contents, unauthored | 0.971 | **0.990** |
| **Set B** — 34 questions in lay wording that avoids the section title | **1.000** | 0.588 |
| median latency (Set A) | 51 ms | 2.9 ms |

**The keyword baseline wins Set A**, and that is the honest headline: when the
query already contains the document's own words, a 30-line BM25 beats a
transformer at ~18× the speed. Dense retrieval earns its cost only on Set B —
questions phrased the way a reader would actually phrase them — where it gains
**+0.412 recall@5**.

Abstention, on 12 questions the textbook cannot answer:

| | refuses out-of-corpus | answers in-corpus |
|---|---|---|
| score gate at 0.35 | **12/12** | **34/34** |
| extractive baseline | 0/12 | 34/34 |

The threshold sits in a real gap, not a tuned one: in-corpus top-1 scores bottom
out at 0.410, out-of-corpus scores top out at 0.277.

**Generation is the trade, not the upgrade** — and it loses the half that matters:

| | grounding | median length | latency |
|---|---|---|---|
| extractive baseline | **1.000** | 5,052 chars | ~0 s |
| generated (Qwen2.5-0.5B-Instruct) | **0.632** | **350 chars** | 31 s median |

14× shorter, but more than a third of the generated answer's content words are
not on the pages it cites. So **the extractive path is the default and
`--generate` is opt-in.** RESULTS.md §5 has the two failure modes, including one
answer that is fluent, correctly cited and factually wrong.

---

## What was here before

This was a folder of nine notebooks, five of them unrelated to RAG, containing
six live API keys. What survived is one notebook —
[`notebooks/original/rag-pipeline.ipynb`](notebooks/original/rag-pipeline.ipynb)
— kept deliberately as the record of what went wrong, with its saved outputs
intact.

That notebook is a CPU adaptation of Daniel Bourke's
[simple-local-rag](https://github.com/mrdbourke/simple-local-rag) (MIT). The
tutorial's own copy was deleted rather than kept; what is here is the run that
was actually done, on CPU, with Gemma-2B. **The chunking defect in RESULTS.md §1
is inherited from that tutorial, not invented locally** — it is worth knowing
that a widely-followed tutorial ships it.

Deleted, and why:

| File | Why |
|---|---|
| `00-simple-local-rag.ipynb` | the upstream tutorial, downloaded, duplicated by the notebook that was actually run |
| `RAG.ipynb` | a Happy/Sad **image classifier**. Not RAG. Also `imhdr.what` (typo) inside a bare `except`, so every file silently failed the format check |
| `Llama2_with_llamaindex.ipynb` | never ran (`ImportError`, missing accelerate); two HuggingFace tokens |
| `summarization.ipynb`, `Untitled.ipynb` | LangChain summarisation, a different task; three live keys; the one run that executed died on a 429 |
| `Untitled1.ipynb` | `facebook/rag-token-nq`, failed on a 70 GB wiki_dpr download |
| `code_1.py` | speech-to-text loop, unrelated |
| `HackathonUdhgam/` | a different project (neural gas clustering); moved out, not deleted |

⚠️ **The deleted files still exist at `C:\move\Code\DSMLDL\RAG` and still contain
the keys.** They were never in git and never pushed, so the exposure is local —
but [the keys must be rotated at their providers](RESULTS.md#0-security)
regardless.

---

## Layout

```
ragpdf/chunk.py      PDF -> chunks.  STATES THE INVARIANT
ragpdf/embed.py      chunks -> index/{chunks.json, embeddings.npy}
ragpdf/retrieve.py   dense() and BM25 -- the model path and the boring path
ragpdf/answer.py     abstain / extract / generate, with degradation rules
ragpdf/cli.py        build · ask · eval · verify

eval/build_titles.py Set A, generated from the PDF's table of contents
eval/questions.json  Set B, authored (see the caveat in RESULTS.md)
eval/out_of_corpus.json  Set C, 12 questions the book cannot answer
eval/run_eval.py     every retrieval and abstention number
eval/run_generation.py   the Phase 3.5 comparison

tests/test_ragpdf.py     14 tests, no network, no PDF, <1 s
tests/original_impl.py   the original chunker, transcribed from the notebook
tests/verify_tests.py    runs the tests against it -- 3/3 must fail
```

Stages take data and return data. No module reads a file at import, connects to
anything, or exits.

## Licence and attribution

Code: MIT. The textbook is CC BY-NC-SA 4.0 and is **not** redistributed here —
`build` expects you to download it. The original notebook derives from
mrdbourke/simple-local-rag (MIT).
