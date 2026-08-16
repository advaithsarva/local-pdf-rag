# RESULTS

Every number here came from a command that is printed next to it. Seeds are
fixed; the retrieval numbers are deterministic. Measured 2026-08-16 on CPU
(no GPU available on this machine), Python 3.12.1, torch 2.7.1+cpu.

---

## 0. Security

The folder contained **six live API keys in plaintext**, across three providers:

| Provider | File |
|---|---|
| OpenAI (`sk-proj-…`) | `Untitled.ipynb`, `.ipynb_checkpoints/Untitled-checkpoint.ipynb` |
| OpenAI (legacy `sk-…`) | `summarization.ipynb` |
| Anthropic (`sk-ant-api03-…`) | `.ipynb_checkpoints/Untitled-checkpoint.ipynb` |
| HuggingFace (`hf_…`, ×2) | `Llama2_with_llamaindex.ipynb` |

```bash
grep -rInoE "(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})" .
```

**The folder was never a git repository and was never pushed**, so the exposure
is local disk only — there is no history to purge and no public URL to retract.
That is the one piece of good news in this section.

The rebuilt project contains no credential. `HF_TOKEN` is read from the
environment with **no fallback default**, and is not required: the default
generator is ungated precisely so that nothing here depends on a secret.

> ⚠️ **Still outstanding: rotate all six at their providers.** The files remain
> at `C:\move\Code\DSMLDL\RAG` and were deliberately not deleted — deleting a
> key is not revoking it, and destroying the only copy of the evidence before
> the owner has acted on it helps nobody.

---

## 1. THE root cause: sentences were re-joined without their whitespace

One defect explains everything wrong with the shipped artifact.

The pipeline split each page into sentences with spaCy, then rebuilt each chunk
by concatenating them:

```python
item["sentences"] = [str(sentence) for sentence in nlp(item["text"]).sents]   # cell 9
joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()    # cell 12
joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # cell 12
```

`str(span)` returns `Span.text`, which **excludes the span's trailing
whitespace**. `"".join(...)` does not put it back. So every sentence boundary
inside every chunk lost its separator, and adjacent words fused.

Reproduced in five lines:

```python
>>> src = "Is that clear? Yes it is."
>>> "".join(str(s) for s in nlp(src).sents)
'Is that clear?Yes it is.'
```

The `re.sub(r'\.([A-Z])', r'. \1', ...)` line on the next line of the notebook is
a **partial repair of this exact bug** — inherited from the upstream tutorial.
It restores the space after a full stop followed by a capital letter, and only
that. It does nothing for `?`, `!`, a lowercase continuation, a URL, a quote, a
bracket or a digit. Its presence is why the defect was survivable enough to ship.

### What it cost, measured against the PDF

For all 1,680 shipped chunks: is the chunk text present on the page it cites?

| | count | rate |
|---|---|---|
| verbatim quote of the cited page | 1,328 | **0.790** |
| same characters, whitespace mangled | 352 | **0.210** |
| text altered beyond whitespace, or wrong page | 0 | 0.000 |

```bash
python eval/measure_original.py     # needs the original CSV + the PDF (both gitignored)
```

**The page citations were 100% correct.** The quoted text was not. For a system
whose product is "here is the answer and here is the page it came from", 21% of
citations pointing at text the page does not literally contain is the whole
problem, and nothing ever raised.

Symptoms, all from this one cause:

1. Fused tokens (`clear?Yes`, `Medicine.https`) go into the embedding model as
   out-of-vocabulary junk, degrading the vector for the chunk that contains them.
2. `chunk_word_count`, computed as `len(text.split(" "))`, undercounts by one
   per sentence boundary — a metric that silently disagreed with the text.
3. A quoted passage could not be located in the source, so no downstream check
   on a citation was possible even in principle.
4. The band-aid regex made the output *look* fine on the common case, which is
   how it survived to being saved to disk and reloaded for months.

### The fix

`ragpdf/chunk.py` never joins. It records `(start, end)` character offsets on
the raw page text and produces the chunk as `page_text[start:end]`. The invariant
is checked at build time and the build aborts if it fails:

```
$ python -m ragpdf.cli build
[build] 1715 chunks | verbatim 1715/1715 = 1.0000
[build] wrote index/ (1715x768 float32) in 456.3s
```

| | original | rebuild |
|---|---|---|
| chunks | 1,680 | 1,715 |
| **verbatim quote of cited page** | **0.790** | **1.0000** |
| citations to the wrong page | 0.000 | 0.000 |

A side effect worth stating: the rebuild reproduces the PDF's *own* text-layer
artifacts faithfully — `"The cardiovascul ar system"` appears on page 87 exactly
like that, because that is what the PDF contains. A citation system that
silently tidies its quotes is a citation system you cannot trust; the tidying
belongs in the display layer, not in the stored chunk.

### A bug in the rewrite, worth recording

The first version of `sentence_spans` ended each span at the *start* of the
boundary match:

```python
_BOUNDARY = re.compile(r'(?<=[.!?])["\'’”)\]]*\s')
end = m.start()          # wrong: the closing ')' or '"' lands in no span at all
```

Closing brackets and quotes therefore belonged to neither the sentence before
them nor the one after. Interior boundaries were unaffected — a chunk is sliced
across them — but the **final** boundary of a chunk truncated the last character:

```
...salt. (Salt is 60 percent chloride.        <- the ')' was gone
```

**9 of 1,715 chunks (0.5%).** It never violated the invariant — the text was
still a verbatim slice, just a slice one character short of the sentence — which
is exactly why the build check did not catch it and a `1.0000` verbatim rate did
not mean the chunker was right. Found by scanning for chunks whose `end` offset
landed immediately before a closing character.

Fixed by making the whitespace a capture group and ending the span at
`m.start(1)`. `test_closing_punctuation_stays_with_its_sentence` covers it. The
rebuilt index has the same 1,715 chunks and the same 1.0000 verbatim rate, and
**every recall figure in §3 was identical before and after** — the fix is about
quote fidelity, not retrieval.

The lesson is the useful part: *an invariant tells you the class of bug it was
designed to exclude is absent, and nothing else.* A green check is not a proof
of correctness.

---

## 2. Things that looked like bugs and were not

Reported because a diagnosis that only finds problems is not a diagnosis.

**The CSV round-trip is fine.** The original wrote each 768-float vector as its
`str()` repr into a CSV and parsed it back with `np.fromstring`. That looks
lossy and is the obvious suspect. Measured against a fresh encode of the same
chunks:

| | | |
|---|---|---|
| max absolute error | **1.192e-07** | 12 chunks, `default_rng(42)` |
| minimum cosine similarity | **0.999999969** | same sample |
| vectors truncated by numpy's `...` repr | **0 / 1,680** | all rows |
| NaN, zero-norm, or wrong-dimension rows | **0** | all rows |
| duplicate chunk texts | **0** | all rows |

```bash
python eval/measure_original.py     # section 2
```

768 elements is under numpy's 1,000-element summarisation threshold, so nothing
was elided. The round trip is *correct*. It is merely expensive — 21.3 MB on
disk and ~20 s to parse, for 5 MB of floats. The rebuild uses `.npy` + `.json`:

| | original | rebuild |
|---|---|---|
| index on disk | 21.3 MB (1 CSV) | **6.8 MB** (5.3 MB `.npy` + 1.5 MB `.json`) |
| load time | ~20 s | **<0.1 s** |

**The minimum-token filter dropped nothing substantive.** 72 book pages have no
surviving chunk. Every one of them has fewer than 60 words of extractable text —
they are plates, blank versos and section dividers. Zero pages with real content
were lost.

**The saved embeddings were genuinely the user's own run.** 1,680 unit-norm
768-dim vectors, no duplicates, no NaNs, and the notebook's saved cell outputs
show a complete end-to-end run on CPU with `google/gemma-2b-it` producing real
answers. This project was not a folder of intentions.

---

## 3. Retrieval: dense vs. the boring baseline

Both retrievers run over the **same 1,715 chunks**, so this compares retrievers
and not pipelines. `recall@5` = the correct page appears among the 5 chunks
returned.

```bash
python -m ragpdf.cli eval
```

### Set A — 104 section titles, taken from the PDF's own table of contents

Nobody wrote these. The document supplies the query (the section title) and the
gold pages (that section's page range, up to the next TOC entry). They cannot be
tuned. This is keyword search's home ground: the query is literally printed on
the target page.

| | recall@5 | median latency |
|---|---|---|
| BM25 (keyword baseline) | **0.990** (103/104) | **2.9 ms** |
| dense, all-mpnet-base-v2 | 0.971 (101/104) | 51.3 ms |

**The baseline wins, at ~18× the speed.** Published because it is true. If your
users paste headings and product names, the transformer is costing you money.

The one query both miss is `"Appendices"` — a two-page navigational stub with no
prose. That is a defect in the query set, not in either retriever; it survived
the "drop navigational titles" filter and is left in rather than quietly removed
after seeing the result.

### Set B — 34 questions in the wording a reader would actually use

Authored. Each question deliberately avoids the target section's title words
("how do you tell whether someone weighs too much for their height", not "body
mass index"), which is exactly the case dense retrieval is bought for.

| | recall@5 | median latency |
|---|---|---|
| dense, all-mpnet-base-v2 | **1.000** (34/34) | 99.8 ms |
| BM25 (keyword baseline) | 0.588 (20/34) | 8.9 ms |

**+0.412 recall@5 for the model.** That gap, and not Set A, is the entire
argument for embedding anything.

> **On the latency figures.** Absolute milliseconds are load-dependent — these
> were taken on a contended CPU, and an earlier run of the same command on the
> same index gave 38.1/2.1 ms (Set A) and 46.9/4.8 ms (Set B). The recall figures
> were **identical in both runs**; only the timings moved. Treat the ratio
> (dense costs roughly 10–20× BM25) as the reproducible quantity, not the
> absolute numbers.

> ### Caveat, stated plainly
>
> **Set B was written by the same person who built the retriever, after the
> retriever existed.** It is honest about the *kind* of query dense retrieval
> handles and keyword search does not; it is not evidence about real user
> traffic, and 34 queries is a small set — one query is worth 0.029 recall.
> Set A exists to carry the load Set B cannot: it is unauthored, four times
> larger, and the baseline beats the model on it.

> ### Two labels were corrected after the first run, and here is the audit trail
>
> The first run scored **dense 0.941 (32/34)**. Both misses were inspected, and
> in both cases the *label* was wrong, not the retrieval:
>
> | Query | Labelled | Actually on | Correction |
> |---|---|---|---|
> | "how many calories does a person burn just staying alive" | 492–500, *Factors Affecting Energy Expenditure* | **p480** — basal metabolism is defined inside *Weight Management* | widened to 472–500 |
> | "why must cholesterol be carried through the blood by carriers" | 305–312, *How Lipids Work* | **pp322–326** — chylomicrons, LDL and HDL are in *Digestion and Absorption of Lipids* | moved to 319–330 |
>
> Both corrections are recorded in the `label_corrected` field of
> `eval/questions.json` and were verified by reading the pages. The section
> *titles* implied one thing and the section *contents* another — labelling from
> a table of contents without opening the pages is how a retriever gets blamed
> for being right. **Pre-correction numbers: dense 0.941, BM25 0.529. The gap
> moved from +0.412 to +0.412 — the correction helped both retrievers equally
> and changed no conclusion.**

---

## 4. Abstention: does it know when to shut up?

12 questions the textbook cannot answer (football, git, TCP/UDP, Moby Dick).

```bash
python -m ragpdf.cli eval        # Set C
```

| top-1 dense score | min | median | max |
|---|---|---|---|
| in-corpus (34 Set B questions) | **0.410** | 0.667 | — |
| out-of-corpus (12 questions) | — | 0.158 | **0.277** |

The two distributions do not overlap. `MIN_SCORE = 0.35` sits in the gap between
0.277 and 0.410; it was read off this table, not tuned per query.

| | refuses out-of-corpus | still answers in-corpus |
|---|---|---|
| score gate at 0.35 | **12/12 = 1.000** | **34/34 = 1.000** |
| extractive baseline (no gate) | 0/12 = 0.000 | 34/34 |

Without the gate the system answers "who won the 2010 FIFA World Cup" with five
passages about nutrition, ranked, scored, and never raising. That is the failure
mode the gate exists for, and the 0.133 margin between the distributions is
narrower than it looks — a longer or more nutrition-adjacent out-of-corpus
question would close it. **This is 12 questions, not a guarantee.**

---

## 5. Generation: the trade, not the upgrade

The Phase 3.5 feature is a **grounded answer with checkable page citations and
abstention**. The baseline it must beat is the extractive path underneath it:
return the 5 retrieved passages verbatim with their page numbers.

```bash
python eval/run_generation.py    # 34 questions, Qwen/Qwen2.5-0.5B-Instruct, CPU
```

34 questions (Set B), same index, same retrieved context for both paths, greedy
decoding so the figure is deterministic rather than one sample.

| | grounding | median length | latency |
|---|---|---|---|
| extractive baseline | **1.000** | 5,052 chars | ~0 s |
| generated (Qwen2.5-0.5B-Instruct) | **0.632** | **350 chars** | 31.3 s median, 102.4 s max |

**The generator loses on the accuracy-shaped number and wins on the
readability-shaped one.** It is a trade, and both halves of it are real:

- **14× shorter.** 350 characters instead of 5,052 — one paragraph instead of
  five passages. That is the whole reason anyone wants this.
- **0.632 grounding.** More than a third of the answer's content words are not in
  the pages it cites. They come from the model's weights. On a 0.5B model that
  is the expected price; it is still a price.
- **31 seconds instead of zero**, on CPU, with a 102-second worst case.

Per the rule this feature was built under — *if the model cannot beat the boring
version on a number you can produce, say so* — **it does not beat it on
correctness, so the extractive path stays the default.** `--generate` is opt-in,
and that is the finding, not an oversight.

### Two failure modes worth naming

**1. The model refuses when the context contains the answer. 1 of 34.**

> *"which fats must come from food because the body cannot make them"*
> → **"The textbook does not cover this."**

Retrieval was correct here — Set B is 34/34, so *Nonessential and Essential Fatty
Acids* was in the context. The 0.5B model declined anyway. This is a **model-level
false refusal**, entirely distinct from the §4 score gate, which never fired on
this query. A system can have a perfectly calibrated abstention threshold and
still refuse for a completely different reason one layer up. Excluding it,
grounding over the remaining 33 is 0.651 (min 0.167, median 0.600, max 1.000,
with 5 answers fully grounded).

**2. A fluent, plausible, subtly wrong answer. Grounding 0.167:**

> *"how does canning or freezing stop food from spoiling"* → *"Canning or freezing
> prevent food spoilage by removing all living organisms and enzymes, thus
> preventing bacterial growth and chemical reactions that cause food to
> deteriorate."*

Freezing removes neither organisms nor enzymes — it suspends them. The answer
reads well, cites pages 1019–1036 which really are about food preservation, and
is wrong. **This is why §8 says factual accuracy is not measured anywhere here.**
Grounding caught this one at 0.167 because the invented claim used vocabulary
absent from the pages, but grounding measures word provenance, not truth, and it
will not catch a wrong claim assembled from the right words.

### Why the default model is not the one the notebook used

The original used `google/gemma-2b-it`, which is gated: it needs a HuggingFace
account, an accepted licence and an `HF_TOKEN`. **A feature nobody else can run
is a feature nobody can check.** The default is therefore an ungated 0.5B model,
so the grounding figure above is reproducible by any reader with no credential.
`--model google/gemma-2b-it` restores the original behaviour and picks up
`HF_TOKEN` if it is set.

---

## 6. Tests

```bash
$ python tests/test_ragpdf.py
ragpdf: 14/14 passed, 0 failed

$ python tests/verify_tests.py
3/3 chunking assertions fail against the original.
rebuild (full suite): 14/14 passed, 0 failed
RESULT: suite is trustworthy
```

The suite runs in under a second: no pytest, no network, no PDF, no model
download, one in-file fixture page containing every join hazard the real
textbook has.

`tests/original_impl.py` is the original chunker transcribed from notebook cells
5, 9 and 12. `verify_tests.py` runs the three chunking assertions against it, and
all three must fail — if they passed, the suite would be decorative.

Two tests are deliberately **excluded** from that 3/3 count, because the
original passes them:

- `test_chunk_carries_its_page` — a sanity check, not a discriminator.
- `test_closing_punctuation_stays_with_its_sentence` — this guards a bug in the
  **rewrite** (§1, "A bug in the rewrite"), not one the original had. It is a
  regression test, and counting it would inflate a number that is supposed to
  measure how well the suite catches the *original's* defects.

Counting either would make the suite look more discriminating than it is.

---

## 7. Dependencies

| | original | rebuild |
|---|---|---|
| declared runtime dependencies | 12 | **3** |

Dropped: `spacy` (the sentencizer's `Span.text` semantics *were* the bug —
`chunk.py` uses a regex and slices), `pandas` (the index is `.npy` + `.json`),
`accelerate`, `bitsandbytes`, `flash-attn` (all GPU quantisation, on a machine
with no GPU), `matplotlib`, `psutil`, `tqdm`.

Remaining: `pymupdf`, `sentence-transformers`, `numpy`. `torch` and
`transformers` arrive under `sentence-transformers` and are not declared
separately because nothing here pins or configures them.

---

## 8. What is not measured

- **The generated answer's factual correctness.** Grounding measures whether the
  answer's words came from the retrieved pages, not whether the claim is true.
  No claim of factual accuracy is made anywhere in this repo.
- **`google/gemma-2b-it`.** The original notebook ran it end to end on CPU and
  its saved outputs are in `notebooks/original/rag-pipeline.ipynb`, but no
  grounding figure was computed for it here. Do not attach one.
- **Recall@k for k ≠ 5**, and **MRR/nDCG.** Only recall@5 was measured.
- **Anything about a corpus other than this one PDF.**
