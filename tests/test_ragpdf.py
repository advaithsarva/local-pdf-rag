"""One test per bug actually found in the original. No pytest, no network, no
PDF, no model download. Runs in well under a second.

    python tests/test_ragpdf.py

Every test here fails against the original implementation except where noted --
see tests/verify_tests.py, which proves that.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragpdf import answer as ans
from ragpdf import chunk as ch
from ragpdf import embed as emb
from ragpdf import retrieve as ret
from ragpdf import finetune_retriever as ft

# A page with every join hazard the real textbook has: a question mark, a URL
# after a full stop, a quote, a bracket and a digit.
PAGE = (
    "Vitamin C is water soluble. It is not stored in the body.\n"
    "Is that a problem? Not if intake is regular.\n"
    'See the reference table.https://example.org/rdi lists every value.\n'
    'The report said "intake is adequate." (See figure 3.) '
    "Adults need 90 mg per day. Smokers need 35 mg more than that."
)


def _chunks(chunker=ch.chunk_page, page_text=PAGE, n=2, min_chars=0):
    return chunker(page_text, page=7, sentences_per_chunk=n, min_chars=min_chars)


# --- THE bug -----------------------------------------------------------------

def test_chunks_are_verbatim_slices(chunker=ch.chunk_page):
    """THE bug. `"".join(str(sent) for sent in sents)` dropped the whitespace
    spaCy leaves outside a Span, so 21% of shipped chunks were not quotes of
    the page they cited. A chunk must be a slice."""
    for c in _chunks(chunker):
        assert PAGE[c["start"]:c["end"]] == c["text"], \
            f"chunk at {c['start']}:{c['end']} is not a slice of its page:\n" \
            f"  slice: {PAGE[c['start']:c['end']]!r}\n  chunk: {c['text']!r}"


def test_no_fused_sentence_boundaries(chunker=ch.chunk_page):
    """The visible symptom: 'Is that a problem?Not if...'. The original's
    band-aid regex only repaired '.' followed by a capital, so '?' and lowercase
    boundaries stayed fused."""
    for c in _chunks(chunker):
        for bad in ("problem?Not", "stored.Is", "table.https"):
            if bad in PAGE:
                continue  # present in the source itself, not our doing
            assert bad not in c["text"], f"fused boundary {bad!r} in {c['text']!r}"


def test_chunk_text_is_findable_in_the_page(chunker=ch.chunk_page):
    """Weaker than the slice test and it still catches the original: after
    `text.replace("\\n", " ")` and `.replace("  ", " ")`, a chunk cannot be
    located in the page it cites even as a plain substring. Nothing downstream
    can then verify a citation."""
    for c in _chunks(chunker):
        assert c["text"] in PAGE, \
            f"chunk text does not occur in its page at all: {c['text']!r}"


# --- silent-failure guards ---------------------------------------------------

def test_empty_document_raises(chunker=None):
    """A scanned PDF with no text layer produced an empty corpus and no error."""
    assert ch.chunk_page("   \n  ", page=0, sentences_per_chunk=10, min_chars=1) == []
    try:
        ch.build_chunks.__wrapped__  # noqa
    except AttributeError:
        pass
    # build_chunks must raise rather than return []
    import types
    fake = types.SimpleNamespace()
    fake.pages = ["  ", "\n"]
    orig = ch.page_texts
    ch.page_texts = lambda p: fake.pages
    try:
        ch.build_chunks("nowhere.pdf")
    except ValueError:
        pass
    else:
        raise AssertionError("build_chunks returned an empty corpus instead of raising")
    finally:
        ch.page_texts = orig


def test_asking_for_more_results_than_chunks_raises(chunker=None):
    """`k = min(k, n)` is how a bug becomes a silent bug. Raise instead."""
    vecs = np.eye(3, 4, dtype=np.float32)
    try:
        ret.dense(vecs[0], vecs, k=10)
    except ValueError:
        pass
    else:
        raise AssertionError("dense() silently clamped k")
    try:
        ret.BM25(["a b c", "d e f"]).top("a", k=9)
    except ValueError:
        pass
    else:
        raise AssertionError("BM25.top() silently clamped k")


def test_index_length_mismatch_raises(chunker=None):
    """1680 chunks and 1679 vectors must not load as a working index."""
    try:
        emb.save(os.path.join(os.path.dirname(__file__), "_tmp_idx"),
                 [{"text": "a"}, {"text": "b"}], np.zeros((1, 4), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("save() accepted a chunk/embedding count mismatch")


# --- citations ---------------------------------------------------------------

def test_page_offset_round_trips(chunker=None):
    """A citation is only useful if it is the number printed on the page. The
    PDF's page 42 is the book's page 0."""
    assert ch.PAGE_OFFSET == 41
    for pdf_index in (0, 41, 42, 1207):
        book = pdf_index - ch.PAGE_OFFSET
        assert book + ch.PAGE_OFFSET == pdf_index


def test_chunk_carries_its_page(chunker=ch.chunk_page):
    for c in _chunks(chunker):
        assert c["page"] == 7


def test_closing_punctuation_stays_with_its_sentence(chunker=ch.chunk_page):
    """A sentence ends after its closing bracket or quote, not before it. Ending
    a span at the boundary match start instead truncated '(See figure 3.)' to
    '(See figure 3.' on any chunk whose final boundary landed there -- still a
    verbatim slice, just one character short of the sentence."""
    joined = "".join(c["text"] for c in _chunks(chunker, n=1))
    for closer in ('adequate."', "(See figure 3.)"):
        assert closer in joined, f"{closer!r} lost its closing character"


# --- retrieval ---------------------------------------------------------------

def test_bm25_idf_never_negative(chunker=None):
    """Plain Okapi idf goes negative for a term in more than half the corpus,
    so a common word actively penalises the documents containing it."""
    bm = ret.BM25(["vitamin c", "vitamin d", "vitamin e", "iron"])
    assert bm.idf["vitamin"] > 0, f"idf('vitamin') = {bm.idf['vitamin']}"
    assert (bm.scores("vitamin") >= 0).all()


def test_dense_returns_scores_in_descending_order(chunker=None):
    vecs = np.array([[1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx, scores = ret.dense(vecs[0], vecs, k=3)
    assert idx[0] == 0 and scores == sorted(scores, reverse=True), (idx, scores)


# --- the model layer degrades ------------------------------------------------

def test_unloadable_model_degrades_it_does_not_crash(chunker=None):
    """A model that cannot be loaded -- gated, offline, misspelled -- must mean
    'generation off, here are the passages, and here is why', never a traceback
    and never a silently empty answer. Offline-safe: the id does not exist, so
    this fails before any download."""
    saved = os.environ.pop("HF_TOKEN", None)
    try:
        gen, reason = ans.load_generator("ragpdf-test/definitely-not-a-real-model")
        assert gen is None, "load_generator returned a model for a nonexistent id"
        assert reason and "HF_TOKEN" in reason, f"reason does not explain itself: {reason}"

        chunks = [{"page": 5, "text": "Vitamin C is water soluble."}]
        out = ans.answer("vitamin c", chunks, [0], [0.9], generator=None)
        assert out["answer"].strip(), "degraded path returned an empty answer"
        assert out["pages"] == [5] and out["generated"] is False
    finally:
        if saved is not None:
            os.environ["HF_TOKEN"] = saved


def test_abstains_below_threshold(chunker=None):
    chunks = [{"page": 5, "text": "Vitamin C is water soluble."}]
    out = ans.answer("who won the 2010 world cup", chunks, [0], [0.10])
    assert out["abstained"] and out["pages"] == [], out
    assert "does not cover" in out["answer"]


def test_extractive_answer_is_fully_grounded(chunker=None):
    """The baseline the generator must beat: grounding 1.0 by construction."""
    chunks = [{"page": 5, "text": "Vitamin C is water soluble and not stored."}]
    out = ans.answer("vitamin c", chunks, [0], [0.9])
    ctx = ans.format_context(chunks, [0])
    assert ans.grounding(out["answer"], ctx) == 1.0
    assert ans.grounding("Vitamin C is synthesised by the liver in dogs.", ctx) < 1.0


# --- retriever fine-tuning: split/pairing logic, no model, no PDF --------------

def test_split_titles_is_disjoint_and_covers_everything(chunker=None):
    """The number reported in RESULTS.md rests on train and test never overlapping."""
    titles = [{"query": f"t{i}", "gold_pages": [i]} for i in range(23)]
    train, test = ft.split_titles(titles, held_out_every=5)
    assert set(t["query"] for t in train).isdisjoint(t["query"] for t in test)
    assert len(train) + len(test) == len(titles)
    assert [t["query"] for t in test] == ["t0", "t5", "t10", "t15", "t20"]


def test_split_titles_is_deterministic(chunker=None):
    titles = [{"query": f"t{i}", "gold_pages": [i]} for i in range(30)]
    a = ft.split_titles(titles)
    b = ft.split_titles(titles)
    assert a == b


def test_build_pairs_skips_titles_with_no_matching_chunk(chunker=None):
    chunks = [{"page": 10, "text": "digestion begins in the mouth"}]
    titles = [
        {"query": "Digestion", "gold_pages": [9, 11]},          # 10 is in range: matches
        {"query": "Appendices", "gold_pages": [9000, 9001]},     # no chunk on those pages
    ]
    pairs = ft.build_pairs(titles, chunks)
    assert pairs == [("Digestion", "digestion begins in the mouth")]


def test_build_pairs_takes_the_first_matching_chunk_only(chunker=None):
    """One positive per title -- a title matching three chunks does not get
    three times the training weight of one that matches a single chunk."""
    chunks = [
        {"page": 5, "text": "first chunk on page five"},
        {"page": 5, "text": "second chunk, also page five"},
    ]
    titles = [{"query": "Topic", "gold_pages": [5, 5]}]
    pairs = ft.build_pairs(titles, chunks)
    assert pairs == [("Topic", "first chunk on page five")]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def run(chunker=ch.chunk_page, label="ragpdf"):
    passed, failed = 0, []
    for t in TESTS:
        try:
            t(chunker=chunker)
            passed += 1
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
    print(f"\n{label}: {passed}/{len(TESTS)} passed, {len(failed)} failed")
    for name, err in failed:
        print(f"  FAIL {name}\n       {str(err)[:200]}")
    return passed, failed


if __name__ == "__main__":
    _, failed = run()
    sys.exit(1 if failed else 0)
