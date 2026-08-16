"""PDF -> chunks.

THE INVARIANT, which the whole project turns on:

    A chunk's `text` is a verbatim slice of the page it cites.
    chunk["text"] == page_text(pdf, chunk["page"])[chunk["start"]:chunk["end"]]

Every chunk is produced by slicing the page string, never by re-joining pieces
of it. The original pipeline rebuilt chunks with `"".join(str(sent) for sent in
sents)`; spaCy's `Span.text` drops the trailing whitespace, so 21% of shipped
chunks were not quotes of the source (see RESULTS.md). Slicing cannot do that.

Enforced by tests/test_ragpdf.py::test_chunks_are_verbatim_slices.
"""

import re

import fitz  # pymupdf

# Page numbering: the PDF's page 42 is the book's page 0. A citation is only
# useful if it matches the number printed on the page the reader opens.
PAGE_OFFSET = 41

# ponytail: naive sentence split, no abbreviation list. A wrong split at "Dr."
# only moves a chunk boundary -- it cannot corrupt the text, because the text
# is a slice. Add an abbreviation list only if boundary quality is measured
# and found wanting.
#
# The capture group is load-bearing: a sentence ends AFTER its closing quote or
# bracket, not before it. Ending the span at the match start instead dropped the
# ')' from '(Salt is 60 percent chloride.)' on the 9 chunks (of 1,715) whose
# final boundary landed on one. Still a verbatim slice either way -- just a
# slice one character short of the sentence.
_BOUNDARY = re.compile(r'(?<=[.!?])["\'’”)\]]*(\s)')


def page_texts(pdf_path: str) -> list[str]:
    """Raw text of every page, in order. Index i is book page i - PAGE_OFFSET."""
    with fitz.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of each sentence in `text`. Whitespace between
    sentences belongs to neither, so slicing across a group keeps it."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.start(1)          # the whitespace, so closers stay with the sentence
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def chunk_page(text: str, page: int, sentences_per_chunk: int, min_chars: int) -> list[dict]:
    """Group a page's sentences into chunks. Text is always `text[start:end]`."""
    spans = sentence_spans(text)
    out = []
    for i in range(0, len(spans), sentences_per_chunk):
        group = spans[i:i + sentences_per_chunk]
        start, end = group[0][0], group[-1][1]
        body = text[start:end]
        if len(body.strip()) < min_chars:
            continue
        out.append({"page": page, "start": start, "end": end, "text": body})
    return out


def build_chunks(pdf_path: str, sentences_per_chunk: int = 10, min_chars: int = 120) -> list[dict]:
    """Every chunk in the document. Raises rather than silently producing none."""
    if sentences_per_chunk < 1:
        raise ValueError(f"sentences_per_chunk must be >= 1, got {sentences_per_chunk}")
    chunks = []
    for i, text in enumerate(page_texts(pdf_path)):
        chunks.extend(chunk_page(text, i - PAGE_OFFSET, sentences_per_chunk, min_chars))
    if not chunks:
        raise ValueError(f"{pdf_path} produced no chunks -- is it a scanned PDF with no text layer?")
    return chunks


def verify_verbatim(chunks: list[dict], pdf_path: str) -> tuple[int, int]:
    """(verbatim, total) -- the invariant, checked against the PDF itself."""
    pages = page_texts(pdf_path)
    ok = 0
    for c in chunks:
        page = pages[c["page"] + PAGE_OFFSET]
        if page[c["start"]:c["end"]] == c["text"]:
            ok += 1
    return ok, len(chunks)
