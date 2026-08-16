"""The original chunker, transcribed verbatim from notebooks/original/rag-pipeline.ipynb
(cells 5, 9, 12), so the test suite can be run against it.

Requires spaCy, which the rebuilt package does not. This file is dev-only --
it exists to prove the tests are not decorative.
"""

import re

from spacy.lang.en import English

_nlp = English()
_nlp.add_pipe("sentencizer")


def chunk_page(text, page, sentences_per_chunk=10, min_chars=120):
    """Notebook cells 5, 9 and 12, with the same signature as ragpdf.chunk.chunk_page.

    Cell 5:  text = text.replace("\\n", " ").strip()
    Cell 9:  item["sentences"] = [str(s) for s in nlp(item["text"]).sents]
    Cell 12: joined = "".join(sentence_chunk).replace("  ", " ").strip()
             joined = re.sub(r'\\.([A-Z])', r'. \\1', joined)
    """
    text = text.replace("\n", " ").strip()                      # cell 5
    sentences = [str(s) for s in _nlp(text).sents]              # cell 9  <-- drops trailing ws
    out = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        joined = "".join(group).replace("  ", " ").strip()      # cell 12 <-- no separator
        joined = re.sub(r"\.([A-Z])", r". \1", joined)          # cell 12 <-- the band-aid
        if len(joined) / 4 <= min_chars / 4:
            continue
        # The original carried no offsets at all; there was nothing to slice back
        # to, which is precisely why the defect was invisible.
        out.append({"page": page, "start": 0, "end": 0, "text": joined})
    return out
