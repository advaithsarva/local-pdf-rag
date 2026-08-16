"""Set A: section titles from the PDF's own table of contents.

Free ground truth -- the document supplies both the query and the gold pages, so
nobody authored this set and nobody can tune it. It measures exact-term lookup,
which is the case keyword search should win. That is the point: it is the
baseline's home ground.

A section's gold pages are its full page range (its TOC page up to the next
section's), not just its first page -- landing on page 3 of a 6-page section is
a correct retrieval, and scoring it wrong would flatter nobody honestly.

Titles that appear more than once in the TOC ("Introduction", "Summary") are
dropped: their ground truth is genuinely ambiguous.
"""

import json
import os
import re
import sys
from collections import Counter

import fitz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragpdf.chunk import PAGE_OFFSET  # noqa: E402

BOILERPLATE = "University of Hawai"
NAVIGATIONAL = re.compile(r"^(Chapter \d+\.|Contents|Preface|Human Nutrition|About the Contributors"
                          r"|Acknowledgements|Appendix|Index|Glossary)", re.I)
MAX_SECTION_PAGES = 30  # a longer "section" is a chapter wrapper, not a topic


def main(pdf="human-nutrition-text.pdf", out="eval/titles.json"):
    doc = fitz.open(pdf)
    toc = [(re.sub(r"\s+", " ", t).strip(), p - 1) for _, t, p in doc.get_toc()]
    counts = Counter(t.lower() for t, _ in toc)

    rows = []
    for i, (title, page0) in enumerate(toc):
        if BOILERPLATE in title or len(title) < 8 or NAVIGATIONAL.match(title):
            continue
        if counts[title.lower()] > 1:
            continue  # ambiguous ground truth
        end = next((p for _, p in toc[i + 1:] if p > page0), doc.page_count)
        if end - page0 > MAX_SECTION_PAGES:
            continue
        rows.append({
            "query": title,
            "gold_pages": list(range(page0 - PAGE_OFFSET, end - PAGE_OFFSET)),
        })

    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    spans = [len(r["gold_pages"]) for r in rows]
    print(f"{len(rows)} title queries -> {out} "
          f"(gold span: min {min(spans)}, median {sorted(spans)[len(spans)//2]}, max {max(spans)})")


if __name__ == "__main__":
    main()
