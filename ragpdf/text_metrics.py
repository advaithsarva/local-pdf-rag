"""BLEU, ROUGE-L and token-F1 -- from scratch, stdlib only, same reasoning as
this account's `telugu-english-codemix` (chrF/BLEU implemented from scratch
rather than adding `sacrebleu`/`rouge-score` for three formulas).

Why these three, and why the generated answer is scored against the
retrieved context rather than a hand-written reference: `answer.grounding`
already measures precision -- the fraction of the generated answer's content
words that appear in the context. `token_f1` here is what that number was
always half of: precision *and* recall against the same reference, the
standard SQuAD-style F1. BLEU and ROUGE-L are the two other metrics named
in the PBL rubric; both are legitimate for this task read as "the model is
summarizing the retrieved context," which is literally what generation does
here -- `ragpdf/answer.py`'s prompt is "answer using only the context items
below."

No gold reference answers exist for Set B (only gold pages + a couple of
evidence keywords -- see `eval/questions.json`), so a hand-written-reference
BLEU/ROUGE run was not possible without inventing text that was never in
the book. Scoring against the retrieved context avoids that: the context is
a verbatim slice of the correct pages by `chunk.py`'s own invariant, so it
is a legitimate reference, not a stand-in for one that should have existed.
"""

import math
import re
from collections import Counter

_STOPWORDS = set("""a an the and or but if then than that this these those of in on at to for from with
by as is are was were be been being it its it's they them their there here what which who whom how why
when where can could may might will would shall should do does did not no yes you your we our i me my
also such about into over under more most some any each other same so very just only own too s t""".split())


def _tokens(text):
    return text.lower().split()


def _content_words(text):
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS]


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(candidate: str, reference: str, max_n: int = 4) -> float:
    """Standard sentence BLEU: geometric mean of 1..max_n-gram precision,
    times a brevity penalty. 0.0 if any n-gram order has zero overlap
    (the standard behaviour -- BLEU is unforgiving of a missing order by
    design, not a bug in this implementation)."""
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(cand, n)
        if not cand_ng:
            precisions.append(0.0)
            continue
        ref_ng = _ngrams(ref, n)
        overlap = sum(min(count, ref_ng.get(g, 0)) for g, count in cand_ng.items())
        precisions.append(overlap / sum(cand_ng.values()))

    if min(precisions) == 0.0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * geo_mean


def _lcs_length(a, b):
    """Standard O(len(a)*len(b)) longest-common-subsequence DP."""
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F-measure: LCS length over candidate and reference tokens,
    turned into precision/recall/F1 the same way any retrieval metric is."""
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(candidate: str, reference: str) -> dict:
    """Content-word set precision/recall/F1 -- the same word filter
    `answer.grounding` uses, extended to recall. Returns all three because
    a single F1 number hides which direction a low score is failing in:
    low precision means invented content, low recall means the context was
    available but the answer left most of it out."""
    cand_words = set(_content_words(candidate))
    ref_words = set(_content_words(reference))
    if not cand_words or not ref_words:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = cand_words & ref_words
    precision = len(overlap) / len(cand_words)
    recall = len(overlap) / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


if __name__ == "__main__":
    # Self-check with hand-computable examples -- run this file directly.
    # Not a replacement for tests/test_ragpdf.py, but these three formulas
    # are exactly the kind of thing worth checking against arithmetic done
    # by hand before trusting them on real data.

    # Identical strings: everything is 1.0.
    assert bleu_score("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert token_f1("the cat sat on the mat", "the cat sat on the mat")["f1"] == 1.0

    # No shared words at all: everything is 0.0.
    assert bleu_score("apple banana cherry", "xylophone zebra yak") == 0.0
    assert rouge_l("apple banana cherry", "xylophone zebra yak") == 0.0
    assert token_f1("apple banana cherry", "xylophone zebra yak")["f1"] == 0.0

    # Hand-computed ROUGE-L: candidate "the cat sat", reference "the big cat
    # sat down" -- LCS is "the cat sat" (length 3). precision = 3/3 = 1.0,
    # recall = 3/5 = 0.6, F1 = 2*1.0*0.6/(1.0+0.6) = 0.75.
    r = rouge_l("the cat sat", "the big cat sat down")
    assert abs(r - 0.75) < 1e-9, r

    # Hand-computed token_f1: candidate content words {cat, sat, mat} (the/on
    # are filtered, one by stopword, one by the 3-letter minimum). Reference
    # content words {cat, dog, sat, mat, rug} ("and"/"a"/"on" filtered the
    # same way). Overlap is all 3 candidate words: precision = 3/3 = 1.0,
    # recall = 3/5 = 0.6, F1 = 2*1.0*0.6/1.6 = 0.75.
    f = token_f1("the cat sat on the mat", "a cat and a dog sat on a mat and rug")
    assert f["precision"] == 1.0 and f["recall"] == 0.6, f
    assert abs(f["f1"] - 0.75) < 1e-3, f

    # BLEU brevity penalty: a candidate that is a perfect-precision prefix of
    # the reference (every 1..4-gram it contains also appears in the
    # reference) should still be penalised below 1.0 for being shorter.
    # Candidate needs at least 4 tokens or the 4-gram precision collapses to
    # 0/0 for an unrelated reason (no 4-grams to score at all).
    short_bleu = bleu_score("the cat sat quietly", "the cat sat quietly on the warm mat today")
    assert 0.0 < short_bleu < 1.0, short_bleu

    print("all text_metrics self-checks passed")
