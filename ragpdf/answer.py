"""Retrieved chunks -> a grounded answer with page citations.

Three rules this module exists to enforce:

1. **It abstains.** If the best chunk scores below `MIN_SCORE`, the question is
   not in the book and the answer says so. It does not summarise the five
   least-irrelevant pages on nutrition.
2. **It degrades.** No network, no local weights, no credential -- the extractive
   answer (the retrieved passages, verbatim, with their page numbers) is still
   returned, with `generated=False` and a stated reason. A missing credential is
   never a crash and never a silent empty result.
3. **It cites.** Every answer carries the pages it drew on, and because of the
   chunk.py invariant those citations are checkable against the PDF.

The default generator is **ungated on purpose**: the original notebook used
`google/gemma-2b-it`, which needs an accepted licence and an `HF_TOKEN`, so its
output could never be measured in CI or by a reader. An unmeasurable feature is
a decoration. Gemma is still available via `--model google/gemma-2b-it`, which
picks up `HF_TOKEN` if it is set -- an env var with no fallback default.

The generator is measured against the extractive baseline it sits on top of;
both numbers are in RESULTS.md.
"""

import os
import re
import textwrap

# Calibrated in eval/ -- see RESULTS.md "Choosing the abstention threshold".
MIN_SCORE = 0.35

# Ungated and CPU-sized, so the grounding number below is reproducible by
# anyone. `--model google/gemma-2b-it` restores what the original notebook used.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_PROMPT = """Answer the query using only the context items below. If the context
does not contain the answer, say "The textbook does not cover this." Do not add
facts that are not in the context. Answer in three sentences or fewer.

Context items:
{context}

Query: {query}
Answer:"""

_STOPWORDS = set("""a an the and or but if then than that this these those of in on at to for from with
by as is are was were be been being it its it's they them their there here what which who whom how why
when where can could may might will would shall should do does did not no yes you your we our i me my
also such about into over under more most some any each other same so very just only own too s t""".split())


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


def grounding(answer: str, context: str) -> float:
    """Fraction of the answer's content words that appear in the context.

    1.0 for any extractive answer by definition. Below 1.0 for a generated one
    is the measurable price of fluency -- report it, do not hide it.
    """
    words = _content_words(answer)
    if not words:
        return 1.0
    return len(words & _content_words(context)) / len(words)


def format_context(chunks: list[dict], indices: list[int]) -> str:
    return "\n\n".join(
        f"[page {chunks[i]['page']}] {' '.join(chunks[i]['text'].split())}" for i in indices
    )


def extractive(chunks: list[dict], indices: list[int], scores: list[float]) -> dict:
    """The boring baseline: hand back what was retrieved. Grounding is 1.0 by
    construction, because the text is a verbatim slice of the cited page."""
    body = "\n\n".join(
        textwrap.fill(" ".join(chunks[i]["text"].split()), 88) for i in indices
    )
    return {
        "answer": body,
        "pages": [chunks[i]["page"] for i in indices],
        "scores": [round(s, 4) for s in scores],
        "generated": False,
        "abstained": False,
        "reason": "extractive baseline",
    }


def abstain(top_score: float) -> dict:
    return {
        "answer": "The textbook does not cover this.",
        "pages": [],
        "scores": [],
        "generated": False,
        "abstained": True,
        "reason": f"best retrieval score {top_score:.3f} < MIN_SCORE {MIN_SCORE}",
    }


def load_generator(model_id: str = MODEL_ID):
    """(tokenizer, model) or (None, reason). Never raises, never exits.

    `HF_TOKEN` is read but not required -- the default model is ungated. Gated
    models get the token if it is present and a stated reason if it is not.
    """
    token = os.environ.get("HF_TOKEN")  # no fallback default, by design
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return None, f"transformers/torch unavailable: {e}"
    try:
        # No device_map, no dtype: CPU float32 is the default, and device_map
        # would drag in `accelerate` for nothing.
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
        return (tok, model), None
    except Exception as e:  # network down, rate limited, licence not accepted
        hint = "" if token else " (HF_TOKEN is unset; gated models need one)"
        return None, f"could not load {model_id}: {type(e).__name__}: {e}{hint}"


def generate(generator, query: str, context: str, max_new_tokens: int = 256,
             temperature: float = 0.7) -> str:
    tok, model = generator
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": _PROMPT.format(context=context, query=query)}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         temperature=temperature, do_sample=temperature > 0)
    # Decode only the new tokens. Decoding the whole sequence and splitting on
    # "Answer:" leaves the chat template's role marker in the text, which then
    # counts as ungrounded content and quietly depresses the grounding score.
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def answer(query: str, chunks: list[dict], indices: list[int], scores: list[float],
           generator=None, min_score: float = MIN_SCORE, temperature: float = 0.7) -> dict:
    """The whole decision, in one place: abstain, generate, or fall back.

    `temperature=0.0` selects greedy decoding. eval/run_generation.py uses it so
    the published grounding figure is deterministic rather than one sample.
    """
    if not indices or scores[0] < min_score:
        return abstain(scores[0] if scores else 0.0)

    base = extractive(chunks, indices, scores)
    if generator is None:
        return base

    context = format_context(chunks, indices)
    text = generate(generator, query, context, temperature=temperature)
    return {
        "answer": text,
        "pages": base["pages"],
        "scores": base["scores"],
        "generated": True,
        "abstained": False,
        "reason": "generated from retrieved context",
        "grounding": round(grounding(text, context), 3),
    }
