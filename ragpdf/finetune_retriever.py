"""Fine-tunes the dense retriever's bi-encoder on this book's own section structure.

Same model family as the base pipeline -- a sentence-transformers bi-encoder
(all-mpnet-base-v2), CPU, no API key -- fine-tuned with in-batch-negative contrastive
loss (MultipleNegativesRankingLoss) on (section title, first chunk of that section)
pairs built from Set A's table-of-contents titles (eval/titles.json).

Split discipline, so the reported numbers are not circular (the sibling rule in
CLAUDE.md #7 -- do not tune the thing you are about to measure):

    train   4 of every 5 titles, by position       -> the only rows the model sees
    test_A  the held-out 5th title                 -> in-domain, unseen at train time
    Set B   eval/questions.json, never touched here -> out-of-domain, unseen throughout

`split_titles` is the single source of truth for the train/test boundary; both this
script and eval/compare_finetuned.py import it, so the two cannot silently disagree
about which titles were trained on.

Usage:
    python -m ragpdf.finetune_retriever --epochs 4 --out models/finetuned-retriever
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE_MODEL = "all-mpnet-base-v2"


def split_titles(titles: list[dict], held_out_every: int = 5):
    """Deterministic split by position -- no shuffling, no seed to lose track of."""
    train = [t for i, t in enumerate(titles) if i % held_out_every != 0]
    test = [t for i, t in enumerate(titles) if i % held_out_every == 0]
    return train, test


def build_pairs(titles: list[dict], chunks: list[dict]) -> list[tuple[str, str]]:
    """(title, chunk text) for the first chunk whose page falls in the title's gold range.

    One positive per title keeps this a small, honest fine-tune rather than a
    silently-inflated one -- a title with no matching chunk (a TOC entry pointing at a
    page with no extracted text) is skipped, not padded with a near-miss.
    """
    pairs = []
    skipped = []
    for t in titles:
        gold = range(t["gold_pages"][0], t["gold_pages"][-1] + 1)
        match = next((c for c in chunks if c["page"] in gold), None)
        if match is None:
            skipped.append(t["query"])
            continue
        pairs.append((t["query"], match["text"]))
    if skipped:
        print(f"  {len(skipped)} title(s) with no matching chunk, skipped: {skipped}")
    return pairs


def finetune(pairs: list[tuple[str, str]], base_model: str = BASE_MODEL,
             epochs: int = 4, batch_size: int = 8):
    from sentence_transformers import InputExample, SentenceTransformer
    from sentence_transformers.sentence_transformer import losses
    from torch.utils.data import DataLoader

    if len(pairs) < batch_size:
        raise ValueError(f"{len(pairs)} training pairs is fewer than batch_size={batch_size}; "
                         f"MultipleNegativesRankingLoss needs enough in-batch negatives")

    model = SentenceTransformer(base_model, device="cpu")
    examples = [InputExample(texts=[a, b]) for a, b in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=max(1, int(0.1 * len(loader) * epochs)),
        show_progress_bar=True,
    )
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titles", default=os.path.join(ROOT, "eval", "titles.json"))
    parser.add_argument("--index", default=os.path.join(ROOT, "index"))
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "finetuned-retriever"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    from ragpdf import embed as emb

    with open(args.titles, encoding="utf-8") as f:
        titles = json.load(f)
    chunks, _ = emb.load(args.index)

    train_titles, test_titles = split_titles(titles)
    print(f"{len(train_titles)} train titles, {len(test_titles)} held out for eval "
          f"(every 5th, by position)")

    pairs = build_pairs(train_titles, chunks)
    print(f"{len(pairs)} (title, chunk) training pairs")

    model = finetune(pairs, epochs=args.epochs, batch_size=args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    print(f"saved fine-tuned retriever to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
