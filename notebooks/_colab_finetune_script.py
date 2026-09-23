"""Run remotely via `colab exec -f` on the pdf-rag-finetune GPU session.

Mirrors ragpdf/finetune_retriever.py's split_titles()/build_pairs() exactly -- kept in
lockstep by hand, same reason as notebooks/finetune_retriever_colab.ipynb.
"""
import json
import shutil

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader


def split_titles(titles, held_out_every=5):
    train = [t for i, t in enumerate(titles) if i % held_out_every != 0]
    test = [t for i, t in enumerate(titles) if i % held_out_every == 0]
    return train, test


def build_pairs(titles, chunks):
    pairs, skipped = [], []
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


with open("/titles.json", encoding="utf-8") as f:
    titles = json.load(f)
with open("/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

train_titles, test_titles = split_titles(titles)
print(f"{len(train_titles)} train titles, {len(test_titles)} held out (every 5th, by position)")
pairs = build_pairs(train_titles, chunks)
print(f"{len(pairs)} (title, chunk) training pairs")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

model = SentenceTransformer("all-mpnet-base-v2", device=device)
examples = [InputExample(texts=[a, b]) for a, b in pairs]
loader = DataLoader(examples, shuffle=True, batch_size=8)
loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=4,
    warmup_steps=max(1, int(0.1 * len(loader) * 4)),
    show_progress_bar=False,
)

model.save("/content/finetuned-retriever")
shutil.make_archive("/content/finetuned-retriever", "zip", "/content/finetuned-retriever")
print("saved and zipped to /content/finetuned-retriever.zip")
