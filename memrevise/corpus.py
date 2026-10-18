"""The passage collections, as published on Hugging Face (config `corpus` of mohammed933/ConvRevise): corpus/<corpus>.jsonl, one passage
per line with doc_id, title, section and text, in the order of the encoded vectors (row r of encoded/<corpus>/ is line r).
"""
import json
import os

from . import config


def path(corpus):
    return os.path.join(config.CORPORA, f"{corpus}.jsonl")


def passages(corpus):
    with open(path(corpus), encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def passage_text(p, corpus):
    """The text the retriever encodes: the page title and section in front of the passage on TopiOCQA, the page title on OR-QuAC, the
    passage alone on CORAL (whose passages have no title)."""
    if corpus == "topiocqa" and p["title"]:
        return f"{p['title']} [SEP] {p['section']} | {p['text']}"
    if corpus == "orquac" and p["title"]:
        return f"{p['title']} [SEP] {p['text']}"
    return p["text"]


def ids_and_titles(corpus):
    """doc_id and page title of every passage, in row order."""
    ids, titles = [], []
    for p in passages(corpus):
        ids.append(p["doc_id"])
        titles.append(p["title"])
    return ids, titles
