"""Paths and the per-corpus settings of the system. Every other module reads its locations from here.

The repository root is the folder that holds this package; MEMREVISE_ROOT overrides it. In git:
    data/benchmark/{train,dev,test}.jsonl   the ConvRevise questions (the same files as on Hugging Face); `eval_set` names each set
    data/writer_training/    the writer's training rows
Large inputs that are not in git live under it:
    corpus/<corpus>.jsonl    the passage collections (from Hugging Face: doc_id, title, section, text; see memrevise.corpus)
    encoded/<corpus>/        the collections encoded by memrevise.encode_corpus (shard_000_of_001/)
    models/base/, models/writer/   the writer's base model and its LoRA adapter (from Hugging Face)
    outputs/                 memories, queries and rankings written by the commands
"""
import json
import os

ROOT = os.environ.get("MEMREVISE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CORPORA = os.path.join(ROOT, "corpus")
ENCODED = os.environ.get("MEMREVISE_ENCODED") or os.path.join(ROOT, "encoded")
OUTPUTS = os.path.join(ROOT, "outputs")
WRITER = os.environ.get("MEMREVISE_WRITER") or os.path.join(ROOT, "models", "writer")
BASE_MODEL = os.environ.get("MEMREVISE_BASE_MODEL") or os.path.join(ROOT, "models", "base")   # Ministral-3-3B-Instruct-2512, text weights
RETRIEVER = "lightonai/Reason-ModernColBERT"

TEST_SETS = ["test_topiocqa", "test_coral", "test_orquac"]
DEV_SETS = ["dev_topiocqa", "dev_coral_natural", "dev_coral_corrections_returns", "dev_coral_returns", "dev_orquac_corrections_returns",
            "dev_orquac_value_revisions"]
TRAIN_SETS = ["train_topiocqa", "train_coral"]

# One entry per collection. titled: passages carry page titles, so the search uses the title prior, the facts vote and the exclusion
# step, and the memory writer runs topic repair. facts_vote: how the facts of the earlier turns about the topic choose a page.
CORPUS = {
    "topiocqa": dict(titled=True, facts_vote="all_nodes_top100"),
    "orquac": dict(titled=True, facts_vote="last3_nodes"),
    "coral": dict(titled=False, facts_vote=None),
}


def corpus_of(eval_set):
    """The collection an evaluation set searches: test_coral -> coral, dev_orquac_value_revisions -> orquac."""
    return eval_set.split("_")[1]


def questions(eval_set):
    """The questions of an evaluation set, in file order, from data/benchmark/<split>.jsonl."""
    with open(os.path.join(DATA, "benchmark", f"{eval_set.split('_')[0]}.jsonl"), encoding="utf-8") as f:
        qs = [q for q in map(json.loads, f) if q["eval_set"] == eval_set]
    if not qs:
        raise SystemExit(f"no questions for '{eval_set}'; the sets are {', '.join(TEST_SETS + DEV_SETS + TRAIN_SETS)}")
    return qs


def turns(q):
    """The conversation the writer reads: the earlier turns, then the question, whose answer is withheld."""
    return q["history"] + [{"turn": len(q["history"]) + 1, "question": q["question"], "answer": ""}]


def encoded_dir(corpus):
    return os.path.join(ENCODED, corpus, "shard_000_of_001")


def output_dir(eval_set, name="writer"):
    return os.path.join(OUTPUTS, name, eval_set)
