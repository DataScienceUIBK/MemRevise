"""What every competitor shares: the questions of a set, the conversation each one reads, and its query file.

Every competitor reads the same conversation as MemRevise: the earlier turns with their answers, each answer cut at ANSWER_BUDGET
characters, then the final question, whose answer is withheld.
"""
import json
import os

from memrevise import config

ANSWER_BUDGET = 600


def add_args(ap):
    ap.add_argument("--set", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th question, from the i-th")
    return ap


def shard_of(a):
    return tuple(int(x) for x in a.shard.split("/"))


def questions(eval_set, shard=(0, 1)):
    """The questions of the set that fall in this shard, each with 'index' = its position in the set."""
    i, n = shard
    return [dict(q, index=k) for k, q in enumerate(config.questions(eval_set)) if k % n == i]


def history(q):
    """[(question, answer)] for the earlier turns, answers cut at ANSWER_BUDGET characters."""
    return [(t["question"], str(t.get("answer") or "")[:ANSWER_BUDGET]) for t in q["history"]]


def write_queries(eval_set, name, shard, rows):
    """rows: {"episode_id": <question id>, "queries": [text, ...], ...}; writes outputs/<name>/<set>/queries.s<i>.jsonl."""
    d = config.output_dir(eval_set, name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"queries.s{shard[0]}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} queries -> {out}", flush=True)
    return out


def training_rows():
    """(question, its row of data/competitor_training/train.jsonl) for every training question, in the order of data/benchmark/train.jsonl."""
    rows = []
    for name in ("benchmark", "competitor_training"):
        with open(os.path.join(config.DATA, name, "train.jsonl"), encoding="utf-8") as f:
            rows.append([json.loads(l) for l in f])
    qs, cs = rows
    assert [q["id"] for q in qs] == [c["id"] for c in cs], "data/competitor_training/train.jsonl is not in the order of the benchmark"
    return list(zip(qs, cs))
