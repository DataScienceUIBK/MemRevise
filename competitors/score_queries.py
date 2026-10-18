"""Rank the passages for a competitor's queries with the frozen retriever that MemRevise uses, so a competitor row and MemRevise differ
only in the query text. Writes rankings in the format of memrevise.search, scored by memrevise.evaluate.

    python -m competitors.score_queries --set test_coral --names t5qr llm_rewrite [--shard 0/1]
    python -m competitors.score_queries --set test_coral --names cmt_rag --prior after       (CMT-RAG: see --prior)

reads outputs/<name>/<set>/queries*.jsonl, one row per question: {"episode_id": <question id>, "queries": [text, ...]}, plus "prior_ids"
for CMT-RAG; writes outputs/<name>/<set>/rankings.s<i>.jsonl.
Several texts for one question (CMT-RAG's sub-questions) are scored one by one; the z-scored passage scores are averaged. TopiOCQA
questions are lower-case; as in the paper, the queries of the rows in SENTENCE_CASED are sentence-cased there, the others are scored as
written.
"""
import argparse
import glob
import json
import os

import torch

from memrevise import config
from memrevise.colbert import Retriever, cap, znorm

SENTENCE_CASED = {"original_query", "t5qr", "convsearch_r1", "llm_rewrite"}


def load_queries(eval_set, name):
    rows = {}
    for f in sorted(glob.glob(os.path.join(config.output_dir(eval_set, name), "queries*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rows[r["episode_id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="the competitors: outputs/<name>/<set>/")
    ap.add_argument("--prior", choices=["none", "after"], default="none",
                    help="after: CMT-RAG's ranking: the passages retrieved this turn (the top 5 of each sub-question) by score, then the "
                         "prior evidence recovered from the conversation's trace graph (prior_ids, in stored order), then the rest")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    corpus = config.corpus_of(a.set)
    qs = [q for i, q in enumerate(config.questions(a.set)) if i % sn == si]
    queries = {name: load_queries(a.set, name) for name in a.names}
    for name, qrows in queries.items():
        missing = [q["id"] for q in qs if not [t for t in (qrows.get(q["id"]) or {}).get("queries", []) if t and t.strip()]]
        if missing:
            raise SystemExit(f"{len(missing)} questions have no query in outputs/{name}/{a.set}/ (first: {missing[0]})")
    R = Retriever(corpus)
    for name, qrows in queries.items():
        norm = cap if corpus == "topiocqa" and name in SENTENCE_CASED else (lambda s: s)
        rank_all(R, qs, qrows, norm, a.prior, os.path.join(config.output_dir(a.set, name), f"rankings.s{si}.jsonl"))


def rank_all(R, qs, qrows, norm, prior, out):
    with open(out, "w") as fo:
        for n, q in enumerate(qs, 1):
            r = qrows[q["id"]]
            texts = [norm(t) for t in r["queries"] if t and t.strip()]
            raw = [R.score(t) for t in texts]
            scores = sum(znorm(v) for v in raw) / len(raw)
            if prior == "after":
                fresh = sorted({int(i) for v in raw for i in torch.topk(v, 5).indices.tolist()})
                bump = torch.zeros_like(scores)
                bump[fresh] = 1e4
                earlier = [R.row[d] for d in r.get("prior_ids") or [] if d in R.row and R.row[d] not in set(fresh)]
                for k, row in enumerate(earlier):
                    bump[row] = 5e3 + (len(earlier) - k)
                scores = scores + bump
            rank, stale_rank, gall = R.rank(scores, q["gold_passages"], q["stale_passages"])
            fo.write(json.dumps({"episode_id": q["id"], "gold_rank": rank, "stale_win": float(stale_rank < rank),
                                 "gold_ranks": sorted(set(gall)), "n_gold": len(gall)}) + "\n")
            if n % 200 == 0:
                print(f"  {n}/{len(qs)}", flush=True)
    print(f"{len(qs)} questions ranked -> {out}", flush=True)


if __name__ == "__main__":
    main()
