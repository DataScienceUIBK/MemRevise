"""The retrieval reward of ConvSearch-R1-FT: the API of the authors' retrieval server (src/retrieval/server.py of their repository) over
the frozen retriever of MemRevise and the ConvRevise collections.

    python -m competitors.convsearch_r1_ft.reward_server [--port 8000] [--gpus topiocqa=0,1 coral=2,3]

POST /retrieve {"queries": [text, ...], "query_ids": [id, ...], "topk": 100} -> {"result": [rank, ...]}: for each query, the 1-based rank
of the best gold passage of its training question among all the passages of that question's collection, or -1 when the rank is beyond
topk, the text is blank or the id is not a training question (their rewrite_r1.py turns the rank into the reward). The text is scored as
it arrives, not sentence-cased. GET /health -> {"ok": true, "n": <queries served>}.
Each GPU in --gpus holds one copy of one collection; the queries of a request are split over the copies of their collection.
Environment: the retriever's (transformers 4.51).
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from memrevise import config
from memrevise.colbert import Retriever


def training_gold():
    """{question id: (collection, gold passage ids)} for data/benchmark/train.jsonl."""
    gold = {}
    with open(os.path.join(config.DATA, "benchmark", "train.jsonl"), encoding="utf-8") as f:
        for q in map(json.loads, f):
            gold[q["id"]] = (config.corpus_of(q["eval_set"]), q["gold_passages"])
    return gold


class Replica:
    def __init__(self, corpus, gpu):
        self.R, self.lock = Retriever(corpus, device=f"cuda:{gpu}"), threading.Lock()

    @torch.no_grad()
    def ranks(self, texts, golds, topk):
        out = []
        with self.lock:
            for text, gold in zip(texts, golds):
                if not text.strip():
                    out.append(-1)
                    continue
                rank = self.R.rank(self.R.score(text), gold)[0]          # 10**9 when no gold passage is in the collection
                out.append(rank + 1 if rank + 1 <= topk else -1)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpus", nargs="+", default=["topiocqa=0,1", "coral=2,3"], help="<collection>=<gpu>,<gpu>...")
    a = ap.parse_args()
    gold = training_gold()
    replicas = {c: [Replica(c, int(g)) for g in gs.split(",")] for c, gs in (x.split("=", 1) for x in a.gpus)}
    served = {"n": 0}

    def retrieve(req):
        texts, ids, topk = req["queries"], req["query_ids"], int(req.get("topk", 100))
        res = [-1] * len(texts)
        jobs = []
        for c, reps in replicas.items():
            idx = [i for i, qid in enumerate(ids) if qid in gold and gold[qid][0] == c]
            for j, rep in enumerate(reps):
                part = idx[j::len(reps)]
                if part:
                    jobs.append((rep, part))
        with ThreadPoolExecutor(max(1, len(jobs))) as ex:
            done = ex.map(lambda job: (job[1], job[0].ranks([texts[i] for i in job[1]], [gold[ids[i]][1] for i in job[1]], topk)), jobs)
            for part, ranks in done:
                for i, r in zip(part, ranks):
                    res[i] = r
        served["n"] += len(res)
        return {"result": res}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.rstrip("/") != "/retrieve":
                return self.send_error(404)
            self._send(retrieve(json.loads(self.rfile.read(int(self.headers["Content-Length"])))))

        def do_GET(self):
            if self.path.rstrip("/") != "/health":
                return self.send_error(404)
            self._send({"ok": True, **served})

        def log_message(self, *args):
            pass

    print(f"[reward] {', '.join(f'{c} x{len(r)}' for c, r in replicas.items())} on port {a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
