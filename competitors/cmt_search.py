"""Retrieval process for CMT-RAG (started by competitors.cmt_rag, in the retriever's Python environment): the passage ids of the top
TOPK passages of the frozen retriever for each sub-question, best first. The query text is scored as written (no sentence case), by the
plain MaxSim score. File protocol in --rpc: the client writes req_<k>.json (a list of texts), this process writes res_<k>.json (a list of
doc_id lists, [] for an empty text); READY when loaded; exits on STOP or when the client is gone."""
import argparse
import glob
import json
import os
import time

import torch

from memrevise import config
from memrevise.colbert import Retriever

TOPK = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--rpc", required=True)
    a = ap.parse_args()
    parent = os.getppid()
    R = Retriever(config.corpus_of(a.set))

    @torch.no_grad()
    def search(text):
        if not text.strip():
            return []
        s = R.score(text)
        return [R.ids[i] for i in torch.topk(s, k=min(TOPK, s.numel())).indices.tolist()]

    open(f"{a.rpc}/READY", "w").write("ok")
    print("[cmt-search] READY", flush=True)
    n = 0
    while not os.path.exists(f"{a.rpc}/STOP") and os.getppid() == parent:
        reqs = sorted(glob.glob(f"{a.rpc}/req_*.json"))
        if not reqs:
            time.sleep(0.05)
            continue
        for rq in reqs:
            out = [search(t) for t in json.load(open(rq))]
            rs = rq.replace("req_", "res_")
            json.dump(out, open(rs + ".tmp", "w"), ensure_ascii=False)
            os.replace(rs + ".tmp", rs)
            os.remove(rq)
            n += 1
    print(f"[cmt-search] done, {n} requests", flush=True)


if __name__ == "__main__":
    main()
