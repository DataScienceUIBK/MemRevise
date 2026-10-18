"""Retrieval process for topic repair (started by memrevise.write_memory, on the same GPU): answers 'on which page does this text live?'
with the page votes of the top 20 passages (weight 1/rank, exact page titles). File protocol in --rpc: the client writes req_<k>.json
(a list of texts), this process writes res_<k>.json ([[title, share], ...top 3] per text); READY when loaded; exits on STOP."""
import argparse
import glob
import json
import os
import time

import torch

from . import config
from .colbert import Retriever, cap, znorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--rpc", required=True)
    a = ap.parse_args()
    R = Retriever(config.corpus_of(a.set))
    assert R.titled, "topic repair needs page titles"

    def locate(text):
        if not (text or "").strip():
            return []
        v = {}
        for rk, r in enumerate(torch.topk(znorm(R.score(cap(text[:600]))), 20).indices.tolist()):
            if R.title[r]:
                v[R.title[r]] = v.get(R.title[r], 0.0) + 1.0 / (rk + 1)
        tot = sum(v.values()) or 1.0
        return [[t, round(x / tot, 3)] for t, x in sorted(v.items(), key=lambda kv: -kv[1])[:3]]

    open(f"{a.rpc}/READY", "w").write("ok")
    print("[locate] READY", flush=True)
    n, last = 0, time.time()
    while not os.path.exists(f"{a.rpc}/STOP") and time.time() - last < 1200:
        reqs = sorted(glob.glob(f"{a.rpc}/req_*.json"))
        if not reqs:
            time.sleep(0.05)
            continue
        for rq in reqs:
            out = [locate(t) for t in json.load(open(rq))]
            rs = rq.replace("req_", "res_")
            json.dump(out, open(rs + ".tmp", "w"), ensure_ascii=False)
            os.replace(rs + ".tmp", rs)
            os.remove(rq)
            n += 1
            last = time.time()
    print(f"[locate] done, {n} requests", flush=True)


if __name__ == "__main__":
    main()
