"""Write the memory of every conversation of an evaluation set, turn by turn, with the trained writer (one model call per turn). Earlier
turns show their answer; the final turn's answer is withheld, and at the final turn the writer also writes the search query.

On collections with page titles (TopiOCQA, OR-QuAC) every history node after the first is checked by topic repair: a retrieval process
says on which page the turn's question and answer live and on which page the node's facts live; when both are the same page, that page is
a name the previous node introduced (or an earlier topic) and it is not the node's topic nor named in the turn's answer, the page becomes
the node's first topic. The retrieval process runs in the retriever's Python environment (MEMREVISE_RETRIEVER_PYTHON).

    python -m memrevise.write_memory --set test_coral [--shard 0/1] [--writer models/writer]
writes outputs/writer/<set>/memories.s<i>.jsonl (one memory per conversation) and queries.s<i>.jsonl (the final query).
"""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

from . import config
from .prompt import messages, query_of
from .schema import ConvGraph, Op, hist_answer, parse_extraction

MAX_PROMPT, MAX_NEW, CHUNK = 16384, 800, 16
core = lambda x: re.sub(r"\s*\(.*?\)", "", (x or "")).strip().lower()


class Locator:
    """Client of memrevise.locate_server: texts -> [[title, share], ...top 3] per text, over a file-based request queue."""

    def __init__(self, eval_set, rpc_dir):
        self.rpc, self.k = rpc_dir, 0
        shutil.rmtree(rpc_dir, ignore_errors=True)
        os.makedirs(rpc_dir)
        py = os.environ.get("MEMREVISE_RETRIEVER_PYTHON", sys.executable)
        self.log = open(os.path.join(rpc_dir, "server.log"), "w")
        self.proc = subprocess.Popen([py, "-m", "memrevise.locate_server", "--set", eval_set, "--rpc", rpc_dir], cwd=config.ROOT,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not os.path.exists(f"{rpc_dir}/READY"):
            if self.proc.poll() is not None or time.time() - t0 > 900:
                raise SystemExit(f"[memory] the retrieval process did not start; see {rpc_dir}/server.log")
            time.sleep(2)

    def __call__(self, texts):
        self.k += 1
        rq = f"{self.rpc}/req_{self.k:06d}.json"
        rs = rq.replace("req_", "res_")
        json.dump(texts, open(rq + ".tmp", "w"), ensure_ascii=False)
        os.replace(rq + ".tmp", rq)
        t0 = time.time()
        while not os.path.exists(rs):
            if time.time() - t0 > 600:
                raise SystemExit("[memory] the retrieval process did not answer in 10 minutes")
            time.sleep(0.05)
        out = json.load(open(rs))
        os.remove(rs)
        return out

    def close(self):
        open(f"{self.rpc}/STOP", "w").write("stop")
        self.proc.wait()


def repair_candidate(node, prev, seen, topics, located, answer):
    """The page name to put first, or '' (the conditions are in the module docstring)."""
    if prev is None or not located:
        return ""
    P = located[0][0]
    pc = core(P)
    if not pc or pc in {core(a) for a in node.asked if a}:
        return ""
    new_names = {e: t for e, t in (prev.types or {}).items() if e not in seen and e not in prev.asked}
    introduced = pc in {core(e) for e in new_names} or pc in " ".join(f.text for f in prev.facts).lower()
    earlier = next((t for t in topics if core(t) == pc), "")
    if not (introduced or earlier):
        return ""
    if pc in (answer or "").lower():
        return ""
    return next((e for e in new_names if core(e) == pc), "") or earlier or P


def repair(node, prev, seen, topics, located_qa, located_facts, answer):
    new = repair_candidate(node, prev, seen, topics, located_qa, answer)
    if not new or not located_facts or located_qa[0][0] != located_facts[0][0]:
        return False
    new_names = {e: t for e, t in (prev.types or {}).items() if e not in seen and e not in prev.asked}
    node.asked = [new] + [a for a in node.asked if a != new]
    node.ops = [Op(op="REACTIVATE" if new in topics else "ADD", entity=new, detail="")] + [o for o in node.ops if o.entity != new]
    if new_names.get(new):
        node.types = dict(node.types or {}, **{new: new_names[new]})
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="evaluation set, e.g. test_topiocqa, test_coral, dev_orquac_value_revisions")
    ap.add_argument("--shard", default="0/1", help="i/n: this process writes every n-th conversation starting at i")
    ap.add_argument("--writer", default=config.WRITER, help="the writer's LoRA adapter")
    ap.add_argument("--base", default=config.BASE_MODEL, help="the writer's base model")
    ap.add_argument("--out", default="", help="output folder (default outputs/writer/<set>)")
    a = ap.parse_args()
    if os.environ.get("CR_NO_CUDNN_SDPA", "1") == "1":
        torch.backends.cuda.enable_cudnn_sdp(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    si, sn = (int(x) for x in a.shard.split("/"))
    W = a.out or config.output_dir(a.set)
    os.makedirs(W, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.writer, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                                          attn_implementation="sdpa"), a.writer).to("cuda:0").eval()
    stats = collections.Counter()

    @torch.no_grad()
    def generate(msgs):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT).to("cuda:0")
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        return [tok.decode(o[L:], skip_special_tokens=True) for o in out]

    eps = [{"episode_id": q["id"], "turns": config.turns(q)} for i, q in enumerate(config.questions(a.set)) if i % sn == si]
    OUT, QOUT = f"{W}/memories.s{si}.jsonl", f"{W}/queries.s{si}.jsonl"
    done = {json.loads(l)["conv_id"] for l in open(OUT)} if os.path.exists(OUT) else set()
    eps = [e for e in eps if e["episode_id"] not in done]
    eps.sort(key=lambda e: (len(e["turns"]), sum(len(t["question"]) + len(str(t.get("answer") or "")) for t in e["turns"])))
    titled = config.CORPUS[config.corpus_of(a.set)]["titled"]
    locate = Locator(a.set, f"{W}/rpc_{si}") if titled and eps else None
    print(f"[memory] {a.set} shard {a.shard}: {len(eps)} conversations to write ({len(done)} done), topic repair {'on' if locate else 'off'}", flush=True)
    with open(OUT, "a") as fo:
        for c in range(0, len(eps), CHUNK):
            chunk = eps[c:c + CHUNK]
            graphs = [ConvGraph(conv_id=e["episode_id"]) for e in chunk]
            queries = {}
            for j in range(max(len(e["turns"]) for e in chunk)):
                live = [i for i, e in enumerate(chunk) if j < len(e["turns"])]
                msgs = []
                for i in live:
                    ts = chunk[i]["turns"]
                    fin = j == len(ts) - 1
                    msgs.append(messages(graphs[i], j + 1, ts[j - 1]["question"] if j else "", ts[j]["question"], "" if fin else hist_answer(ts[j]), fin))
                new_nodes = {}
                for i, raw in zip(live, generate(msgs)):
                    node = parse_extraction(raw, j + 1)
                    new_nodes[i] = node
                    if j == len(chunk[i]["turns"]) - 1:
                        queries[i] = query_of(raw, node, chunk[i]["turns"][j]["question"])
                if locate:
                    hist = [i for i in live if 0 < j < len(chunk[i]["turns"]) - 1]
                    if hist:
                        located_qa = locate([f"{chunk[i]['turns'][j]['question']} {hist_answer(chunk[i]['turns'][j])}".strip() for i in hist])
                        ctx, need = {}, []
                        for k, i in enumerate(hist):
                            prev = graphs[i].turns.get(j)
                            seen = {e for tn, nd in graphs[i].turns.items() if tn < j for e in (nd.types or {})}
                            topics = list(dict.fromkeys(a_ for tn in sorted(graphs[i].turns) for a_ in graphs[i].turns[tn].asked if a_))
                            ans = hist_answer(chunk[i]["turns"][j])
                            ctx[i] = (prev, seen, topics, located_qa[k], ans)
                            if repair_candidate(new_nodes[i], prev, seen, topics, located_qa[k], ans):
                                need.append(i)
                        located_facts = locate([" ".join(f.text for f in new_nodes[i].facts).strip() for i in need]) if need else []
                        for k, i in enumerate(need):
                            prev, seen, topics, lq, ans = ctx[i]
                            stats["topic repairs"] += int(repair(new_nodes[i], prev, seen, topics, lq, located_facts[k], ans))
                for i in live:
                    graphs[i].add_turn(new_nodes[i])
            with open(QOUT, "a") as fq:
                for i, g in enumerate(graphs):
                    fq.write(json.dumps({"episode_id": g.conv_id, "query": queries.get(i, "")}, ensure_ascii=False) + "\n")
            for g in graphs:
                fo.write(json.dumps(g.to_record(), ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[memory] {c + len(chunk)}/{len(eps)} conversations {dict(stats)}", flush=True)
    if locate:
        locate.close()
    print("[memory] DONE", flush=True)


if __name__ == "__main__":
    main()
