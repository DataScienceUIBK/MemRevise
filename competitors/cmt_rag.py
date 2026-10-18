"""CMT-RAG (Zhou et al., 2026), re-implemented with a prompted LLM: every turn of the conversation is decomposed into sub-question
traces, the traces form a session DAG, and the final turn's resolved sub-questions are the queries.

    python -m competitors.cmt_rag --set test_coral [--shard 0/1] [--model Qwen/Qwen2.5-32B-Instruct] [--tp 2]
                                                        -> outputs/cmt_rag/test_coral/queries.s0.jsonl
    then: python -m competitors.score_queries --set test_coral --names cmt_rag --prior after

Per turn, in order: the trace generator (the LLM, greedy) reads the trace namespace of the dialogue (the earlier resolved sub-questions,
never their answers), the last three user questions and the current one, and writes 1 to 4 drafts "[SubQ] ... [KW] ... [DEP] [k, ...]";
[A_k] in a sub-question is replaced by the answer of trace k. Each resolved sub-question retrieves its top 5 passages (the fresh
evidence); its prior evidence is the fresh evidence of its dependency traces and of at most one trace found by keyword lookup (Ochiai
coefficient of the keyword sets > 0.05, traces older than the preceding 10 turns, ties to the smaller id). The reader (the same LLM)
answers the sub-question from the fresh then the prior passages, deduplicated in first-occurrence order, and the trace joins the DAG.
The model only reads the questions of the conversation: answers enter through the reader.
Output per question: queries = the final turn's resolved sub-questions, prior_ids = their prior evidence, in order.

The fresh retrieval runs in the retriever's Python environment (MEMREVISE_RETRIEVER_PYTHON), as competitors.cmt_search, on the GPU
after vLLM's (--search-gpu; its log is outputs/cmt_rag/<set>/rpc_<i>/server.log); vLLM (writer environment) uses the first --tp GPUs.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

from memrevise import config
from memrevise.corpus import passage_text, passages

from . import common

MODEL = "Qwen/Qwen2.5-32B-Instruct"
TP = 2
RECENT, LOOKUP_WINDOW, OCHIAI_MIN, MAX_SUBQ = 3, 10, 0.05, 4
MAX_PROMPT = 7000                                        # prompt tokens kept (model window 8192)
GEN_NEW, READ_NEW = 320, 32
GEN_SYS = ("You are the trace generator of a conversational retrieval system. For the current user query, write retrieval-oriented "
           "sub-questions as trace drafts. Output ONLY lines of the form\n[SubQ] <sub-question> [KW] <space-separated keywords> [DEP] [<ids>]\n"
           "Rules: write one line per sub-question (1 to 4 lines) in the order they must be answered; a sub-question may refer to the answer of an "
           "earlier trace with the placeholder [A_k] (k = the trace id) and must then list k in [DEP]; [DEP] [] when there is no dependency; "
           "keywords are a few normalised content words of the sub-question; resolve pronouns and omissions using the traces, never ask the "
           "user. Do not write anything else.")
READ_SYS = "Answer the question using only the provided paragraphs. Reply with only the answer span, yes, or no. Do not explain."
LINE = re.compile(r"\[SubQ\]\s*(.*?)\s*\[KW\]\s*(.*?)\s*\[DEP\]\s*\[(.*?)\]", re.S)


class Search:
    """Client of competitors.cmt_search: texts -> [[doc_id, ...top 5] per text], over a file-based request queue."""

    def __init__(self, eval_set, rpc_dir, gpu):
        self.rpc, self.k = rpc_dir, 0
        shutil.rmtree(rpc_dir, ignore_errors=True)
        os.makedirs(rpc_dir)
        py = os.environ.get("MEMREVISE_RETRIEVER_PYTHON", sys.executable)
        self.log = open(os.path.join(rpc_dir, "server.log"), "w")
        self.proc = subprocess.Popen([py, "-m", "competitors.cmt_search", "--set", eval_set, "--rpc", rpc_dir], cwd=config.ROOT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=self.log, stderr=subprocess.STDOUT)

    def wait_ready(self):
        t0 = time.time()
        while not os.path.exists(f"{self.rpc}/READY"):
            if self.proc.poll() is not None or time.time() - t0 > 900:
                raise SystemExit(f"[cmt-rag] the retrieval process did not start; see {self.rpc}/server.log")
            time.sleep(2)

    def __call__(self, texts):
        if not texts:
            return []
        self.k += 1
        rq = f"{self.rpc}/req_{self.k:06d}.json"
        rs = rq.replace("req_", "res_")
        json.dump(texts, open(rq + ".tmp", "w"), ensure_ascii=False)
        os.replace(rq + ".tmp", rq)
        while not os.path.exists(rs):
            if self.proc.poll() is not None:
                raise SystemExit(f"[cmt-rag] the retrieval process stopped; see {self.rpc}/server.log")
            time.sleep(0.05)
        out = json.load(open(rs))
        os.remove(rs)
        return out

    def close(self):
        open(f"{self.rpc}/STOP", "w").write("stop")
        self.proc.wait()


def toks(s):
    return set(re.findall(r"\w+", s.lower()))


def parse_drafts(text, dag, next_id):
    """Drafts in order; a dependency must be an existing trace or an earlier draft of this turn."""
    out = []
    for m in LINE.finditer(text):
        q, kw, deps = m.group(1).strip(), m.group(2).strip(), m.group(3)
        if not q:
            continue
        my_id = next_id + len(out)
        ids = [int(x) for x in re.findall(r"\d+", deps)]
        ids = [k for k in dict.fromkeys(ids) if k in dag or next_id <= k < my_id]
        out.append({"q_decom": q, "kw": kw, "deps": ids, "id": my_id})
        if len(out) >= MAX_SUBQ:
            break
    return out


def resolve(q, dag):
    """[A_k] -> the answer of trace k (also written [A_T_k] / [A_Tk], mixing in the namespace label)."""
    def rep(m):
        k = int(m.group(1))
        return dag[k]["answer"] if k in dag and dag[k]["answer"] else m.group(0)
    return re.sub(r"\[A_(?:T_?)?(\d+)\]", rep, q)


def keyword_lookup(kw, dag, t, exclude):
    """The trace of turns older than the preceding LOOKUP_WINDOW turns with the best Ochiai coefficient > OCHIAI_MIN; ties -> smaller id."""
    Q, best = toks(kw), None
    for k, tr in sorted(dag.items()):
        if k in exclude or tr["turn"] > t - LOOKUP_WINDOW - 1:
            continue
        K = toks(tr["kw"])
        s = len(Q & K) / math.sqrt(len(Q) * len(K)) if Q and K else 0.0
        if s > OCHIAI_MIN and (best is None or s > best[0]):
            best = (s, k)
    return [best[1]] if best else []


def main():
    ap = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=TP, help="vLLM tensor parallel size: the first --tp visible GPUs")
    ap.add_argument("--search-gpu", type=int, default=None, help="index of the visible GPU for the retrieval process (default: --tp)")
    a = ap.parse_args()
    sh = common.shard_of(a)
    qs = common.questions(a.set, sh)
    corpus = config.corpus_of(a.set)
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    k = a.tp if a.search_gpu is None else a.search_gpu
    if visible and k >= len(visible):
        raise SystemExit(f"[cmt-rag] {a.tp} GPUs for vLLM and one for the retrieval process: GPU index {k} is not visible")
    search = Search(a.set, os.path.join(config.output_dir(a.set, "cmt_rag"), f"rpc_{sh[0]}"), visible[k] if visible else str(k))
    # the reader reads a passage as the paper's search server returned it: the page title in front on OR-QuAC only
    text = {p["doc_id"]: passage_text(p, corpus) if corpus == "orquac" else p["text"] for p in passages(corpus)}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=8192)
    sp_gen, sp_read = SamplingParams(temperature=0.0, max_tokens=GEN_NEW), SamplingParams(temperature=0.0, max_tokens=READ_NEW)
    search.wait_ready()
    trim = {"gen": 0, "read": 0}

    def chat(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True)

    def gen_prompt(c, t):
        """Over the window, the oldest traces leave the namespace first."""
        turns = c["turns"]
        lines = [f"[T_{k}] {tr['q_sub']}" for k, tr in sorted(c["dag"].items())]
        recent = "\n".join(f"- {turns[j]['question']}" for j in range(max(0, t - RECENT), t)) or "(none)"
        while True:
            ns = "\n".join(lines) or "(no traces yet)"
            p = chat(GEN_SYS, f"Trace namespace of this dialogue (earlier sub-questions; their answers are available as [A_k]):\n{ns}\n\n"
                              f"Recent user questions:\n{recent}\n\nCurrent user query (turn {t + 1}): {turns[t]['question']}\n\nTrace drafts:")
            if not lines or len(tok(p).input_ids) <= MAX_PROMPT:
                return p
            lines = lines[1:]
            trim["gen"] += 1

    def read_prompt(paras, q):
        """Over the window, the last paragraphs are dropped first; the question is never cut."""
        while True:
            p = chat(READ_SYS, "Paragraphs:\n" + "\n\n".join(f"[{i + 1}] {x}" for i, x in enumerate(paras)) + f"\n\nQuestion: {q}")
            if len(paras) <= 1 or len(tok(p).input_ids) <= MAX_PROMPT:
                return p
            paras = paras[:-1]
            trim["read"] += 1

    convs = [{"id": q["id"], "turns": config.turns(q), "dag": {}, "next": 1, "final": None} for q in qs]
    T = max(len(c["turns"]) for c in convs)
    t0 = time.time()
    for t in range(T):
        live = [c for c in convs if t < len(c["turns"])]
        outs = llm.generate([gen_prompt(c, t) for c in live], sp_gen)
        drafts = [parse_drafts(o.outputs[0].text, c["dag"], c["next"]) for c, o in zip(live, outs)]
        for c, d in zip(live, drafts):                   # no parsable line -> the query itself is the single sub-question
            if not d:
                question = c["turns"][t]["question"]
                d.append({"q_decom": question, "kw": " ".join(sorted(toks(question))[:8]), "deps": [], "id": c["next"]})
            c["next"] += len(d)
        nsub = 0
        for j in range(MAX_SUBQ):                        # the j-th draft of every conversation, after its earlier drafts are complete
            flat = [(c, d[j]) for c, d in zip(live, drafts) if j < len(d)]
            if not flat:
                break
            for c, dr in flat:
                dr["q_sub"] = resolve(dr["q_decom"], c["dag"])
            hits = search([dr["q_sub"] for _, dr in flat])
            read_prompts = []
            for (c, dr), h in zip(flat, hits):
                dr["local"] = h
                prior_tr = list(dict.fromkeys(dr["deps"] + keyword_lookup(dr["kw"], c["dag"], t, set(dr["deps"]))))
                dr["prior"] = list(dict.fromkeys(pid for k in prior_tr for pid in c["dag"][k]["local"]))
                read_prompts.append(read_prompt([text[pid] for pid in dict.fromkeys(dr["local"] + dr["prior"])], dr["q_sub"]))
            for (c, dr), o in zip(flat, llm.generate(read_prompts, sp_read)):
                dr["answer"] = o.outputs[0].text.strip().split("\n")[0][:200]
                c["dag"][dr["id"]] = {"q_sub": dr["q_sub"], "kw": dr["kw"], "deps": dr["deps"], "local": dr["local"], "answer": dr["answer"],
                                      "turn": t}
            nsub += len(flat)
        for c, d in zip(live, drafts):
            if t == len(c["turns"]) - 1:                 # the final turn
                c["final"] = {"episode_id": c["id"], "queries": [dr["q_sub"] for dr in d],
                              "prior_ids": list(dict.fromkeys(pid for dr in d for pid in dr["prior"]))}
        print(f"[cmt-rag] turn {t + 1}/{T}: {len(live)} conversations, {nsub} sub-questions, {time.time() - t0:.0f}s", flush=True)
    search.close()
    print(f"[cmt-rag] window guard: {trim['gen']} namespace lines and {trim['read']} reader paragraphs dropped", flush=True)
    common.write_queries(a.set, "cmt_rag", sh, [c["final"] for c in convs])


if __name__ == "__main__":
    main()
