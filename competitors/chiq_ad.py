"""CHIQ-AD (Mo et al., EMNLP 2024) in its default configuration: a prompted LLM first enhances the history, then rewrites the final
question on the enhanced history.

    python -m competitors.chiq_ad --set test_coral [--shard 0/1] [--model Qwen/Qwen2.5-32B-Instruct]
                                                        -> outputs/chiq_ad/test_coral/queries.s0.jsonl

Stage 1, history enhancement with the authors' prompts (their generation/enhancement_generation.py), for the final question u:
explain_question (u', u made unambiguous), single_response (a one-sentence pseudo response to u), new_topic (does u start a new topic?)
and explain_response (the last answer of the history, expanded). Stage 2, the default configuration of their Sec. 3.3: the earlier
turns with the last answer replaced by its expanded form; on a topic switch only the last turn is kept, otherwise the context is the
summary of that enhanced history (prompt summarize_context); then the rewriting prompt (search_query, which asks for {"query": ...}) on
the context and u u' plus the pseudo response. A query that is empty after parsing falls back to u.
Sampling as in the paper: temperature 0.7, at most 512 new tokens, one fixed seed per stage, so a rerun does not reproduce the text
token for token. Writer environment (vLLM), tensor parallel over the first --tp GPUs.
"""
import argparse
import json
import re

from . import common

MODEL = "Qwen/Qwen2.5-32B-Instruct"
TP = 4
TEMPERATURE, MAX_NEW = 0.7, 512
ENHANCE_SEED, REWRITE_SEED = 20261002, 20261004
ENHANCE_WINDOW, REWRITE_WINDOW = 8192, 16384           # model window (tokens) of the enhancement run and of the rewriting run

# the authors' prompts, verbatim, white space collapsed as their code does
PROMPT = {
    "single_response":
        """
        Given a series of question-and-answer pairs, along with a new question, your task is to give a one
        sentence response to the new question.
        """,
    "search_query":
        """
        Given a series of question-and-answer pairs as context, along with a new question, your task is to convert
         the new question into a search engine query that can be used to retrieve relevant documents. The output
          should be placed in a JSON dictionary as follow: {"query": ""}
        """,
    "explain_response":
        """
        You are given a question-and-answer pair, where the answer is not clear. Your goal is to write a long version of
        the answer based on its given context. The generated answer should be one sentence only and less than 20 words.
        """,
    "explain_question":
        """
        You are given a set of question-answers pairs and a new question that is ambiguous. Your goal is to re-write the
        question so it became clear. Just write the new question without any introduction.
        """,
    "summarize_context":
        """
        You are given a context in the form of question-answer pairs. Your goal is to write a paragraph that summarizes
         the information in the context. The summary should be short with one sentence for each question answer pair.
        """,
    "new_topic":
        """
        Given a series of question-and-answer pairs, along with a new question, your task is to determine whether the new
        question continues the discussion on an existing topic or introduces a new topic. Please respond with either
        "new_topic" or "old_topic" as appropriate.
        """,
}
PROMPT = {k: re.sub(r"\s+", " ", v).strip() + "\n\n" for k, v in PROMPT.items()}


def qmark(q):
    """Their `q + "?" if not q.endswith("?") else q`."""
    q = q.strip()
    return q if q.endswith("?") else q + "?"


def enhancement_requests(q):
    """[(prompt name, user message)]: the stage-1 generations stage 2 reads, laid out as their _generate_prompt_1 does."""
    hist = common.history(q)
    pairs = [f"{qmark(u)} {a}" for u, a in hist]
    head, u = ["### Context\n"], qmark(q["question"])
    req = [("explain_question", head + pairs + ["\n", "### Ambiguous Question", u]),
           ("single_response", head + pairs + ["\n", "### Question", u]),
           ("new_topic", head + pairs + ["\n", "### Question", u])]
    if hist:                                             # the last answered turn, with the turns before it as context
        req.append(("explain_response", head + pairs[:-1] + ["\n", "### Question and Answer to explain", pairs[-1]]))
    return [(name, "\n".join(body)) for name, body in req]


def unjson(s):
    """The query of the rewriting prompt's {"query": ...}: the JSON parsed (escaped quotes kept), else the first "query" string literal
    with its escapes, else the bare text without code fences."""
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s.strip())
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        try:
            v = json.loads(cand)
            if isinstance(v, dict) and isinstance(v.get("query"), str):
                return v["query"].strip()
        except Exception:
            pass
    m = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.S)
    if m:
        try:
            return json.loads('"' + m.group(1) + '"').strip()
        except Exception:
            return m.group(1).replace('\\"', '"').strip()
    return t.strip()


def main():
    ap = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=TP, help="vLLM tensor parallel size")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sh = common.shard_of(a)
    qs = common.questions(a.set, sh)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=REWRITE_WINDOW)

    def chat(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True)

    # stage 1. The enhancement run had an 8192-token window: a generation stops when prompt + output reach it.
    reqs = [(q["id"], name, chat(PROMPT[name], user)) for q in qs for name, user in enhancement_requests(q)]
    params = []
    for _, _, p in reqs:
        room = ENHANCE_WINDOW - len(tok(p).input_ids)
        if room <= 0:
            raise SystemExit(f"an enhancement prompt is longer than the {ENHANCE_WINDOW}-token window")
        params.append(SamplingParams(temperature=TEMPERATURE, max_tokens=min(MAX_NEW, room), seed=ENHANCE_SEED))
    enh = {}
    for (qid, name, _), o in zip(reqs, llm.generate([p for _, _, p in reqs], params)):
        enh.setdefault(qid, {})[name] = o.outputs[0].text.strip()
    print(f"{len(reqs)} enhancement generations", flush=True)

    # stage 2: the enhanced history, the summary where there is no topic switch, then the rewriting prompt
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW, seed=REWRITE_SEED)
    items = []
    for q in qs:
        e = enh[q["id"]]
        hist = [f"{qmark(u)} {x}" for u, x in common.history(q)]
        if hist and e.get("explain_response"):          # RE: only the last answer is expanded
            hist[-1] = f"{qmark(q['history'][-1]['question'])} {e['explain_response']}"
        switch = "new" in e["new_topic"].lower()
        if switch:                                       # TS: only the last turn is kept
            hist = hist[-1:]
        items.append({"q": q, "u": qmark(q["question"]), "u2": e["explain_question"], "pr": e["single_response"], "switch": switch,
                      "hist": hist, "summary": ""})
    need = [it for it in items if not it["switch"] and it["hist"]]
    if need:                                             # HS: the summary of the enhanced history
        outs = llm.generate([chat(PROMPT["summarize_context"], "### Context\n" + "\n".join(it["hist"])) for it in need], sp)
        for it, o in zip(need, outs):
            it["summary"] = o.outputs[0].text.strip()
    prompts = []
    for it in items:
        ctx = it["summary"] if not it["switch"] and it["summary"] else "\n".join(it["hist"])
        u = " ".join(x for x in [it["u"], it["u2"], it["pr"]] if x)
        prompts.append(chat(PROMPT["search_query"], "### Context\n" + (ctx or "(none)") + "\n\n### Question\n" + u))
    rows, fallback = [], 0
    for it, o in zip(items, llm.generate(prompts, sp)):
        query = unjson(o.outputs[0].text)
        if not query:                                    # their empty-generation fallback
            query, fallback = it["u"], fallback + 1
        rows.append({"episode_id": it["q"]["id"], "queries": [query]})
    print(f"topic switch in {sum(it['switch'] for it in items)}/{len(items)}; empty query -> question: {fallback}", flush=True)
    common.write_queries(a.set, "chiq_ad", sh, rows)


if __name__ == "__main__":
    main()
