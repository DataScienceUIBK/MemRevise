"""Search: rank the passages for the final question of every conversation of an evaluation set, from the memory and the query the
writer wrote (memrevise.write_memory). Steps, per question:
  1. two-part query: 0.67 x the writer's query + 0.33 x '<current topic>. <user question>' (topic tokens weigh twice), each part
     divided by its total token weight, then standardised;
  2. number rule: a final turn that only replaces a number ('the one in 1863') swaps it into the topic name and the writer's query,
     which is then searched alone with the title prior on the new name;
  3. title prior: +1.5 on the passages of the page named by the final topic (titled collections);
  4. facts vote: the facts stored for the topic vote for a page, whose passages gain +1.0 (TopiOCQA: every earlier node, within the top
     100; OR-QuAC: the last three nodes about the topic);
  5. exclusion step: when the final turn rejects a topic and a same-name page exists, the rejected page loses 1.5 and its namesakes gain 1.5.

    python -m memrevise.search --set test_coral [--memory outputs/writer/test_coral] [--shard 0/1]
writes <memory dir>/rankings.s<i>.jsonl: one row per question (gold_rank = 0-based rank of the best gold passage).
"""
import argparse
import glob
import json
import re

import torch

from . import config
from .colbert import Retriever, cap, core_title, znorm
from .schema import ConvGraph, same_entity

COMPOSITE_TEMPLATE_SHARE = 0.33
TITLE_PRIOR, FACTS_VOTE, EXCLUSION = 1.5, 1.0, 1.5

# ---- number rule ------------------------------------------------------------------------------------------------------------------
ROMAN = r"(?:x{0,3})(?:ix|iv|v?i{0,3})"


def _vals(s):
    out = []
    for m in re.finditer(r"'?\d+s?\b|\b" + ROMAN + r"\b", s.lower()):
        v = m.group(0)
        if v and v not in ("i", "v", "x"):
            out.append(v)
    return out


def _kind(v):
    return "roman" if re.fullmatch(ROMAN, v) else "decade" if v.endswith("s") else "num"


def number_swap(name, turn):
    """The topic name with its number replaced by the one the final turn asks for, or None. Fires only on a short turn without a question
    word that names exactly one wanted number, decade or Roman numeral of the same kind as one in the name."""
    tl = turn.lower()
    w = re.findall(r"[a-z0-9']+", tl)
    if len(w) > 12 or any(x in w for x in ("what", "who", "when", "where", "which", "how", "why")):
        return None
    if re.search(r"\d+(st|nd|rd|th)\b", tl):
        return None
    tv, nv = _vals(turn), _vals(name)
    if not tv or not nv:
        return None
    rej = [v for v in re.findall(r"\b(?:not|rather than|instead of) (?:the )?('?\w+)", tl) if v in tv]
    want = [v for v in tv if v not in rej]
    if len(want) != 1:
        return None
    x = want[0]
    if x in nv:
        return None
    if rej and not any(y in nv for y in rej):
        return None
    olds = [y for y in rej if y in nv] or [v for v in nv if _kind(v) == _kind(x)]
    if not olds:
        return None
    out = re.sub(r"(?i)(?<![\w'])" + re.escape(olds[0]) + r"(?![\w])", x.upper() if _kind(x) == "roman" else x, name, count=1)
    return out if out != name else None


# ---- the search -----------------------------------------------------------------------------------------------------------------
def _asked0(node):
    return (node.get("asked") or [""])[0]


def _facts_text(node):
    return " ".join(f.get("text", "") for f in (node.get("facts") or []))


def facts_vote_topiocqa(R, rec, name, scores):
    """The earlier nodes about the topic (or, if none, every earlier node) are searched one by one; their top 20 passages vote with weight
    1/rank, the votes are summed per page, and the winning page's passages within the current top 100 gain FACTS_VOTE."""
    nodes = rec["turns"][:-1]
    texts = [_facts_text(nd) for nd in nodes]
    about = [t for nd, t in zip(nodes, texts) if name and _asked0(nd) and (_asked0(nd) == name or same_entity(_asked0(nd), name))]
    votes = {}
    for tx in (about if about else texts):
        if not tx.strip():
            continue
        for rk, r in enumerate(torch.topk(znorm(R.score(cap(tx[:600]))), 20).indices.tolist()):
            votes[r] = votes.get(r, 0.0) + 1.0 / (rk + 1)
    pages = {}
    for r, x in votes.items():
        if R.title[r]:
            pages[R.title[r]] = pages.get(R.title[r], 0.0) + x
    if not pages:
        return scores
    top = set(torch.topk(scores, 100).indices.tolist())
    rows = [r for r in R.rows_of(max(pages, key=pages.get), exact=True) if r in top]
    return scores + FACTS_VOTE * R.mask(scores, rows) if rows else scores


def facts_vote_orquac(R, rec, name, scores):
    """The facts of the last three earlier nodes about the topic form one search; its top 20 passages vote for a page (weight 1/rank),
    whose passages gain FACTS_VOTE."""
    found = []
    for nd in reversed(rec["turns"][:-1]):
        a0, ft = _asked0(nd), _facts_text(nd).strip()
        if a0 and (a0 == name or same_entity(a0, name)) and ft:
            found.append(ft)
        if len(found) >= 3:
            break
    text = " ".join(reversed(found))
    if not text:
        return scores
    pages = {}
    for rk, r in enumerate(torch.topk(znorm(R.score(cap(text[:600]))), 20).indices.tolist()):
        if R.title[r]:
            pages[R.title[r]] = pages.get(R.title[r], 0.0) + 1.0 / (rk + 1)
    return scores + FACTS_VOTE * R.mask(scores, R.rows_of(max(pages, key=pages.get), exact=True)) if pages else scores


def exclusion(R, rec, name, scores):
    """When the final turn rejects a topic: the facts of the last three earlier nodes find the rejected page (top passage); if another page
    shares its name once the bracketed part is removed, the rejected page loses EXCLUSION and its namesakes (not the current topic's
    page) gain EXCLUSION."""
    last = rec["turns"][-1]
    rejected = [r for r in dict.fromkeys([c for c in (last.get("contradicts") or []) if c] +
                                         [o.get("entity") for o in (last.get("ops") or []) if o.get("op") in ("ABANDON", "RETRACT") and o.get("entity")])
                if r != name]
    if not rejected:
        return scores
    text = " ".join(_facts_text(nd).strip() for nd in rec["turns"][:-1][::-1][:3])[:600]
    if not text.strip(" ."):
        return scores
    top = int(torch.argmax(znorm(R.score(cap(text)))))
    page, page_core = R.title[top], R.title_core[top]
    if not (page_core and page and len({R.title[r] for r in R.rows_of_core.get(page_core, [])}) >= 2):
        return scores
    out_rows = R.rows_of_title[page]
    cur = set(R.rows_of(name)) if name else set()
    up = sorted(set(R.rows_of_core[page_core]) - set(out_rows) - cur)
    return scores + EXCLUSION * R.mask(scores, up) - EXCLUSION * R.mask(scores, sorted(out_rows))


def search_one(R, rec, query, question):
    """Scores of every passage for one final question. rec = the memory (graph record); query = the writer's query."""
    g = ConvGraph.from_record(rec)
    topic = g.focus()
    template = f"{topic}. {question}".strip()
    st, wt = R.token_scores(cap(template), topic, 2.0)
    sq, wq = R.token_scores(cap(query), "", 1.0)
    base = znorm(COMPOSITE_TEMPLATE_SHARE * st / max(wt, 1e-6) + (1.0 - COMPOSITE_TEMPLATE_SHARE) * sq / max(wq, 1e-6))
    name = _asked0(rec["turns"][-1])
    prevname = _asked0(rec["turns"][-2]) if len(rec["turns"]) > 1 else ""
    swapped = old = None
    for cand in dict.fromkeys([name, prevname]):
        if cand and number_swap(cand, question):
            swapped, old = number_swap(cand, question), cand
            break
    if swapped:                                      # the writer's query with the new number, and the title prior on the new name
        wq_ = query.replace(old, swapped) if old in query else (query if swapped.lower() in query.lower() else f"{swapped}: {query}")
        return znorm(R.score(cap(wq_))) + (TITLE_PRIOR * R.mask(base, R.rows_of(swapped)) if R.titled else 0)
    scores = base + (TITLE_PRIOR * R.mask(base, R.rows_of(name)) if R.titled else 0)
    kind = R.cfg["facts_vote"]
    if kind == "all_nodes_top100":
        scores = facts_vote_topiocqa(R, rec, name, scores)
    elif kind == "last3_nodes" and name:
        scores = facts_vote_orquac(R, rec, name, scores)
    if R.titled:
        scores = exclusion(R, rec, name, scores)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="evaluation set, e.g. test_topiocqa, test_coral, dev_orquac_value_revisions")
    ap.add_argument("--memory", default="", help="folder with memories.jsonl and queries.jsonl (default outputs/writer/<set>)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process searches every n-th question starting at i")
    a = ap.parse_args()
    mdir = a.memory or config.output_dir(a.set)
    si, sn = (int(x) for x in a.shard.split("/"))
    R = Retriever(config.corpus_of(a.set))
    G = {json.loads(l)["conv_id"]: json.loads(l) for f in sorted(glob.glob(f"{mdir}/memories.s*.jsonl")) for l in open(f)}
    Q = {json.loads(l)["episode_id"]: json.loads(l)["query"] for f in sorted(glob.glob(f"{mdir}/queries.s*.jsonl")) for l in open(f)}
    eps = [q for i, q in enumerate(config.questions(a.set)) if i % sn == si]
    with open(f"{mdir}/rankings.s{si}.jsonl", "w") as fo:
        for q in eps:
            eid, gold = q["id"], q["gold_passages"]
            if not any(d in R.row for d in gold) or eid not in G:
                continue
            scores = search_one(R, G[eid], Q[eid], q["question"])
            rank, stale_rank, gall = R.rank(scores, gold, q["stale_passages"])
            fo.write(json.dumps({"episode_id": eid, "gold_rank": rank, "stale_win": float(stale_rank < rank), "gold_ranks": gall,
                                 "n_gold": len([d for d in gold if d in R.row])}) + "\n")
    print(f"[search] {a.set} shard {a.shard}: {len(eps)} questions -> {mdir}/rankings.s{si}.jsonl", flush=True)


if __name__ == "__main__":
    main()
