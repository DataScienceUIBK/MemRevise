"""The writer's prompt. The writer never sees the conversation: at turn j it reads the MEMORY built from its own nodes of turns 1..j-1,
the previous question, and the new turn (its answer too, except at the final turn). It writes one JSON node, and the search query at the
final turn. The same functions build the training prompts and the inference prompts.
"""
import json
import os
from .schema import ConvGraph

SYSTEM_PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "writer_prompt.txt"), encoding="utf-8").read()
STATUS = {"FOCUS": "CURRENT", "FILL": "EARLIER", "SUPPRESS": "DROPPED"}


def render_memory(g, j):
    """g = the memory after turns 1..j-1. Every topic in the order it came up, with its kind, the turns that discussed it, its status and
    its stored facts; then one log line per turn with its operations."""
    if not g.turns:
        return "MEMORY\n(empty: this is the first turn)"
    wp = g.weight_program(j)
    order, facts, types, where, seen = [], {}, {}, {}, set()
    for tn in sorted(g.turns):
        n = g.turns[tn]
        for a in n.asked:
            if a and a not in order:
                order.append(a)
            if a:
                where.setdefault(a, []).append(tn)
        types.update(n.types or {})
        for f in n.facts:
            k = f.text.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            if f.entity and f.entity not in order:
                order.append(f.entity)
            facts.setdefault(f.entity or "", []).append(f.text.strip())
    out = ["MEMORY"]
    for i, e in enumerate(order, 1):
        ty = types.get(e, "")
        kind = f", a {ty}" if ty and ty != "other" else ""
        tn = where.get(e)
        out.append(f"{i}. {e}{kind} | {('turns ' if len(tn) > 1 else 'turn ') + ', '.join(map(str, tn)) if tn else 'only in facts'} | "
                   f"{STATUS.get(wp.get(e, ''), 'EARLIER')}")
        for t in facts.get(e, []):
            out.append(f"     - {t}")
    if facts.get(""):
        out.append("(facts without a topic)")
        for t in facts[""]:
            out.append(f"     - {t}")
    out += ["", "TURN LOG"]
    for tn in sorted(g.turns):
        n = g.turns[tn]
        acts = [f"{o.op} {o.entity}" + (f" — {o.detail}" if o.detail else "") for o in n.ops] or \
               [f"about {', '.join(a for a in n.asked if a) or '(nothing)'}"]
        out.append(f"- turn {tn}: " + "; ".join(acts) +
                   (f" | builds on turn{'s' if len(n.depends_on) > 1 else ''} {', '.join(map(str, n.depends_on))}" if n.depends_on else ""))
    return "\n".join(out)


def messages(g, j, last_question, question, answer, final):
    """The chat messages for turn j (1-based). answer = '' at the final turn, whose answer is not known."""
    ctx = (render_memory(g, j) + "\n\n" + f"LAST QUESTION IN MEMORY: {last_question or '(none)'}" + "\n\n"
           + f"NEW TURN (turn {j})\nQ: {question}\nA: {answer if answer else '(not known yet)'}"
           + "\n\nWRITE THE QUERY: " + ("yes" if final else "no"))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx}]


def graph_from_nodes(conv_id, nodes):
    """Node dicts (facts as {entity, fact|text}) for turns 1..k -> ConvGraph."""
    turns = []
    for i, nd in enumerate(nodes):
        turns.append({"turn_no": i + 1, "asked": nd.get("asked", []), "depends_on": nd.get("depends_on", []),
                      "contradicts": nd.get("contradicts", []), "types": nd.get("types", {}) or {}, "ops": nd.get("ops", []),
                      "facts": [{"entity": f.get("entity", ""), "text": f.get("text", f.get("fact", "")), "turn": i + 1}
                                for f in (nd.get("facts") or [])]})
    return ConvGraph.from_record({"conv_id": conv_id, "turns": turns})


def query_of(raw, node, question):
    """The search query of a final node; a missing or unreadable query falls back to '<topic>. <question>'."""
    import re
    s_, e_ = raw.find("{"), raw.rfind("}")
    q = ""
    if s_ >= 0 and e_ > s_:
        try:
            q = str(json.loads(raw[s_:e_ + 1]).get("query") or "").strip()
        except Exception:
            q = ""
    if not q:
        m_ = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        q = m_.group(1).replace('\\"', '"').strip() if m_ else ""
    if q:
        return q.split("\n")[0]
    a0 = (node.asked or [""])[0]
    return f"{a0}. {question}" if a0 else question
