"""The memory: one node per turn, and the statuses replayed from the nodes' operations.

A node records the topics the turn is about (asked), the facts its question and answer establish, the earlier turns it builds on
(depends_on), the topics it rejects (contradicts) and its operations:
    ADD a new topic, REFINE the current one, REVISE a value, ABANDON a topic, REACTIVATE an earlier topic, RETRACT a false premise.
Replaying the operations in turn order gives every topic a status: current (FOCUS), earlier (FILL) or dropped (SUPPRESS).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import re

ENTITY_TYPES = ("person", "work", "place", "organization", "event", "date", "concept", "object", "other")
OPS = ("ADD", "REFINE", "REVISE", "ABANDON", "REACTIVATE", "RETRACT")


@dataclass
class Fact:
    entity: str
    text: str
    turn: int


@dataclass
class Op:
    op: str
    entity: str
    detail: str = ""


@dataclass
class TurnNode:
    turn_no: int
    asked: list = field(default_factory=list)
    facts: list = field(default_factory=list)
    depends_on: list = field(default_factory=list)
    contradicts: list = field(default_factory=list)
    types: dict = field(default_factory=dict)
    ops: list = field(default_factory=list)
    correction: bool = False          # kept for the record format; the writer does not set it
    query: str = ""


@dataclass
class ConvGraph:
    conv_id: str = ""
    turns: dict = field(default_factory=dict)        # turn_no -> TurnNode

    def add_turn(self, node: TurnNode) -> None:
        self.turns[node.turn_no] = node

    def to_record(self) -> dict:
        return {"conv_id": self.conv_id, "turns": [asdict(self.turns[k]) for k in sorted(self.turns)]}

    @classmethod
    def from_record(cls, r: dict) -> "ConvGraph":
        g = cls(conv_id=r.get("conv_id", ""))
        for nd in r.get("turns", []):
            facts = [Fact(entity=f.get("entity", ""), text=f.get("text", ""), turn=int(f.get("turn", nd["turn_no"])))
                     for f in nd.get("facts", []) if f.get("text")]
            ops = [Op(op=o.get("op", ""), entity=o.get("entity", ""), detail=o.get("detail", ""))
                   for o in nd.get("ops", []) if isinstance(o, dict) and o.get("op") in OPS and o.get("entity")]
            g.add_turn(TurnNode(turn_no=int(nd["turn_no"]), asked=nd.get("asked", []), facts=facts, depends_on=nd.get("depends_on", []),
                                contradicts=nd.get("contradicts", []), types=nd.get("types", {}), ops=ops, query=nd.get("query", ""),
                                correction=bool(nd.get("correction", False))))
        return g

    def prior_turns(self, t: int) -> list:
        return [self.turns[j] for j in sorted(self.turns) if j < t]

    def memory_state(self, t: int) -> dict:
        """Replay the operations of the turns before t: topic -> ACTIVE | DORMANT | ABANDONED | RETRACTED.
        ADD / REFINE / REVISE / REACTIVATE make the topic current (the previous current one becomes earlier); ABANDON drops a topic;
        RETRACT marks it false, and a false topic is never brought back."""
        state: dict = {}
        for n in self.prior_turns(t):
            for o in n.ops:
                if o.op in ("ADD", "REFINE", "REVISE", "REACTIVATE"):
                    if o.op == "REACTIVATE" and state.get(o.entity) == "RETRACTED":
                        continue
                    for e, st in state.items():
                        if st == "ACTIVE" and e != o.entity:
                            state[e] = "DORMANT"
                    state[o.entity] = "ACTIVE"
                elif o.op == "ABANDON":
                    state[o.entity] = "ABANDONED"
                elif o.op == "RETRACT":
                    state[o.entity] = "RETRACTED"
        return state

    def weight_program(self, t: int) -> dict:
        """topic -> FOCUS (current) | FILL (earlier) | SUPPRESS (dropped)."""
        chan = {"ACTIVE": "FOCUS", "DORMANT": "FILL", "ABANDONED": "SUPPRESS", "RETRACTED": "SUPPRESS"}
        return {e: chan[st] for e, st in self.memory_state(t).items()}

    def focus(self) -> str:
        """The current topic after every stored turn ('' when none)."""
        T = (max(self.turns) + 1) if self.turns else 1
        return next((e for e, c in self.weight_program(T).items() if c == "FOCUS"), "")


def parse_extraction(raw: str, turn_no: int) -> TurnNode:
    """The writer's JSON output -> a node. Tolerant to text around the JSON and to a truncated JSON (fields recovered by pattern)."""
    s, e = raw.find("{"), raw.rfind("}")
    o = {}
    if s >= 0 and e > s:
        try:
            o = json.loads(raw[s:e + 1])
        except Exception:
            o = {}
    asked = [a.strip() for a in (o.get("asked") or []) if isinstance(a, str) and a.strip()]
    facts = []
    for f in (o.get("facts") or []):
        if isinstance(f, dict):
            ent = (f.get("entity") or "").strip()
            txt = (f.get("fact") or f.get("text") or "").strip()
            if txt:
                facts.append(Fact(entity=ent, text=txt, turn=turn_no))
        elif isinstance(f, str) and f.strip() and f.strip().lower() not in ("null", "none"):
            facts.append(Fact(entity="", text=f.strip(), turn=turn_no))
    dep = sorted({int(x) for x in (o.get("depends_on") or []) if str(x).isdigit() and int(x) < turn_no})
    contr = [c.strip() for c in (o.get("contradicts") or []) if isinstance(c, str) and c.strip() and c.strip().lower() not in ("null", "none")]
    types = {k: v for k, v in (o.get("types") or {}).items() if isinstance(k, str) and v in ENTITY_TYPES}
    ops = []
    for x in (o.get("ops") or []):
        if isinstance(x, dict) and str(x.get("op", "")).upper() in OPS and str(x.get("entity", "")).strip():
            ops.append(Op(op=str(x["op"]).upper(), entity=str(x["entity"]).strip(), detail=str(x.get("detail", "") or "").strip()[:200]))
    for op in ops:                                   # a dropped or false topic is also a rejected one
        if op.op in ("ABANDON", "RETRACT") and op.entity not in contr:
            contr.append(op.entity)
    if not facts:                                    # truncated JSON: recover the facts written before the cut
        for blk in re.findall(r'\{[^{}]*"fact"[^{}]*\}', raw):
            fm = re.search(r'"(?:fact|text)"\s*:\s*"((?:[^"\\]|\\.)*)"', blk)
            if not fm:
                continue
            em = re.search(r'"entity"\s*:\s*"((?:[^"\\]|\\.)*)"', blk)

            def _u(s2):
                try:
                    return json.loads('"' + s2 + '"')
                except Exception:
                    return s2
            txt = _u(fm.group(1)).strip()
            if txt:
                facts.append(Fact(entity=(_u(em.group(1)).strip() if em else ""), text=txt, turn=turn_no))
    if not asked:
        am = re.search(r'"asked"\s*:\s*\[([^\]]*)\]', raw)
        if am:
            asked = [a for a in (re.sub(r'\\(.)', r'\1', x) for x in re.findall(r'"((?:[^"\\]|\\.)*)"', am.group(1))) if a.strip()]
    if not dep:
        dm = re.search(r'"depends_on"\s*:\s*\[([^\]]*)\]', raw)
        if dm:
            dep = sorted({int(x) for x in re.findall(r'\d+', dm.group(1)) if int(x) < turn_no})
    seen_f, uniq = set(), []                         # drop exact repeats of a fact within the turn
    for f in facts:
        k = (f.entity.strip().lower(), f.text.strip().lower())
        if k not in seen_f:
            seen_f.add(k)
            uniq.append(f)
    facts = uniq
    q = o.get("query")
    q = q.strip() if isinstance(q, str) else ""
    if not q:
        qm_ = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if qm_:
            try:
                q = json.loads('"' + qm_.group(1) + '"').strip()
            except Exception:
                q = qm_.group(1).strip()
    _c = o.get("correction", False)
    if isinstance(_c, str):
        _c = _c.strip().lower() in ("true", "yes", "1")
    return TurnNode(turn_no=turn_no, asked=asked, facts=facts, depends_on=dep, contradicts=contr, types=types, ops=ops,
                    query=q[:600], correction=bool(_c))


# ---- entity names ---------------------------------------------------------------------------------------------------------------
def norm_entity(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _whole_word_in(needle, hay):
    if not needle or not hay:
        return False
    return re.search(rf"(?:^|\s){re.escape(needle)}(?:$|\s)", hay) is not None


def same_entity(a, b):
    """Two names denote the same topic: equal after normalisation, or one is a whole-word extension of the other."""
    na, nb = norm_entity(a), norm_entity(b)
    if not na or not nb:
        return False
    return na == nb or _whole_word_in(na, nb) or _whole_word_in(nb, na)


ANSWER_CHARS = 600


def hist_answer(turn):
    """An earlier turn's answer, cut at 600 characters."""
    return str(turn.get("answer", "") or "")[:ANSWER_CHARS]
