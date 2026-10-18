"""The frozen retriever: Reason-ModernColBERT with its 128-dim projection, scoring by MaxSim against a pre-encoded collection.

    Retriever(corpus)            loads the encoder and the encoded collection onto the GPU, and the page titles when the collection has them
    .score(text, boost="")       one score per passage; the tokens of `boost` inside the query weigh twice
    znorm(v)                     zero mean, unit variance over the collection (so two scores can be added)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from . import config
from .corpus import ids_and_titles as corpus_titles

DOC_CHUNK = 100000
SKIPLIST = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


class ColBERTEncoder(nn.Module):
    """ModernBERT + the 1_Dense projection, ColBERT conventions: '[Q] ' / '[D] ' prefixes, queries padded with [MASK] to the query
    length, L2-normalised token vectors, punctuation tokens of documents masked out."""

    def __init__(self, repo_id=config.RETRIEVER, attn_implementation="sdpa"):
        super().__init__()
        local = Path(snapshot_download(repo_id=repo_id))
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        self.encoder = AutoModel.from_pretrained(local, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16)
        dcfg = json.loads((local / "1_Dense" / "config.json").read_text())
        self.dense = nn.Linear(dcfg["in_features"], dcfg["out_features"], bias=bool(dcfg.get("bias", False)))
        dstate = load_file(local / "1_Dense" / "model.safetensors")
        self.dense.weight.data = dstate["linear.weight" if "linear.weight" in dstate else next(iter(dstate))].to(self.dense.weight.dtype)
        if dcfg.get("bias", False):
            self.dense.bias.data = dstate.get("linear.bias", torch.zeros_like(self.dense.bias))
        st = json.loads((local / "config_sentence_transformers.json").read_text()) if (local / "config_sentence_transformers.json").exists() else {}
        self.query_prefix = st.get("query_prefix", "[Q] ")
        self.doc_prefix = st.get("document_prefix", "[D] ")
        self.query_len = int(st.get("query_length", 128))
        self.doc_len = int(st.get("document_length", 8192))
        self.proj_dim = self.dense.out_features
        ids = set()
        for w in st.get("skiplist_words", SKIPLIST):
            ids.update(int(i) for i in self.tokenizer(w, add_special_tokens=False)["input_ids"])
        self.register_buffer("skiplist_ids", torch.tensor(sorted(ids), dtype=torch.long), persistent=False)
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts, is_query, device):
        prefix, max_len = (self.query_prefix, self.query_len) if is_query else (self.doc_prefix, self.doc_len)
        enc = self.tokenizer([prefix + (t or "") for t in texts], padding="max_length" if is_query else True, truncation=True,
                             max_length=max_len, return_tensors="pt")
        input_ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        if is_query and self.tokenizer.mask_token_id is not None:          # query expansion: padding becomes [MASK]
            input_ids = torch.where(input_ids == self.tokenizer.pad_token_id, torch.full_like(input_ids, self.tokenizer.mask_token_id), input_ids)
        h = self.encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        v = F.normalize(self.dense(h.to(self.dense.weight.dtype)), dim=-1)
        valid = attn.bool()
        if not is_query:
            valid = valid & ~torch.isin(input_ids, self.skiplist_ids.to(device))
            v = v * valid.unsqueeze(-1).to(v.dtype)
        return v, valid


def znorm(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def core_title(s):
    """A page title without its bracketed qualifier, lower-cased: 'Jayson Blair (actor)' -> 'jayson blair'."""
    return re.sub(r"\s*\(.*?\)", "", (s or "")).strip().lower()


def cap(s):
    """Sentence case, as every query is scored."""
    return re.sub(r"(^\s*|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s or "")


class Retriever:
    def __init__(self, corpus, device="cuda:0"):
        self.corpus, self.device, self.cfg = corpus, device, config.CORPUS[corpus]
        self.bb = ColBERTEncoder().to(device).eval()
        sd = Path(config.encoded_dir(corpus))
        meta = json.loads((sd / "meta.json").read_text())
        n, T, P = int(meta["n_local"]), int(meta["T_max"]), int(meta["P"])
        self.vectors = torch.from_numpy(np.fromfile(sd / "vectors.f16.bin", dtype=np.int16).reshape(n, T, P)).view(torch.bfloat16).to(device)
        self.doc_mask = torch.from_numpy(np.fromfile(sd / "mask.u8.bin", dtype=np.uint8).reshape(n, T)).bool().to(device)
        ids, titles = corpus_titles(corpus)
        self.ids = ids[:n]
        self.row = {pid: i for i, pid in enumerate(self.ids)}
        self.titled = self.cfg["titled"]
        if self.titled:
            self.title = titles[:n]                                         # exact page title of each passage
            self.title_core = [core_title(t) for t in self.title]
            self.rows_of_title, self.rows_of_core = {}, {}
            for r, t in enumerate(self.title):
                self.rows_of_title.setdefault(t, []).append(r)
            for r, t in enumerate(self.title_core):
                self.rows_of_core.setdefault(t, []).append(r)

    @torch.no_grad()
    def _per_token(self, text):
        if not text or not text.strip():
            return torch.zeros(1, self.bb.proj_dim, device=self.device), torch.zeros(1, dtype=torch.bool, device=self.device)
        v, m = self.bb.encode([text], True, self.device)
        return v[0], m[0]

    def _spans(self, text, names, L):
        """Token positions of each name inside the query (case-insensitive character match)."""
        full = self.bb.query_prefix + (text or "")
        offs = self.bb.tokenizer(full, truncation=True, max_length=self.bb.query_len, return_offsets_mapping=True)["offset_mapping"]
        low, spans = full.lower(), {}
        for nm in names:
            nm = (nm or "").strip()
            if len(nm) < 3:
                continue
            idxs = set()
            for m in re.finditer(re.escape(nm.lower()), low):
                cs, ce = m.start(), m.end()
                for ti, (a, b) in enumerate(offs):
                    if ti >= L:
                        break
                    if a is None or (a == b == 0):
                        continue
                    if a < ce and b > cs:
                        idxs.add(ti)
            if idxs:
                spans[nm] = sorted(idxs)
        return spans

    @torch.no_grad()
    def token_scores(self, text, boost="", boost_weight=2.0):
        """(per-passage weighted MaxSim sum, total token weight). Query tokens weigh 1, the tokens of `boost` weigh `boost_weight`."""
        qv, qm = self._per_token(text)
        qv = qv.to(self.vectors.dtype); outs = []
        for a in range(0, self.vectors.shape[0], DOC_CHUNK):                 # MaxSim over documents in chunks (memory only)
            per = torch.einsum("lp,ntp->lnt", qv, self.vectors[a:a + DOC_CHUNK])
            outs.append(per.masked_fill(~self.doc_mask[None, a:a + DOC_CHUNK, :], float("-inf")).max(dim=-1).values.float())
            del per
        pq = torch.cat(outs, dim=1)
        w = qm.float().clone()
        if boost:
            for nm, idxs in self._spans(text, [boost], qv.shape[0]).items():
                if nm == boost:
                    w[idxs] = boost_weight
        return w @ pq, float(w.sum())

    def score(self, text, boost=""):
        return self.token_scores(text, boost)[0]

    def rows_of(self, title, exact=False):
        return (self.rows_of_title if exact else self.rows_of_core).get(title if exact else core_title(title), []) if self.titled else []

    def vote(self, text, k=20):
        """Pages voted by the top k passages for `text`, weight 1/rank: {exact title: vote}."""
        v = {}
        if not self.titled or not (text or "").strip(" ."):
            return v
        for rk, r in enumerate(torch.topk(znorm(self.score(cap(text[:600]))), k).indices.tolist()):
            if self.title[r]:
                v[self.title[r]] = v.get(self.title[r], 0.0) + 1.0 / (rk + 1)
        return v

    def mask(self, like, rows):
        m = torch.zeros_like(like)
        if rows:
            m[torch.tensor(rows, device=like.device)] = 1.0
        return m

    def rank(self, scores, gold_ids, stale_ids=()):
        """0-based rank of the best gold passage, of the best stale passage, and of every gold passage."""
        gold = [self.row[d] for d in gold_ids if d in self.row]
        stale = [self.row[d] for d in stale_ids if d in self.row]
        order = torch.argsort(-scores)
        pos = torch.empty_like(order)
        pos[order] = torch.arange(order.numel(), device=order.device)
        gall = sorted(int(pos[r]) for r in gold)
        return (gall[0] if gall else 10 ** 9), (min(int(pos[r]) for r in stale) if stale else 10 ** 9), gall
