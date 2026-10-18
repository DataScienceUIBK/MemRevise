"""ChatRetriever (Mao et al., 2024): Qwen-7B-Chat with the authors' LoRA adapter (kyriemao/chatretriever) as a dense retriever that encodes
the whole conversation into one vector. Row chatretriever is the released adapter; row chatretriever_ft is the adapter trained further by
competitors.train_chatretriever. It has its own encoder, so it encodes the collection itself and ranks the passages itself.

    python -m competitors.chatretriever --encode topiocqa --shard 0/8      -> encoded/chatretriever/topiocqa/part_0_of_8.pt (one GPU per part)
    python -m competitors.chatretriever --set test_topiocqa [--shard 0/4]  -> outputs/chatretriever/test_topiocqa/rankings.s0.jsonl
    add --adapter models/chatretriever_ft --name chatretriever_ft to both for the fine-tuned row

As in the authors' code (model_type qwen_chat_cot_lora): a passage is '<|im_start|>assistant\n<passage><|im_end|>', a conversation is its
ChatML turns joined by newlines, the earlier turns as user and assistant messages and the question as the last user message; both end in
<|extra_0|><|extra_1|><|extra_2|>. A conversation has at most 1024 tokens and a passage 384; after truncation the last three tokens are set
back to the <|extra_*|> tokens (their ensure_suffix_in_inputs). The embedding is the final hidden state of the last token, L2-normalised
and stored in fp16; the score is the dot product. A conversation that does not fit drops its oldest messages two at a time, never the
question. The passage text is the one the frozen retriever encodes (memrevise.corpus.passage_text). The paper encoded each collection in 8
parts (passage k in part k mod 8, 24 passages per batch) and ranked over the parts in order.
The rankings have the format of competitors.score_queries (0-based ranks) and are scored by memrevise.evaluate; a question none of whose
gold passages is in the collection is skipped.
Environment: the retriever's (transformers 4.51; Qwen-7B-Chat's remote code also needs tiktoken and einops, and fails on transformers 5).
"""
import argparse
import glob
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from memrevise import config
from memrevise.corpus import passage_text, passages
from . import common

QWEN = "Qwen/Qwen-7B-Chat"
ADAPTER = "kyriemao/chatretriever"
SUFFIX = "<|extra_0|><|extra_1|><|extra_2|>"
MAX_Q, MAX_P, BATCH = 1024, 384, 24


def load(adapter, trainable=False, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
    tok.pad_token_id = tok.eod_id if hasattr(tok, "eod_id") else 151643
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return tok, model


def embeddings(tok, model, texts, max_len, device="cuda:0"):
    """Last-token embeddings, L2-normalised, in float32 (with the graph, for training)."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    suf = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    real = enc["attention_mask"].sum(dim=1)
    rows = torch.arange(enc["input_ids"].size(0), device=enc["input_ids"].device)
    for i in range(1, len(suf) + 1):                       # their ensure_suffix_in_inputs: truncation must not cut the suffix
        enc["input_ids"][rows, real - i] = suf[len(suf) - i]
    h = model(**enc, output_hidden_states=True).hidden_states[-1]
    idx = enc["attention_mask"].sum(dim=1) - 1
    return torch.nn.functional.normalize(h[torch.arange(h.size(0)), idx].float(), p=2, dim=-1)


def passage_input(text):
    return f"<|im_start|>assistant\n{text}<|im_end|>{SUFFIX}"


def session_input(tok, q):
    """Their convsearch_dataset.py fit rule: while history + question > MAX_Q - 30 tokens, drop the two oldest messages."""
    hist = []
    for question, answer in common.history(q):
        hist.append(("user", question))
        if answer:
            hist.append(("assistant", answer))
    sizes = [len(tok(x, add_special_tokens=False)["input_ids"]) for _, x in hist]
    qlen = len(tok(q["question"], add_special_tokens=False)["input_ids"])
    while hist and sum(sizes) + qlen > MAX_Q - 30:
        hist, sizes = hist[2:], sizes[2:]
    parts = [f"<|im_start|>{r}\n{x}<|im_end|>" for r, x in hist]
    parts.append(f"<|im_start|>user\n{q['question']}<|im_end|>")
    return "\n".join(parts) + SUFFIX


def encoded_dir(name, corpus):
    return os.path.join(config.ENCODED, name, corpus)


@torch.no_grad()
def encode(tok, model, corpus, shard, name, adapter):
    i, n = shard
    rows = [(p["doc_id"], passage_text(p, corpus)) for k, p in enumerate(passages(corpus)) if k % n == i]
    print(f"[chatretriever] {corpus} part {i}/{n}: {len(rows)} passages", flush=True)
    ids, embs = [], []
    for b in range(0, len(rows), BATCH):
        batch = rows[b:b + BATCH]
        embs.append(embeddings(tok, model, [passage_input(t) for _, t in batch], MAX_P).half().cpu())
        ids.extend(d for d, _ in batch)
        if (b // BATCH) % 500 == 0:
            print(f"  {b + len(batch)}/{len(rows)}", flush=True)
    d = encoded_dir(name, corpus)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"part_{i}_of_{n}.pt")
    torch.save({"ids": ids, "embs": torch.cat(embs), "adapter": adapter}, out)
    print(f"[chatretriever] -> {out}", flush=True)


def load_collection(name, corpus, device):
    files = glob.glob(os.path.join(encoded_dir(name, corpus), "part_*_of_*.pt"))
    parts = {int(m.group(1)): int(m.group(2)) for m in (re.search(r"part_(\d+)_of_(\d+)\.pt$", f) for f in files)}
    n = set(parts.values())
    if len(n) != 1 or sorted(parts) != list(range(next(iter(n)))):
        raise SystemExit(f"{encoded_dir(name, corpus)}: the parts are incomplete ({sorted(parts)}); run --encode {corpus} for every part")
    ids, embs = [], []
    for k in range(len(parts)):                            # part order: the rows of part 0, then part 1, ...
        d = torch.load(os.path.join(encoded_dir(name, corpus), f"part_{k}_of_{len(parts)}.pt"))
        ids.extend(d["ids"])
        embs.append(d["embs"])
    return ids, torch.cat(embs).float().to(device)


@torch.no_grad()
def rank(tok, model, eval_set, shard, name, device="cuda:0"):
    ids, embs = load_collection(name, config.corpus_of(eval_set), device)
    row = {pid: i for i, pid in enumerate(ids)}
    qs = common.questions(eval_set, shard)
    out = os.path.join(config.output_dir(eval_set, name), f"rankings.s{shard[0]}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(out, "w") as fo:
        for q in qs:
            gold = [row[d] for d in q["gold_passages"] if d in row]
            stale = [row[d] for d in q["stale_passages"] if d in row]
            if not gold:
                continue
            v = embeddings(tok, model, [session_input(tok, q)], MAX_Q).half().cpu().float().to(device)
            scores = (v @ embs.T).squeeze(0)
            order = torch.argsort(-scores)
            pos = torch.empty_like(order)
            pos[order] = torch.arange(order.numel(), device=device)
            gall = sorted(int(pos[r]) for r in gold)
            s = min(int(pos[r]) for r in stale) if stale else 10 ** 9
            fo.write(json.dumps({"episode_id": q["id"], "gold_rank": gall[0], "gold_ranks": gall, "stale_win": float(s < gall[0]),
                                 "n_gold": len(gall)}) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(qs)}", flush=True)
    print(f"{n} of {len(qs)} questions ranked -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", help="rank the questions of this set")
    ap.add_argument("--encode", metavar="CORPUS", choices=sorted(config.CORPUS), help="encode this collection instead")
    ap.add_argument("--shard", default="0/1", help="i/n: questions (or passages) k with k mod n = i")
    ap.add_argument("--adapter", "--model", dest="adapter", default=ADAPTER, help="the released adapter (default) or models/chatretriever_ft")
    ap.add_argument("--name", default="chatretriever", help="the row: encoded/<name>/ and outputs/<name>/")
    a = ap.parse_args()
    if bool(a.set) == bool(a.encode):
        ap.error("give --set or --encode")
    sh = common.shard_of(a)
    tok, model = load(a.adapter)
    model.eval()
    if a.encode:
        encode(tok, model, a.encode, sh, a.name, a.adapter)
    else:
        rank(tok, model, a.set, sh, a.name)


if __name__ == "__main__":
    main()
