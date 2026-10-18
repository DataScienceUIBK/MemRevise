"""Train ChatRetriever's released LoRA adapter (kyriemao/chatretriever on Qwen-7B-Chat) further on the ConvRevise training conversations:
row chatretriever_ft.

    python -m competitors.train_chatretriever            (one GPU)   -> models/chatretriever_ft/ and models/chatretriever_ft/epoch_<k>/
reads data/benchmark/train.jsonl (the conversations) and data/competitor_training/train.jsonl (positives and negatives), row by row.

The recipe follows the authors' released training command where this setting allows: their ranking loss (cross-entropy of each
conversation against the positives of the batch followed by the sampled negatives of the batch, at temperature 0.01); 4 hard negatives
sampled per conversation from its mined negatives that are not stale passages (all of them when it has fewer); learning rate 1e-4, 50
warm-up steps then linear decay, AdamW without weight decay, gradients clipped at norm 1; batches of 8 conversations, 4 accumulated per
optimizer step; 3 epochs, the rows shuffled again every epoch; gradient checkpointing; seed 20260921. A conversation is encoded as in
competitors.chatretriever; a passage is the text of the first positive or of a negative, without its title. The adapter is saved after
every epoch; the paper used epoch 3, chosen on dev, which is also the adapter saved at the end (672 batches: no partial accumulation is left).
Not as theirs: their masked instruction-tuning loss is not used, and the in-batch contrast is one GPU's batch.
Environment: the retriever's (transformers 4.51).
"""
import argparse
import json
import os
import random

import torch

from memrevise import config
from . import chatretriever as cr

EPOCHS, LR, BATCH, ACCUM, NEGS, TEMP, WARMUP, WEIGHT_DECAY, SEED = 3, 1e-4, 8, 4, 4, 0.01, 50, 0.0, 20260921


def training_rows(tok):
    with open(os.path.join(config.DATA, "benchmark", "train.jsonl"), encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    with open(os.path.join(config.DATA, "competitor_training", "train.jsonl"), encoding="utf-8") as f:
        ct = [json.loads(l) for l in f]
    if [q["id"] for q in qs] != [r["id"] for r in ct]:
        raise SystemExit("data/competitor_training/train.jsonl is not in the order of data/benchmark/train.jsonl")
    rows = []
    for q, r in zip(qs, ct):
        pos = [p for p in r["positives"] if p.get("text")]
        negs = [n["text"] for n in r["negatives"] if isinstance(n, dict) and n.get("text")]
        if pos and negs:
            rows.append({"session": cr.session_input(tok, q), "pos": pos[0]["text"], "negs": negs})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(config.ROOT, "models", "chatretriever_ft"))
    a = ap.parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok, model = cr.load(cr.ADAPTER, trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()          # under LoRA the frozen embeddings would otherwise cut every checkpointed segment
    model.config.use_cache = False
    model.train()
    train = training_rows(tok)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = sum(1 for i in range(0, len(train), BATCH) if len(train[i:i + BATCH]) >= 2)
    total = max(1, (EPOCHS * n_batches + ACCUM - 1) // ACCUM)
    print(f"[chatretriever-ft] {len(train)} conversations, {n_batches} batches per epoch, {total} optimizer steps", flush=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(1, WARMUP)) if k < WARMUP else max(0.0, (total - k) / max(1, total - WARMUP)))
    rnd = random.Random(SEED)
    step = 0
    for ep in range(EPOCHS):
        rnd.shuffle(train)
        for i in range(0, len(train), BATCH):
            batch = train[i:i + BATCH]
            if len(batch) < 2:                  # the loss needs a second conversation in the batch
                continue
            q = cr.embeddings(tok, model, [b["session"] for b in batch], cr.MAX_Q)
            docs = [cr.passage_input(b["pos"]) for b in batch]
            for b in batch:
                docs += [cr.passage_input(t) for t in rnd.sample(b["negs"], min(NEGS, len(b["negs"])))]
            d = cr.embeddings(tok, model, docs, cr.MAX_P)
            scores = q.mm(d.T) / TEMP           # the first len(q) passages are the positives
            loss = torch.nn.functional.cross_entropy(scores, torch.arange(len(q), device=scores.device))
            (loss / ACCUM).backward()
            step += 1
            if step % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"  epoch {ep + 1} step {step} loss {loss.item():.4f}", flush=True)
        d = os.path.join(a.out, f"epoch_{ep + 1}")
        model.save_pretrained(d)
        tok.save_pretrained(d)
        print(f"[chatretriever-ft] epoch {ep + 1} -> {d}", flush=True)
    if step % ACCUM:                            # a partial accumulation group at the end is applied
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"[chatretriever-ft] {step} batches -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
