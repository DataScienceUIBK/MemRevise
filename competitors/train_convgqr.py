"""Train one of ConvGQR's two generators (the authors' src/train_GQR.py) on the final turns of the training conversations. google-t5/t5-base
learns to decode the hand-written standalone query of the turn (--decode oracle; the turns without one are skipped) or the gold answer of
the turn (--decode answer). The input is that of competitors.convgqr.

loss = decoder cross-entropy + 0.5 x MSE(first encoder state, ANCE embedding of the turn's first gold passage), the frozen passage encoder
being castorini/ance-msmarco-passage (first token, embeddingHead, LayerNorm). The gold passage text is 'title | text' on TopiOCQA (the
title holds 'Title [SEP] Section'), the text alone elsewhere, cut to 384 tokens. AdamW lr 1e-5, eps 1e-8, no weight decay, linear schedule
without warm-up, gradient clipping 1.0, batch 8, 15 epochs, seed 42, the turns reshuffled every epoch. As in the authors' code, the model
is saved whenever the loss of a batch is the lowest seen so far; the last of these saves is the model the row uses.

    python -m competitors.train_convgqr --decode oracle          (one GPU each; the two runs are independent)
    python -m competitors.train_convgqr --decode answer
reads data/benchmark/train.jsonl and data/competitor_training/train.jsonl; writes models/convgqr/<decode>/ (with saved_at.json: the epoch,
step and batch loss of the save).
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)

from memrevise import config
from .convgqr import MAX_QUERY, MAX_RESPONSE, flat_concat
from .common import training_rows

T5 = "google-t5/t5-base"
ANCE_MODEL = "castorini/ance-msmarco-passage"
MAX_DOC, LR, BATCH, EPOCHS, ALPHA, SEED = 384, 1e-5, 8, 15, 0.5, 42


class ANCE(RobertaForSequenceClassification):
    """The authors' ANCE passage encoder: RoBERTa, first token, Linear(768, 768), LayerNorm."""

    def __init__(self, cfg):
        RobertaForSequenceClassification.__init__(self, cfg)
        self.embeddingHead = nn.Linear(cfg.hidden_size, 768)
        self.norm = nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask):
        h = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.norm(self.embeddingHead(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode", required=True, choices=["oracle", "answer"], help="oracle: the rewrite; answer: the answer")
    ap.add_argument("--out", default=os.path.join(config.ROOT, "models", "convgqr"), help="writes <out>/<decode>/")
    a = ap.parse_args()
    out_dir = os.path.join(a.out, a.decode)
    dev = "cuda:0"
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = T5Tokenizer.from_pretrained(T5)
    model = T5ForConditionalGeneration.from_pretrained(T5).to(dev)
    cfg = RobertaConfig.from_pretrained(ANCE_MODEL, finetuning_task="MSMarco")
    ptok = RobertaTokenizer.from_pretrained(ANCE_MODEL, do_lower_case=True)
    ance = ANCE.from_pretrained(ANCE_MODEL, config=cfg).to(dev).eval()

    ex, skipped = [], 0
    for q, c in training_rows():
        if a.decode == "oracle":
            target, keep = c["rewrite"], bool(c["rewrite"])
        else:
            target = str(c["answer"] or "")
            keep = bool(target.strip())
        if not keep:
            skipped += 1
            continue
        pos = c["positives"][0]
        ptxt = f"{pos['title']} | {pos['text']}" if q["source"].startswith("topiocqa") and pos.get("title") else pos["text"]
        ids, mask = flat_concat(tok, q)
        lab = tok(target, padding="max_length", max_length=MAX_QUERY if a.decode == "oracle" else MAX_RESPONSE, truncation=True).input_ids
        lab = [-100 if x == tok.pad_token_id else x for x in lab]
        pd = ptok(ptxt, padding="max_length", max_length=MAX_DOC, truncation=True)
        ex.append((ids, mask, lab, pd.input_ids, pd.attention_mask))
    print(f"[train_convgqr] decode {a.decode}: {len(ex)} training turns (skipped {skipped})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=EPOCHS * ((len(ex) + BATCH - 1) // BATCH))
    best, step, rnd = float("inf"), 0, random.Random(SEED)
    for ep in range(EPOCHS):
        model.train()
        rnd.shuffle(ex)
        tot, n = 0.0, 0
        for i in range(0, len(ex), BATCH):
            b = ex[i:i + BATCH]
            T = lambda k: torch.tensor([x[k] for x in b], device=dev)   # noqa: E731
            out = model(input_ids=T(0), attention_mask=T(1), labels=T(2))
            with torch.no_grad():
                pemb = ance(T(3), T(4)).detach()
            kd = nn.functional.mse_loss(out.encoder_last_hidden_state[:, 0], pemb)
            loss = out.loss + ALPHA * kd
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            tot += float(loss.detach())
            n += 1
            if step % 100 == 0:
                print(f"[train_convgqr] epoch {ep + 1} step {step} decode {out.loss.item():.4f} kd {kd.item():.4f} total {loss.item():.4f}",
                      flush=True)
            if loss.item() < best:                      # the authors' checkpoint rule: the lowest batch loss so far
                best = loss.item()
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
                json.dump({"epoch": ep + 1, "step": step, "loss": best}, open(os.path.join(out_dir, "saved_at.json"), "w"))
        print(f"[train_convgqr] epoch {ep + 1} mean loss {tot / max(n, 1):.4f} (best batch {best:.4f})", flush=True)
    print(f"[train_convgqr] {step} steps; kept {out_dir} ({json.load(open(os.path.join(out_dir, 'saved_at.json')))})", flush=True)


if __name__ == "__main__":
    main()
