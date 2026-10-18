"""T5QR (Lin et al., 2020): the public CANARD rewriter castorini/t5-base-canard turns the final question into a standalone query. The same
generator serves the row T5QR-FT, the rewriter fine-tuned on the training conversations by competitors.train_t5qr.

The input is the CANARD convention 'q1 ||| a1 ||| ... ||| final question'. When it is longer than 512 tokens the oldest history parts are
dropped until it fits, so the final question is always kept. Decoding as published: 10 beams, at most 64 new tokens, early stopping.

    python -m competitors.t5qr --set test_coral --shard 0/4                                       -> outputs/t5qr/test_coral/queries.s0.jsonl
    python -m competitors.t5qr --set test_coral --shard 0/4 --model models/t5qr_ft --name t5qr_ft -> outputs/t5qr_ft/test_coral/queries.s0.jsonl
The questions of a shard are rewritten --batch at a time (T5QR 16, T5QR-FT 1). Beam search over a padded batch can change an output, so the
paper's queries are reproduced with its shards (4 per set) and these batch sizes.
"""
import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import common

CANARD = "castorini/t5-base-canard"
MAX_INPUT, MAX_NEW, BEAMS = 512, 64, 10
BATCH = {"t5qr": 16, "t5qr_ft": 1}


def canard_source(tok, q):
    """'q1 ||| a1 ||| ... ||| final question', the oldest history parts dropped until it fits MAX_INPUT tokens."""
    hist = [x for qa in common.history(q) for x in qa]
    while hist and len(tok(" ||| ".join(hist + [q["question"]])).input_ids) > MAX_INPUT:
        hist = hist[1:]
    return " ||| ".join(hist + [q["question"]])


def main():
    ap = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default=CANARD, help="the rewriter (models/t5qr_ft for T5QR-FT)")
    ap.add_argument("--name", default="t5qr", help="the row the queries are written for: t5qr or t5qr_ft")
    ap.add_argument("--batch", type=int, default=0, help="questions per generate call (default: 16 for t5qr, 1 for t5qr_ft)")
    a = ap.parse_args()
    sh = common.shard_of(a)
    batch = a.batch or BATCH.get(a.name, 16)
    tok = AutoTokenizer.from_pretrained(CANARD)          # the fine-tune keeps the CANARD tokeniser
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to("cuda:0").eval()
    qs = common.questions(a.set, sh)
    src = [canard_source(tok, q) for q in qs]
    print(f"[t5qr] {a.set} shard {a.shard}: {len(qs)} questions, {a.model}, batch {batch}", flush=True)
    rows = []
    with torch.no_grad():
        for i in range(0, len(qs), batch):
            enc = tok(src[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT).to("cuda:0")
            out = model.generate(**enc, max_new_tokens=MAX_NEW, num_beams=BEAMS, early_stopping=True)
            rows += [{"episode_id": q["id"], "queries": [tok.decode(o, skip_special_tokens=True).strip()]}
                     for q, o in zip(qs[i:i + batch], out)]
    common.write_queries(a.set, a.name, sh, rows)


if __name__ == "__main__":
    main()
