"""ConvGQR (Mo et al., ACL 2023; github fengranMark/ConvGQR), trained on the training conversations by competitors.train_convgqr: one
T5-base generator writes a standalone rewrite of the final question, a second one writes an answer to it, and the search query is the
rewrite, a space, then the answer (the authors' eval_type "oracle+answer"). Greedy decoding, at most 64 tokens per generator, batches of 32.

The input is the authors' flat concatenation (flat_concat_for_test): "question: <final question>", then the earlier questions and answers
from the most recent backwards, the first of them prefixed "context: ", each cut to 64 tokens, the whole cut to 512 tokens.

    python -m competitors.convgqr --set test_coral --shard 0/4 [--model models/convgqr]   -> outputs/convgqr/test_coral/queries.s0.jsonl
--model holds the two generators: oracle/ (the rewrite) and answer/.
"""
import argparse
import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from memrevise import config
from . import common

MAX_QUERY, MAX_RESPONSE, MAX_CONCAT, BATCH = 64, 64, 512, 32


def flat_concat(tok, q):
    """Token ids and attention mask, padded to MAX_CONCAT: the final question, then the history newest first."""
    ctx = [x for qa in common.history(q) for x in qa]
    ids = tok.encode("question: " + q["question"], add_special_tokens=True, max_length=MAX_QUERY, truncation=True)
    first = True
    for j in range(len(ctx) - 1, -1, -1):
        ml = MAX_RESPONSE if j % 2 == 1 else MAX_QUERY
        t = ("context: " + ctx[j]) if first else ctx[j]
        first = False
        u = tok.encode(t, add_special_tokens=True, max_length=ml, truncation=True)
        if len(ids) + len(u) > MAX_CONCAT:
            # the authors' cut keeps the end-of-sequence token; when the budget is already full nothing is added
            room = MAX_CONCAT - len(ids)
            if room >= 2:
                ids += u[:room - 1] + [u[-1]]
            break
        ids += u
    mask = [1] * len(ids) + [0] * (MAX_CONCAT - len(ids))
    return ids + [tok.pad_token_id] * (MAX_CONCAT - len(ids)), mask


def main():
    ap = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default=os.path.join(config.ROOT, "models", "convgqr"), help="folder with oracle/ and answer/")
    a = ap.parse_args()
    sh = common.shard_of(a)
    gens = {d: (T5Tokenizer.from_pretrained(os.path.join(a.model, d)),
                T5ForConditionalGeneration.from_pretrained(os.path.join(a.model, d)).to("cuda:0").eval()) for d in ("oracle", "answer")}
    qs = common.questions(a.set, sh)
    print(f"[convgqr] {a.set} shard {a.shard}: {len(qs)} questions, {a.model}", flush=True)
    rows = []
    with torch.no_grad():
        for i in range(0, len(qs), BATCH):
            b = qs[i:i + BATCH]
            outs = {}
            for d, (tok, model) in gens.items():
                enc = [flat_concat(tok, q) for q in b]
                ids = torch.tensor([x[0] for x in enc], device="cuda:0")
                mask = torch.tensor([x[1] for x in enc], device="cuda:0")
                o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_length=MAX_QUERY if d == "oracle" else MAX_RESPONSE)
                outs[d] = [s.strip() for s in tok.batch_decode(o, skip_special_tokens=True)]
            for k, q in enumerate(b):
                rows.append({"episode_id": q["id"], "queries": [(outs["oracle"][k] + " " + outs["answer"][k]).strip()],
                             "rewrite": outs["oracle"][k], "answer": outs["answer"][k]})
    common.write_queries(a.set, "convgqr", sh, rows)


if __name__ == "__main__":
    main()
