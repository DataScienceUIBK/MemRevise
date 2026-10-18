"""Train T5QR-FT: castorini/t5-base-canard continued on the final turns of the training conversations, with the hand-written standalone
query of the turn as the target (the turns without one are skipped). The input is that of competitors.t5qr. Learning rate 3e-4, batch 8,
20 epochs, bf16, seed 0; targets cut to 96 tokens. The kept epoch has the lowest loss on a held-out 20% of the training conversations
(random.Random(0) over the sorted conversation names), never on a dev or test set.

    python -m competitors.train_t5qr                (one GPU)
reads data/benchmark/train.jsonl and data/competitor_training/train.jsonl; writes models/t5qr_ft/ (the kept model, the generator's
--model) and models/t5qr_ft_run/ (the epoch checkpoints).
"""
import argparse
import json
import os
import random
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from memrevise import config
from .common import training_rows
from .t5qr import CANARD, MAX_INPUT, canard_source

LR, BATCH, EPOCHS, MAX_TARGET, HELD_OUT = 3e-4, 8, 20, 96, 0.20
# the paper drew the held-out conversations from their working names, which sort differently; renaming gives the paper's split
WORKING_NAME = {"topiocqa_train:": "train:", "coral_train:": "coraltrain:", "coral_spliced_c:": "v4hard:"}


def working_name(conversation_id):
    return re.sub(r"^[a-z_]+:", lambda m: WORKING_NAME.get(m.group(0), m.group(0)), conversation_id)


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(r["src"], truncation=True, max_length=MAX_INPUT)
        x["labels"] = self.tok(text_target=r["tgt"], truncation=True, max_length=MAX_TARGET)["input_ids"]
        return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(config.ROOT, "models", "t5qr_ft"), help="the kept model; checkpoints go to <out>_run")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CANARD)
    rows, skipped = [], 0
    for q, c in training_rows():
        if not c["rewrite"]:
            skipped += 1
            continue
        rows.append({"conv": working_name(q["conversation_id"]), "src": canard_source(tok, q), "tgt": c["rewrite"]})
    # held out by conversation, not by turn: the turns of one conversation share their gold passages
    convs = sorted({r["conv"] for r in rows})
    random.Random(0).shuffle(convs)
    val = set(convs[:max(1, int(HELD_OUT * len(convs)))])
    tr = [r for r in rows if r["conv"] not in val]
    va = [r for r in rows if r["conv"] in val]
    print(f"[train_t5qr] {len(rows)} turns with a hand-written query (skipped {skipped}) / {len(convs)} conversations -> "
          f"train {len(tr)}, held out {len(va)}", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(CANARD)
    args = Seq2SeqTrainingArguments(output_dir=f"{a.out}_run", learning_rate=LR, per_device_train_batch_size=BATCH,
                                    per_device_eval_batch_size=BATCH, num_train_epochs=EPOCHS, eval_strategy="epoch", save_strategy="epoch",
                                    save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                                    logging_steps=20, report_to=[], seed=0, predict_with_generate=False, bf16=True)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=Rows(tr, tok), eval_dataset=Rows(va, tok),
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    losses = [(h.get("epoch"), h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]
    json.dump({"base": CANARD, "n_train": len(tr), "n_held_out": len(va), "eval_loss_by_epoch": losses},
              open(os.path.join(a.out, "train_report.json"), "w"), indent=1)
    print(f"[train_t5qr] saved {a.out}; held-out loss by epoch {[(round(e or 0), round(l, 4)) for e, l in losses]}", flush=True)


if __name__ == "__main__":
    main()
