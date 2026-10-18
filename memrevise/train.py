"""Train the writer: a fresh LoRA adapter (rank 16, alpha 32, dropout 0.05, q/k/v/o projections) on Ministral-3-3B, 3 epochs, learning rate
5e-5 with a cosine schedule and 5% warm-up, effective batch 12, prompt tokens masked. A checkpoint is saved and evaluated every 300 steps;
the kept one has the lowest held-out loss on the fields written before the query (depends_on, contradicts, ops, asked).

    torchrun --nproc_per_node=4 -m memrevise.train            (4 GPUs x batch 1 x accumulation 3 = 12)
    python -m memrevise.train --grad-accum 12                 (one GPU, same effective batch)
reads data/writer_training/train.jsonl.gz and dev.jsonl and writes models/writer/ (the adapter) and models/writer_run/ (checkpoints).
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from . import config


def read_jsonl(path):
    import gzip
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        return [json.loads(l) for l in f]


class Rows(Dataset):
    """Chat prompt + JSON target; loss on the target tokens. decision_only=True keeps only the tokens up to the end of "asked"."""

    def __init__(self, rows, tok, max_length=6144, decision_only=False):
        self.rows, self.tok, self.max_length, self.donly = rows, tok, max_length, decision_only

    def __len__(self):
        return len(self.rows)

    def _decision_mask(self, target, n_tok):
        s = target.find('"asked":')
        if s < 0:
            return None
        e = target.find("]", s)
        if e < 0:
            return None
        enc = self.tok(target, add_special_tokens=False, return_offsets_mapping=True)
        return [1 if a < e + 1 else 0 for a, b in enc["offset_mapping"]][:n_tok]

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = self.tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(r["target"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids, labels = p_ids + t_ids, [-100] * len(p_ids) + t_ids
        weights = None
        if self.donly:
            dm = self._decision_mask(r["target"], len(t_ids)) or []
            dm = dm + [0] * (len(t_ids) - len(dm))
            weights = [0.0] * len(p_ids) + [1.0 if m else 0.0 for m in dm]
        if len(ids) > self.max_length:                                       # keep the tail: the target stays whole
            ids, labels = ids[-self.max_length:], labels[-self.max_length:]
            weights = weights[-self.max_length:] if weights is not None else None
        out = {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}
        if weights is not None:
            out["weights"] = weights
        return out


class Pad:
    """Right padding (left padding gives fully masked rows under SDPA)."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": torch.tensor([b["input_ids"] + [self.pad_id] * (m - len(b["input_ids"])) for b in batch]),
               "attention_mask": torch.tensor([b["attention_mask"] + [0] * (m - len(b["input_ids"])) for b in batch]),
               "labels": torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])}
        if all("weights" in b for b in batch):
            out["weights"] = torch.tensor([b["weights"] + [0.0] * (m - len(b["weights"])) for b in batch])
        return out


class WriterTrainer(Trainer):
    """Cross-entropy in fp32 (bf16 over a large vocabulary overflows on long rows); weighted when the batch carries weights."""

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs = dict(inputs)
        labels, weights = inputs.pop("labels"), inputs.pop("weights", None)
        out = model(**inputs)
        logits = out.logits[..., :-1, :].float().contiguous()
        sl = labels[..., 1:].contiguous()
        if weights is None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), sl.view(-1), ignore_index=-100, reduction="sum") \
                / (sl != -100).sum().clamp_min(1)
        else:
            ce = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), sl.view(-1), ignore_index=-100, reduction="none")
            w = weights[..., 1:].contiguous().view(-1).to(ce.dtype) * (sl.view(-1) != -100).to(ce.dtype)
            loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(config.DATA, "writer_training"))
    ap.add_argument("--base", default=config.BASE_MODEL)
    ap.add_argument("--out", default=config.WRITER)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test)")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    from peft import LoraConfig, get_peft_model
    run = Path(str(a.out).rstrip("/") + "_run")
    run.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(os.path.join(a.data, "train.jsonl.gz"))
    dev = read_jsonl(os.path.join(a.data, "dev.jsonl"))
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():                                              # a rare non-finite gradient is zeroed, not propagated
        if p.requires_grad:
            p.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
    nodes = [r for r in dev if r.get("row") in ("history", "final", "hidden-twin")]
    eval_sets = {"all": Rows(dev, tok), "decision": Rows(nodes, tok, decision_only=True)}
    args = TrainingArguments(output_dir=str(run), num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=1,
                             gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                             logging_steps=20, save_strategy="steps", save_steps=300, eval_strategy="steps", eval_steps=300,
                             save_total_limit=2, per_device_eval_batch_size=1, load_best_model_at_end=True,
                             metric_for_best_model="eval_decision_loss", greater_is_better=False, bf16=True, seed=a.seed, report_to=[],
                             remove_unused_columns=False)
    trainer = WriterTrainer(model=model, train_dataset=Rows(rows, tok), eval_dataset=eval_sets, data_collator=Pad(tok.pad_token_id), args=args)
    trainer.train()
    print(f"[train] best checkpoint {trainer.state.best_model_checkpoint} (decision loss {trainer.state.best_metric})", flush=True)
    if int(os.environ.get("RANK", "0")) == 0:
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        print(f"[train] writer saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
