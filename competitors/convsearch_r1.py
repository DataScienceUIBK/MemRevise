"""ConvSearch-R1 (Zhu et al., 2025): a 3B query rewriter trained by reinforcement learning on the rank of the gold passage. Row
convsearch_r1 is the authors' released checkpoint; row convsearch_r1_ft is that checkpoint trained further with their recipe on the
ConvRevise training conversations (competitors/convsearch_r1_ft/).

    python -m competitors.convsearch_r1 --set test_coral [--shard 0/4]          -> outputs/convsearch_r1/test_coral/queries.s0.jsonl
    python -m competitors.convsearch_r1 --set test_coral --model models/convsearch_r1_ft --name convsearch_r1_ft

The prompt is the template of the authors' README: their instruction, the earlier turns as 'Q1: ... A1: ...' between '### Context Begin ###'
and '### Context End ###', then 'Query: <question>' and 'Rewrite:', as one user message in the model's chat template. Decoding is that of
their src/infer/infer.py: sampling at temperature 0.7, top_p 1.0, up to 4096 new tokens. Each question has its own seed, SEED plus its
1-based position in the set, so a rerun on the same GPU type and library versions writes the same text. The query is the whole text
between <rewrite> and </rewrite>, or, without those tags, the text after the last </think>; an empty output writes no query.
Environment: the writer's (transformers 5.11).
"""
import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import common

MODEL = "BeastyZ/Qwen2.5-3B-ConvSearch-R1-TopiOCQA"
SEED = 20261002
TEMPERATURE, TOP_P, MAX_NEW, MAX_PROMPT = 0.7, 1.0, 4096, 6144
INSTRUCTION = ("Given a query and its context, you must first think about the reasoning process "
               "in the mind to decontextualize the query by resolving coreference and omission "
               "issues. Then, provide the user with a rewrite that retains its original meaning "
               "and is as informative as possible to help search engines retrieve relevant "
               "documents effectively. The reasoning process and rewrite should be enclosed "
               "within <think> </think> and <rewrite> </rewrite> tags, respectively, i.e., "
               "<think> reasoning process here </think>\n<rewrite> rewrite here </rewrite>.\n\n")


def user_message(q):
    """The content of the one user message (the GRPO training data uses it too)."""
    ctx = []
    for i, (question, answer) in enumerate(common.history(q), 1):
        ctx += [f"Q{i}: {question}", f"A{i}: {answer}"]
    return (INSTRUCTION + "### Context Begin ###\n" + "\n".join(ctx) + "\n### Context End ###\n\nQuery: " + q["question"]
            + "\nRewrite:")


def rewrite_of(text):
    m = re.search(r"<rewrite>(.*?)</rewrite>", text, re.S)
    return (m.group(1) if m else text.split("</think>")[-1]).strip()


def main():
    ap = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default=MODEL, help="the released checkpoint (default) or models/convsearch_r1_ft")
    ap.add_argument("--name", default="convsearch_r1", help="the row: outputs/<name>/")
    a = ap.parse_args()
    sh = common.shard_of(a)
    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device).eval()
    qs = common.questions(a.set, sh)
    rows = []
    with torch.no_grad():
        for n, q in enumerate(qs, 1):
            prompt = tok.apply_chat_template([{"role": "user", "content": user_message(q)}], tokenize=False, add_generation_prompt=True)
            enc = tok([prompt], return_tensors="pt", truncation=True, max_length=MAX_PROMPT).to(device)
            torch.manual_seed(SEED + q["index"] + 1)
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = rewrite_of(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if text:
                rows.append({"episode_id": q["id"], "queries": [text]})
            if n % 25 == 0:
                print(f"  {n}/{len(qs)}", flush=True)
    common.write_queries(a.set, a.name, sh, rows)


if __name__ == "__main__":
    main()
