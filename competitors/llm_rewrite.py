"""LLM Rewrite: a prompted LLM rewrites the final question as one standalone search query, reading the conversation.

    python -m competitors.llm_rewrite --set test_coral [--shard 0/1] [--model Qwen/Qwen2.5-32B-Instruct]
                                                        -> outputs/llm_rewrite/test_coral/queries.s0.jsonl

The user message holds the earlier turns (USER: question / SYSTEM: answer cut at 600 characters) and the final question. Greedy decoding
with vLLM, at most 200 new tokens; the output, stripped of white space and then of quotation marks, is the query. Writer environment
(vLLM), tensor parallel over the first --tp GPUs.
"""
import argparse

from . import common

MODEL = "Qwen/Qwen2.5-32B-Instruct"
TP = 4
SYSTEM = ("Rewrite the user's FINAL turn as one standalone search query, resolving "
          "all references using the conversation. Output only the query.")
MAX_NEW = 200


def messages(q):
    convo = "\n".join(f"USER: {u}\nSYSTEM: {a}" for u, a in common.history(q))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"CONVERSATION:\n{convo}\n\nFINAL TURN: {q['question']}"}]


def main():
    ap = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=TP, help="vLLM tensor parallel size")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sh = common.shard_of(a)
    qs = common.questions(a.set, sh)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=8192)
    prompts = [tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True) for q in qs]
    rows = []
    for q, o in zip(qs, llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_NEW))):
        text = o.outputs[0].text.strip().strip('"')
        if text:                                     # an empty rewrite writes no row (as in the paper's run)
            rows.append({"episode_id": q["id"], "queries": [text]})
    if len(rows) < len(qs):
        print(f"{len(qs) - len(rows)} questions got an empty rewrite and have no query", flush=True)
    common.write_queries(a.set, "llm_rewrite", sh, rows)


if __name__ == "__main__":
    main()
