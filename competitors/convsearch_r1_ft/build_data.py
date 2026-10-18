"""The GRPO training data of ConvSearch-R1-FT in the authors' parquet format (verl/examples/data_preprocess/topiocqa.py of
github.com/BeastyZ/ConvSearch-R1): one row per question of data/benchmark/train.jsonl, in file order (1,790), whose prompt is the user
message of competitors.convsearch_r1, whose ground truth is its gold passage ids joined by '|' and whose extra_info.query_id is its id, by
which competitors.convsearch_r1_ft.reward_server finds the gold passages. val.parquet (the first 32 rows) exists only because the trainer
reads one; it is never evaluated (test_freq -1).

    python -m competitors.convsearch_r1_ft.build_data         -> outputs/convsearch_r1_ft/grpo/{train,val}.parquet
Needs pandas and pyarrow (the verl environment has both).
"""
import argparse
import json
import os

import pandas as pd

from memrevise import config
from ..convsearch_r1 import user_message

OUT = os.path.join(config.OUTPUTS, "convsearch_r1_ft", "grpo")


def rows():
    out = []
    with open(os.path.join(config.DATA, "benchmark", "train.jsonl"), encoding="utf-8") as f:
        for q in map(json.loads, f):
            corpus = config.corpus_of(q["eval_set"])
            out.append({"data_source": "convrevise_" + corpus, "prompt": [{"role": "user", "content": user_message(q)}], "ability": "rewrite",
                        "reward_model": {"style": "rule", "ground_truth": "|".join(q["gold_passages"])},
                        "extra_info": {"split": "train", "index": len(out), "question": q["question"], "query_id": q["id"]}})
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    r = rows()
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(r).to_parquet(os.path.join(OUT, "train.parquet"))
    pd.DataFrame(r[:32]).to_parquet(os.path.join(OUT, "val.parquet"))
    print(f"{len(r)} rows -> {OUT}/train.parquet", flush=True)


if __name__ == "__main__":
    main()
