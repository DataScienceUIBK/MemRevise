"""Original Query: the final question alone.

    python -m competitors.original_query --set test_coral          -> outputs/original_query/test_coral/queries.s0.jsonl
"""
import argparse

from . import common


def main():
    a = common.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    sh = common.shard_of(a)
    common.write_queries(a.set, "original_query", sh, [{"episode_id": q["id"], "queries": [q["question"]]} for q in common.questions(a.set, sh)])


if __name__ == "__main__":
    main()
