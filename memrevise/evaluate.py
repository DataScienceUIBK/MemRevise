"""Scores the rankings written by memrevise.search with the metrics of the paper, per set and per tier, and the mean over the sets.

    python -m memrevise.evaluate --sets test_topiocqa test_coral test_orquac                    # one run: outputs/writer/
    python -m memrevise.evaluate --names seed_1 seed_2 seed_3                                   # the per-question mean of several runs
    python -m memrevise.evaluate --names writer --compare other                                 # writer minus other, with a 95% interval

Metrics (binary relevance, every gold passage of the question; ranks are 0-based):
    MRR@10     1 / (rank of the best gold passage + 1) if it is in the top 10, else 0
    NDCG@3     DCG of the gold passages in the top 3 / the ideal DCG of min(number of golds, 3) golds
    R@10, R@100   share of the gold passages in the top 10 / top 100
    stale-win  1 if a stale passage (one about a topic the user has left) is ranked above every gold passage, else 0; lower is better
A set reports the mean over its questions (all) and the mean of its tiers with at least 30 questions (macro).
--compare: the paired difference in MRR@10 (--names minus --compare) with a 95% bootstrap interval over conversations: the questions of
one conversation are resampled together (fixed seed); * = the interval excludes zero; tiers with fewer than 30 questions are not compared.

Input: outputs/<name>/<set>/rankings*.jsonl, one row per question: {"episode_id": <question id>, "gold_rank": int, "gold_ranks": [int, ...],
"stale_win": 0 or 1}. Rankings of any other system written in this format are scored the same way.
"""
import argparse
import collections
import glob
import json
import math
import os
import random

from . import config

METRICS = ("MRR@10", "NDCG@3", "R@10", "R@100", "stale-win")
MIN_TIER = 30


def scores(row):
    """The five metrics of one question."""
    best, g = row["gold_rank"], row["gold_ranks"]
    n = max(1, len(g))
    ndcg = sum(1 / math.log2(p + 2) for p in g if p < 3) / sum(1 / math.log2(i + 2) for i in range(min(n, 3)))
    return (1.0 / (best + 1) if 0 <= best < 10 else 0.0, ndcg, sum(p < 10 for p in g) / n, sum(p < 100 for p in g) / n,
            float(row["stale_win"]))


def load(eval_set, name):
    out = {}
    for f in glob.glob(os.path.join(config.output_dir(eval_set, name), "rankings*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[r["episode_id"]] = scores(r)
    return out


def run(eval_set, names, ids):
    """Per-question metrics, averaged over the runs in names; None (with a message) if a run misses a question."""
    runs = [load(eval_set, n) for n in names]
    for n, r in zip(names, runs):
        if any(i not in r for i in ids):
            print(f"   {n}: incomplete, {sum(i in r for i in ids)} of {len(ids)} questions ranked")
            return None
    return {i: tuple(sum(r[i][k] for r in runs) / len(runs) for k in range(len(METRICS))) for i in ids}


def mean(x, ids, k):
    return sum(x[i][k] for i in ids) / len(ids)


def interval(a, b, conv, draws, seed=13):
    """Mean of a - b over the questions (MRR@10), and the 2.5 and 97.5 percentiles of its bootstrap over conversations."""
    keys = list(conv)
    d = lambda I: sum(a[i][0] - b[i][0] for i in I) / len(I)
    rnd = random.Random(seed)
    ds = sorted(d([i for _ in keys for i in conv[keys[rnd.randrange(len(keys))]]]) for _ in range(draws))
    return d([i for k in keys for i in conv[k]]), ds[int(0.025 * draws)], ds[int(0.975 * draws) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=config.TEST_SETS)
    ap.add_argument("--names", "--name", nargs="+", default=["writer"], help="the outputs/<name>/ folders to score (several: their mean)")
    ap.add_argument("--compare", nargs="+", help="outputs/<name>/ folders to compare against (several: their mean)")
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    per_set = []
    for s in a.sets:
        qs = config.questions(s)
        ids = [q["id"] for q in qs]
        conv_of = {q["id"]: q["conversation_id"] for q in qs}
        tiers = collections.defaultdict(list)
        for q in qs:
            tiers[q["tier"]].append(q["id"])
        print(f"\n== {s}  ({len(ids)} questions, {len(set(conv_of.values()))} conversations)")
        x = run(s, a.names, ids)
        y = run(s, a.compare, ids) if a.compare else None
        if x is None or (a.compare and y is None):
            continue
        big = [t for t in sorted(tiers) if len(tiers[t]) >= MIN_TIER]
        head = f"   {'':34s}" + "".join(f"{m:>10s}" for m in METRICS)
        print(head + (f"   MRR@10 minus {'+'.join(a.compare)} [95% interval]" if y else ""))
        rows = [("all", ids)] + [(f"{t} (n={len(tiers[t])}{'' if t in big else ', not in macro'})", tiers[t]) for t in sorted(tiers)]
        for label, I in rows:
            line = f"   {label:34s}" + "".join(f"{mean(x, I, k):10.4f}" for k in range(len(METRICS)))
            if y and len(I) < MIN_TIER:
                line += "   (fewer than 30 questions: not compared)"
            elif y:
                c = collections.defaultdict(list)
                for i in I:
                    c[conv_of[i]].append(i)
                m, lo, hi = interval(x, y, c, a.draws)
                line += f"   {m:+.4f} [{lo:+.3f}, {hi:+.3f}]{' *' if not lo <= 0 <= hi else ''}"
            print(line)
        macro = [sum(mean(x, tiers[t], k) for t in big) / len(big) for k in range(len(METRICS))]
        print(f"   {'macro (tiers with >= 30 questions)':34s}" + "".join(f"{v:10.4f}" for v in macro))
        per_set.append([mean(x, ids, k) for k in range(len(METRICS))])
    if len(per_set) == len(a.sets) > 1:
        print(f"\n== mean over the {len(a.sets)} sets")
        print(f"   {'':34s}" + "".join(f"{m:>10s}" for m in METRICS))
        print(f"   {'all':34s}" + "".join(f"{sum(v[k] for v in per_set) / len(per_set):10.4f}" for k in range(len(METRICS))))


if __name__ == "__main__":
    main()
