"""Encode a passage collection with the frozen Reason-ModernColBERT (256 document tokens, 128 dimensions) for memrevise.search.

    python -m memrevise.encode_corpus --corpus topiocqa        (one GPU; TopiOCQA 434k passages take about two hours)
reads corpus/<corpus>.jsonl (memrevise.corpus), writes encoded/<corpus>/shard_000_of_001/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .colbert import ColBERTEncoder
from .corpus import passage_text, passages, path

DOC_TOKENS, DIM = 256, 128


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=sorted(config.CORPUS))
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    out = Path(config.encoded_dir(a.corpus))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("status") == "done":
        print(f"[encode] already done: {out}")
        return
    try:
        bb = ColBERTEncoder(attn_implementation="flash_attention_2").to("cuda:0")
    except (ImportError, TypeError, ValueError):
        bb = ColBERTEncoder(attn_implementation="sdpa").to("cuda:0")
    bb.doc_len = min(bb.doc_len, DOC_TOKENS)
    written, t0 = 0, time.time()
    with open(out / "vectors.f16.bin", "wb") as vf, open(out / "mask.u8.bin", "wb") as mf, \
            open(out / "passage_ids.i64.bin", "wb") as idf, open(out / "lengths.i32.bin", "wb") as lf:
        def flush(texts, ids):
            v, m = bb.encode(texts, False, "cuda:0")
            v, m = v[:, :DOC_TOKENS], m[:, :DOC_TOKENS]
            pad = DOC_TOKENS - v.shape[1]
            if pad > 0:
                v, m = torch.nn.functional.pad(v, (0, 0, 0, pad)), torch.nn.functional.pad(m, (0, pad))
            vf.write(v.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().tobytes())
            mf.write(m.cpu().numpy().astype(np.uint8).tobytes())
            idf.write(np.array(ids, dtype=np.int64).tobytes())
            lf.write(m.sum(dim=1).cpu().numpy().astype(np.int32).tobytes())
            return v.shape[0]
        ids, texts = [], []
        for pid, p in enumerate(passages(a.corpus)):
            ids.append(pid)
            texts.append(passage_text(p, a.corpus))
            if len(texts) >= a.batch_size:
                written += flush(texts, ids)
                ids, texts = [], []
        if texts:
            written += flush(texts, ids)
    (out / "meta.json").write_text(json.dumps({"status": "done", "n_local": written, "T_max": DOC_TOKENS, "P": DIM,
                                               "vector_dtype": "bfloat16-as-int16", "encode_seconds": time.time() - t0, "corpus": path(a.corpus)}, indent=2))
    print(f"[encode] {written} passages -> {out}", flush=True)


if __name__ == "__main__":
    main()
