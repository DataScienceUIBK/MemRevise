"""MemRevise: a revisable memory for conversational search.

    memrevise.write_memory   the writer reads each turn once and updates the memory; at the final turn it writes the search query
    memrevise.search         ranks the passages for the final question from the memory and the query
    memrevise.evaluate       MRR@10, NDCG@3, Recall@10/100 and stale-win per set and tier; seed means and paired intervals
    memrevise.train          trains the writer (LoRA on Ministral-3-3B)
    memrevise.encode_corpus  encodes a passage collection with the frozen Reason-ModernColBERT
"""
