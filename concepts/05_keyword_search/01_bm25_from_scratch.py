"""Stage 5: keyword search (BM25), from scratch.

Scored on the same 300 SciFact queries as the Stage 3 dense baseline, so the
numbers are directly comparable.

What to look for: BM25 is a 1994 algorithm with no learned parameters, running
in pure Python. If it lands anywhere near a modern neural bi-encoder, that tells
you something important about how much of retrieval is just term matching --
and it explains why hybrid works, because the two methods fail on DIFFERENT
queries rather than the same ones.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import evaluate, load_corpus, load_queries  # noqa: E402
from rag.sparse import BM25  # noqa: E402

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")

    start = time.perf_counter()
    bm25 = BM25(docs)
    index_time = time.perf_counter() - start
    print(f"BM25 index: {len(bm25)} docs, {len(bm25.postings):,} unique terms, "
          f"{index_time:.2f}s  (no GPU, no model)\n")

    def retrieve(question: str, k: int) -> list[str]:
        return [c.doc_id for c, _ in bm25.search(question, k=k)]

    start = time.perf_counter()
    report = evaluate(queries, retrieve, ndcg_ks=(10,))
    elapsed = time.perf_counter() - start

    print(f"BM25          {report}")
    print(f"              {1000 * elapsed / len(queries):.1f} ms/query\n")
    print("Stage 3 dense baselines for comparison:")
    print("  bge-base title, no prefix    nDCG@10=0.7588  R@10=0.8802  MRR=0.7246")
    print("  bge-base title + prefix      nDCG@10=0.7449  R@10=0.8709  MRR=0.7105")
