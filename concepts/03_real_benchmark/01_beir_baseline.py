"""Stage 3: a document-level baseline on a real benchmark.

NO CHUNKING YET. Each SciFact abstract is embedded whole. Two jobs:

1. VALIDATE THE PIPELINE. bge-base-en-v1.5 has published SciFact numbers.
   If we land near them, the embedding call, the normalisation, the top-k and
   the nDCG implementation are all probably right. If we land far off, we have
   a bug -- and finding that out now is worth more than any technique
   later in the ladder, because every later result inherits this code.

2. ESTABLISH THE REFERENCE POINT that everything after must beat. "Chunking helped"
   is only meaningful against "didn't chunk at all".

We also run two ablations that look like trivia and are not:

  title    -- index "title. text" vs text alone
  prefix   -- the BGE query instruction, on queries only

Both are conventions baked into published numbers. Skipping either silently
costs you points and makes your pipeline look broken when it isn't.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import BGE_QUERY_PREFIX, Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402

CONFIGS = [
    ("title + prefix   (BEIR convention)", True, True),
    ("title, no prefix", True, False),
    ("no title, prefix", False, True),
    ("no title, no prefix", False, False),
]

if __name__ == "__main__":
    queries = load_queries(split="test")
    print(f"SciFact: {len(load_corpus())} docs, {len(queries)} test queries")
    print(f"relevant docs per query: {sum(len(q.relevant) for q in queries) / len(queries):.2f}\n")

    embedder = Embedder()
    results = {}

    for label, with_title, with_prefix in CONFIGS:
        docs = load_corpus(with_title=with_title)
        print(f"[{label}]")

        start = time.perf_counter()
        store = VectorStore(docs, embedder.embed([d.text for d in docs], progress=True))
        index_time = time.perf_counter() - start

        prefix = BGE_QUERY_PREFIX if with_prefix else ""

        def retrieve(question: str, k: int) -> list[str]:
            qv = embedder.embed([question], prefix=prefix)[0]
            return [c.doc_id for c, _ in store.search(qv, k=k)]

        start = time.perf_counter()
        report = evaluate(queries, retrieve, ndcg_ks=(10,))
        results[label] = report
        print(f"  {report}")
        print(f"  index {index_time:.1f}s | eval {time.perf_counter() - start:.1f}s\n")

    print("=" * 78)
    print(f"{'config':<38} {'nDCG@10':>9} {'R@10':>8} {'MRR':>8}")
    print("-" * 78)
    for label, r in sorted(results.items(), key=lambda kv: -kv[1].ndcg[10]):
        print(f"{label:<38} {r.ndcg[10]:>9.4f} {r.recall[10]:>8.4f} {r.mrr:>8.4f}")
    print("=" * 78)
    print("\nCompare the top row against the published bge-base-en-v1.5 SciFact\n"
          "nDCG@10 on the MTEB/BEIR leaderboard. Close => pipeline is sound.")
