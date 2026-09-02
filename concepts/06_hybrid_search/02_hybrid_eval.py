"""Stage 6b: hybrid retrieval, RRF and weighted fusion.

6a measured the ceiling: dense alone reaches recall@10 = 0.890, and perfect
fusion could reach 0.920. So there is +0.030 on the table, and no more.

This measures how much of it real fusion captures -- and, just as important,
what it COSTS. Fusion can lose queries that dense already had, by letting BM25
push a wrong document above a right one. The net is what matters, not the win
column.

Both retrievers' rankings are computed ONCE per query and reused across every
fusion setting, so the whole sweep costs one retrieval pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.fusion import reciprocal_rank_fusion, weighted_fusion  # noqa: E402
from rag.sparse import BM25  # noqa: E402

CANDIDATES = 100   # depth pulled from each retriever before fusing

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs], progress=True))
    bm25 = BM25(docs)

    # Precompute both rankings per query once.
    cache = {}
    for q in queries:
        qv = embedder.embed([q.question])[0]
        cache[q.qid] = (
            [(c.doc_id, s) for c, s in store.search(qv, k=CANDIDATES)],
            [(c.doc_id, s) for c, s in bm25.search(q.question, k=CANDIDATES)],
        )

    def make(fn):
        def retrieve(question, k, _fn=fn):
            qid = next(q.qid for q in queries if q.question == question)
            return _fn(*cache[qid])[:k]
        return retrieve

    configs = [
        ("dense only", lambda d, s: [i for i, _ in d]),
        ("bm25 only", lambda d, s: [i for i, _ in s]),
        ("RRF k=60", lambda d, s: reciprocal_rank_fusion(
            [[i for i, _ in d], [i for i, _ in s]], k=60)),
        ("RRF k=10", lambda d, s: reciprocal_rank_fusion(
            [[i for i, _ in d], [i for i, _ in s]], k=10)),
        ("RRF k=60 w=2:1", lambda d, s: reciprocal_rank_fusion(
            [[i for i, _ in d], [i for i, _ in s]], k=60, weights=[2.0, 1.0])),
        ("weighted 0.5/0.5", lambda d, s: weighted_fusion([d, s], [0.5, 0.5])),
        ("weighted 0.7/0.3", lambda d, s: weighted_fusion([d, s], [0.7, 0.3])),
        ("weighted 0.9/0.1", lambda d, s: weighted_fusion([d, s], [0.9, 0.1])),
    ]

    results = {}
    for label, fn in configs:
        results[label] = evaluate(queries, make(fn), ndcg_ks=(10,))

    base = results["dense only"].ndcg[10]
    print(f"\n{'config':<20} {'nDCG@10':>9} {'vs dense':>10} {'R@10':>8} {'MRR':>8}")
    print("-" * 60)
    for label, r in results.items():
        delta = "" if label == "dense only" else f"{r.ndcg[10] - base:+.4f}"
        print(f"{label:<20} {r.ndcg[10]:>9.4f} {delta:>10} {r.recall[10]:>8.4f} {r.mrr:>8.4f}")
    print("-" * 60)
    print(f"perfect-fusion recall@10 ceiling from 6a: 0.920")
