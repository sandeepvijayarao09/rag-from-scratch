"""Stage 7: two-stage retrieval with a cross-encoder reranker.

ENGINEERING NOTE, which is itself the lesson. The naive way to sweep depth is a
rerank pass per depth: top-20, then top-50, then top-100. That triples the work
for no reason, because the top-20 scores are a SUBSET of the top-100 scores.

So: score every (query, candidate) pair ONCE at max depth, cache it, and derive
every shallower config by slicing. Same principle as the embedding cache in
Stage 2 -- a cross-encoder score is a pure function of (model, query, document).
Measured cost of getting this wrong: 27 minutes instead of 12.

Sweeps the two decisions that matter:

  DEPTH  -- candidates handed over by stage one. Deeper raises the ceiling the
            reranker can reach and costs linearly more forward passes. The
            first stage's recall@depth is a HARD CAP: the reranker can only
            reorder what it was given, never recover a document stage one missed.

  MODEL  -- MiniLM-L6 (90MB, 6 layers, 264 pairs/s) vs bge-reranker-base
            (1.1GB, 12 layers, 48 pairs/s). A 5.5x latency difference.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.fusion import weighted_fusion  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402
from rag.sparse import BM25  # noqa: E402

MAX_DEPTH = 100
DEPTHS = [10, 20, 50, 100]

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")
    by_id = {d.id: d for d in docs}

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    bm25 = BM25(docs)

    print("building first-stage candidates...", flush=True)
    dense_c, hybrid_c = {}, {}
    for q in queries:
        qv = embedder.embed([q.question])[0]
        d = [(c.doc_id, s) for c, s in store.search(qv, k=MAX_DEPTH)]
        s = [(c.doc_id, sc) for c, sc in bm25.search(q.question, k=MAX_DEPTH)]
        dense_c[q.qid] = d
        hybrid_c[q.qid] = [(i, 0.0) for i in weighted_fusion([d, s], [0.7, 0.3])][:MAX_DEPTH]

    qid_of = {q.question: q.qid for q in queries}
    results = {}

    def first_stage(candidates, depth):
        def retrieve(question, k):
            return [i for i, _ in candidates[qid_of[question]][:depth]][:k]
        return retrieve

    print("\n--- first stages, no reranking ---", flush=True)
    for label, cand in [("dense", dense_c), ("hybrid 0.7/0.3", hybrid_c)]:
        r = evaluate(queries, first_stage(cand, MAX_DEPTH), ks=(1, 3, 5, 10, 100))
        results[label] = (r, 0.0)
        print(f"{label:<36} {r}", flush=True)
        print(f"{'':36} recall@{MAX_DEPTH}={r.recall[100]:.4f}  <- reranker ceiling")

    for name, model in [("MiniLM-L6", CrossEncoderReranker.MINI),
                        ("bge-base", CrossEncoderReranker.BGE)]:
        print(f"\n--- {name} ---", flush=True)
        reranker = CrossEncoderReranker(model)

        # Score the UNION of everything any config could ask for, once.
        cache: dict[tuple[str, str], float] = {}
        start = time.perf_counter()
        for n, q in enumerate(queries, 1):
            need = list(dict.fromkeys(
                [i for i, _ in dense_c[q.qid]] + [i for i, _ in hybrid_c[q.qid]]))
            scored = reranker.rerank(q.question, [(by_id[i], 0.0) for i in need],
                                     top_k=len(need))
            for chunk, sc in scored:
                cache[(q.qid, chunk.doc_id)] = sc
            if n % 100 == 0:
                print(f"  scored {n}/{len(queries)} queries "
                      f"({time.perf_counter()-start:.0f}s)", flush=True)
        pairs = len(cache)
        elapsed = time.perf_counter() - start
        print(f"  {pairs:,} pairs in {elapsed:.0f}s ({pairs/elapsed:.0f}/s), "
              f"{1000*elapsed/len(queries):.0f}ms/query at depth {MAX_DEPTH}", flush=True)

        def reranked(candidates, depth, _c=cache):
            def retrieve(question, k):
                qid = qid_of[question]
                ids = [i for i, _ in candidates[qid][:depth]]
                return sorted(ids, key=lambda i: -_c[(qid, i)])[:k]
            return retrieve

        for depth in DEPTHS:
            r = evaluate(queries, reranked(dense_c, depth), ks=(1, 3, 5, 10, 100))
            ms = 1000 * elapsed / len(queries) * depth / MAX_DEPTH
            results[f"dense top-{depth} -> {name}"] = (r, ms)
            print(f"dense top-{depth:<3} -> {name:<12} {r}", flush=True)

        r = evaluate(queries, reranked(hybrid_c, 50), ks=(1, 3, 5, 10, 100))
        results[f"hybrid top-50 -> {name}"] = (r, 1000 * elapsed / len(queries) * 0.5)
        print(f"hybrid top-50 -> {name:<12} {r}", flush=True)

    base = results["dense"][0].ndcg[10]
    print("\n" + "=" * 86)
    print(f"{'config':<34} {'nDCG@10':>9} {'vs dense':>10} {'R@10':>8} {'MRR':>8} {'ms/q':>8}")
    print("-" * 86)
    for label, (r, ms) in sorted(results.items(), key=lambda kv: -kv[1][0].ndcg[10]):
        d = "" if label == "dense" else f"{r.ndcg[10] - base:+.4f}"
        print(f"{label:<34} {r.ndcg[10]:>9.4f} {d:>10} {r.recall[10]:>8.4f} "
              f"{r.mrr:>8.4f} {ms:>8.0f}")
    print("=" * 86)
