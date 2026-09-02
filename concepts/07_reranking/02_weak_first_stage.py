"""Stage 7b: when DOES a reranker help?

7a found both rerankers hurting on top of dense retrieval, and the sanity check
ruled out a bug: bge-reranker separates gold from non-gold cleanly (+0.575 vs
+0.055) but moves the gold document up as often as down.

The explanation: dense already places gold at mean rank 2.33 out of 20. There is
almost nothing left to repair, and the reranker's judgement is CORRELATED with
the bi-encoder's rather than superior to it -- both are BGE models trained on
similar data. Adding a second correlated opinion adds variance, not accuracy.

That explanation makes a prediction: give the reranker a WEAK first stage and it
should help a lot. BM25 scores 0.665 nDCG@10 versus dense's 0.759, and it fails
on different queries (Stage 6), so its ordering has much more room to improve
and much less correlation with the reranker.

If reranking BM25 shows a large gain, the explanation holds and the rule is
"rerankers repair weak orderings, not strong ones". If it also degrades, the
explanation is wrong and something about this dataset defeats reranking.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402
from rag.sparse import BM25  # noqa: E402

DEPTH = 50

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")
    by_id = {d.id: d for d in docs}
    qid_of = {q.question: q.qid for q in queries}

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    bm25 = BM25(docs)

    stages = {
        "BM25 (weak)": {q.qid: [c.doc_id for c, _ in bm25.search(q.question, k=DEPTH)]
                        for q in queries},
        "dense (strong)": {},
    }
    for q in queries:
        qv = embedder.embed([q.question])[0]
        stages["dense (strong)"][q.qid] = [c.doc_id for c, _ in store.search(qv, k=DEPTH)]

    reranker = CrossEncoderReranker(CrossEncoderReranker.BGE)
    print(f"reranking depth {DEPTH}, bge-reranker-base\n", flush=True)

    rows = []
    for name, cand in stages.items():
        before = evaluate(queries, lambda qs, k, _c=cand: _c[qid_of[qs]][:k], ndcg_ks=(10,))

        cache = {}
        start = time.perf_counter()
        for q in queries:
            for chunk, s in reranker.rerank(q.question,
                                            [(by_id[i], 0.0) for i in cand[q.qid]],
                                            top_k=DEPTH):
                cache[(q.qid, chunk.doc_id)] = s
        print(f"  {name}: scored in {time.perf_counter()-start:.0f}s", flush=True)

        def rr(question, k, _c=cand, _s=cache):
            qid = qid_of[question]
            return sorted(_c[qid], key=lambda i: -_s[(qid, i)])[:k]

        after = evaluate(queries, rr, ndcg_ks=(10,))
        rows.append((name, before, after))

    print("\n" + "=" * 78)
    print(f"{'first stage':<18} {'before':>9} {'after rerank':>14} {'delta':>10} {'MRR delta':>12}")
    print("-" * 78)
    for name, b, a in rows:
        print(f"{name:<18} {b.ndcg[10]:>9.4f} {a.ndcg[10]:>14.4f} "
              f"{a.ndcg[10]-b.ndcg[10]:>+10.4f} {a.mrr-b.mrr:>+12.4f}")
    print("=" * 78)
