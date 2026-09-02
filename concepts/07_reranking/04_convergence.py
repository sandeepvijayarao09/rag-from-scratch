"""Stage 7d: two dead ends, then the simple answer.

  7b hypothesis: "rerankers repair WEAK orderings."
     Falsified by Stage 8b -- multi-query's ordering is weak and reranking still lost.

  7c hypothesis: "rerankers add value when DECORRELATED from the first stage."
     Falsified by 7c itself -- BM25 correlated MOST with the reranker (0.442)
     and was the only stage that gained.

The pattern both hypotheses missed is in the outputs, not the inputs. Three
first stages spanning 0.665 to 0.759 nDCG all land at 0.69-0.73 after reranking.

  SIMPLE HYPOTHESIS: reranking REPLACES the first stage's ordering with the
  reranker's own. So the result is the RERANKER's ranking quality, bounded by
  what the candidate set contains. The first stage stops mattering as an
  ordering and matters only as a recall filter.

  Prediction: reranked nDCG should track candidate-set RECALL and be roughly
  independent of the first stage's own nDCG.

This is measured with all three stages on the same queries at the same depth,
reporting first-stage nDCG, candidate recall, and reranked nDCG together.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.fusion import reciprocal_rank_fusion  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402
from rag.sparse import BM25  # noqa: E402

SAMPLE_N, SEED, DEPTH = 50, 0, 50

if __name__ == "__main__":
    qs = load_queries(split="test")
    random.Random(SEED).shuffle(qs)
    queries = sorted(qs[:SAMPLE_N], key=lambda q: int(q.qid))
    qid_of = {q.question: q.qid for q in queries}

    docs = load_corpus(with_title=True)
    by_id = {d.id: d for d in docs}
    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    bm25 = BM25(docs)
    tcache = json.loads(Path("results/llm_transforms.json").read_text())

    stages = {"BM25": {}, "dense": {}, "multi-query": {}}
    for q in queries:
        stages["BM25"][q.qid] = [c.doc_id for c, _ in bm25.search(q.question, k=DEPTH)]
        stages["dense"][q.qid] = [c.doc_id for c, _ in
                                  store.search(embedder.embed([q.question])[0], k=DEPTH)]
        rankings = [[c.doc_id for c, _ in store.search(embedder.embed([v])[0], k=100)]
                    for v in tcache[f"multi_query:{q.qid}"]]
        stages["multi-query"][q.qid] = reciprocal_rank_fusion(rankings, k=60)[:DEPTH]

    reranker = CrossEncoderReranker(CrossEncoderReranker.BGE)
    rows = []
    for name, cand in stages.items():
        before = evaluate(queries, lambda s, k, _c=cand: _c[qid_of[s]][:k],
                          ks=(1, 10, DEPTH), ndcg_ks=(10,))
        scores = {}
        for q in queries:
            for c, s in reranker.rerank(q.question,
                                        [(by_id[i], 0.0) for i in cand[q.qid]],
                                        top_k=DEPTH):
                scores[(q.qid, c.doc_id)] = s
        after = evaluate(queries, lambda s, k, _c=cand, _s=scores: sorted(
            _c[qid_of[s]], key=lambda i: -_s[(qid_of[s], i)])[:k],
            ks=(1, 10, DEPTH), ndcg_ks=(10,))
        rows.append((name, before, after))
        print(f"  {name} done", flush=True)

    print("\n" + "=" * 84)
    print(f"{'first stage':<14} {'stage nDCG':>11} {'cand recall@' + str(DEPTH):>16} "
          f"{'reranked nDCG':>14} {'delta':>10}")
    print("-" * 84)
    for name, b, a in sorted(rows, key=lambda r: r[1].ndcg[10]):
        print(f"{name:<14} {b.ndcg[10]:>11.4f} {b.recall[DEPTH]:>16.4f} "
              f"{a.ndcg[10]:>14.4f} {a.ndcg[10]-b.ndcg[10]:>+10.4f}")
    print("=" * 84)
    spread_in = max(r[1].ndcg[10] for r in rows) - min(r[1].ndcg[10] for r in rows)
    spread_out = max(r[2].ndcg[10] for r in rows) - min(r[2].ndcg[10] for r in rows)
    print(f"\nfirst-stage nDCG spread : {spread_in:.4f}")
    print(f"reranked  nDCG spread   : {spread_out:.4f}")
    print("\nIf the output spread is much smaller than the input spread, reranking\n"
          "is overwriting the ordering with the reranker's own, and the first stage\n"
          "matters only for what it puts INSIDE the candidate set.")
