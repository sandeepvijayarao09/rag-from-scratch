"""Stage 8b: composing a recall technique with a precision technique.

The prediction comes from two earlier results read together.

  Stage 8a: multi-query raised recall@10 from 0.8433 to 0.8833 while nDCG
             stayed flat and MRR dipped. It FINDS more gold documents and
             ranks them no better. RRF fuses four rankings by rank alone, which
             flattens the ordering.

  Stage 7b: a reranker HELPED a weak ordering (BM25, +0.041) and HURT a strong
             one (dense, -0.033). Its value is marginal information over the
             first stage, not absolute quality.

Multi-query output is exactly the profile that Stage 7b says to rerank: high recall,
weak ordering, and an ordering produced by rank-fusion rather than by the
bi-encoder, so it is far less correlated with the reranker's judgement.

PREDICTION: multi-query + rerank beats both alone, and beats dense + rerank,
because the reranker finally has both room to improve and documents worth
promoting.

FALSIFICATION: if it lands at or below dense + rerank (0.7256 on the full set),
the Stage 7b rule is not the whole story and something else drives when reranking
works.

Four arms, same 50 paired queries:
    dense                     strong ordering, baseline recall
    dense       -> rerank     Stage 7a said this hurts
    multi-query               high recall, weak ordering
    multi-query -> rerank     the prediction
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.fusion import reciprocal_rank_fusion  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402

SAMPLE_N, SEED, DEPTH = 50, 0, 50
TRANSFORMS = Path("results/llm_transforms.json")

if __name__ == "__main__":
    qs = load_queries(split="test")
    random.Random(SEED).shuffle(qs)
    queries = sorted(qs[:SAMPLE_N], key=lambda q: int(q.qid))
    qid_of = {q.question: q.qid for q in queries}

    docs = load_corpus(with_title=True)
    by_id = {d.id: d for d in docs}
    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    cache = json.loads(TRANSFORMS.read_text())

    # Two first stages, same depth.
    dense_c, mq_c = {}, {}
    for q in queries:
        dense_c[q.qid] = [c.doc_id for c, _ in
                          store.search(embedder.embed([q.question])[0], k=DEPTH)]
        rankings = [[c.doc_id for c, _ in store.search(embedder.embed([v])[0], k=100)]
                    for v in cache[f"multi_query:{q.qid}"]]
        mq_c[q.qid] = reciprocal_rank_fusion(rankings, k=60)[:DEPTH]

    reranker = CrossEncoderReranker(CrossEncoderReranker.BGE)
    rows = []

    for name, cand in [("dense", dense_c), ("multi-query", mq_c)]:
        before = evaluate(queries, lambda qs_, k, _c=cand: _c[qid_of[qs_]][:k],
                          ndcg_ks=(10,))

        scores, start = {}, time.perf_counter()
        for q in queries:
            for c, s in reranker.rerank(q.question,
                                        [(by_id[i], 0.0) for i in cand[q.qid]],
                                        top_k=DEPTH):
                scores[(q.qid, c.doc_id)] = s
        print(f"  reranked {name} in {time.perf_counter()-start:.0f}s", flush=True)

        after = evaluate(queries, lambda qs_, k, _c=cand, _s=scores: sorted(
            _c[qid_of[qs_]], key=lambda i: -_s[(qid_of[qs_], i)])[:k], ndcg_ks=(10,))
        rows.append((name, before, after))

    print("\n" + "=" * 80)
    print(f"{'first stage':<16} {'':<10} {'nDCG@10':>9} {'R@10':>8} {'MRR':>8} {'rerank Δ':>11}")
    print("-" * 80)
    for name, b, a in rows:
        print(f"{name:<16} {'alone':<10} {b.ndcg[10]:>9.4f} {b.recall[10]:>8.4f} "
              f"{b.mrr:>8.4f} {'':>11}")
        print(f"{'':<16} {'+ rerank':<10} {a.ndcg[10]:>9.4f} {a.recall[10]:>8.4f} "
              f"{a.mrr:>8.4f} {a.ndcg[10]-b.ndcg[10]:>+11.4f}")
    print("=" * 80)
    print(f"n={SAMPLE_N} paired subsample, rerank depth {DEPTH}, bge-reranker-base.")
    print("\nPrediction: the rerank delta is NEGATIVE for dense and POSITIVE for\n"
          "multi-query, because Stage 7b says rerankers repair weak orderings only.")
