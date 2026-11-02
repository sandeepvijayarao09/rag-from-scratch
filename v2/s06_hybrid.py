"""v2 Stage 6: hybrid retrieval, and when RRF earns its default status.

v1 found RRF hurt (every variant) and weighted 0.7/0.3 helped by +0.010, with
the explanation that RRF trusts a much weaker BM25 as much as dense. That
explanation predicts something v1 could not test with one embedding model: the
weaker the dense model, the more fusion should help. v2 runs the same sweep on
all three.

Arms:
  6a  complementarity: how many queries only BM25 finds (the ceiling for fusion)
  6b  the v1 sweep, unchanged: RRF k=60, k=10, k=60 at 2:1, weighted 0.5/0.7/0.9
  6c  the repairs, tuned on train: weighted RRF (BM25 weight and k), and weighted
      score fusion with the weight tuned instead of hand-picked
"""

from _common import banner, embeddings_from_args

from rag.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.v2.bench import Bench, recall10

STAGE = "s06_hybrid"
DEPTH = 100


def candidates(bench, qs):
    vecs = bench.qvecs([q.question for q in qs])
    return {q.qid: (bench.dense_vec(v, DEPTH), bench.sparse(q.question, DEPTH)) for q, v in zip(qs, vecs)}


def rrf(pair, k, w_bm25=1.0):
    d, s = pair
    return reciprocal_rank_fusion([[i for i, _ in d], [i for i, _ in s]], k=k, weights=[1.0, w_bm25])


def wsum(pair, w_dense):
    return weighted_fusion(list(pair), [w_dense, 1 - w_dense])


if __name__ == "__main__":
    for name in embeddings_from_args(__doc__):
        banner(f"Stage 6, {name}")
        bench = Bench(name)
        test, train = candidates(bench, bench.test), candidates(bench, bench.train)

        # 6a complementarity at recall@10
        both = dense_only = bm25_only = neither = 0
        for q in bench.test:
            d = recall10([i for i, _ in test[q.qid][0]], q.relevant) > 0
            s = recall10([i for i, _ in test[q.qid][1]], q.relevant) > 0
            both += d and s; dense_only += d and not s; bm25_only += s and not d
            neither += not d and not s
        meta = {"both": both, "dense_only": dense_only, "bm25_only": bm25_only, "neither": neither}
        print(f"  found by: both {both}, dense only {dense_only}, BM25 only {bm25_only}, "
              f"neither {neither}", flush=True)

        # 6b the v1 sweep
        bench.record(STAGE, "dense", {q: [i for i, _ in p[0]] for q, p in test.items()}, meta=meta)
        bench.record(STAGE, "bm25", {q: [i for i, _ in p[1]] for q, p in test.items()})
        for label, fn in [("RRF k=60", lambda p: rrf(p, 60)), ("RRF k=10", lambda p: rrf(p, 10)),
                          ("RRF k=60 w=2:1", lambda p: rrf(p, 60, 0.5)),
                          ("weighted 0.5/0.5", lambda p: wsum(p, 0.5)),
                          ("weighted 0.7/0.3", lambda p: wsum(p, 0.7)),
                          ("weighted 0.9/0.1", lambda p: wsum(p, 0.9))]:
            bench.record(STAGE, label, {q: fn(p) for q, p in test.items()})

        # 6c repairs, tuned on train only
        grid = [(w, k) for w in [0.1, 0.2, 0.3, 0.5, 0.7] for k in [5, 10, 20, 60]]
        (w, k), tr = bench.tune(lambda prm, qs: {q.qid: rrf(train[q.qid], prm[1], prm[0]) for q in qs},
                                grid)
        bench.record(STAGE, "repaired: weighted RRF (tuned)",
                     {q: rrf(p, k, w) for q, p in test.items()},
                     meta={"bm25_weight": w, "k": k, "train_ndcg10": tr})
        wd, tr = bench.tune(lambda prm, qs: {q.qid: wsum(train[q.qid], prm) for q in qs},
                            [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        bench.record(STAGE, "repaired: weighted fusion (tuned)",
                     {q: wsum(p, wd) for q, p in test.items()},
                     meta={"dense_weight": wd, "train_ndcg10": tr})
