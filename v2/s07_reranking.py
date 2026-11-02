"""v2 Stage 7: cross-encoder reranking, tested against its own rule.

v1's most useful result: a reranker *overwrites* the first-stage ordering, so
the outcome is the reranker's quality, not a blend. bge-reranker-base helped on
BM25 (+0.041) and hurt on dense (-0.033). The rule it implies is testable with
more models: a reranker should help exactly when it is better than the stage it
reorders.

So v2 crosses three rerankers (v1's base model, bge-reranker-v2-m3,
Qwen3-Reranker-4B) with each first stage (BM25 and each embedding model's
dense retrieval), all at depth 50, and reports two modes:

  replace      the reranker's order, as in v1
  interpolate  the repair: blend normalised reranker and first-stage scores,
               with the blend weight tuned on a 150-query train subsample

Reranker scores are cached on (query, document text), so the same pair is
never scored twice across first stages or embedding models.
"""

import time

from _common import banner, embeddings_from_args

from rag.fusion import weighted_fusion
from rag.v2.bench import Bench, subsample
from rag.v2.models import RERANKER_ORDER
from rag.v2.rerankers import Reranker

STAGE = "s07_reranking"
DEPTH, TRAIN_N = 50, 150


def rerank_all(bench, rr, first, qs):
    out, start = {}, time.perf_counter()
    for i, q in enumerate(qs, 1):
        cand = [(d, bench.by_id[d].text) for d, _ in first[q.qid][:DEPTH]]
        out[q.qid] = rr.rerank(q.question, cand)
        if i % 50 == 0:
            print(f"    {i}/{len(qs)} ({time.perf_counter() - start:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    for name in embeddings_from_args(__doc__):
        banner(f"Stage 7, {name}")
        bench = Bench(name)
        train = subsample(bench.train, TRAIN_N)
        firsts = {}
        for split, qs in [("test", bench.test), ("train", train)]:
            vecs = bench.qvecs([q.question for q in qs])
            firsts[split] = {
                "dense": {q.qid: bench.dense_vec(v, DEPTH) for q, v in zip(qs, vecs)},
                "bm25": {q.qid: bench.sparse(q.question, DEPTH) for q in qs},
            }
        for fs in ["dense", "bm25"]:
            bench.record(STAGE, f"{fs} (no rerank)",
                         {q: [d for d, _ in v] for q, v in firsts["test"][fs].items()})

        for rname in RERANKER_ORDER:
            rr = Reranker(rname)
            print(f"  reranker {rname}", flush=True)
            for fs in ["dense", "bm25"]:
                scored = {split: rerank_all(bench, rr, firsts[split][fs],
                                            bench.test if split == "test" else train)
                          for split in ["test", "train"]}
                bench.record(STAGE, f"{fs} -> {rname} (replace)",
                             {q: [d for d, _ in v] for q, v in scored["test"].items()},
                             meta={"depth": DEPTH})

                def blend(a, qs, split):
                    return {q.qid: weighted_fusion([firsts[split][fs][q.qid], scored[split][q.qid]],
                                                   [1 - a, a])[:10] for q in qs}
                a, tr = bench.tune(lambda p, qs: blend(p, qs, "train"),
                                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], queries=train)
                bench.record(STAGE, f"{fs} -> {rname} (interpolate, tuned)",
                             blend(a, bench.test, "test"),
                             meta={"depth": DEPTH, "reranker_weight": a, "train_ndcg10": tr,
                                   "train_n": len(train)})
