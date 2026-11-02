"""v2 Stage 5: BM25, with its two knobs tuned on train.

v1 used the textbook defaults (k1=1.2, b=0.75). Those are reasonable, but they
are a choice, and every later fusion and reranking result depends on how good
BM25 is. v2 keeps the defaults as one arm and adds a train-tuned arm.

BM25 does not depend on the embedding model, so this runs once and the result
is copied into each embedding's folder for the later comparisons.
"""

import itertools
import shutil

from _common import banner, embeddings_from_args

from rag.sparse import BM25
from rag.v2.bench import Bench

STAGE = "s05_bm25"

if __name__ == "__main__":
    names = embeddings_from_args(__doc__)
    banner("Stage 5, BM25 (embedding-independent)")
    bench = Bench(names[0])
    default = BM25(bench.docs)

    def rank_with(index, qs):
        return {q.qid: [c.doc_id for c, _ in index.search(q.question, k=100)] for q in qs}

    best, best_score = None, -1.0
    for k1, b in itertools.product([0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9]):
        idx = BM25(bench.docs, k1=k1, b=b)
        m = bench.mean(rank_with(idx, bench.train), "train")
        if m > best_score:
            best, best_score = (k1, b), m
    print(f"  tuned on train: k1={best[0]}, b={best[1]} (train nDCG@10 {best_score:.4f})", flush=True)

    bench.record(STAGE, "bm25 default (k1=1.2, b=0.75)", rank_with(default, bench.test))
    tuned = BM25(bench.docs, k1=best[0], b=best[1])
    bench.record(STAGE, "bm25 tuned", rank_with(tuned, bench.test), meta={"k1": best[0], "b": best[1]})

    src = bench.out / f"{STAGE}.json"
    for other in names[1:]:
        (bench.out.parent / other).mkdir(parents=True, exist_ok=True)
        shutil.copy(src, bench.out.parent / other / f"{STAGE}.json")
