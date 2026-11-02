"""v2 Stage 10: late interaction with a model trained for it, on the full corpus.

v1 ran MaxSim over raw bge-base token vectors on an 800-document subset and
got +0.011 for 241x the index and ~28x the latency. Two things limited that
result: bge-base was never trained to make its token vectors comparable, and
768-dim float32 tokens forced the subset.

v2 uses GTE-ModernColBERT (trained for MaxSim, 128-dim tokens, float16), so the
whole 5,183-document corpus fits, and the comparison is against each
embedding model's dense retrieval on all 300 test queries. Two arms:

  colbert            full-corpus MaxSim retrieval
  dense -> colbert   MaxSim used as a reranker over dense top-100, the way
                     late interaction is usually deployed

Also reported: index size and median query latency for each.
"""

import statistics
import time

from _common import banner, embeddings_from_args

from rag.v2.bench import Bench
from rag.v2.colbert import ColbertIndex

STAGE = "s10_late_interaction"

if __name__ == "__main__":
    names = embeddings_from_args(__doc__)
    banner("Stage 10, building the ColBERT index (once, shared by all embeddings)")
    first = Bench(names[0])
    index = ColbertIndex()
    index.build(first.docs)
    pos = {d: i for i, d in enumerate(index.doc_ids)}

    runs, lat = {}, []
    for q in first.test:
        t = time.perf_counter()
        runs[q.qid] = [d for d, _ in index.search(q.question, k=100)]
        lat.append(time.perf_counter() - t)
    q_embs = {q.qid: index.model.encode([q.question], is_query=True, show_progress_bar=False)[0]
              for q in first.test}
    meta = {"index_mb": index.index_bytes() / 1e6, "median_query_ms": statistics.median(lat) * 1e3,
            "docs": len(index.doc_ids)}
    print(f"  index {meta['index_mb']:.0f} MB, median query {meta['median_query_ms']:.0f} ms", flush=True)

    for name in names:
        banner(f"Stage 10, {name}")
        bench = first if name == names[0] else Bench(name)
        vecs = bench.qvecs([q.question for q in bench.test])
        dense = {q.qid: bench.dense_vec(v, 100) for q, v in zip(bench.test, vecs)}
        dense_mb = bench.store.vectors.nbytes / 1e6
        bench.record(STAGE, "dense", {q: [d for d, _ in v] for q, v in dense.items()},
                     meta={"index_mb": dense_mb})
        bench.record(STAGE, "colbert (full corpus)", runs, meta=meta)

        import numpy as np
        rer = {}
        for q in bench.test:
            qe = np.asarray(q_embs[q.qid], dtype=np.float32)
            cand = [d for d, _ in dense[q.qid]]
            sc = [float((qe @ index.vectors[pos[d]].astype(np.float32).T).max(axis=1).sum())
                  for d in cand]
            rer[q.qid] = [cand[i] for i in np.argsort(-np.asarray(sc))]
        bench.record(STAGE, "dense -> colbert rerank (top 100)", rer,
                     meta={"index_mb": dense_mb + meta["index_mb"]})
