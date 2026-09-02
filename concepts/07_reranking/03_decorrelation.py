"""Stage 7c: the rule refined by a FAILED prediction.

7b concluded "rerankers repair weak orderings, not strong ones", from bge-reranker
helping BM25 (+0.041) and hurting dense (-0.033).

Stage 8b tested that on multi-query, whose ordering is ALSO weak (MRR 0.662 vs dense's
0.668) but has higher recall. The rule predicted a gain. It lost 0.011.

So weakness is not the mechanism. The candidate explanation is DECORRELATION:
a reranker adds value when its judgement is INDEPENDENT of the first stage's,
and BM25 is the only first stage here built on a different signal (lexical term
matching) rather than on the same bi-encoder.

This measures it directly with Spearman rank correlation between each first
stage's ordering and the reranker's ordering over the same candidates.

PREDICTION:  BM25 correlates LEAST with the reranker (and gained).
             dense and multi-query correlate MORE (and both lost).
If instead all three correlate similarly, decorrelation is not the mechanism
either, and the honest answer is that we do not yet know.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_corpus, load_queries  # noqa: E402
from rag.fusion import reciprocal_rank_fusion  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402
from rag.sparse import BM25  # noqa: E402

SAMPLE_N, SEED, DEPTH = 50, 0, 50


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb)))


if __name__ == "__main__":
    qs = load_queries(split="test")
    random.Random(SEED).shuffle(qs)
    queries = sorted(qs[:SAMPLE_N], key=lambda q: int(q.qid))

    docs = load_corpus(with_title=True)
    by_id = {d.id: d for d in docs}
    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    bm25 = BM25(docs)
    cache = json.loads(Path("results/llm_transforms.json").read_text())
    reranker = CrossEncoderReranker(CrossEncoderReranker.BGE)

    stages = {"BM25": [], "dense": [], "multi-query": []}

    for q in queries:
        dense_hits = store.search(embedder.embed([q.question])[0], k=DEPTH)
        rankings = [[c.doc_id for c, _ in store.search(embedder.embed([v])[0], k=100)]
                    for v in cache[f"multi_query:{q.qid}"]]
        mq = reciprocal_rank_fusion(rankings, k=60)[:DEPTH]

        candidates = {
            "BM25": [(c.doc_id, s) for c, s in bm25.search(q.question, k=DEPTH)],
            "dense": [(c.doc_id, s) for c, s in dense_hits],
            # RRF scores are not returned, so use inverted rank as the ordering signal.
            "multi-query": [(d, -i) for i, d in enumerate(mq)],
        }

        for name, cand in candidates.items():
            if len(cand) < 5:
                continue
            rr = {c.doc_id: s for c, s in reranker.rerank(
                q.question, [(by_id[i], 0.0) for i, _ in cand], top_k=len(cand))}
            stages[name].append(spearman([s for _, s in cand],
                                         [rr[i] for i, _ in cand]))

    print("\n" + "=" * 72)
    print(f"{'first stage':<16} {'Spearman vs reranker':>22} {'rerank delta':>16}")
    print("-" * 72)
    known = {"BM25": "+0.0406", "dense": "-0.0332", "multi-query": "-0.0109"}
    for name, vals in stages.items():
        print(f"{name:<16} {np.mean(vals):>22.3f} {known[name]:>16}")
    print("=" * 72)
    print("\nIf BM25 correlates least AND is the only one that gained, decorrelation\n"
          "is the mechanism and 'weak ordering' was a proxy for it.")
