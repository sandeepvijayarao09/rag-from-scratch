"""Combining rankings from multiple retrievers.

The hard part of hybrid search is not running both retrievers, it is that their
scores are not comparable. Cosine similarity lives in [-1, 1] and clusters
tightly around 0.7-0.9 for anything plausible. BM25 is unbounded, corpus- and
query-dependent, and a score of 18 means nothing on its own. Adding them
directly is meaningless.

Two ways out:

RECIPROCAL RANK FUSION (RRF)
    score(d) = SUM over retrievers r:  1 / (k + rank_r(d))

    Throws the scores away entirely and uses only RANK. That sidesteps
    normalisation completely, needs no tuning, and is robust when one retriever
    is much better calibrated than the other. k=60 is the value from the
    original paper; it damps the difference between rank 1 and rank 2 so a
    single confident retriever cannot dominate.

    This is the default for a reason: nothing to tune, hard to get wrong.

WEIGHTED SCORE FUSION
    Normalise each retriever's scores to [0,1] per query, then blend with a
    weight. Keeps magnitude information that RRF discards -- the gap between
    rank 1 and rank 2 is real signal and RRF ignores it. In exchange you now
    have a weight to tune and a normalisation scheme that can misbehave when
    one retriever returns a flat score distribution.
"""


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    span = hi - lo
    # A flat distribution carries no ranking information; treat it as neutral
    # rather than dividing by zero or amplifying noise.
    return [0.5] * len(scores) if span < 1e-9 else [(s - lo) / span for s in scores]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60,
                           weights: list[float] | None = None) -> list[str]:
    """`rankings` is one ranked list of doc ids per retriever, best first."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def weighted_fusion(scored: list[list[tuple[str, float]]],
                    weights: list[float] | None = None) -> list[str]:
    """`scored` is one list of (doc_id, score) per retriever, best first."""
    weights = weights or [1.0] * len(scored)
    fused: dict[str, float] = {}
    for results, weight in zip(scored, weights):
        ids = [d for d, _ in results]
        for doc_id, norm in zip(ids, _minmax([s for _, s in results])):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * norm
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
