"""Turning chunk hits into document rankings.

The moment you chunk, your index and your relevance judgements stop speaking
the same language. The store returns CHUNKS; BEIR judges DOCUMENTS. Something
has to pool one into the other, and that pooling is a real design decision,
not glue code -- it changes your scores.

  max  -- a document scores as its single best chunk. The standard choice, and
          the right default: one strongly-matching passage is exactly what you
          want, and it doesn't punish long documents for having filler.
  sum  -- add every matching chunk's score. Rewards documents that match in
          several places, but systematically favours LONG documents, which have
          more chunks and therefore more chances to accumulate score.
  mean -- average the chunk scores. Overcorrects the other way: one perfect
          passage gets diluted by the rest of the document.

OVERFETCH is the other trap. To rank k documents you must fetch more than k
chunks, because several top chunks routinely come from the SAME document --
especially with overlap, where near-duplicate neighbours crowd the top of the
list. Fetch exactly k chunks and you can silently return fewer than k docs,
which quietly caps your recall@k. Fetching k * overfetch is cheap insurance.
"""


def doc_level_retriever(store, embedder, prefix: str = "", pool: str = "max",
                        overfetch: int = 5):
    """Return `retrieve(question, k) -> list of doc_ids`, best first."""
    assert pool in ("max", "sum", "mean")

    def retrieve(question: str, k: int) -> list[str]:
        query_vector = embedder.embed([question], prefix=prefix)[0]
        hits = store.search(query_vector, k=k * overfetch)

        scores: dict[str, list[float]] = {}
        for chunk, score in hits:
            scores.setdefault(chunk.doc_id, []).append(score)

        pooled = {
            doc_id: max(s) if pool == "max" else sum(s) if pool == "sum" else sum(s) / len(s)
            for doc_id, s in scores.items()
        }
        return [d for d, _ in sorted(pooled.items(), key=lambda kv: -kv[1])][:k]

    return retrieve
