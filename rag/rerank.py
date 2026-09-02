"""Cross-encoder reranking -- the second stage of two-stage retrieval.

THE ARCHITECTURAL DIFFERENCE, which is the whole point:

  Bi-encoder (what Stages 2-6 used)
      embed(query) and embed(doc) are computed SEPARATELY, then compared with
      a dot product. The document never sees the query. That is what lets you
      precompute the whole corpus once and search it in milliseconds -- and
      also what limits it. All the interaction between query and document has
      to survive being compressed into a single 768-dim vector each.

  Cross-encoder (this file)
      [query, doc] go through the transformer TOGETHER, so self-attention can
      compare individual terms directly, and the model outputs one relevance
      score. Far more accurate. Also impossible to precompute: you pay a full
      forward pass per (query, document) pair.

That cost is why reranking is a SECOND stage. Scoring 5,183 documents per query
is out of the question; scoring the top 50 from a cheap retriever is routine.
Stage one optimises recall (get the right doc in the candidate set at all),
stage two optimises precision (get it to rank 1).

WHY THIS SHOULD WORK HERE, predicted before running it: Stage 2 found r@1
collapsing while r@5 held, and Stage 3 shows dense recall@10 = 0.880 against
nDCG@10 = 0.759. The right document is usually already retrieved and simply
ranked too low. That is the exact failure a reranker repairs, and the exact
failure a better bi-encoder does not.
"""

import numpy as np


class CrossEncoderReranker:
    # Small and fast vs larger and stronger. Measuring both is the point:
    # reranker choice is a quality/latency decision, not a default.
    MINI = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~90MB, 6 layers
    BGE = "BAAI/bge-reranker-base"                  # ~1.1GB, 12 layers

    def __init__(self, model_name: str = MINI, device: str | None = None):
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, candidates: list, top_k: int = 10,
               batch_size: int = 64) -> list:
        """`candidates` is [(Chunk, first_stage_score)]; returns the same shape,
        reordered, carrying the CROSS-ENCODER score instead."""
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]
