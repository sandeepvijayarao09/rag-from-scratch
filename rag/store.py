"""A brute-force vector store, done properly.

Still O(N) per query -- we have not changed the algorithm. What changes is the
constant factor, and two of those changes matter far past this toy corpus:

1. Normalize every vector ONCE at index time. Cosine similarity is
   (a.b)/(|a||b|); if both sides are unit length it collapses to just a.b.
   Stage 1's naive_rag.py recomputed each chunk's norm on every single
   query, which is work that can never change. Now the whole search is one matmul: (N,D) @ (D,) .

2. Top-k via argpartition, not a full sort. Sorting is O(N log N) to find k
   items you could get in O(N). Irrelevant at N=149, very relevant at N=10^6.

Exact brute force like this stays viable much further than people assume --
a few hundred thousand vectors is still tens of milliseconds. Stage 15 is where
we measure exactly when it stops being enough and switch to approximate search.
"""

import numpy as np

from .corpus import Chunk


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        assert len(chunks) == vectors.shape[0]
        self.chunks = chunks
        self.vectors = normalize(vectors.astype(np.float32))

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "VectorStore":
        return cls(chunks, embedder.embed([c.text for c in chunks]))

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the k most similar chunks, highest score first."""
        scores = self.vectors @ normalize(query_vector.astype(np.float32))

        k = min(k, len(self.chunks))
        # Partition puts the k best anywhere in the first k slots, then we
        # sort only those k -- not all N.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [(self.chunks[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.chunks)
