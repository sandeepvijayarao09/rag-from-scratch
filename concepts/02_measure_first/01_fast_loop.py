"""Stage 2a: make the iteration loop fast enough to actually experiment in.

Three changes vs Stage 1's naive_rag.py, each teaching one thing:
  - batch the embedding calls        -> stop paying HTTP overhead 150x
  - cache embeddings by content hash -> re-running an experiment costs nothing
  - normalize once, then matmul      -> cosine similarity is just a dot product

Run it twice. The second run is the point.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_cat_facts  # noqa: E402


def timed(label, fn):
    start = time.perf_counter()
    result = fn()
    print(f"{label:<34} {time.perf_counter() - start:7.3f}s")
    return result


if __name__ == "__main__":
    chunks = load_cat_facts()
    print(f"corpus: {len(chunks)} chunks\n")

    embedder = Embedder()
    store = timed("index (batched + cached)", lambda: VectorStore.build(chunks, embedder))

    query_vector = embedder.embed(["why do cats purr?"])[0]
    timed("query x100 (numpy matmul)",
          lambda: [store.search(query_vector, k=3) for _ in range(100)])

    print()
    for chunk, score in store.search(query_vector, k=3):
        print(f"  [{chunk.id:>3}] {score:.3f}  {chunk.text[:78]}")
