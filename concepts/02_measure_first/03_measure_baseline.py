"""Stage 2c: the baseline number that every later stage gets compared to.

This is the least glamorous file in the repo and the most important one.
From here on, "this technique is better" is a claim with a number attached
or it is not a claim.

We score the SAME retriever twice -- once on direct questions, once on
paraphrased ones. Watch the gap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_cat_facts  # noqa: E402
from rag.metrics import evaluate, load_evalset  # noqa: E402


def build_retriever():
    embedder = Embedder()
    store = VectorStore.build(load_cat_facts(), embedder)

    def retrieve(question: str, k: int) -> list[int]:
        query_vector = embedder.embed([question])[0]
        return [chunk.id for chunk, _ in store.search(query_vector, k=k)]

    return retrieve, store


if __name__ == "__main__":
    retrieve, store = build_retriever()
    print(f"corpus: {len(store)} chunks\n")

    print("DENSE RETRIEVAL BASELINE  (bge-base-en-v1.5, exact cosine)")
    print("-" * 62)

    reports = {}
    for variant in ("direct", "paraphrased"):
        queries = load_evalset(variant)
        reports[variant] = evaluate(queries, retrieve)
        print(f"{variant:<12} {reports[variant]}")

    gap = reports["direct"].recall[1] - reports["paraphrased"].recall[1]
    print("-" * 62)
    print(f"leakage gap @1: {gap:+.3f}")
    print(
        "\nThat gap is vocabulary overlap, not retrieval skill. The paraphrased\n"
        "row is the honest baseline -- optimise against that one."
    )
