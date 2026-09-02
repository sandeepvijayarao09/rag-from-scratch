"""Stage 6a: will hybrid even help? Measure before you build.

Hybrid retrieval only pays off if dense and sparse fail on DIFFERENT queries.
If they fail on the same ones, fusing them just averages two identical opinions
and you have added a second index for nothing.

So decompose the 300 queries by which retriever found the gold document in its
top 10:

    BOTH        -- fusion cannot help, already solved
    DENSE ONLY  -- BM25 missed it; hybrid keeps the win if fusion is sane
    BM25 ONLY   -- dense missed it; THIS is the upside hybrid is buying
    NEITHER     -- fusion cannot help, needs a different technique entirely

`BM25 ONLY` is the number that matters. It is the headroom. If it is near zero,
skip hybrid and spend the effort somewhere else.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_corpus, load_queries  # noqa: E402
from rag.sparse import BM25  # noqa: E402

K = 10

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs], progress=True))
    bm25 = BM25(docs)

    buckets = {"both": [], "dense only": [], "bm25 only": [], "neither": []}

    for q in queries:
        qv = embedder.embed([q.question])[0]
        dense_hits = {c.doc_id for c, _ in store.search(qv, k=K)}
        sparse_hits = {c.doc_id for c, _ in bm25.search(q.question, k=K)}
        gold = set(q.relevant)

        d, s = bool(dense_hits & gold), bool(sparse_hits & gold)
        key = "both" if d and s else "dense only" if d else "bm25 only" if s else "neither"
        buckets[key].append(q)

    n = len(queries)
    print(f"\nWhere the gold document was found, top-{K}, n={n}\n")
    for key in ("both", "dense only", "bm25 only", "neither"):
        count = len(buckets[key])
        bar = "#" * round(50 * count / n)
        print(f"  {key:<11} {count:>4}  {100*count/n:>5.1f}%  {bar}")

    upside = len(buckets["bm25 only"])
    ceiling = len(buckets["both"]) + len(buckets["dense only"]) + upside
    print(f"\n  dense alone recall@{K}: "
          f"{(len(buckets['both']) + len(buckets['dense only']))/n:.3f}")
    print(f"  perfect-fusion ceiling: {ceiling/n:.3f}  "
          f"(+{upside/n:.3f} from the {upside} bm25-only queries)")
    print(f"  unreachable by fusion : {len(buckets['neither'])/n:.3f}\n")

    if buckets["bm25 only"]:
        print("Queries only BM25 found (look at the vocabulary):")
        for q in buckets["bm25 only"][:6]:
            print(f"  - {q.question[:96]}")
