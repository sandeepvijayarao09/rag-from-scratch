"""Stage 10: late interaction (ColBERT-style MaxSim).

CORPUS SUBSET, and the reason IS the lesson. One vector per token means
5,183 abstracts x ~250 tokens x 768 dims x 4 bytes is roughly 4 GB, versus
16 MB for single-vector. A 250x blowup. We index 800 documents, and report the
extrapolation rather than pretending we ran the full corpus.

Queries are restricted to those whose gold document survives in the subset, so
the comparison against a single-vector baseline on the SAME subset is fair.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.late_interaction import LateInteractionIndex  # noqa: E402

SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 800
SEED = 0

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")

    # Keep every gold document, then pad with random distractors to SUBSET.
    gold = {d for q in queries for d in q.relevant}
    keep = [d for d in docs if d.id in gold]
    rest = [d for d in docs if d.id not in gold]
    random.Random(SEED).shuffle(rest)
    subset = keep + rest[:max(0, SUBSET - len(keep))]
    ids = {d.id for d in subset}
    queries = [q for q in queries if set(q.relevant) & ids]

    print(f"subset: {len(subset)} docs ({len(keep)} gold + distractors), "
          f"{len(queries)} queries\n")

    embedder = Embedder()
    store = VectorStore(subset, embedder.embed([d.text for d in subset]))
    single = evaluate(queries,
                      lambda q, k: [c.doc_id for c, _ in
                                    store.search(embedder.embed([q])[0], k=k)],
                      ndcg_ks=(10,))
    single_bytes = store.vectors.nbytes
    print(f"single-vector  {single}")
    print(f"               index {single_bytes/1e6:.1f} MB\n")

    index = LateInteractionIndex()
    start = time.perf_counter()
    index.add(subset, progress=True)
    build = time.perf_counter() - start

    start = time.perf_counter()
    late = evaluate(queries, lambda q, k: [d for d, _ in index.search(q, k=k)],
                    ndcg_ks=(10,))
    ms = 1000 * (time.perf_counter() - start) / len(queries)
    multi_bytes = index.index_bytes()

    print(f"\nlate interaction {late}")
    print(f"               index {multi_bytes/1e6:.1f} MB, build {build:.0f}s, {ms:.0f}ms/query\n")

    print("=" * 74)
    print(f"{'':<18} {'nDCG@10':>9} {'R@10':>8} {'index MB':>10} {'ms/query':>10}")
    print("-" * 74)
    print(f"{'single-vector':<18} {single.ndcg[10]:>9.4f} {single.recall[10]:>8.4f} "
          f"{single_bytes/1e6:>10.1f} {'<1':>10}")
    print(f"{'late interaction':<18} {late.ndcg[10]:>9.4f} {late.recall[10]:>8.4f} "
          f"{multi_bytes/1e6:>10.1f} {ms:>10.0f}")
    print("=" * 74)
    ratio = multi_bytes / single_bytes
    print(f"\nindex size ratio: {ratio:.0f}x")
    print(f"extrapolated to all 5,183 docs: {multi_bytes/len(subset)*5183/1e9:.2f} GB "
          f"vs {single_bytes/len(subset)*5183/1e6:.0f} MB")
    print("\nCAVEAT: these token vectors come from bge-base, trained for SINGLE-vector\n"
          "retrieval with CLS pooling. Real ColBERT is trained end-to-end under the\n"
          "MaxSim objective, so its token vectors are shaped for this operation.\n"
          "Any shortfall below is the cost of borrowing an architecture without its\n"
          "training objective -- which is the point worth taking away.")
