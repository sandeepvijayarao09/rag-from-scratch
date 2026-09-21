"""Stage 9: contextual retrieval (Anthropic, 2024).

THE PROBLEM. Split a document and each piece stops being self-describing:

    "The company's revenue grew by 3% over the previous quarter."

Which company? Which quarter? Unfindable by any query naming either, because
neither appears in the chunk. No retrieval technique fixes this: the
information is not in the indexed text.

THE FIX. Before embedding, ask an LLM to write a short blurb situating the
chunk inside its parent document, and prepend it.

MODEL CHOICE, measured before running. One LLM call per chunk is the expensive
shape, so the generator matters more here than anywhere else in the repo:

    Llama-3.2-1B     0.4s/chunk
    gemma4:e4b       9.2s/chunk      23x slower, comparable blurbs

At 9.2s the full corpus would take 11 hours. At 0.4s it is minutes. This is why
the technique is normally run against a small fast model with prompt caching,
and it is the single most important implementation decision in this stage.

THREE ARMS, same subset, same queries, doc-level max pooling:

    whole           no chunking at all, the Stage 4 control
    chunked         chunks embedded as-is
    contextualised  chunks with an LLM blurb prepended

PREDICTION, written before the run. Stage 4 showed SciFact abstracts are short
(p50 = 204 words) and self-contained, with only 2.8% exceeding the encoder
window. There is very little lost context to restore, so I expect contextual
retrieval to recover some of what chunking destroyed but not to beat `whole`.
If it does beat `whole`, the blurb is adding retrievable signal rather than
merely restoring it, which would be the more interesting result.

Resumable: blurbs are cached to disk by (model, document, chunk).
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag import chunking  # noqa: E402
from rag.contextual import ContextualIndexer  # noqa: E402
from rag.retrieve import doc_level_retriever  # noqa: E402

SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SEED, CHUNK, OVERLAP = 0, 100, 20
FAST_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")

    gold = {d for q in queries for d in q.relevant}
    keep = [d for d in docs if d.id in gold]
    rest = [d for d in docs if d.id not in gold]
    random.Random(SEED).shuffle(rest)
    subset = keep + rest[:max(0, SUBSET - len(keep))]
    ids = {d.id for d in subset}
    queries = [q for q in queries if set(q.relevant) & ids]
    parents = {d.id: d.text for d in subset}

    print(f"subset: {len(subset)} docs ({len(keep)} gold + distractors), "
          f"{len(queries)} queries\n", flush=True)

    embedder = Embedder()
    rows = []

    whole = VectorStore(subset, embedder.embed([d.text for d in subset]))
    rows.append(("whole (no chunking)", len(subset),
                 evaluate(queries, doc_level_retriever(whole, embedder), ndcg_ks=(10,))))
    print(f"{rows[-1][0]:<24} {rows[-1][2]}", flush=True)

    chunks = chunking.apply(subset, "fixed", CHUNK, OVERLAP)
    plain = VectorStore(chunks, embedder.embed([c.text for c in chunks]))
    rows.append((f"chunked {CHUNK}/{OVERLAP}", len(chunks),
                 evaluate(queries, doc_level_retriever(plain, embedder), ndcg_ks=(10,))))
    print(f"{rows[-1][0]:<24} {rows[-1][2]}", flush=True)

    print(f"\ncontextualising {len(chunks)} chunks with {FAST_MODEL.split('/')[-1]}...",
          flush=True)
    start = time.perf_counter()
    indexer = ContextualIndexer(model=FAST_MODEL)
    ctx_chunks = indexer.apply(chunks, parents, progress=True)
    elapsed = time.perf_counter() - start
    print(f"  {elapsed:.0f}s ({elapsed/max(1,len(chunks)):.2f}s/chunk)", flush=True)

    ctx = VectorStore(ctx_chunks, embedder.embed([c.text for c in ctx_chunks], progress=True))
    rows.append(("contextualised", len(ctx_chunks),
                 evaluate(queries, doc_level_retriever(ctx, embedder), ndcg_ks=(10,))))
    print(f"{rows[-1][0]:<24} {rows[-1][2]}", flush=True)

    base = rows[0][2].ndcg[10]
    print("\n" + "=" * 80)
    print(f"{'arm':<24} {'units':>8} {'nDCG@10':>9} {'vs whole':>10} {'R@10':>8} {'MRR':>8}")
    print("-" * 80)
    for label, n, r in rows:
        d = "" if label.startswith("whole") else f"{r.ndcg[10] - base:+.4f}"
        print(f"{label:<24} {n:>8} {r.ndcg[10]:>9.4f} {d:>10} {r.recall[10]:>8.4f} {r.mrr:>8.4f}")
    print("=" * 80)
    recovered = rows[2][2].ndcg[10] - rows[1][2].ndcg[10]
    print(f"\ncontextualising recovered {recovered:+.4f} over plain chunks")
    print("and still {} `whole`.".format(
        "beats" if rows[2][2].ndcg[10] > base else "trails"))

    sample = next((c for c in ctx_chunks if c.meta.get("blurb")), None)
    if sample:
        print(f"\n--- example blurb ---\n{sample.meta['blurb'][:300]}")
