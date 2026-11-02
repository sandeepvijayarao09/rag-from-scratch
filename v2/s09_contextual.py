"""v2 Stage 9: contextual retrieval with a 12B blurb writer.

v1 ran Anthropic's contextual retrieval with Llama-3.2-1B writing the blurbs,
because at ~9s per chunk a bigger model made the full corpus an 11-hour job.
It recovered only part of what chunking lost (-0.006 vs no chunking). The open
question was whether a 1B model's blurbs were simply too weak.

v2 answers that: same 600-document subset (every gold document plus seeded
distractors, so arms are comparable), same 100/20 chunks, blurbs from
gemma4:12b. Blurbs are cached by (model, document, chunk) and shared by all
three embedding models, so the expensive part runs once.

    python v2/s09_contextual.py generate     # 12B blurbs, the slow part
    python v2/s09_contextual.py score        # all three embedding models
"""

import random
import sys
import time

from _common import banner, embeddings_from_args

from rag import chunking
from rag.beir import load_corpus, load_queries
from rag.contextual import ContextualIndexer
from rag.embed import Embedder
from rag.store import VectorStore
from rag.v2.bench import Bench
from rag.v2.models import GENERATOR

STAGE = "s09_contextual"
SUBSET, SEED, CHUNK, OVERLAP = 600, 0, 100, 20


def subset_docs(docs, queries):
    gold = {d for q in queries for d in q.relevant}
    keep = [d for d in docs if d.id in gold]
    rest = [d for d in docs if d.id not in gold]
    random.Random(SEED).shuffle(rest)
    return keep + rest[:max(0, SUBSET - len(keep))]


def contextualised_chunks():
    docs, queries = load_corpus(with_title=True), load_queries(split="test")
    sub = subset_docs(docs, queries)
    chunks = chunking.apply(sub, "fixed", CHUNK, OVERLAP)
    indexer = ContextualIndexer(model=GENERATOR, cache_dir="cache/v2", think=False,
                                options={"temperature": 0.0, "seed": 0, "num_predict": 200})
    return sub, chunks, indexer, {d.id: d.text for d in sub}


def pooled(store, vecs, qs):
    out = {}
    for q, v in zip(qs, vecs):
        best = {}
        for c, s in store.search(v, k=50):
            best[c.doc_id] = max(best.get(c.doc_id, -1e9), s)
        out[q.qid] = [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "score"
    sub, chunks, indexer, parents = contextualised_chunks()
    if mode == "generate":
        banner(f"Stage 9, writing {len(chunks)} blurbs with {GENERATOR}")
        start = time.perf_counter()
        indexer.apply(chunks, parents, progress=True)
        print(f"  {time.perf_counter() - start:.0f}s", flush=True)
        sys.exit()

    ctx = indexer.apply(chunks, parents)
    missing = sum(1 for c in ctx if not c.meta.get("blurb"))
    if missing:
        print(f"  warning: {missing} chunks have no blurb; run 'generate' first", flush=True)
    for name in embeddings_from_args(__doc__):
        banner(f"Stage 9, {name}")
        bench = Bench(name)
        qs = bench.test
        vecs = bench.qvecs([q.question for q in qs])
        meta = {"subset_docs": len(sub), "chunks": len(chunks), "blurb_model": GENERATOR}

        whole = VectorStore(sub, bench.embedder.embed([bench.doc_text(d) for d in sub]))
        bench.record(STAGE, "whole (600-doc subset)", pooled(whole, vecs, qs), meta=meta)

        emb = Embedder(model=bench.spec.ollama, cache_dir=f"cache/v2/emb/{name}/s09")
        plain = VectorStore(chunks, emb.embed([bench.doc_text(bench.by_id[c.doc_id], c.text)
                                               for c in chunks], progress=True))
        bench.record(STAGE, f"chunked {CHUNK}/{OVERLAP}", pooled(plain, vecs, qs), meta=meta)

        texts = [bench.doc_text(bench.by_id[c.doc_id], c.text) for c in ctx]
        cstore = VectorStore(ctx, emb.embed(texts, progress=True))
        bench.record(STAGE, "contextualised (gemma4:12b blurbs)", pooled(cstore, vecs, qs), meta=meta)
