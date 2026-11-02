"""v2 Stage 4: chunking, and whether it ever helps on SciFact.

v1's diagnostic was the key move: only 2.8% of abstracts overflow bge-base's
512-token window, so chunking had almost nothing to rescue, and all six
configs lost. The window is a property of the embedding model, and the v2
ladder spans 512 (bge-base), 2,048 (embeddinggemma) and 32k (qwen3), so the
diagnostic predicts chunking matters even less for the other two.

Arms:
  4a  truncation diagnostic per model (word-based estimate, ~1.3 tokens/word)
  4b  the v1 sweep: fixed/sentence/recursive at 200 and 100 words, max-pooled
  4c  the repair, tuned on train: chunk ONLY the documents the model truncates,
      and blend each document's whole-text score with its best chunk's score
"""

import numpy as np

from _common import banner, embeddings_from_args

from rag import chunking
from rag.embed import Embedder
from rag.store import VectorStore, normalize
from rag.v2.bench import Bench

STAGE = "s04_chunking"
WINDOW = {"bge-base": 512, "embeddinggemma-300m": 2048, "qwen3-embedding-4b": 32768}
CONFIGS = [("fixed", 200, 40), ("sentence", 200, 1), ("recursive", 200, 40),
           ("fixed", 100, 20), ("recursive", 100, 20), ("sentence", 100, 1)]
TOKENS_PER_WORD = 1.3


def chunk_embedder(bench, tag):
    return Embedder(model=bench.spec.ollama, cache_dir=f"cache/v2/emb/{bench.name}/chunks-{tag}")


def pooled(store, vecs, qs, k=50):
    out = {}
    for q, v in zip(qs, vecs):
        best = {}
        for c, s in store.search(v, k=k):
            best[c.doc_id] = max(best.get(c.doc_id, -1e9), s)
        out[q.qid] = [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    return out


if __name__ == "__main__":
    for name in embeddings_from_args(__doc__):
        banner(f"Stage 4, {name}")
        bench = Bench(name)
        limit = WINDOW[name]
        words = np.array([len(d.text.split()) for d in bench.docs])
        over = words * TOKENS_PER_WORD > limit
        meta = {"window_tokens": limit, "docs_over_window": int(over.sum()),
                "pct_over_window": float(over.mean() * 100)}
        print(f"  {over.sum()} of {len(bench.docs)} docs ({over.mean():.1%}) exceed the "
              f"{limit}-token window", flush=True)
        tvecs = bench.qvecs([q.question for q in bench.test])
        bench.record(STAGE, "whole documents", {q.qid: [d for d, _ in bench.dense_vec(v, 10)]
                                                for q, v in zip(bench.test, tvecs)}, meta=meta)

        # 4b the v1 sweep
        for strategy, size, overlap in CONFIGS:
            label = f"{strategy} {size}/{overlap}"
            chunks = chunking.apply(bench.docs, strategy, size, overlap)
            emb = chunk_embedder(bench, label.replace("/", "-").replace(" ", "_"))
            texts = [bench.doc_text(bench.by_id[c.doc_id], c.text) for c in chunks]
            store = VectorStore(chunks, emb.embed(texts, progress=True))
            bench.record(STAGE, label, pooled(store, tvecs, bench.test), meta={"chunks": len(chunks)})

        # 4c the repair: chunk only what gets truncated, blend whole + best chunk
        long_docs = [d for d, o in zip(bench.docs, over) if o]
        if not long_docs:
            print("  repair skipped: no document exceeds this model's window", flush=True)
            continue
        chunks = chunking.apply(long_docs, "sentence", 200, 1)
        emb = chunk_embedder(bench, "repair")
        cvecs = normalize(emb.embed([bench.doc_text(bench.by_id[c.doc_id], c.text) for c in chunks],
                                    progress=True))
        index = {d.id: i for i, d in enumerate(bench.docs)}
        owner = np.array([index[c.doc_id] for c in chunks])

        def precompute(qs):
            vecs = bench.qvecs([q.question for q in qs])
            pre = {}
            for q, v in zip(qs, vecs):
                whole = bench.all_scores(v)
                best = whole.copy()
                np.maximum.at(best, owner, cvecs @ v)
                pre[q.qid] = (whole, best)
            return pre

        def rank(a, qs, pre):
            out = {}
            for q in qs:
                whole, best = pre[q.qid]
                s = (1 - a) * whole + a * best
                top = np.argpartition(-s, 10)[:10]
                out[q.qid] = [bench.docs[i].id for i in top[np.argsort(-s[top])]]
            return out

        pre_train, pre_test = precompute(bench.train), precompute(bench.test)
        a, tr = bench.tune(lambda p, qs: rank(p, qs, pre_train), [0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        bench.record(STAGE, "repaired: chunk long docs only, blended",
                     rank(a, bench.test, pre_test),
                     meta={"chunk_weight": a, "train_ndcg10": tr, "chunks": len(chunks),
                           "long_docs": len(long_docs)})
