"""v2: what each reranker costs per query, next to what it buys.

Stage 7 reports quality. A product team also needs the price: milliseconds per
query at the depth Stage 7 used (50 candidates from dense bge-base). This times
each reranker on the same 40 test queries, bypassing the score cache, after a
warm-up batch so model loading and kernel compilation are not counted.

Numbers are for one laptop (Apple Silicon, MPS, fp16 for the 4B model), batch
16 for the cross-encoders and 8 for Qwen3. Treat them as relative costs.

    python v2/latency.py
"""

import json
import statistics
import time

from _common import banner

from rag.v2.bench import RESULTS, Bench, subsample
from rag.v2.models import RERANKER_ORDER
from rag.v2.rerankers import Reranker

DEPTH, N, WARMUP = 50, 40, 3


def time_reranker(rr, bench, qs, first):
    rr._load()
    out = []
    for i, q in enumerate(qs):
        texts = [bench.by_id[d].text for d, _ in first[q.qid]]
        pairs = [(q.question, t) for t in texts]
        start = time.perf_counter()
        for s in range(0, len(pairs), rr.batch_size):
            chunk = pairs[s:s + rr.batch_size]
            if rr.spec.kind == "cross-encoder":
                rr._model.predict(chunk, batch_size=rr.batch_size, show_progress_bar=False)
            else:
                rr._score_qwen3(chunk)
        if i >= WARMUP:
            out.append((time.perf_counter() - start) * 1000)
    return out


if __name__ == "__main__":
    banner("Reranker latency, depth 50 over dense bge-base")
    bench = Bench("bge-base", verbose=False)
    qs = subsample(bench.test, N + WARMUP)

    # Time the query embedding through Ollama directly: bench.qvecs would hit
    # the embedding cache and report ~0 ms.
    import ollama
    texts = [bench.query_text(q.question) for q in qs]
    ollama.embed(model=bench.spec.ollama, input=texts[:1])          # load the model
    start = time.perf_counter()
    for t in texts:
        ollama.embed(model=bench.spec.ollama, input=[t])
    embed_ms = (time.perf_counter() - start) * 1000 / len(qs)
    vecs = bench.qvecs([q.question for q in qs])
    start = time.perf_counter()
    first = {q.qid: bench.dense_vec(v, DEPTH) for q, v in zip(qs, vecs)}
    search_ms = (time.perf_counter() - start) * 1000 / len(qs)

    result = {"depth": DEPTH, "queries": N, "first_stage": {
        "query_embedding_ms": round(embed_ms, 1), "dense_search_ms": round(search_ms, 2)}}
    print(f"  first stage: embed {embed_ms:.1f} ms/query, search {search_ms:.2f} ms/query")
    for name in RERANKER_ORDER:
        rr = Reranker(name)
        ms = time_reranker(rr, bench, qs, first)
        q = statistics.quantiles(ms, n=10)
        result[name] = {"median_ms": round(statistics.median(ms)), "p90_ms": round(q[-1]),
                        "batch_size": rr.batch_size}
        print(f"  {name:22s} median {statistics.median(ms):7.0f} ms/query   p90 {q[-1]:7.0f}")
        del rr

    (RESULTS / "latency.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {RESULTS / 'latency.json'}")
