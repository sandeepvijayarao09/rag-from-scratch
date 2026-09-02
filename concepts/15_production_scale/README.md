# Stage 15 — Production scale

**Tier: Pro** · prerequisite: Stage 14 · status: planned

Orthogonal to every technique above. These do not change *what* you retrieve,
they change whether the system survives contact with real traffic and real data.

## The scale ladder

| Corpus | What you need | What breaks |
|---|---|---|
| 10²–10⁴ | numpy brute force, in-process | Nothing. **Most real products live here** |
| 10⁴–10⁶ | HNSW / FAISS, or pgvector | Latency; you accept *approximate* recall for the first time |
| 10⁶–10⁸ | Qdrant / Vespa / Turbopuffer, int8 or PQ | Memory, index build time, reindexing becomes an ops problem |
| Beyond | Sharding, multi-stage retrieval, streaming ingest | Freshness and consistency, not similarity search |

**The non-obvious lesson**, and this repo has the measurement to back it: exact
brute force in numpy searched 5,183 documents in well under a millisecond, and
scales to a few hundred thousand vectors in tens of milliseconds. People reach
for a vector database one to two orders of magnitude before they need one.

## Topics

- **ANN indexing** — HNSW vs IVF-PQ. Every ANN index trades exact recall for
  latency, and the first thing to measure is *what you lost*, not how fast it got
- **Quantization** — int8, binary, Matryoshka. 4-32× memory reduction for a
  measurable recall cost. Relevant immediately if you took Stage 10 seriously
- **Incremental ingestion** — corpora change. Index builds, freshness lag,
  deletes, reindex cost
- **Caching** — query, embedding, generation. Stage 2's embedding cache is the
  cheapest version of this and it already saved hours
- **Observability** — per-query retrieval traces. Without them, production
  failures are unfixable, because you cannot tell a retrieval miss from a
  generation failure after the fact
- **Guardrails** — Stage 11's citation and abstention machinery, wired to alerts
