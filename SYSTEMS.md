# The RAG Taxonomy

Every system below exists because a simpler one failed in a specific way.
Organised by *what problem it solves*, not by how fashionable it is.

The failure mode column is the useful one. Reach for a system when you have
measured its failure mode, not because it appeared in a blog post.

---

## Layer 0 — Baselines

| System | What it does | When it is the right answer |
|---|---|---|
| **No-RAG (long context)** | Stuff the entire corpus into the prompt | Corpus fits in context and cost is acceptable. Always measure this first: it is often better than bad RAG and it bounds what retrieval can buy you |
| **Naive RAG** | Embed chunks, cosine top-k, stuff, generate | Small, clean, single-topic corpus. The thing every tutorial builds |

**Naive RAG fails when:** queries use different vocabulary than documents, exact
terms matter, documents are long or multi-topic, or the answer requires more
than one document.

---

## Layer 1 — Retrieval quality, single shot

The query is taken at face value; only the matching improves.

| System | Mechanism | Fixes |
|---|---|---|
| **Sparse / BM25** | Term frequency + inverse document frequency, no neural net | Exact terms, IDs, error codes, rare proper nouns. Dense embeddings are systematically bad at these |
| **Hybrid** | Run dense and sparse, fuse the rankings (RRF or weighted) | The union of both failure modes. Usually the single largest quality jump in real systems |
| **Reranking (two-stage)** | Cheap retriever gets top-50, a cross-encoder rescores them | Ranking, not recall. A cross-encoder sees query and document *together* so it can judge relevance a bi-encoder cannot |
| **Late interaction (ColBERT)** | One vector per token, score by MaxSim over token pairs | Precision without a rerank pass, at large index cost |

**Key distinction, and the one people get wrong:** recall problems and ranking
problems need different fixes. If the right doc is in your top-50 but not your
top-3, a reranker helps and a better embedding model mostly does not.

---

## Layer 2 — Indexing and representation

Change *what you store*, not how you search it.

| System | Mechanism | Fixes |
|---|---|---|
| **Chunking strategies** | Fixed / sentence / recursive / structural, with overlap | Documents longer than the encoder window, or mixing topics |
| **Small-to-big** | Retrieve small precise units, return their larger parent for generation | The precision/context tradeoff. Small chunks match better, large chunks answer better. You stop having to choose |
| **Contextual retrieval** | Prepend an LLM-written summary of the document to each chunk before embedding | Chunks that are meaningless out of context ("This rose 3%" - what did?) |
| **RAPTOR** | Recursively cluster and summarise into a tree; retrieve at any level | Questions needing whole-document or cross-document synthesis, not one passage |
| **Metadata filtering** | Structured predicates alongside the vector search | Tenancy, recency, permissions, "only 2024 docs". Cheap and enormously useful |

---

## Layer 3 — Query understanding

The documents are fine. The query is the problem.

| System | Mechanism | Fixes |
|---|---|---|
| **Query rewriting** | LLM rewrites the raw query into a retrieval-friendly one | Conversational, vague, or misspelled input |
| **Multi-query** | Generate N paraphrases, retrieve for each, fuse | Vocabulary mismatch, by brute force. Directly attacks the leakage gap measured in Stage 1 |
| **HyDE** | LLM writes a *hypothetical answer*, embed that instead of the query | The asymmetry between short queries and long passages. You search answer-space with an answer |
| **Step-back** | Ask a more general question first, retrieve for both | Narrow questions whose context lives in broader documents |
| **Decomposition** | Split a compound question into sub-questions | "Compare X and Y" - one retrieval cannot serve both halves |
| **Routing** | Classify the query, send it to the right index or tool | Mixed corpora, or when retrieval should not fire at all |

---

## Layer 4 — Control flow

The model decides what to do, instead of a fixed pipeline.

| System | Mechanism | Fixes |
|---|---|---|
| **Self-RAG** | Model decides *whether* to retrieve, then critiques its own output | Retrieval firing on questions that do not need it, and unsupported claims |
| **CRAG (corrective)** | Grade retrieved docs; if weak, re-retrieve or fall back | Silent failure. Naive RAG answers confidently from irrelevant context |
| **Agentic / multi-hop** | Loop: retrieve, reason, retrieve again, until satisfied | Questions whose answer requires chaining facts across documents |

**The shared idea:** a fixed `retrieve -> stuff -> generate` pipeline cannot
recover from a bad retrieval. These add a feedback edge.

---

## Layer 5 — Knowledge structure

| System | Mechanism | Fixes |
|---|---|---|
| **GraphRAG** | Extract entities and relations, build a graph, summarise communities | Global questions ("what are the themes?") that no single passage answers |
| **Conversational / memory RAG** | Retrieve over dialogue history plus the corpus | Follow-ups, pronouns, context carried across turns |
| **Multi-modal RAG** | Embed images, tables, audio into the same or linked space | Corpora where the answer is in a figure or a table |

---

## Layer 6 — Production, orthogonal to all of the above

| Concern | What changes |
|---|---|
| **ANN indexing** | HNSW / IVF-PQ. You trade exact recall for latency, and must measure what you lost |
| **Quantization** | int8 / binary / Matryoshka. 4-32x memory reduction for a measurable recall cost |
| **Incremental ingestion** | Corpora change. Index build, freshness lag, deletes, reindex cost |
| **Caching** | Query cache, embedding cache, generation cache |
| **Guardrails** | Citation, abstention, faithfulness scoring, hallucination detection |
| **Observability** | Per-query retrieval traces. Without them production failures are unfixable |

---

## Ordering of impact, in practice

Roughly, for a system that is already working but not good enough:

1. **Hybrid retrieval** - largest single jump, low complexity
2. **Reranking** - second largest, moderate cost per query
3. **Better chunking / small-to-big** - large if your documents are long
4. **Query transformation** - large when query and document vocabulary diverge
5. **Contextual retrieval** - large on fragmented corpora, expensive to index
6. **Agentic / graph** - only for genuinely multi-hop or global questions

The first three are almost always worth doing. The last two are frequently
adopted before the first three have been measured, which is how teams end up
with an agentic pipeline that underperforms BM25.
