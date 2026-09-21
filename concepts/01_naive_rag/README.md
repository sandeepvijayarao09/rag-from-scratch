# Stage 1 — Naive RAG

**Tier: Beginner** · no prerequisites

The loop every tutorial teaches, in 60 lines. Run it once to feel the shape of
the thing:

```bash
./myenv/bin/python concepts/01_naive_rag/naive_rag.py
```

Embed the corpus → embed the query → rank by cosine → stuff the top-k into the
prompt → generate. It genuinely works, and internalising this loop puts you
ahead of most people who "know RAG" from blog posts.

## Attribution

This stage is adapted from a widely-circulated "RAG from scratch" tutorial:
the cat-facts corpus, Ollama, `bge-base` embeddings, and pure-Python cosine
similarity. It is the starting point, not original work, and I have not been
able to pin down the single canonical source since the same 150-line example
circulates in several places.

Everything from Stage 2 onward is original: the eval harness, the BEIR
integration, BM25, the fusion and reranking analysis, and every measurement.

## What it hides

Every stage after this one exists because of a line in this table.

| Blind spot | Where it bites | Fixed in |
|---|---|---|
| No chunking | `readlines()` only works because cat-facts is one self-contained fact per line | Stage 4 |
| No evaluation | Every change is unverifiable; improvements are vibes | Stage 2 |
| Dense-only retrieval | Embeddings miss exact terms, IDs, error codes, rare proper nouns | Stage 5 |
| No abstention or citation | The prompt says "don't make up information" with no mechanism to enforce it | Stage 11 |
| Re-embeds every run | 150 serial HTTP calls per experiment; you stop experimenting | Stage 2 |
| Pure-Python O(N) scan | Recomputes every chunk's vector norm on every query | Stage 2 |

## The one to internalise

**The corpus is doing the work, not the code.** Cat-facts is one self-contained
fact per line, which is why `readlines()` passes for an ingestion pipeline and
why retrieval looks nearly perfect. Change the corpus to real documents and this
implementation degrades immediately.

Most RAG tutorials are demonstrations on data chosen to make the demonstration
succeed. Stage 3 moves to a benchmark you cannot tune against.
