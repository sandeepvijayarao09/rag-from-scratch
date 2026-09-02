# Stage 2 — Measure first

**Tier: Beginner** · prerequisite: Stage 1

The stage that makes every other stage possible. Two problems with Stage 1:
it was too slow to experiment in, and it produced no number you could argue with.

## Files

| File | Concept |
|---|---|
| `01_fast_loop.py` | Batching, caching, and vectorized similarity |
| `02_generate_evalset.py` | Building labelled eval data — and its bias |
| `03_measure_baseline.py` | recall@k / MRR, and the leakage gap |

## 2a — The loop

| | Stage 1 | Stage 2a | Why |
|---|---|---|---|
| Index, cold | 2.631s | 1.144s | Batched embedding — one HTTP round-trip per 32 chunks, not per chunk |
| Index, re-run | 2.631s | 0.000s | Content-hash cache — embeddings are a pure function of (model, text) |
| 100 queries | 0.565s | 0.003s | Normalize once at index time; cosine collapses to one matmul |

Stage 1 recomputed every chunk's vector norm on every query — work whose
answer can never change. At 150 chunks none of these absolute numbers matter.
The **ratios** are what survive to 10⁶, and the 0.000s re-index is what makes
Stages 5-8 possible: you'll re-run the pipeline dozens of times comparing
retrieval techniques, and re-embedding the corpus each time would stop you.

Exact brute-force search stays viable much further than people assume — a few
hundred thousand vectors is still tens of milliseconds. Stage 15 measures
exactly where it breaks, rather than guessing.

## 2b/2c — The scoreboard, and why one number lies

For each fact, gemma4 writes two questions: a `direct` one phrased as someone
who just read the fact would, and a `paraphrased` one that deliberately avoids
the fact's distinctive vocabulary. The gold label is free — the question was
generated *from* chunk `i`, so chunk `i` is the correct retrieval.

Same retriever, same corpus, both question sets:

```
direct       n=150  r@1=0.927  r@3=0.987  r@5=0.987  MRR@10=0.955  nDCG@10=0.965
paraphrased  n=150  r@1=0.680  r@3=0.947  r@5=0.960  MRR@10=0.809  nDCG@10=0.853
                        ^^^^^
                    leakage gap @1: +0.247
```

**A quarter of the "accuracy" was vocabulary leakage.** The direct questions
inherit rare words from their source chunk — "bezoar", "clowder" — so cosine
similarity is handed the answer. Strip those words and top-1 retrieval falls
off a cliff.

This is the most common way RAG evaluations lie, and it is almost always
unintentional: generate questions from chunks, report 98%, ship, watch it fail
on real users who don't know your corpus's vocabulary.

Two things worth noticing beyond the headline:

- **r@1 collapsed but r@5 barely moved** (0.98 → 0.96). The right chunk is
  usually still *retrieved*, just not ranked first. That's a ranking problem,
  not a recall problem — which is precisely the failure a reranker fixes, and
  why Stage 7 will watch MRR rather than recall@5.
- **MRR fell 0.983 → 0.798**, tracking that reordering. Recall@5 alone would
  have told you nothing was wrong.

`paraphrased` is the honest baseline. Every later stage optimizes against
that row.
