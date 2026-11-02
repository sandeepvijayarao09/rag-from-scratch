# RAG from 0 to 1

A decision procedure, not a tutorial order. Every branch below is backed by a
measurement in this repo, and the point of each step is to tell you whether the
*next* step is worth taking.

The short version: most RAG advice is a list of techniques. What you actually
need is a way to decide which ones apply to you, because on my benchmark, tested
with paired significance tests, LLM query rewriting made things significantly
worse, a strong reranker made them significantly better, and most of the other
recommended techniques made no detectable difference at all.

---

## Step 0 — Do you need retrieval at all?

Before building anything, check whether your corpus fits in a context window.

| Corpus | Do this |
|---|---|
| Fits in context, cost acceptable | Stuff it. Skip RAG entirely. |
| Too large, or cost matters | Continue |
| Some queries need no lookup at all | Add a router later (Stage 12) |

A long-context baseline is also the number RAG has to beat. Skipping it means
you never learn whether your retrieval is adding anything.

---

## Step 1 — Build the dumbest thing, then make it measurable

Naive RAG first (Stage 1): embed, cosine top-k, stuff, generate. It works, and
you need something to measure.

Then, **before any technique**, build these three (Stage 2):

1. **A labelled query set.** 30 queries minimum, with the correct document
   marked.
2. **recall@k, MRR, nDCG@k.** Retrieval and generation get separate scoreboards.
3. **An embedding cache.** Re-indexing has to be free or you will stop
   experimenting. Mine went 2.6s to 0.00s and that is the only reason the rest
   of this repo exists.

> **Trap.** If you generate eval questions with an LLM from your own chunks,
> they inherit rare vocabulary from the source and your scores inflate. Mine
> went from recall@1 = 0.680 to 0.927 on the same retriever, purely from
> leakage. Generate a paraphrased variant that strips distinctive terms, and
> report the gap between them.

---

## Step 2 — Validate the pipeline before you trust it

Run a public benchmark where your embedding model has a published score
(Stage 3). BEIR SciFact, NFCorpus, FiQA are all small enough to do locally.

If you land near the published number, your embedding call, normalisation,
top-k and metrics are all correct. If you land far off, you have a bug, and
every technique you evaluate from here inherits it.

I got 0.745 against a published 0.741. That check is worth an afternoon.

While you are here, **ablate the conventions** rather than inheriting them. The
BGE query-instruction prefix is treated as mandatory and cost me 0.014 nDCG on
this corpus.

---

## Step 3 — Diagnose before you optimise

This is the step nearly everyone skips, and it is the one that saves the most
time. Each diagnostic below takes minutes and predicts whether a technique
worth hours will help.

| Question to answer | How to check | Decision |
|---|---|---|
| Are documents longer than the encoder window? | Token length distribution vs model limit | <5% over → **do not chunk**. Mine was 2.8% and all 6 configs lost |
| Do documents mix unrelated topics? | Read twenty of them | No → do not chunk |
| Does lexical search find things dense misses? | Run BM25, count queries only it got | Mine: 9 of 300. That capped hybrid at +0.03 |
| Is the gold doc retrieved but ranked low? | Mean gold rank inside your top-20 | Already ~2 of 20 → **skip reranking** |
| Do queries look like your documents? | Read ten of each | Yes → skip query rewriting |
| Is retrieval or generation failing? | recall@k vs answer quality, separately | recall high + answers bad → the generator is your ceiling |

**Only proceed to a technique whose diagnostic came back positive.**

---

## Step 4 — Retrieval techniques, in value order

Measured on SciFact (300 test queries) with paired bootstrap 95% CIs and Holm
correction, on three embedding models. Baseline `bge-base-en-v1.5`, 0.759
nDCG@10. Full tables: [results/v2/RESULTS.md](results/v2/RESULTS.md).

### First: choose the embedding model
The biggest single lever measured. `embeddinggemma-300m` alone scores 0.799,
more than any technique adds on top of bge-base. Use each model's documented
query and document formats, and check them on your data.

### Then: reranking (Stage 7), with a reranker stronger than your first stage
| First stage | bge-reranker-base | Qwen3-Reranker-4B |
|---|---|---|
| BM25 (0.665) | **+0.041** (significant) | **+0.110** (significant) |
| dense bge-base (0.759) | −0.033 (not significant) | **+0.044** (significant) |

Reranking *replaces* your ordering, so you end up with the reranker's quality
bounded by candidate recall. Two consequences:

- Rerank only if the reranker ranks better than your first stage **on your data**.
  A small reranker behind a strong dense model is wasted latency, or worse.
- Under reranking, optimise the first stage for **recall**, not ordering. Its
  ordering is about to be discarded.

bge-reranker-base costs 1,515ms/query at depth 50; a 4B reranker costs more.
Set depth deliberately.

### Cheap, but measure before keeping: weighted hybrid (Stage 6)
Weighted 0.7/0.3 was +0.010, not significant. It is nearly free (a BM25 index
builds in 0.21s and adds 0.7ms per query), so it is worth trying where your
queries depend on rare exact terms, which SciFact's mostly do not. **Weight it.**
Unweighted RRF, the "needs no tuning" default, was −0.009 to −0.024 across three
variants on bge-base and significantly worse (−0.057) on the stronger
embeddinggemma-300m, because it treats a weaker retriever's rank-1 as equally
credible.

### Late interaction (Stage 10): probably not
A trained ColBERT on the full corpus was +0.002 against bge-base (not
significant) and significantly worse than embeddinggemma-300m (−0.038), for a
much larger index.

### Chunking (Stage 4): only if Step 3 said yes
All six configs made no detectable difference (−0.017 to −0.003) on a corpus
where only 2.8% of documents exceed the model's window. The strategy barely
matters either: at size 200 fixed, sentence and recursive spanned 0.0026.

If you do chunk, remember your index now returns chunks while your labels judge
documents. Pool with `max`, and overfetch `k × 5` or you silently cap recall@k.

### Query rewriting (Stage 8): avoid on claim-like queries
With a 12B rewriter, on all 300 queries:

| | Δ (bge-base) |
|---|---|
| multi-query | +0.009 (not significant) |
| HyDE | **−0.040** (significant) |
| step-back | **−0.039** (significant) |
| decomposition | **−0.087** (significant) |

Techniques that **replace** the original query lose; blending the rewrite back
in with the original at a tuned weight only recovers a tie. HyDE fails when your
queries are already written like your documents, because there is no asymmetry
to bridge and only the hallucinated specifics remain.

---

## Step 5 — The generation half

Retrieval metrics cannot see the failures here (Stage 11).

Build a three-condition test:

| Context given | Correct behaviour |
|---|---|
| the right document | answer, with a citation |
| top-ranked docs, right one removed | say "not enough information" |
| nothing | say "not enough information" |

Require citations, because an uncited sentence is one nobody can check. Give the
model an explicit blessed way to refuse, because otherwise the strongest force
in the prompt is the instruction to answer.

> **What I found was the opposite of the standard warning.** `gemma4:e4b`
> refused 83% of distractor contexts and 100% of empty ones, and also refused
> 60% of cases where it held the correct document. Too conservative, not
> reckless. Reframing the task and relaxing the prompt each bought 0.07 and then
> stopped, so the generator was the ceiling. A system with *perfect* retrieval
> and this model still fails on more than half of SciFact.

Both over-answering and over-refusing are failures. They need opposite fixes,
and only one of them gets written about.

---

## Step 6 — Control flow, only when the questions demand it

| Add | When |
|---|---|
| Router (Stage 12) | Some queries need no retrieval. Cheap, high value. |
| CRAG grading (Stage 12) | You need to *detect* "all retrieved docs are irrelevant" |
| Multi-hop (Stage 13) | Answers genuinely require chaining across documents |
| GraphRAG (Stage 14) | Global questions no single passage can answer |

Each adds LLM calls to the query path. Naive RAG is one call, CRAG is two or
three, multi-hop is one per hop. Verify your questions actually need this before
paying for it on every request.

---

## Step 7 — Scale, when you get there

| Corpus size | What you need |
|---|---|
| 10²–10⁴ | numpy brute force. **Most products live here** |
| 10⁴–10⁶ | HNSW/FAISS or pgvector. You accept approximate recall for the first time |
| 10⁶–10⁸ | Qdrant/Vespa, int8 or PQ quantization, offline index builds |

Exact numpy search handled 5,183 documents in under a millisecond and scales to
a few hundred thousand vectors in tens of milliseconds. People adopt a vector
database one to two orders of magnitude before they need one.

---

## The one thing to take away

The techniques are not the skill. Every one of them is a few hours of work and
most of them are in a library already.

The skill is **refusing to add a technique until you have measured the gap it
is supposed to close.** On this benchmark that discipline was the difference
between +0.041 and −0.048 from the same family of ideas.

See [NOTES.md](NOTES.md) for what broke along the way, including the two
hypotheses I got wrong before landing the reranking rule.
