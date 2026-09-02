# Decision Record

Every step and every decision in building this RAG ladder, with the reasoning,
the alternatives rejected, and the evidence. Read top to bottom to reconstruct
how the system got here.

Convention: **D<n>** is a decision, **S<n>** is a step that produced evidence.

---

## Stage 0 — Starting point

Stage 1, `concepts/01_naive_rag/naive_rag.py`: 150 lines of cat facts, one line per chunk, brute-force
cosine similarity in pure Python, Ollama for embedding and generation.

It works, and it teaches the real loop: embed corpus, embed query, rank by
similarity, stuff context, generate. What it hides is everything that makes RAG
hard, because the tutorial corpus is one self-contained fact per line. That
single property removes the need for chunking, makes retrieval nearly trivial,
and lets `readlines()` pass for an ingestion pipeline.

Gaps identified before writing any code:

| Gap | Why it matters |
|---|---|
| No chunking | `readlines()` only works on one-fact-per-line data |
| No evaluation | Every change is unverifiable |
| Dense-only retrieval | Embeddings miss exact terms, IDs, rare words |
| No abstention or citation | Prompt says "don't make things up" with no mechanism |
| Re-embeds every run | No index lifecycle |
| One HTTP call per chunk | Throughput left on the floor |

---

## D1 — Fix the loop before fixing the retrieval

**Chose:** build the scoreboard and the fast iteration path first, ahead of any
retrieval technique.

**Rejected:** going straight to hybrid search or reranking, which is the fun part.

**Why:** two blockers made exploration impossible, not just slow. A single
experiment cost 150 serial HTTP round-trips, and there was no number to compare
a result against. Any technique added before that would be adopted or rejected
on vibes. The order is forced: you cannot tune what you cannot measure.

---

## S1 — Measure the loop

Three changes, each isolating one mechanism.

| | Stage 1 | Stage 2 | mechanism |
|---|---|---|---|
| Index, cold | 2.631s | 1.144s | Batched embedding, `ollama.embed` takes a list |
| Index, re-run | 2.631s | 0.000s | Content-hash disk cache |
| 100 queries | 0.565s | 0.003s | Normalize once, cosine collapses to one matmul |

**The mechanism behind the 190x:** Stage 1 recomputed every chunk's vector
norm on every query. That value can never change. Normalizing at index time
turns cosine similarity into a plain dot product, and the whole search becomes
`(N,D) @ (D,)`.

**The one that actually mattered** is the 0.000s re-index, not the 190x. Stage 3
will re-run this pipeline dozens of times comparing techniques. At 2.6s of
re-embedding per run you stop experimenting. At 0.00s you do not.

At 150 chunks all these absolute times are meaningless. The ratios are what
survive to 10^6.

**Also chose:** `argpartition` over a full sort for top-k. Irrelevant at N=150,
O(N) vs O(N log N) at N=10^6. Cost nothing to do right the first time.

---

## D2 — How to build the labelled eval set

**Chose:** LLM-generate a question per fact, then adversarially rewrite it to
strip shared vocabulary. Keep both variants and report the gap.

**Rejected:**
- *Generate straight, no rewriting.* Fastest. Questions derived from a chunk
  inherit its exact wording, so cosine similarity is handed the answer.
- *Hand-write 30 questions.* Highest signal per query, and what production teams
  actually do, but slow and small.

**Why:** the gap between the two variants is itself the measurement. A single
synthetic number tells you nothing about whether your retriever works or whether
your eval leaked.

**Implementation:** gemma4 writes a `direct` and a `paraphrased` question per
fact. The gold label is free, since the question was generated *from* chunk `i`.
Resumable and cached, so it runs once.

---

## S2 — The leakage gap

Same retriever, same corpus, two phrasings:

```
direct       n=150  r@1=0.927  r@3=0.987  r@5=0.987  MRR@10=0.955  nDCG@10=0.965
paraphrased  n=150  r@1=0.680  r@3=0.947  r@5=0.960  MRR@10=0.809  nDCG@10=0.853
                        ^^^^^
                    leakage gap @1: +0.247
```

The gap held steady as the eval set grew (+0.320 at n=50, +0.245 at n=110,
+0.247 at n=150), so it is a property of the method, not a small-sample artifact.

A quarter of the apparent accuracy was vocabulary overlap. Fact 3 is the clearest
case: `direct` asks for "the technical term for a cat's hairball", `paraphrased`
asks about "those clumps of fur that cats throw up". The word *bezoar* never
appears in the second, and top-1 retrieval falls off a cliff.

**The subtler finding, which is worth more than the headline:** r@1 collapsed
(0.927 to 0.680) while r@5 barely moved (0.987 to 0.960). The correct chunk is
still being *retrieved*, just no longer ranked first. That is a ranking problem,
not a recall problem. Rerankers fix ranking. Better embeddings fix recall. You
now know which one you need before building either.

MRR fell 0.955 to 0.809, tracking that reordering. Recall@5 alone would have
reported that nothing was wrong.

**Decision that follows:** `paraphrased` is the honest baseline. Every later
stage optimizes against that row.

---

## D3 — Corpus for the chunking stage

**Chose:** a standard benchmark, BEIR SciFact. 5,183 scientific abstracts,
300 labelled test queries.

**Rejected:**
- *Reflow cat-facts into prose.* Controlled and keeps ground truth, but the
  ground truth is still synthetic and still ours.
- *Real messy documents.* Realistic, but loses labels entirely, and chunking's
  effect gets tangled with corpus noise.

**Why the benchmark wins, and this is the part that is easy to miss:**
`bge-base-en-v1.5` has *published* SciFact scores. That converts the eval from
"is this technique better?" into "is my pipeline even correct?" Very few learning
setups give you an external correctness check on your own code.

---

## D4 — Metrics: generalize to nDCG and multi-relevant

**Chose:** rewrite `metrics.py` for `{doc_id: grade}` judgement sets, add nDCG@10.

**Why:** Stage 1 had exactly one gold chunk per query. SciFact has 339 qrels
across 300 queries, up to 5 relevant docs for a single claim. Single-gold
recall@k cannot express that, and nDCG@10 is BEIR's headline metric, which is
what makes comparison against published numbers possible at all.

Three metrics kept, because they fail differently:

- `recall@k`: did the generator get a chance? If this is 0.6, no prompt
  engineering helps.
- `MRR@k`: how high did the first hit land? Cheap, ignores everything after it.
- `nDCG@k`: order-sensitive, counts the 2nd and 3rd relevant doc. Comparable to
  leaderboards.

Stage 1 was re-run after the refactor and produced identical numbers, confirming
the generalization did not change behaviour.

**Also chose:** string document ids everywhere. Stage 1 used ints, BEIR ids look
like `"31715818"`. One id type across the repo beats the mild ugliness of `str(0)`.

---

## S3 — Truncation analysis, before choosing a chunking strategy

```
SciFact: 5183 documents
  words     min=33   p50=204   p90=309   p99=473   max=1541
  ~tokens            p90=402   p99=615   max=2003

  over the 512-token limit: 145/5183 (2.8%) - silently truncated
  estimated tokens never embedded: 20,688 (1.4% of corpus)

  queries: p50=12 words, max=29 words
```

**Mechanism worth knowing:** bge-base has a 512-token window. Longer input is not
rejected, it is silently truncated. You get a vector back, it looks fine, and it
was built from a prefix of your text. No error, no warning.

**The finding, which is a negative one:** only 2.8% of documents overflow, and
1.4% of corpus tokens are lost. On this corpus chunking has almost nothing to
rescue.

That is a prediction made from data before running the experiment. Chunking is
not a universal good. It is a response to two specific problems: documents longer
than the encoder window, or documents mixing unrelated topics. SciFact abstracts
have neither. They are short, single-topic, self-contained.

Note also the asymmetry: 12-word claims searching 250-word abstracts. That length
mismatch is the reason the BGE query prefix exists, which motivates the ablation
in S4.

---

## D5 — Chunk-to-document pooling

**Chose:** `max` pooling with 5x overfetch.

**Rejected:** `sum` (systematically favours long documents, which have more
chunks and therefore more chances to accumulate score) and `mean` (dilutes one
perfect passage with the rest of the document).

**Why this is a design decision and not glue code:** the moment you chunk, your
index and your relevance judgements stop speaking the same language. The store
returns chunks, BEIR judges documents. How you pool changes your score.

**The overfetch trap:** to rank k documents you must fetch more than k chunks,
because several top chunks routinely come from the same document, especially
with overlap where near-duplicate neighbours crowd the top. Fetch exactly k and
you can silently return fewer than k documents, quietly capping recall@k.

---

## S4 — Document-level baseline and the convention ablation

No chunking. Each abstract embedded whole. Two ablations that look like trivia
and are not.

```
config                              nDCG@10     R@10      MRR
title, no prefix                     0.7588   0.8802   0.7246
title + prefix  (BEIR convention)    0.7449   0.8709   0.7105
no title, no prefix                  0.7416   0.8697   0.7042
no title, prefix                     0.7346   0.8752   0.6961
```

**Result 1, the pipeline is validated.** The BEIR-convention config scored
nDCG@10 = 0.745 against a published bge-base-en-v1.5 SciFact figure of roughly
0.741. Close enough to conclude the embedding call, normalization, top-k, and
nDCG implementation are all sound. Everything built on top of this inherits that
confidence.

**Result 2, titles help.** +0.017 with no prefix, +0.010 with prefix. Consistent
direction, and it makes sense: an abstract's title is a dense summary of it.

**Result 3, the interesting one. The query prefix hurt.** -0.014 with title,
-0.007 without. Consistent direction in both conditions.

Everyone treats the BGE instruction prefix as mandatory. On SciFact it costs
points. A plausible reason: SciFact queries are declarative *claims*
("0-dimensional biomaterials lack inductive properties"), not questions, and the
prefix reads "Represent this sentence for searching relevant passages" which is
tuned for question-shaped queries.

**Honest caveat, and this is itself a lesson.** With n=300, a nDCG@10 delta of
0.014 is small relative to sampling noise. The direction being consistent across
both title conditions is weak corroboration, not proof. The correct next step is
a paired significance test over per-query scores, not a bigger claim. Small eval
sets produce noisy deltas, and a table of four numbers invites you to over-read
the ordering.

---

## Open prediction (not yet run)

`03_chunking_sweep.py` compares whole / fixed / sentence / recursive across
sizes and overlaps, doc-level pooled, on the same 300 queries.

**Predicted from S3:** chunking does nothing here, or actively hurts by
fragmenting the context that made an abstract matchable. If that holds, it is
worth more than a win somewhere else, because chunking is the most cargo-culted
step in RAG and this measures the case where it does not apply.

---

## Method rules extracted

1. Never claim an improvement without a number. Every stage re-runs the same eval.
2. Measure retrieval and generation separately. They fail differently and are
   fixed differently.
3. Distrust any single metric, especially from synthetic data. Report the gap
   between an easy and a hard variant.
4. Ablate conventions instead of inheriting them. The BGE prefix is convention
   and it cost points here.
5. Predict from data before running the experiment. A confirmed negative result
   is a real result.
6. State the noise floor. A delta smaller than sampling error is not a finding.
