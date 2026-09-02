# Stage 7, Cross-encoder reranking

**Tier: Intermediate** · prerequisite: Stage 6

Reranking is the second-most-recommended upgrade in RAG. On SciFact it made
things worse. Chasing down why took two wrong guesses and produced the most
useful rule in this repo.

## The architecture

| | interaction | precompute? | cost |
|---|---|---|---|
| **Bi-encoder** (Stages 2-6) | after encoding, one dot product | whole corpus, once | ~0ms/query |
| **Cross-encoder** (this) | query + doc through the transformer *together* | impossible | one forward pass per pair |

A bi-encoder compresses a whole abstract into 768 numbers before it has ever
seen the query. A cross-encoder lets self-attention compare individual terms
across both. Strictly more expressive, and strictly not precomputable, which is
why it can only ever be a *second* stage over a shortlist.

## 7a, Both rerankers hurt

```
config                        nDCG@10   vs dense    R@10     ms/q
hybrid 0.7/0.3                 0.7689    +0.0101   0.8899       0
dense                          0.7588              0.8802       0
dense top-10  -> bge-base      0.7474    -0.0114   0.8802     303
dense top-20  -> bge-base      0.7383    -0.0205   0.8734     606
dense top-50  -> bge-base      0.7256    -0.0332   0.8544    1515
dense top-100 -> bge-base      0.7172    -0.0416   0.8444    3030
dense top-10  -> MiniLM-L6     0.7276    -0.0312   0.8802      39
dense top-100 -> MiniLM-L6     0.6908    -0.0680   0.8122     389
```

Two models, four depths, monotonically worse. **The monotonic decline is the
diagnostic.** A reranker that helps gets *better* with depth, because deeper
candidates raise the ceiling it can reach into. A reranker worse than the
ordering it replaces gets *steadily worse*, because every extra candidate is
another chance to promote something wrong.

Note `dense top-10 -> rerank` leaves recall@10 at exactly 0.8802, unchanged. A
reranker can only reorder what stage one handed it. `recall@depth` is a hard
cap and belongs printed next to every rerank result.

## The sanity check, before believing it

```
mean reranker score, GOLD docs : +0.575
mean reranker score, non-gold  : +0.055
separation                     : +0.520

gold mean rank BEFORE rerank   : 2.33
gold mean rank AFTER  rerank   : 2.51
gold moved UP: 10   DOWN: 10   same: 37
```

Not a bug. The reranker discriminates cleanly. It just moves documents up as
often as down.

**Why:** dense already places gold at rank 2.33 out of 20. There is almost
nothing left to repair, and both models are BGE trained on similar data, so the
reranker's judgement is *correlated with* the bi-encoder's rather than superior
to it. A second correlated opinion adds variance, not accuracy.

## 7b, The prediction, and the rule

That explanation is falsifiable: give the reranker a *weak* first stage and it
should help. BM25 scores 0.665, fails on different queries (Stage 6), and its
ordering is far less correlated with the reranker's.

```
first stage             before     after      delta    MRR delta
BM25 (weak)             0.6652    0.7058    +0.0406      +0.0435
dense (strong)          0.7588    0.7256    -0.0332      -0.0326
```

Same reranker, same depth, same 300 queries. **Opposite signs.**

> **Rerankers repair weak orderings, not strong ones.**

That rule is *directionally* right and turns out to be a proxy for something
simpler. See 7c/7d below, where it gets falsified and replaced.

## 7c/7d — Two failed hypotheses, then the real answer

The 7b rule was tested against multi-query in Stage 8b, whose ordering is also
weak (MRR 0.662 vs dense 0.668) but with higher recall. The rule predicted a
gain. **It lost 0.011.** Hypothesis one falsified.

Next candidate: DECORRELATION. Maybe a reranker helps when its judgement is
independent of the first stage, and BM25 is the only stage built on a different
signal (lexical, not the same bi-encoder). Measured with Spearman rank
correlation between each stage's ordering and the reranker's:

```
first stage        Spearman vs reranker     rerank delta
BM25                              0.442          +0.0406
dense                             0.384          -0.0332
multi-query                       0.362          -0.0109
```

**Backwards.** BM25 correlates MOST with the reranker and is the only stage that
gained. Hypothesis two falsified.

Both hypotheses were looking at the inputs. The pattern was in the outputs:

```
first stage     stage nDCG   cand recall@50  reranked nDCG      delta
BM25                0.6518           0.8833         0.6738    +0.0221
dense               0.7007           0.9600         0.6844    -0.0163
multi-query         0.7018           0.9600         0.6909    -0.0109

first-stage nDCG spread : 0.0501
reranked  nDCG spread   : 0.0171
```

Three first stages spanning 0.050 nDCG collapse to a 0.017 spread after
reranking, a 3x compression. The two stages with IDENTICAL candidate recall
(0.9600) land at nearly identical reranked scores; BM25, with lower recall
(0.8833), lands lower.

> **Reranking OVERWRITES the first stage's ordering with the reranker's own.**
> The result is the reranker's ranking quality, bounded by what the candidate
> set contains. The first stage stops mattering as an ordering and matters only
> as a recall filter.

Both earlier hypotheses were proxies for this. BM25's ordering was weak, so
overwriting it helped. Dense's was strong, so overwriting it hurt. Correlation
was a red herring.

## Two rules I would actually use

**Two rules, both actionable:**

1. **Rerank only if the reranker ranks better than your first stage.** Since
   reranking replaces your ordering, the outcome is the reranker's quality. A
   great leaderboard score is irrelevant if your retriever is already better on
   your data. Test it directly: rerank a sample and compare, before committing
   to the latency.

2. **Under reranking, optimise your first stage for RECALL, not ordering.** Its
   ordering is about to be discarded. This inverts the usual instinct: the
   first stage should cast a wide, cheap net, and BM25 or a fast bi-encoder at
   high depth may beat a better-ranked but narrower candidate set.

**On method:** two hypotheses were stated, tested, and falsified before the
third held. Both wrong guesses were published in the repo rather than quietly
deleted, because the sequence is the lesson. A rule that survives one experiment
is a guess; the falsification is what turns it into knowledge.

Latency is the other half. bge-base at depth 50 costs 1,515ms/query against
~0ms for the bi-encoder. On SciFact that buys -0.033 nDCG. Even where reranking
helps, depth is a latency dial you should set deliberately.
