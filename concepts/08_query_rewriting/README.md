# Stage 8, Query rewriting

**Tier: Intermediate** · prerequisite: Stage 7

Every stage so far assumed the query was fine and only the matching needed work.
This one assumes the opposite, and Stage 2 gave direct evidence that the failure
is real: paraphrasing the questions dropped recall@1 from 0.927 to 0.680 with
the corpus untouched. The documents were always findable. The words used to look
for them were wrong.

## Results

```
technique          nDCG@10   vs baseline     R@10      MRR   s/query
baseline            0.7007                 0.8433   0.6682       0.0
multi_query         0.7018       +0.0011   0.8833   0.6618       3.1
hyde                0.6526       -0.0481   0.8033   0.6210       0.9
step_back           0.6347       -0.0660   0.8033   0.5941       0.9
decompose           0.6060       -0.0948   0.7833   0.5557       2.4
```

n=50 fixed seeded subsample, baseline scored on the *same* queries so deltas are
paired. Absolute values are not comparable to the full-300 results elsewhere.

## Read the ordering, it is not random

```
multi_query   +0.0011   keeps the original, adds 3 near-paraphrases
hyde          -0.0481   replaces the original entirely
step_back     -0.0660   adds one semantically distant query
decompose     -0.0948   replaces with fragments
```

**Multi-query is the only survivor, and the interesting part is not its nDCG.**
recall@10 rose 0.8433 → 0.8833 while MRR *dipped*. It finds four more gold
documents per hundred and ranks them no better, because RRF fuses by rank alone
and flattens the ordering. Recall and precision moved independently, which is
exactly why this repo tracks both.

**HyDE (−0.048).** Its premise is that queries and documents occupy different
embedding regions, so you write a fake answer to bridge the gap. But SciFact
queries are declarative claims *already written in document-space*. There is no
gap, so all HyDE contributes is fabrication. For the claim *"1/2000 in UK have
abnormal PrP positivity"* it generated *"…estimated prevalence at 4 ± 1 per
m[illion]"*, numbers contradicting the claim, plus stray LaTeX.

**Step-back (−0.066).** The broader question retrieves general background. But
SciFact is claim *verification*: you need the one specific study, not context
around it.

**Decompose (−0.095).** SciFact claims are atomic. Splitting one into
sub-questions fragments a coherent query into pieces that individually match
nothing well.

## 8b — Composition, and a failed prediction

Multi-query produces a high-recall, weak ordering. Stage 7b's rule said that is
exactly what to rerank. Prediction: multi-query + rerank beats both alone.

```
first stage                   nDCG@10     R@10      MRR    rerank Δ
dense            alone         0.7007   0.8433   0.6682
                 + rerank      0.6844   0.8633   0.6412     -0.0163
multi-query      alone         0.7018   0.8833   0.6618
                 + rerank      0.6909   0.8833   0.6438     -0.0109
```

**Wrong.** Still negative. That falsification is what drove Stage 7c/7d and the
correct rule, reranking *overwrites* the ordering, so the outcome is the
reranker's quality, not a blend.

## When this is worth the latency

**Query rewriting targets a mismatch between how queries are phrased and how
documents are written.** Measure that mismatch before reaching for it. If your
users type keywords and your corpus is prose, or your queries are conversational
and your documents are formal, there is a real gap. If your queries already
resemble your documents, rewriting is pure cost, and every technique here costs
an LLM call per query, forever, on the latency path.

**And prefer additive over replacing.** Keeping the original ranking in the fuse
bounds your downside. Replacing it, as HyDE and decompose do, bets everything on
the rewrite being better.
