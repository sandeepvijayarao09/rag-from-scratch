# Stage 3 — Move to a real benchmark

**Tier: Beginner** · prerequisite: Stage 2

Cat facts got you a working loop. It cannot tell you whether your loop is
*correct*, because you invented the labels yourself.

Corpus from here on: **BEIR SciFact** — 5,183 scientific abstracts, 300 test
queries, 339 relevance judgements (1.13 relevant docs per query, up to 5).

Chosen for one reason: `bge-base-en-v1.5` has **published** SciFact scores. That
converts the eval from *"is this technique better?"* into *"is my pipeline even
correct?"* Very few learning setups give you an external check on your own code.

## Result: the pipeline is sound

```
config                              nDCG@10     R@10      MRR
title, no prefix                     0.7588   0.8802   0.7246
title + prefix  (BEIR convention)    0.7449   0.8709   0.7105   <- published ~0.741
no title, no prefix                  0.7416   0.8697   0.7042
no title, prefix                     0.7346   0.8752   0.6961
```

The convention config scored **0.745** against a published **~0.741**. The
embedding call, normalisation, top-k, and nDCG implementation all check out.
Everything built on top inherits that confidence.

## Two ablations that look like trivia and are not

**Titles help.** +0.017 without the prefix, +0.010 with it. An abstract's title
is a dense summary of it, and every published BEIR number indexes `"title. text"`,
not text alone.

**The query prefix hurt.** −0.014 with title, −0.007 without. Consistent
direction in both conditions.

The BGE instruction prefix (`"Represent this sentence for searching relevant
passages: "`) is treated as mandatory. On SciFact it costs points. Likely
because SciFact queries are declarative *claims* ("0-dimensional biomaterials
lack inductive properties"), not questions, while the prefix is tuned for
question-shaped input.

**Conventions are defaults, not laws. Ablate them.**

## Caveat, and it is part of the lesson

With n=300, a 0.014 nDCG delta is small relative to sampling noise. Consistent
direction across both title conditions is weak corroboration, not proof. The
correct next move is a paired significance test over per-query scores, not a
louder claim.

Small eval sets produce noisy deltas, and a four-row table invites you to
over-read the ordering.

## Also worth noticing

Queries are ~12 words; abstracts are ~250. That asymmetry is exactly the gap the
query prefix was built to bridge, which is what made this ablation worth running
and what Stage 8's HyDE will attack from the other side.
