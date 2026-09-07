# Stage 10 — Multi-vector / late interaction

**Tier: Advanced** · prerequisite: Stage 9

The third point on a spectrum whose other two ends are already in this repo.

| | interaction happens | precomputable | index per doc |
|---|---|---|---|
| Bi-encoder (Stages 2-6) | after encoding, one dot product | yes, whole corpus | ~3 KB |
| Cross-encoder (Stage 7) | inside the transformer, jointly | no | n/a |
| Late interaction (here) | at query time, over stored vectors | yes | ~768 KB |

```
MaxSim(q, d) = SUM over query tokens qi:  MAX over doc tokens dj: qi · dj
```

Every query token finds its single best match anywhere in the document, and
those maxima are summed. A rare term that appears once in a long abstract still
contributes fully, which is exactly what mean-pooling into one vector destroys.

## Results

800-document subset (every gold document plus random distractors), 300 queries.
Both methods scored on the same subset, so the comparison is fair even though
the absolute numbers are higher than the full-corpus tables elsewhere.

```
                     nDCG@10     R@10   index MB   ms/query
single-vector         0.8804   0.9516        2.5         <1
late interaction      0.8911   0.9667      592.5         28
```

It wins: +0.011 nDCG@10 and +0.015 recall@10.

It also costs a **241x larger index** and roughly 28x the query latency.
Extrapolated to the full 5,183 abstracts that is 3.84 GB against 16 MB.

## Putting that next to the other two things that worked

Three techniques in this repo improved retrieval. Two of them bought about the
same amount:

| Technique | Δ nDCG@10 | What it costs |
|---|---|---|
| Weighted hybrid (Stage 6) | +0.010 | one BM25 index: 0.21s build, 0.7ms/query |
| Late interaction (Stage 10) | +0.011 | 241x index size, 28ms/query |
| Reranking on BM25 (Stage 7) | +0.041 | 1,515ms/query |

Hybrid and late interaction deliver an indistinguishable gain, and one of them
is essentially free while the other needs three orders of magnitude more
storage. If you only read the accuracy column you would never see that.

This is the argument for reporting cost alongside every delta. A leaderboard
sorted by nDCG puts these two techniques next to each other and tells you
nothing about which one you should build.

## An honest surprise

I expected this to underperform, because these token vectors come from
`bge-base`, which was trained for single-vector retrieval with CLS pooling.
Real ColBERT checkpoints are trained end-to-end under the MaxSim objective, so
their token vectors are shaped for exactly this operation. Borrowing the
architecture without its training objective should have cost something.

It won anyway. Token-level matching is apparently robust enough to help even
with representations that were not optimised for it. A trained ColBERT would
presumably do better still, and I have not tested one.

## Why the subset

Full-corpus late interaction needs ~3.84 GB of float32 in RAM. The subset keeps
every gold document so no query becomes unanswerable, then pads with random
distractors. Real ColBERTv2 attacks the memory problem with residual
quantization at roughly 2 bits per dimension plus centroid pruning, which gets
the blowup down to about 20x rather than 241x. That compression is most of what
makes the approach practical, and it is not implemented here.
