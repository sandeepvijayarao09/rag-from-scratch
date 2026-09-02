# Stage 10, Multi-vector / late interaction

**Tier: Advanced** · prerequisite: Stage 9 · status: code written, not yet run

The third point on a spectrum you have now seen both ends of.

| | interaction | precompute | index size / doc |
|---|---|---|---|
| **Bi-encoder** (Stages 2-6) | after encoding, one dot product | whole corpus | ~3 KB |
| **Cross-encoder** (Stage 7) | query+doc through the transformer together | impossible |, |
| **Late interaction** (here) | deferred, but over *stored* vectors | whole corpus | ~768 KB |

```
MaxSim(q, d) = SUM over query tokens qi:  MAX over doc tokens dj: qi · dj
```

Each query token finds its single best match anywhere in the document, and those
maxima are summed. A rare term appearing once in a long abstract still
contributes fully, exactly what mean-pooling destroys.

## The cost is the lesson

For SciFact's 5,183 abstracts: **~16 MB vs ~4 GB.** A 250× index blowup. Real
ColBERTv2 attacks this with residual quantization (~2 bits/dim) plus centroid
pruning, getting it to roughly 20×.

This is why `01_maxsim.py` indexes 800 documents (keeping every gold document
plus random distractors) and reports the extrapolation rather than faking a
full-corpus run.

## Caveat, stated up front

These token vectors come from `bge-base`, trained for *single-vector* retrieval
with CLS pooling. Real ColBERT checkpoints are trained end-to-end under the
MaxSim objective, so their token vectors are shaped for this operation.

Expect the mechanism to work and the numbers to underperform trained ColBERT.
**That gap is the point: an architecture is not separable from the objective it
was trained under.**
