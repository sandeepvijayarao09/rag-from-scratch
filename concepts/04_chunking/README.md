# Stage 4, Chunking

**Tier: Core retrieval** · prerequisite: Stage 3

Chunking is the most cargo-culted step in RAG. This stage measures whether it
earns its place, and the answer here is a clean no, predicted in advance.

## 4a, Run the cheap diagnostic first

```
SciFact: 5183 documents
  words     min=33   p50=204   p90=309   p99=473   max=1541
  ~tokens            p90=402   p99=615   max=2003

  over the 512-token limit: 145/5183 (2.8%) - silently truncated
  estimated tokens never embedded: 20,688 (1.4% of corpus)
```

**Mechanism worth knowing:** `bge-base` has a 512-token window, and longer input
is not rejected — it is **silently truncated**. You get a vector back, it looks
fine, and it was built from a prefix of your text. No error, no warning. 145
abstracts are being quietly cut off.

**The finding is negative, and that is what makes it useful.** Only 2.8% of
documents overflow and 1.4% of tokens are lost. On this corpus chunking has
almost nothing to rescue.

So before running 4b I wrote down the prediction: chunking does nothing here,
or hurts by fragmenting the context that made an abstract matchable.

## 4b, The sweep, and the prediction confirmed

```
strategy                 chunks   nDCG@10   vs whole     R@10      MRR
whole                      5183    0.7588              0.8802   0.7246
recursive 200/40           8091    0.7488    -0.0100   0.8592   0.7204
fixed 200/40               8188    0.7477    -0.0111   0.8642   0.7152
fixed 100/20              15209    0.7464    -0.0123   0.8598   0.7163
sentence 200/1             8205    0.7462    -0.0126   0.8569   0.7168
sentence 100/1            17643    0.7401    -0.0187   0.8509   0.7104
recursive 100/20          14679    0.7384    -0.0204   0.8526   0.7081
```

**Six configurations, six losses.**

**1. The strategy barely matters; the decision to chunk does.** At size 200 the
three strategies span 0.0026 nDCG. The penalty for chunking at all is 0.011 —
more than four times the spread between strategies. People spend enormous energy
choosing fixed vs sentence vs recursive. Here that choice was worth a fifth as
much as the choice not to chunk.

**2. Smaller chunks are strictly worse.** Every split discards surrounding
context, and a SciFact claim is matched by the abstract as a whole: methods,
results and conclusion together.

**3. You pay for the privilege.** `sentence 100/1` triples the index (17,643 vs
5,183 chunks) and costs 0.019 nDCG. More storage, more embedding time, more
query work, worse results.

## The pooling problem chunking creates

The moment you chunk, your index returns **chunks** while your relevance
judgements score **documents**. Something must pool one into the other, and that
choice changes your score. See [`../../rag/retrieve.py`](../../rag/retrieve.py).

- `max`, a doc scores as its best chunk. The right default.
- `sum`, systematically favours long documents; they have more chunks and more
  chances to accumulate score.
- `mean` — dilutes one perfect passage with the rest of the document.

**Overfetch trap:** to rank k documents you must fetch more than k chunks,
because several top chunks routinely come from the same document. Fetch exactly
k and you silently return fewer than k docs, capping recall@k.

## When chunking is actually worth it

Chunking is not a technique you apply, it is a **response to a diagnosed
problem**. The diagnostic is cheap: measure your document length distribution
against your encoder's window, and ask whether your documents mix unrelated
topics. If both answers are no, chunking is pure cost.

Where it genuinely pays: legal contracts, technical manuals, long reports,
transcripts, long, multi-topic, answer-in-one-section documents. SciFact
abstracts are the opposite of all three.

**The habit:** run the cheap diagnostic that predicts the result before running
the expensive experiment. 4a took seconds and told us what 17 minutes of
embedding would confirm.
