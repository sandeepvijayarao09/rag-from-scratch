# Stage 5 — Keyword search (BM25)

**Tier: Core retrieval** · prerequisite: Stage 4

No neural network, no GPU, no learned parameters. A 1994 algorithm in ~60 lines
of Python.

```
BM25 index: 5183 docs, 35,704 unique terms, 0.21s
BM25        n=300  r@1=0.528  r@5=0.724  r@10=0.783  MRR@10=0.635  nDCG@10=0.665
            0.7 ms/query
```

Published BM25 on SciFact lands right around 0.665 as well, so the
implementation checks out.

## The comparison that matters

| | index | per query | nDCG@10 |
|---|---|---|---|
| BM25 | 0.21s | 0.7ms | 0.665 |
| bge-base dense | 83s | ~85ms | 0.759 |

Dense wins by 0.094 nDCG. It also costs ~400x more to index and ~120x more per
query. On a corpus where BM25 got you to 0.665 for free, that is the real
tradeoff, and it is worth knowing before you provision a GPU.

## The three ideas inside BM25

```
                                tf(t,d) * (k1 + 1)
    idf(t) * ------------------------------------------------
             tf(t,d) + k1 * (1 - b + b * len(d) / avg_doc_len)
```

- **idf** — a term in 5 of 5,000 documents is far more informative than one in
  4,000. This is precisely what dense embeddings are worst at. A rare gene name,
  error code, or identifier carries enormous signal, and a 768-dim vector
  trained on general text has no particular reason to preserve it.
- **k1 = 1.2** — saturation. `tf/(tf + k1)` is concave, so the 10th occurrence
  of a term adds much less than the 2nd. Without it, keyword spam wins.
- **b = 0.75** — partial length normalisation. Long documents contain more of
  everything. `b=1` normalises fully, `b=0` not at all.

## Why an inverted index

A dense 5,183 x 35,704 term-document matrix would be 622MB of mostly zeros. The
postings list (`term -> [(doc, tf)]`) touches only documents that actually
contain a query term, which is a handful. This is why lexical search scaled to
the web decades before anyone had a GPU.
