# Stage 6, Hybrid retrieval

**Tier: Core retrieval** · prerequisite: Stage 5

Hybrid is the most-recommended upgrade in RAG, so I wanted to know whether it
earns that on SciFact. The answer turned out more interesting than yes.

## 6a, Measure the ceiling first

Hybrid only pays if dense and sparse fail on *different* queries. So decompose
the 300 queries by which retriever found the gold document in its top 10:

```
  both         233   77.7%  #######################################
  dense only    34   11.3%  ######
  bm25 only      9    3.0%  ##
  neither       24    8.0%  ####

  dense alone recall@10 : 0.890
  perfect-fusion ceiling: 0.920   (+0.030 from 9 bm25-only queries)
  unreachable by fusion : 0.080
```

**`bm25 only` is the entire headroom: 9 queries, +0.030 recall@10.** No fusion
scheme can beat that, and a real one will capture less.

The mechanism is visible in those 9 queries:

> "Recurrent mutations occur frequently within **CTCF anchor sites** adjacent to oncogenes."
> "**De novo assembly** of sequence data has more specific **contigs**…"
> "**Statins** increase blood cholesterol."

Rare technical terms. Exactly dense retrieval's predicted failure mode, and
exactly what idf is built to exploit. The theory and the data agree.

**The habit worth stealing:** measuring the ceiling took one script and told us
hybrid was worth at most +0.03 here. That is the difference between an
informed build and a hopeful one.

## 6b, What fusion actually captured

```
config                 nDCG@10   vs dense     R@10      MRR
dense only              0.7588              0.8802   0.7246
bm25 only               0.6652    -0.0936   0.7833   0.6349
RRF k=60                0.7347    -0.0241   0.8394   0.7082
RRF k=10                0.7499    -0.0089   0.8772   0.7164
RRF k=60 w=2:1          0.7464    -0.0123   0.8567   0.7179
weighted 0.5/0.5        0.7528    -0.0060   0.8879   0.7164
weighted 0.7/0.3        0.7689    +0.0101   0.8899   0.7358
weighted 0.9/0.1        0.7663    +0.0075   0.8852   0.7323
```

**RRF, the universally recommended default, made things worse.** -0.024 at
k=60, -0.009 at k=10, negative in every variant tried.

The reason is structural. RRF throws scores away and uses only rank, which makes
it robust when two retrievers are *comparably strong*. Here they are not: dense
is 0.759, BM25 is 0.665. RRF treats "BM25's rank 1" as just as credible as
"dense's rank 1", so BM25's wrong answers get promoted over dense's right ones.
The 9 queries it rescues cost more than they buy.

**Weighted fusion at 0.7/0.3 won**, +0.0101 nDCG and +0.0097 recall@10. Keeping
score magnitude *and* down-weighting the weaker retriever preserves dense's
advantage while still picking up rare-term hits.

Recall@10 went 0.8802 → 0.8899 against a ceiling of 0.920, so fusion captured
about a third of the available headroom.

**Caveat, same as Stage 3:** +0.010 nDCG at n=300 is within sampling noise for a
single comparison. The RRF *deficit* is larger and consistent across four
variants, so that finding is stronger than the weighted-fusion win.

## So when should you use RRF

"Use RRF, it needs no tuning" is good advice when your retrievers are close in
strength. When one clearly dominates, unweighted RRF drags it down. Neither
choice is universal, and the only way to know which regime you are in is to
measure both retrievers separately first.
