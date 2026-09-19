# Learning RAG, basic to advanced

A guided path through this repo. Read a section, run its code, look at what
comes back, then move on. Everything here is measured on the same benchmark, so
you can check every claim yourself.

**Who this is for.** You can write Python and you have heard of embeddings. You
do not need to know what BM25 or nDCG is; both are explained below.

**How to use it.** Do not read this straight through. Each part names one thing
to run. Run it. The numbers on your machine should match the ones here, and if
they do not, that is worth chasing.

```bash
pip install -r requirements.txt
./scripts/download_data.sh          # BEIR SciFact, ~8MB
```

You need [Ollama](https://ollama.com) with `bge-base-en-v1.5` (embeddings) and
`gemma4:e4b` (generation).

---

# Part 1 — What RAG is, and what breaks

## The idea

A language model knows what was in its training data. It does not know your
documents. RAG fixes that the dumbest way possible: find the relevant documents
first, paste them into the prompt, then ask the question.

That is the whole thing. Four steps:

```
1. INDEX     turn each document into a vector, store it
2. RETRIEVE  turn the query into a vector, find the nearest documents
3. AUGMENT   paste those documents into the prompt
4. GENERATE  ask the model, which now has the context it needed
```

## The mechanism: why vectors work at all

An embedding model maps text to a list of numbers, typically 768 of them, such
that texts *about similar things* land near each other. "Cats sleep a lot" and
"felines are frequently dormant" share almost no words but end up close
together, which is exactly what keyword search cannot do.

"Near" means cosine similarity: the angle between two vectors, ignoring their
length.

```
cos(a, b) = (a · b) / (|a| · |b|)
```

If you normalise every vector to length 1 up front, that collapses to just the
dot product `a · b`, which is one multiply-add per dimension. This is the single
most useful optimisation in the repo and it shows up in Part 2.

## Run this

```bash
python concepts/01_naive_rag/naive_rag.py
```

Ask it something about cats. It works.

## What it hides

It works because the corpus was chosen to make it work: 150 cat facts, one
self-contained fact per line. That single property means no chunking is needed,
retrieval is nearly trivial, and `readlines()` passes for an ingestion pipeline.

Six things this version has no answer for:

| Problem | Why it matters | Part |
|---|---|---|
| No chunking | Real documents are not one fact per line | 4 |
| No evaluation | You cannot tell an improvement from a regression | 2 |
| Dense-only | Embeddings are bad at exact terms, IDs, rare words | 5 |
| No abstention | The prompt says "do not make things up" with no mechanism | 7 |
| Re-embeds every run | 150 HTTP calls per experiment, so you stop experimenting | 2 |
| Recomputes norms | Work whose answer can never change | 2 |

**Take away:** most RAG tutorials are demonstrations on data chosen so the
demonstration succeeds.

---

# Part 2 — Measurement, the part everyone skips

This is the most important part of the guide. Skip it and everything after is
guesswork.

## The idea

You are about to be offered a dozen techniques. Each takes a few hours. Without
a number, you cannot tell which ones helped, and "it seems better" is how people
end up shipping a pipeline that is worse than where they started.

## Three metrics, and why you need all three

Say the correct document for a query is ranked 4th.

**recall@k** — was it in the top k? At k=3 no, at k=5 yes. This answers *did the
generator even get a chance?* If recall@5 is 0.6, no amount of prompt
engineering helps: 40% of the time the answer is not in the context window.

**MRR** — mean reciprocal rank, `1/rank`. Here 1/4 = 0.25. This answers *how
high did it land?*

**nDCG@k** — like MRR but it accounts for multiple relevant documents and
discounts lower positions smoothly. This is the standard in information
retrieval and the one to quote if you want to compare against published numbers.

They fail differently, which is the point. In this repo, paraphrasing the
queries barely moved recall@5 (0.987 → 0.960) while MRR fell hard (0.955 →
0.809). The right document was still being *retrieved*, just ranked lower. That
is a ranking problem, not a recall problem, and the two have different fixes.

## The trap that catches nearly everyone

You need labelled data: queries paired with their correct document. The obvious
shortcut is to have an LLM read each chunk and write a question about it. The
label is free, since you know which chunk it came from.

This inflates your scores, and it is not subtle.

Questions generated from a chunk inherit that chunk's rare vocabulary. Ask "what
is the technical term for a cat's hairball" and the word *bezoar* is right there
in both query and document. Cosine similarity is being handed the answer.

```
direct       recall@1 = 0.927
paraphrased  recall@1 = 0.680
             leakage gap: 0.247
```

Same retriever, same corpus. A quarter of the apparent accuracy was vocabulary
overlap. The paraphrased set asks "what do you call those clumps of fur cats
throw up" instead, which is what a real user types.

**Always generate both variants and report the gap.** A single synthetic number
tells you nothing.

## Make iteration free

Embeddings are a pure function of `(model, text)`. Hash the input, cache the
output, and re-indexing costs nothing.

```
                    before    after
index, cold          2.631s   1.144s   (batching)
index, re-run        2.631s   0.000s   (cache)
100 queries          0.565s   0.003s   (normalise once, then matmul)
```

The 190x on query looks like the headline. The 0.000s re-index is what actually
matters, because Parts 4 through 8 re-run this pipeline dozens of times. At 2.6s
per run you would stop experimenting.

## Run this

```bash
python concepts/02_measure_first/01_fast_loop.py       # the speedup
python concepts/02_measure_first/03_measure_baseline.py # the leakage gap
```

## Exercise

Write ten queries by hand for your own documents and label the correct answer
for each. Ten is enough to catch a catastrophic regression. Most teams have
zero.

---

# Part 3 — Check that your pipeline is correct

## The idea

You now have a number. Is it the *right* number? A bug in your normalisation or
your nDCG would produce plausible-looking scores and silently corrupt every
experiment after it.

There is a cheap way to find out. Public benchmarks exist where your exact
embedding model has a published score. Run one. If you land near it, your whole
pipeline is probably correct.

```
this repo:  nDCG@10 = 0.745
published:  nDCG@10 ≈ 0.741
```

That check is worth an afternoon and I would not build on an unvalidated
pipeline again.

## While you are here, question the defaults

BGE models ship with an instruction to prefix queries with "Represent this
sentence for searching relevant passages:". Every tutorial repeats it. On this
corpus it *cost* 0.014 nDCG.

Probably because SciFact queries are declarative claims rather than questions,
and the instruction was tuned for question-shaped input. The delta is small
enough that I would want a significance test before making a strong claim, and I
have not run one.

**Take away:** conventions are defaults, not laws. Ablate them.

## Run this

```bash
python concepts/03_real_benchmark/01_beir_baseline.py
```

---

# Part 4 — Chunking, and when not to

## The idea

Real documents are longer than the encoder's input window. `bge-base` accepts
512 tokens. Longer input is not rejected, it is **silently truncated**: you get
a vector back, it looks fine, and it was built from a prefix of your text.

So you split documents into chunks. The strategies, in increasing respect for
the text:

- **fixed** — N words with overlap. Ignores structure, will cut a sentence in half
- **sentence** — pack whole sentences to a budget
- **recursive** — try paragraph breaks, then sentences, then words. This is what
  LangChain's default splitter does and it is a sensible default

**Overlap** exists because a boundary can orphan context. "This effect was not
observed in the control group" is useless if the previous chunk said which
effect.

## The part nobody tells you

Chunking is not free, and it is not always right. Before splitting anything,
measure your document lengths against your model's window.

```
SciFact: 5183 documents
  over the 512-token limit: 145 (2.8%)
  tokens never embedded: 1.4% of corpus
```

Only 2.8% overflow. So chunking has almost nothing to rescue here, and I
predicted before running it that all configs would lose. All six did.

```
whole                      0.7588
recursive 200/40           0.7488   -0.0100
fixed 200/40               0.7477   -0.0111
sentence 100/1             0.7401   -0.0187
recursive 100/20           0.7384   -0.0204
```

Two things in that table worth more than the headline. Smaller chunks are
strictly worse, because every split discards context that made the abstract
matchable. And at size 200 the three *strategies* span 0.0026 while the penalty
for chunking at all is 0.011. People argue endlessly about which splitter to
use. Here that argument was worth a fifth of the decision nobody argues about.

## The problem chunking creates

Your index now returns chunks, but your labels judge documents. Something has
to pool one into the other, and that choice changes your score:

- `max` — a document scores as its best chunk. The right default
- `sum` — favours long documents, they have more chunks to accumulate score
- `mean` — dilutes one perfect passage with the rest

And **overfetch**: to rank k documents you must fetch more than k chunks,
because several top chunks routinely come from the same document. Fetch exactly
k and you silently cap recall@k.

## Run this

```bash
python concepts/04_chunking/01_truncation_check.py     # the diagnostic
python concepts/04_chunking/02_chunking_sweep.py report
```

## Exercise

Run the truncation check on your own corpus first. If under 5% of documents
overflow and they are single-topic, do not chunk.

---

# Part 5 — Keyword search, and combining it with vectors

## The idea

Embeddings are bad at exact terms. A product SKU, an error code, a gene name, a
rare proper noun: a 768-dimensional vector trained on general text has no
particular reason to preserve them.

BM25 is the opposite. It is a 1994 algorithm with no learned parameters that
scores documents by how often the query's terms appear, weighted by how rare
each term is.

```
                                tf(t,d) · (k1 + 1)
    idf(t) · ------------------------------------------------
             tf(t,d) + k1 · (1 - b + b · len(d) / avg_len)
```

Three ideas:

- **idf** — a term appearing in 5 of 5,000 documents is far more informative
  than one in 4,000. This is precisely the signal embeddings lose
- **k1 = 1.2** — saturation. The 10th occurrence of a word adds much less than
  the 2nd. Without this, keyword spam wins
- **b = 0.75** — partial length normalisation, because long documents contain
  more of everything

It scores 0.665 here against dense retrieval's 0.759, while indexing in 0.21s
and answering in 0.7ms with no GPU.

## Combining them: measure the ceiling first

Hybrid search only pays if the two methods fail on *different* queries. So check
before you build:

```
both         233   77.7%
dense only    34   11.3%
bm25 only      9    3.0%     <- the entire headroom
neither       24    8.0%
```

Nine queries out of 300. Hybrid could gain at most 0.030 recall@10 here.

Look at what those nine contain: "CTCF anchor sites", "de novo assembly...
contigs", "statins". Rare technical terms, exactly the predicted failure mode.
The theory and the data agreed.

## Fusion, and why the default lost

You cannot just add the scores. Cosine lives in [-1,1] and clusters near 0.8;
BM25 is unbounded. Two ways out:

**RRF** throws scores away and uses rank only: `score(d) = Σ 1/(k + rank)`. No
tuning, no normalisation, universally recommended.

**Weighted fusion** normalises each retriever's scores per query, then blends
with a weight you choose.

```
dense only          0.7588
RRF k=60            0.7347   -0.0241
RRF k=10            0.7499   -0.0089
weighted 0.7/0.3    0.7689   +0.0101
```

RRF lost in all four variants I tried. It treats a weaker retriever's rank-1 as
equally credible as a stronger one's, so BM25's wrong answers got promoted over
dense's right ones.

**RRF is good advice when your retrievers are comparable in strength.** Dense
(0.759) clearly dominates BM25 (0.665) here, so equal weighting drags it down.

## Run this

```bash
python concepts/05_keyword_search/01_bm25_from_scratch.py
python concepts/06_hybrid_search/01_complementarity.py
python concepts/06_hybrid_search/02_hybrid_eval.py
```

---

# Part 6 — Reranking, and the rule that took three tries

## The idea

Everything so far is a **bi-encoder**: query and document are embedded
*separately*, then compared with a dot product. That is what lets you precompute
the corpus once. It is also the limitation, because all the interaction between
query and document has to survive being compressed into one vector each.

A **cross-encoder** puts query and document through the transformer *together*,
so attention can compare individual terms. Much more accurate, and impossible to
precompute: one forward pass per pair. So it can only be a second stage over a
shortlist.

The standard advice is: retrieve 50 cheaply, rerank them precisely.

## What happened

It made things worse. Both models, every depth, monotonically.

```
dense (no rerank)          0.759
dense top-10  → rerank     0.747
dense top-50  → rerank     0.726
dense top-100 → rerank     0.717
```

That monotonic decline is a diagnostic worth remembering: a reranker that helps
gets *better* with depth, because more candidates raise the ceiling it can reach.
One that is worse than your ordering gets steadily worse, because every extra
candidate is another chance to promote something wrong.

## Three hypotheses

I checked for a bug first. Gold documents scored +0.575 against +0.055 for
everything else, so the reranker discriminates fine. It just moves the gold
document up as often as down.

**Guess 1: rerankers repair weak orderings.** Tested against multi-query, whose
ordering is weak. Predicted a gain, got −0.011. Wrong.

**Guess 2: rerankers help when decorrelated from the first stage.** Measured
rank correlation. BM25 correlated *most* with the reranker and was the only
stage that gained. Wrong, and backwards.

**Guess 3, which held.** Reranking *overwrites* your ordering. You end up with
the reranker's ranking quality, bounded by what the candidate set contains:

```
first stage     stage nDCG   cand recall   reranked nDCG
BM25                0.6518        0.8833          0.6738
dense               0.7007        0.9600          0.6844
multi-query         0.7018        0.9600          0.6909

input spread  0.0501
output spread 0.0171
```

Three first stages spanning 0.050 collapse to 0.017. The two with identical
candidate recall land at nearly identical scores.

## Two rules you can act on

1. **Rerank only if the reranker ranks better than your first stage on your
   data.** Since it replaces your ordering, that is the whole question. BM25 at
   0.665 gained +0.041. Dense at 0.759 lost 0.033. Same reranker.
2. **Under reranking, optimise your first stage for recall, not ordering.** Its
   ordering is about to be discarded. This inverts the usual instinct.

## Run this

```bash
python concepts/07_reranking/01_rerank_eval.py
python concepts/07_reranking/04_convergence.py
```

---

# Part 7 — Query rewriting, and the generation half

## Fixing the query instead of the index

Part 2 proved the failure mode is real: paraphrasing the queries dropped
recall@1 from 0.927 to 0.680 with the corpus untouched. So rewrite the query.

- **multi-query** — generate paraphrases, retrieve each, fuse
- **HyDE** — have the model write a hypothetical *answer* and embed that instead
- **step-back** — ask a broader question, retrieve for both
- **decomposition** — split a compound question into parts

```
multi_query   +0.0011   nDCG, but +0.040 recall
hyde          -0.0481
step_back     -0.0660
decompose     -0.0948
```

The ordering is not random. **Techniques that add to the original query survive;
techniques that replace it do not.**

HyDE assumes queries and documents live in different regions of embedding space.
SciFact claims are already written like abstract sentences, so there is no gap to
close and only the hallucination remains. For "1/2000 in UK have abnormal PrP
positivity" it invented "4 ± 1 per million", off by a factor of 250.

Multi-query is worth understanding: it raised recall by 0.040 while nDCG stayed
flat. It *finds* more and ranks no better. Recall and precision moved
independently, which is why you track both.

## The generation half

Retrieval metrics cannot see failures here. Build a three-condition test:

| Context | Correct behaviour |
|---|---|
| the right document | answer, with a citation |
| top-ranked documents, right one removed | say "not enough information" |
| nothing | say "not enough information" |

```
gold                      0.40   (want a verdict)
distractor                0.83   (want NOT_ENOUGH)
empty                     1.00   (want NOT_ENOUGH)
```

Every guide warns that models answer confidently from irrelevant context. This
one does the opposite: it refuses correctly on distractors, and *also* refuses
on 60% of cases where it holds the right document.

Reframing the task and relaxing the prompt each bought 0.07 and then stopped, so
the **generator** is the ceiling, not the retrieval or the wording. A system with
perfect retrieval and this model still fails on more than half of SciFact.

Both over-answering and over-refusing are failures. They need opposite fixes,
and only one of them gets written about.

## Run this

```bash
python concepts/08_query_rewriting/01_transform_eval.py report
python concepts/11_grounded_generation/02_claim_verification.py 10
```

---

# Part 8 — Advanced: multi-vector, self-correction, and scale

## Late interaction

One vector per *token* instead of per document. Each query token finds its best
match anywhere in the document, and those maxima are summed.

```
MaxSim(q,d) = Σ over query tokens qi:  max over doc tokens dj: qi · dj
```

It wins: 0.8911 against 0.8804. It also needs a **241x larger index** and 28ms
per query instead of under 1ms.

Put that next to hybrid search, which bought the same +0.01 for a 0.21s BM25
build. Identical accuracy gain, three orders of magnitude apart in cost. A
leaderboard sorted by nDCG puts them adjacent and tells you nothing.

**Report cost alongside every delta.**

## Self-correction

Two mechanisms, opposite outcomes on the same model.

**Self-RAG routing** asks whether a query needs retrieval at all. 16 for 16.

**CRAG grading** asks whether a retrieved document is actually relevant:

```
flag rate | retrieval SUCCEEDED    0.875
flag rate | retrieval FAILED       0.875
separation                        +0.000
balanced accuracy                  0.500
```

It flags 87.5% of documents regardless of whether retrieval worked. Read alone,
"0.88 failure detection" looks excellent. It is an artifact of saying bad to
almost everything.

**A conditional metric is meaningless without its base rate.** This repo produced
two such numbers, here and in the faithfulness score of Part 7, and both
dissolved once the other half was measured.

Why one works and the other does not: routing is surface classification, which a
small model reads easily. Grading needs the same expert inference the generator
ceiling already blocked. **Spend LLM calls where the task is classification, not
where it is expertise.**

## Scale

| Corpus | What you need |
|---|---|
| 10²–10⁴ | numpy brute force. Most products live here |
| 10⁴–10⁶ | HNSW or pgvector. You accept approximate recall for the first time |
| 10⁶–10⁸ | Qdrant/Vespa, int8 or PQ quantization |

Exact numpy search handled 5,183 documents in under a millisecond and scales to
a few hundred thousand vectors in tens of milliseconds. People adopt a vector
database one to two orders of magnitude before they need one.

---

# Common misconceptions, collected

| Belief | What I measured |
|---|---|
| Chunking is a standard preprocessing step | All 6 configs lost on a corpus that did not need it |
| RRF is the safe fusion default | Lost in all 4 variants; weighted won |
| Adding a reranker improves things | −0.033 on dense, +0.041 on BM25. Same reranker |
| HyDE fixes query/document mismatch | −0.048 when there is no mismatch to fix |
| Models hallucinate from bad context | This one over-refused instead |
| LLM graders can filter bad retrievals | Balanced accuracy 0.500 |
| You need a vector database | 5,183 docs search in under 1ms with numpy |

None of these mean the technique is bad. They mean it is **conditional**, and
the condition is almost always whether your current pipeline is already good at
the thing the technique fixes.

---

# Where to go next

- [FLOW.md](FLOW.md) — the same material as a decision procedure, once you know
  the techniques and need to choose between them
- [NOTES.md](NOTES.md) — what broke and what I got wrong, including both dead
  ends in Part 6
- [SYSTEMS.md](SYSTEMS.md) — the full taxonomy, including systems this repo has
  not measured

**The one thing to take away.** The techniques are not the skill. Each is a few
hours of work and most are already in a library. The skill is refusing to add
one until you have measured the gap it is supposed to close. On this benchmark
that discipline was the difference between +0.041 and −0.048 from the same
family of ideas.
