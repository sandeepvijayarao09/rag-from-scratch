# Lab notebook

Running log of what broke, what I got wrong, and things I want to remember.
Not organised. Newest at the bottom.

All of this was built in a small number of long sessions rather than spread over
weeks, which is why the commit dates all land on the same day. The commits are
grouped by stage, not by hour.

---

**Ollama died mid-sweep.**

Chunking sweep blew up on config 2 of 4 with `ConnectionError: Failed to connect
to Ollama`. Config 1 had passed, which confused me for a minute until I realised
config 1 was entirely cache hits and never touched the server at all. The daemon
had been killed earlier when I cleaned up some background jobs. Restarted it,
reran, fine.

Mildly annoying that the failure surfaced two minutes into a run rather than at
startup. A health check before a long job would have caught it. Not adding one
yet; noting it.

---

**Killed my own eval-set generation with `head -4`.**

Piped a 20-minute generation script through `head -4` to peek at the output. head
exits after 4 lines, the write side gets SIGPIPE, python dies. Got 30 of 150
question pairs before it went down.

Cost me nothing because the script checkpoints every 10 items and resumes, which
I had written in for a completely different reason (I expected to interrupt it
manually, not to shoot it). Resumable-by-default earned its keep the first time
it was tested.

---

**Sized a job by guessing. It took 2.7x the budget.**

Queued the reranker sweep at four depths across two models and it blew through a
10-minute limit. Only then did I benchmark the cross-encoder: 264 pairs/sec for
MiniLM, 47.6 for bge-base. The sweep was asking for 132,000 forward passes.

The real fix was not a longer timeout. Reranking the top 20 uses a *subset* of
the scores from reranking the top 100, so scoring every pair once and slicing
gives every depth for free. 27 minutes down to 13, and all four depths instead of
one.

Same idea as the embedding cache from Stage 2. I had already written that cache
and still did not think to apply the pattern one layer up. Benchmark before you
queue.

---

**bge query prefix made things worse.**

Did not expect this. The BGE model card says to prefix queries with "Represent
this sentence for searching relevant passages:" and every tutorial repeats it.
On SciFact it costs 0.014 nDCG@10, consistently, in both title conditions.

Best guess: SciFact queries are declarative claims, not questions, and the
instruction was tuned for question-shaped input. Not certain. The delta is small
enough relative to n=300 that I would want a paired test before making a strong
claim, and I have not run one.

Left it in the README as a caveat rather than a finding.

---

**Chunking lost six times out of six.**

I checked the token distribution first and only 2.8% of abstracts exceed the
512-token window, so I expected chunking to do roughly nothing. It did worse than
nothing. Every configuration lost.

The part I did not predict: at size 200 the three strategies (fixed, sentence,
recursive) land within 0.0026 of each other, while the penalty for chunking at
all is 0.011. I have read a lot of arguments about which splitter to use. On this
corpus that argument was worth about a fifth of the decision nobody argues about.

---

**Two wrong hypotheses about reranking.**

Both rerankers hurt the dense first stage, monotonically worse with depth. First
thought was a bug, so I checked: gold documents score +0.575 against +0.055 for
everything else. The reranker works fine. It just moves the gold document up as
often as down.

Guess 1: rerankers repair weak orderings. Tested it against multi-query, whose
ordering is weak. Predicted a gain, got -0.011. Wrong.

Guess 2: rerankers help when their signal is decorrelated from the first stage.
Measured Spearman correlation. BM25 correlated *most* with the reranker (0.442)
and was the only stage that gained. Wrong, and backwards.

What actually holds is duller than either. Reranking overwrites the ordering, so
you end up with the reranker's ranking quality bounded by candidate recall.
Three first stages spanning 0.050 nDCG collapse to 0.017 after reranking.

Kept both dead ends in the repo. The sequence is more useful than the conclusion,
and deleting failed hypotheses is how you end up with a repo that looks like
everything worked the first time.

---

**HyDE hallucinated numbers that contradicted the query.**

Claim was "1/2000 in UK have abnormal PrP positivity". HyDE generated a passage
asserting prevalence of "4 ± 1 per million", plus some stray LaTeX. Off by a
factor of 250 and pointed the embedding somewhere unhelpful.

The premise of HyDE is that queries and documents live in different regions of
embedding space. SciFact claims are already written like abstract sentences, so
there is no gap to close and the fabrication is all you get. -0.048.

---

**Format bug survived a whole refactor.**

Changed `Chunk.id` from int to str back in Stage 4 to match BEIR's document ids.
`01_fast_loop.py` still had `f"{chunk.id:3d}"` and crashed. It sat broken for
several stages because I never reran the earlier scripts.

Found it only because I ran everything after renaming the folders. Renaming did
not find the bug; running everything afterwards did. Worth doing periodically
rather than assuming untouched code still works.

---

**Cache was 240MB before I looked.**

`cache/embeddings.npz` holds every embedding across every chunking config, which
adds up faster than expected. Gitignored. Split the small result JSONs out into
`results/` so the actual findings are version controlled and the derived blobs
are not.

---

## Open questions

- The prefix and hybrid deltas (0.010 to 0.014) are inside the noise floor at
  n=300. Want a paired bootstrap over per-query scores before treating either as
  real. Not done.
- Every negative result here is on one corpus with a well-matched embedding
  model. SciFact abstracts are short, single-topic and written in the same
  register as the queries. I would expect chunking and query rewriting to flip
  sign on long multi-topic documents, and I have not tested that.
- Multi-query raised recall 0.040 while nDCG stayed flat. Something should be
  able to convert that extra recall into ranking. Reranking did not. Unresolved.
