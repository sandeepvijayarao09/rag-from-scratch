# RAG From Scratch

## In 30 seconds
- **What it is:** A study of RAG (an AI looking up documents before it answers): I built ten commonly recommended techniques and measured each on one benchmark.
- **Why it matters:** Teams often add these techniques by default; this shows which ones help, which hurt, and which make no measurable difference.
- **Key result:** Tested with paired significance tests on 300 queries: LLM query rewriting (HyDE, step-back, decomposition) reliably made search worse; a strong reranker reliably made it better (+0.044 on top of dense search, +0.110 on top of keyword search); the other techniques made no detectable difference. My first version claimed "seven of ten made it worse" from raw means; the tests showed that was overstated, and this README is the corrected version. One corpus only (BEIR SciFact).
- **What I did:** Solo. Stage 1 adapts a public tutorial; everything after is mine.
- **Built with:** Python, NumPy, Ollama, bge-base / embeddinggemma / Qwen3 embeddings, Qwen3 and BGE rerankers, Gemma, PyTorch, pytest (no LangChain, no vector database).
- **Try it:** [sandeepvijayarao09.github.io/RAG](https://sandeepvijayarao09.github.io/RAG/)

I implemented ten recommended RAG techniques and measured each one against the
same benchmark, then re-tested every result with paired significance tests,
three embedding models and three rerankers. Three techniques reliably made
retrieval worse, reranking reliably helped once the reranker was strong enough,
and the rest made no difference I could detect.

This repo is the code, the numbers, and the reasoning for why. It is a
fifteen-stage ladder from naive RAG to production concerns; stages 1 to 12 are
built and measured, 13 and 14 have code but no measurements yet, and 15 is
planned.

[![CI](https://github.com/sandeepvijayarao09/rag-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeepvijayarao09/rag-from-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE)

![Change in nDCG@10 from each technique on BEIR SciFact with bge-base, with 95% confidence intervals: query decomposition, HyDE and step-back are significantly worse; reranking with Qwen3-Reranker-4B, and reranking BM25, are significantly better; the rest are not distinguishable from zero](docs/results.svg)

## The results

Corpus is BEIR SciFact: 5,183 scientific abstracts, 300 labelled test queries.
Metric is nDCG@10. Baseline is dense retrieval with `bge-base-en-v1.5` at
0.759, which I validated against a published figure of roughly 0.741 before
building anything on top of it.

Every row below is a paired comparison on the same 300 queries: a bootstrap
95% confidence interval over per-query scores, Holm-corrected within each
stage. Anything tuned (fusion weights, blend weights) was tuned on the 809
SciFact train queries and scored once on test.

| Technique | Δ nDCG@10 (bge-base) | 95% CI | Verdict |
|---|---|---|---|
| Query decomposition | −0.087 | [−0.120, −0.055] | **worse** |
| HyDE | −0.040 | [−0.069, −0.012] | **worse** |
| Step-back prompting | −0.039 | [−0.064, −0.015] | **worse** |
| Reranking on dense, bge-reranker-base | −0.033 | [−0.062, −0.004] | no detectable difference after Holm |
| RRF fusion (3 variants) | −0.024 to −0.009 | | no detectable difference |
| Chunking (6 configs) | −0.017 to −0.003 | | no detectable difference |
| BGE query instruction prefix | −0.014 | [−0.031, +0.003] | no detectable difference |
| Late interaction (ColBERT, full corpus) | +0.002 | [−0.027, +0.031] | no detectable difference |
| Contextual retrieval (600-doc subset) | +0.001 | [−0.009, +0.011] | no detectable difference |
| Multi-query (3 paraphrases + RRF) | +0.009 | [−0.010, +0.028] | no detectable difference |
| Weighted hybrid 0.7/0.3 | +0.010 | [−0.006, +0.027] | no detectable difference |
| **Reranking on dense, Qwen3-Reranker-4B** | **+0.044** | [+0.019, +0.070] | **better** |
| Reranking on BM25, bge-reranker-base (vs BM25 alone) | +0.041 | [+0.016, +0.067] | **better** |
| **Reranking on BM25, Qwen3-Reranker-4B (vs BM25 alone)** | **+0.110** | [+0.082, +0.140] | **better** |

Query rewriting used `gemma4:12b` at temperature 0. Full per-model tables,
including the repaired variants: [`results/v2/RESULTS.md`](results/v2/RESULTS.md).

What holds up:

- **Rewriting the query with an LLM hurts on this corpus.** With a 12B
  rewriter, the three rewrites are significantly worse in 8 of 9
  technique-by-embedder cases (HyDE on qwen3-embedding-4b, −0.019, is the
  exception). SciFact claims are already the
  best query for their own evidence: a hypothetical answer invents details, a
  broader question retrieves background, and splitting an atomic claim loses
  it. Blending the rewrite back in with the original claim, tuned on train,
  only recovers a tie (+0.002 to +0.014, none significant).
- **Reranking is about the reranker.** The small `bge-reranker-base` cannot
  improve a strong dense ranking, which is what v1 measured; on the two
  stronger embedders it is significantly worse (−0.062 and −0.044). `Qwen3-Reranker-4B`
  improves it on 54 queries and worsens it on 12 (with a tuned blend), and
  lifts BM25 from 0.665 to 0.776.
- **The embedding model matters more than any technique.** `embeddinggemma-300m`
  alone scores 0.799, more than any technique adds on top of bge-base.

What does not hold up: chunking, RRF, the query prefix, multi-query, weighted
hybrid, contextual retrieval and late interaction all sit inside their
confidence intervals on every embedding model, with one exception: on
`embeddinggemma-300m`, the strongest embedder, unweighted RRF (−0.057, and
−0.038 with 2:1 weights) and ColBERT (−0.035 to −0.038) are significantly
worse, because they pull a 0.799 ranking toward weaker ones. They are not shown to hurt; they are shown to do nothing measurable
on a corpus of short, single-topic abstracts.

### What v1 claimed, and what changed

The first version of this repo reported raw mean differences with no
significance tests and said "seven of ten techniques made retrieval worse."
The raw numbers reproduce almost exactly (same 0.759 baseline, same −0.033
reranker loss, same +0.041 on BM25), but the conclusion went further than the
data:

| v1 claim | Re-tested |
|---|---|
| Chunking, RRF, query prefix made it worse | Inside the noise; no detectable difference |
| Reranking on dense made it worse (−0.033) | Not significant after correction, and a stronger reranker makes it +0.044 |
| HyDE, step-back, decomposition made it worse | Confirmed, with a 3x larger rewriter, on all 300 queries |
| "Seven of ten" | Reranking was counted as two of the ten techniques, and "4 RRF variants" was 3 |
| Stage 8 rows in the headline table | Came from a fixed 50-query subsample (noted under Limitations, not in the table) next to full-300 rows; v2 uses all 300 |
| Late interaction +0.011 | Raw token vectors on an 800-doc subset; a trained ColBERT on the full corpus is +0.002 |

The full list of changes between the two versions is in [`v2/README.md`](v2/README.md).

Stage 12 tested self-correction. Routing queries that need no retrieval worked
(16 of 16 test prompts), but CRAG-style document grading flagged 87.5% of
retrievals as bad whether or not they had actually failed: balanced accuracy
0.50, which is chance. Details in
[`concepts/12_self_correction/`](concepts/12_self_correction/).

What this does and does not show: it is not evidence that these techniques
are bad. It is evidence that they are conditional, and that the condition is
usually whether the current pipeline is already good at the thing the
technique fixes. The cheap diagnostic predicted every null result: only nine
of 300 queries are found by BM25 and missed by dense (so fusion has little to
add), and only 2.8% of abstracts exceed bge-base's 512-token window (so
chunking has little to rescue). A reranker helps when it is stronger than the
first stage it sits behind, and not otherwise.

### The generation half is a separate problem

Stage 11 measures what the model does with the documents, and it does not
behave the way the guides warn about.

```
context         gave verdict   NOT_ENOUGH
gold                      12           18    0.40  (want a verdict)
distractor                 5           25    0.83  (want NOT_ENOUGH)
empty                      0           30    1.00  (want NOT_ENOUGH)
```

Handed the correct document, `gemma4:e4b` declines to commit 60% of the time.
Handed irrelevant documents it correctly refuses 83% of the time. It is too
conservative, not reckless, which is the opposite of the failure everyone
writes about.

Reframing the task and relaxing the prompt each bought 0.07 and then stopped,
so the generator is the ceiling for this stage rather than the retrieval or the
wording. A system with perfect retrieval and this model would still fail on more
than half of SciFact. That is the argument for Method Rule 2: had I only
measured end-to-end accuracy I would have gone off tuning retrieval, which was
never the problem.

## Setup

Install [Ollama](https://ollama.com), then pull the two models the code uses.
The embedding model is the GGUF build of `bge-base-en-v1.5` on Hugging Face;
`ollama pull bge-base-en-v1.5` will not find it.

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf   # embeddings (68 MB)
ollama pull gemma4:e4b                                   # generation (9.6 GB)

pip install -r requirements.txt     # or requirements-core.txt to skip torch
./scripts/download_data.sh          # BEIR SciFact, ~8 MB
python concepts/01_naive_rag/naive_rag.py
```

`requirements-core.txt` (numpy, ollama, pydantic) covers every stage except 7
and 10, which need torch and pull cross-encoder weights from Hugging Face on
first run. Stage 9 also uses `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` for
the per-chunk blurbs.

To check the headline baselines with one command (stages 3, 5 and 6, a few
minutes cold):

```bash
./scripts/reproduce.sh              # writes results/logs/*.txt
```

Unit tests for the pure-Python core (metrics, BM25, fusion, chunking) need no
Ollama or data:

```bash
pip install -r requirements-dev.txt && pytest
```

To re-run the tested comparisons (about a day on a laptop, most of it 12B
generation; `fast` skips the LLM stages):

```bash
./v2/run_all.sh fast && python v2/report.py
```

Everything runs locally. Embeddings, LLM outputs and cross-encoder scores are
cached to disk, so re-running a stage costs nothing and only new work is
computed.

## Where to start

The same material is published as a guide at
[sandeepvijayarao09.github.io/RAG](https://sandeepvijayarao09.github.io/RAG/)
(generated by `scripts/build_site.py`).

Two entry points, depending on why you are here.

**Learning RAG?** [**LEARN.md**](LEARN.md) is a guided path from "what is an
embedding" to late interaction and self-correction. Eight parts, each with
something to run and a result to check against. It explains the mechanisms
rather than just reporting the numbers.

**Already know the techniques and deciding which to use?**
[**FLOW.md**](FLOW.md) is a build order with a diagnostic gate before each step,
so you can work out what applies to your corpus instead of trying all of it.

## Layout

```
LEARN.md      guided path, basic to advanced. Start here if learning.
FLOW.md       0-to-1 decision procedure. Start here if building.
concepts/     15 stages, beginner to pro. Code plus a README per stage.
rag/          shared library: retrieval, metrics, chunking, fusion, graph
results/      measured outputs and reproduce logs; results/v2/ holds per-query scores
tests/        unit tests for rag/ (no Ollama needed)
v2/           the re-test: three embedders, three rerankers, paired statistics
NOTES.md      lab notebook: what broke, what I got wrong
SYSTEMS.md    taxonomy of RAG systems by the problem each solves
DECISIONS.md  design decisions with alternatives rejected
```

Start at [`concepts/`](concepts/) for the ladder, or read
[`concepts/07_reranking/`](concepts/07_reranking/) if you want the most
interesting single stage.

## No frameworks

No LangChain, no LlamaIndex, no vector database. BM25, cosine search, RRF,
MaxSim, label propagation and all the metrics are written directly. Partly
because I wanted to understand the mechanisms rather than configure them, and
partly because a framework would have hidden most of the findings above. The
`bge` query prefix and the RRF `k` parameter are both defaults you would never
question if a library set them for you.

Exact numpy brute-force search handles this corpus in well under a millisecond
and scales to a few hundred thousand vectors in tens of milliseconds. A vector
database would have added an approximate index and a recall loss to measure, for
a corpus that fits in 16MB of RAM.

## Method

Seven rules, arrived at mostly by violating them first.

1. No claimed improvement without a number.
2. Measure retrieval and generation separately. They fail differently and get
   fixed differently.
3. Distrust any single metric, especially on synthetic data. Report the gap
   between an easy variant and a hard one.
4. Ablate conventions rather than inheriting them.
5. Run the cheap diagnostic that predicts the result before running the
   expensive experiment.
6. State the noise floor. A delta smaller than sampling error is not a finding.
7. A confirmed negative result is a result.

Rule 3 came out of Stage 2, where LLM-generated eval questions inherited rare
vocabulary from their source chunks and inflated recall@1 from 0.680 to 0.927.
Rule 5 came out of Stage 4, where a two-second check on token lengths predicted
seventeen minutes of chunking experiments.

## Limitations

Worth saying plainly, because most of the negative results above are weaker than
they look:

- **One corpus.** SciFact abstracts are short, single-topic, and written in the
  same register as the queries. That is close to the best case for a dense
  bi-encoder and close to the worst case for chunking and query rewriting. I
  expect several of these results to flip on long multi-topic documents. Not
  tested.
- **n=300.** Paired tests on 300 queries detect differences of roughly 0.02 to
  0.03 nDCG. "No detectable difference" means smaller than that, not zero.
- **The stage READMEs are the v1 record.** `concepts/` keeps the original
  analysis and numbers, with a note at the top of each affected stage linking
  to the re-tested result. The v1 Stage 8 tables use a fixed 50-query
  subsample; the v2 results use all 300.
- **Subsets for the expensive stages.** Contextual retrieval (Stage 9) runs on
  a 600-document subset, and Stages 11 and 12 on 16 to 30 sampled queries, so
  those results are directional. The sample sizes are in each stage README.
- **Stages 13 to 15 are not measured.** Agentic multi-hop retrieval
  (`rag/adaptive.py`) and GraphRAG (`rag/graph.py`) have code but no eval
  runner yet; Stage 15 is a plan. Their READMEs state the expected result
  before the run so the predictions stay falsifiable.

## Attribution

Stage 1 is adapted from a public "RAG from scratch" tutorial (the cat-facts
corpus with Ollama and `bge-base`). Everything from Stage 2 onward is my own:
the eval harness, the BEIR integration, BM25, the fusion and reranking analysis,
and every measurement in this repo.

Built with AI assistance, which is recorded in the commit trailers.
