# RAG From Scratch

**I implemented ten recommended RAG techniques and measured each one on a public
benchmark. Eight of them made retrieval worse.**

This repo is the evidence, the code, and the reasoning for why.

Corpus: BEIR SciFact (5,183 abstracts, 300 labelled queries). Metric: nDCG@10.
Baseline: `bge-base-en-v1.5`, **0.759** — validated against a published ~0.741,
so the pipeline is known-correct before anything is built on it.

| Technique | Δ nDCG@10 |
|---|---|
| Chunking *(6 configs, all of them)* | −0.010 to −0.020 |
| RRF fusion *(4 variants, all of them)* | −0.009 to −0.024 |
| Cross-encoder reranking on dense | −0.033 |
| HyDE | −0.048 |
| Step-back prompting | −0.066 |
| Query decomposition | −0.095 |
| **Weighted hybrid 0.7/0.3** | **+0.010** |
| **Cross-encoder reranking on BM25** | **+0.041** |

The two that won are the two where I diagnosed the gap *before* applying the
technique. Everything else was a correlated addition to an already-strong
baseline.

No LangChain, no LlamaIndex, no vector database. BM25, cosine search, RRF,
MaxSim, label propagation and the metrics are implemented directly, because the
point is to understand the mechanisms rather than configure them.

## Setup

```bash
pip install -r requirements.txt
./scripts/download_data.sh
python concepts/01_naive_rag/naive_rag.py
```

Requires [Ollama](https://ollama.com) with `bge-base-en-v1.5` and `gemma4:e4b`.

## Start here

| | |
|---|---|
| **[concepts/](concepts/)** | The ladder: 15 stages, code + findings |
| [SYSTEMS.md](SYSTEMS.md) | The taxonomy, organised by the problem each system solves |
| [DECISIONS.md](DECISIONS.md) | Every design decision, alternatives rejected, evidence |

```bash
./myenv/bin/python concepts/01_naive_rag/naive_rag.py
```

## The ladder

| Tier | Stages | |
|---|---|---|
| **Beginner** | 01 naive_rag · 02 measure_first · 03 real_benchmark | make it work, then make it measurable |
| **Core** | 04 chunking · 05 keyword_search · 06 hybrid_search | the retrieval fundamentals |
| **Intermediate** | 07 reranking · 08 query_rewriting | two-stage and query-side |
| **Advanced** | 09 context_enrichment · 10 multi_vector · 11 grounded_generation | representation and grounding |
| **Pro** | 12 self_correction · 13 agentic_retrieval · 14 graph_rag · 15 production_scale | control flow and structure |

## The setup

- **Corpus** — BEIR SciFact: 5,183 scientific abstracts, 300 labelled queries
- **Models** — `bge-base-en-v1.5` (embed), `gemma4:e4b` (generate), `bge-reranker-base` (rerank)
- **Metric** — nDCG@10 (BEIR standard), plus recall@k and MRR
- **All local**, via Ollama and MPS

SciFact was chosen because `bge-base-en-v1.5` has *published* scores on it. That
turns the eval from "is this technique better?" into "is my pipeline correct?"
We measured **nDCG@10 = 0.745** against a published **~0.741**, so everything
built on top inherits that confidence.

## Results

Dense baseline: **nDCG@10 = 0.759** (title, no query prefix).

| Stage | Technique | Result |
|---|---|---|
| 03 | BGE query prefix *(convention)* | **−0.014** |
| 04 | Chunking, all 6 configs | **−0.010 to −0.020** |
| 06 | RRF fusion, all 4 variants | **−0.009 to −0.024** |
| 06 | Weighted hybrid 0.7/0.3 | **+0.010** ✅ |
| 07 | Reranking on dense | **−0.033** |
| 07 | Reranking on **BM25** | **+0.041** ✅ |
| 08 | HyDE | **−0.048** |
| 08 | Step-back | **−0.066** |
| 08 | Decomposition | **−0.095** |
| 08 | Multi-query | flat nDCG, **+0.040 recall** |

**Almost every recommended upgrade lost.** The two that won are the two where
the gap was diagnosed *before* the technique was applied — hybrid after measuring
a 9-query complementarity headroom, reranking only after establishing the first
stage was weak.

This is **not** "these techniques don't work." They are conditional, and the
condition is nearly always *"is my current stage already good at the thing this
fixes?"* `bge-base` on SciFact is a strong, well-matched baseline, so most
additions are correlated noise. Reranking already flipped sign once, from −0.033
on dense to +0.041 on BM25.

## The three findings worth stealing

**1. Synthetic evals leak vocabulary.** LLM-generated questions inherit rare
words from their source chunk. Direct questions scored recall@1 = 0.927;
paraphrased ones stripped of those words scored 0.680. A quarter of the apparent
accuracy was leakage. *(Stage 2)*

**2. Reranking overwrites your ordering.** Three first stages spanning 0.050
nDCG collapsed to a 0.017 spread after reranking. You get the reranker's ranking
quality, bounded by candidate-set recall — so rerank only if the reranker beats
your first stage, and optimise that first stage for **recall**, not ordering.
*(Stage 7, after two falsified hypotheses)*

**3. Chunking is a response to a diagnosed problem, not a default step.** Only
2.8% of SciFact abstracts exceed the encoder window, so chunking had nothing to
rescue — and all six configs lost. The strategy choice mattered 4× less than the
choice not to chunk. *(Stage 4)*

## Method rules

1. Never claim an improvement without a number.
2. Measure retrieval and generation separately. They fail differently.
3. Distrust any single metric, especially from synthetic data.
4. Ablate conventions instead of inheriting them.
5. Run the cheap diagnostic that predicts the result before the expensive experiment.
6. State the noise floor. A delta smaller than sampling error is not a finding.
7. A confirmed negative result is a real result.

## Running

Scripts run from the repo root and add it to `sys.path` themselves. Embeddings,
LLM transformations and cross-encoder scores are cached under `cache/`, so
re-running costs nothing and only new work is computed.
