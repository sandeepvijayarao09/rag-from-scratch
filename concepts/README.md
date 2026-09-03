# The Ladder, beginner to pro

Fifteen stages. Each one is a distinct kind of RAG system, ordered so that every
stage only needs what came before it.

Each folder holds runnable code plus a README with the finding. The shared
library lives in [`../rag/`](../rag/) and grows as stages need it.

---

## Tier 1, Beginner: make it work, then make it measurable

| # | Stage | What you learn | Status |
|---|---|---|---|
| 01 | [`01_naive_rag`](01_naive_rag/) | The loop every tutorial teaches, and its six blind spots | ✅ |
| 02 | [`02_measure_first`](02_measure_first/) | Eval harness, recall/MRR/nDCG, and how synthetic evals lie | ✅ |
| 03 | [`03_real_benchmark`](03_real_benchmark/) | Validate your pipeline against published scores | ✅ |

**Why this order:** you cannot tune what you cannot measure, and you cannot
trust a measurement you have not validated. Stage 2 alone found that a quarter
of our apparent accuracy was vocabulary leakage.

---

## Tier 2, Core retrieval: the fundamentals

| # | Stage | What you learn | Status |
|---|---|---|---|
| 04 | [`04_chunking`](04_chunking/) | When splitting documents helps, and when it is pure cost | ✅ |
| 05 | [`05_keyword_search`](05_keyword_search/) | BM25 from scratch: idf, saturation, length normalisation | ✅ |
| 06 | [`06_hybrid_search`](06_hybrid_search/) | Fusing dense + sparse; RRF vs weighted, and when RRF fails | ✅ |

---

## Tier 3 — Intermediate: two-stage and query-side

| # | Stage | What you learn | Status |
|---|---|---|---|
| 07 | [`07_reranking`](07_reranking/) | Cross-encoders, and the rule that took two falsified hypotheses | ✅ |
| 08 | [`08_query_rewriting`](08_query_rewriting/) | Multi-query, HyDE, step-back, decomposition | ✅ |

---

## Tier 4, Advanced: representation and grounding

| # | Stage | What you learn | Status |
|---|---|---|---|
| 09 | [`09_context_enrichment`](09_context_enrichment/) | Contextual retrieval, small-to-big | code written |
| 10 | [`10_multi_vector`](10_multi_vector/) | Late interaction / ColBERT MaxSim, and its 250x index cost | code written |
| 11 | [`11_grounded_generation`](11_grounded_generation/) | Citations, abstention, and a generator ceiling | ✅ |

---

## Tier 5, Pro: control flow and knowledge structure

| # | Stage | What you learn | Status |
|---|---|---|---|
| 12 | [`12_self_correction`](12_self_correction/) | CRAG grading, Self-RAG retrieve-or-not | code written |
| 13 | [`13_agentic_retrieval`](13_agentic_retrieval/) | Multi-hop loops: retrieve, reason, retrieve again | code written |
| 14 | [`14_graph_rag`](14_graph_rag/) | Entity graphs, communities, global questions | code written |
| 15 | [`15_production_scale`](15_production_scale/) | ANN, quantization, ingestion, observability | planned |

---

## Rules of the ladder

1. **Never claim an improvement without a number.** Every stage re-runs the same eval.
2. **Measure retrieval and generation separately.** They fail differently.
3. **Distrust any single metric**, especially from synthetic data. Report the gap
   between an easy and a hard variant.
4. **Ablate conventions** instead of inheriting them.
5. **Run the cheap diagnostic** that predicts the result before the expensive experiment.
6. **State the noise floor.** A delta smaller than sampling error is not a finding.
7. **A confirmed negative result is a real result.**

## Running

```bash
./myenv/bin/python concepts/02_measure_first/01_fast_loop.py
```

Scripts run from the repo root and add it to `sys.path` themselves. Embeddings,
LLM transformations and cross-encoder scores are cached under `cache/`, so
re-running costs nothing and only new work is computed.
