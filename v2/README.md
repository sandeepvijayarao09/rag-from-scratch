# v2: the same ladder, tested properly

v1 built stages 1 to 12 with one model in each slot and reported mean scores. Its own stage READMEs flag the same weakness several times: n=300 is small, deltas of about 0.01 might be noise, and the right next step is a paired test. v2 is that next step, plus the follow-up the v1 results point to: if most techniques lost because the baseline was already strong, vary the baseline.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Embedding model | bge-base only | embeddinggemma-300m, bge-base, qwen3-embedding-4b |
| Reranker | bge-reranker-base | the same, plus bge-reranker-v2-m3 and Qwen3-Reranker-4B |
| Generator | gemma4:e4b (4B), temperature 0.3 to 0.7 | gemma4:12b, temperature 0, fixed seed, cached |
| Late interaction | raw bge-base token vectors, 800-doc subset | GTE-ModernColBERT (trained for MaxSim), full corpus |
| Query rewriting | 50-query subsample | all 300 test queries |
| Tuned settings | chosen on the test queries | chosen on the train split (809 queries), scored once on test |
| Scores kept | means only | per query, every config, every model |
| Statistics | none | paired bootstrap 95% CI, Holm-corrected within each stage |
| Failed techniques | reported | reported, plus one repaired variant each |

## The repairs

Each loss in v1 had the same shape: the technique replaced or diluted a dense ranking that was already good. So each repair keeps the dense signal and adds the technique at a weight, tuned on train.

| Technique | Repair |
|---|---|
| Chunking | Chunk only the documents the model truncates, and blend whole and best-chunk scores |
| RRF | Weight the stronger retriever and tune k |
| Reranking | Interpolate reranker and first-stage scores instead of replacing |
| HyDE | Embed the claim plus the hypothetical document, not the document alone |
| Step-back, decomposition | Keep the original claim and add the rewrites at a low weight |

A repair that still does not beat dense is reported as exactly that.

## Running it

```bash
./v2/run_all.sh          # everything, about a day (the 12B steps are most of it)
./v2/run_all.sh fast     # stages 3-7 and 10 only, no 12B generation
python v2/report.py      # rebuild results/v2/RESULTS.md from whatever has run
```

Every step resumes from its caches (embeddings, reranker scores and LLM outputs are all stored on disk), so an interrupted run carries on where it stopped.

## Files

- `rag/v2/models.py`: the model ladder, with each model card's query and document formats
- `rag/v2/bench.py`: data loading, retrieval, per-query recording, bootstrap and Holm
- `rag/v2/llm.py`, `rerankers.py`, `colbert.py`: cached generation, the three rerankers, and late interaction
- `v2/s0N_*.py`: one script per stage
- `results/v2/<embedding>/<stage>.json`: per-query nDCG@10, recall@10, MRR and top-10 lists
- `results/v2/RESULTS.md`, `summary.json`: the tested comparisons
