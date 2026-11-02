# v2 results

Δ nDCG@10 against each stage's control, on the 300 SciFact test queries.
▼ significantly worse · ▲ significantly better · · no detectable difference
(paired bootstrap 95% CI, Holm-corrected within each stage). Repairs and any
tuned setting were chosen on the train split only.
= 0 same top 10 as the control on every query · n/a not applicable (the chunk
repair only runs on models that truncate documents; only bge-base does, 2.8%).

## The ten techniques, by embedding model

| Technique | Form | embeddinggemma-300m | bge-base | qwen3-embedding-4b |
|---|---|---|---|---|
| Chunking | recursive 200/40 | · -0.006 | · -0.009 | · +0.003 |
| Chunking | sentence 100/1 | · +0.000 | · -0.014 | · +0.001 |
| Chunking | **repaired: chunk long docs only, blended** | n/a | = 0 | n/a |
| RRF fusion | RRF k=60 | ▼ -0.057 | · -0.024 | · -0.030 |
| RRF fusion | RRF k=10 | · -0.027 | · -0.009 | · -0.010 |
| RRF fusion | **repaired: weighted RRF (tuned)** | · +0.002 | · +0.011 | · +0.001 |
| Rerank after dense | dense -> bge-reranker-base (replace) | ▼ -0.062 | · -0.033 | ▼ -0.044 |
| Rerank after dense | dense -> bge-reranker-v2-m3 (replace) | · -0.036 | · -0.007 | · -0.018 |
| Rerank after dense | dense -> qwen3-reranker-4b (replace) | · +0.013 | ▲ +0.044 | ▲ +0.033 |
| Rerank after dense | **dense -> qwen3-reranker-4b (interpolate, tuned)** | · +0.010 | ▲ +0.042 | ▲ +0.035 |
| HyDE | hyde (v1 form) | ▼ -0.070 | ▼ -0.040 | · -0.019 |
| HyDE | **hyde (repaired, tuned)** | · +0.005 | · +0.014 | · +0.003 |
| Step-back | step_back (v1 form) | ▼ -0.071 | ▼ -0.039 | ▼ -0.060 |
| Step-back | **step_back (repaired, tuned)** | · -0.002 | · +0.004 | · +0.003 |
| Decomposition | decompose (v1 form) | ▼ -0.052 | ▼ -0.087 | ▼ -0.050 |
| Decomposition | **decompose (repaired, tuned)** | · +0.002 | · +0.002 | · +0.003 |
| Query instruction | query format: alt | = 0 | · -0.014 | · -0.004 |
| Multi-query | multi_query (v1 form) | · -0.002 | · +0.009 | · +0.002 |
| Weighted hybrid | weighted 0.7/0.3 | · +0.005 | · +0.010 | · +0.008 |
| Weighted hybrid | **repaired: weighted fusion (tuned)** | · +0.002 | · +0.014 | · +0.008 |
| Rerank after BM25 | bm25 -> bge-reranker-base (replace) | ▲ +0.041 | ▲ +0.041 | ▲ +0.041 |
| Rerank after BM25 | bm25 -> bge-reranker-v2-m3 (replace) | ▲ +0.067 | ▲ +0.067 | ▲ +0.067 |
| Rerank after BM25 | bm25 -> qwen3-reranker-4b (replace) | ▲ +0.110 | ▲ +0.110 | ▲ +0.110 |
| Rerank after BM25 | **bm25 -> qwen3-reranker-4b (interpolate, tuned)** | ▲ +0.107 | ▲ +0.107 | ▲ +0.107 |

## embeddinggemma-300m

### s03_benchmark (control: dense baseline, 0.7987)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| query format: card | 0.7921 | -0.0066 | [-0.0168, +0.0034] | 0.3920 | 22/29 | no detectable difference |
| no titles | 0.7897 | -0.0090 | [-0.0233, +0.0055] | 0.3920 | 23/43 | no detectable difference |

### s04_chunking (control: whole documents, 0.7987)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| fixed 200/40 | 0.7924 | -0.0063 | [-0.0151, +0.0020] | 0.7590 | 13/25 | no detectable difference |
| sentence 200/1 | 0.7927 | -0.0059 | [-0.0135, +0.0011] | 0.6732 | 18/27 | no detectable difference |
| recursive 200/40 | 0.7927 | -0.0060 | [-0.0148, +0.0028] | 0.7590 | 14/29 | no detectable difference |
| fixed 100/20 | 0.7918 | -0.0069 | [-0.0205, +0.0065] | 0.9564 | 23/39 | no detectable difference |
| recursive 100/20 | 0.8017 | +0.0030 | [-0.0110, +0.0171] | 1.0000 | 30/35 | no detectable difference |
| sentence 100/1 | 0.7988 | +0.0001 | [-0.0127, +0.0129] | 1.0000 | 25/35 | no detectable difference |

### s05_bm25 (control: bm25 default (k1=1.2, b=0.75), 0.6652)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 tuned | 0.6624 | -0.0028 | [-0.0100, +0.0040] | 0.4257 | 11/21 | no detectable difference |

### s06_hybrid (control: dense, 0.7987)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 | 0.6652 | -0.1334 | [-0.1735, -0.0930] | 0.0009 | 31/109 | worse |
| RRF k=60 | 0.7417 | -0.0570 | [-0.0869, -0.0274] | 0.0016 | 37/71 | worse |
| RRF k=10 | 0.7715 | -0.0272 | [-0.0509, -0.0032] | 0.1560 | 35/68 | no detectable difference |
| RRF k=60 w=2:1 | 0.7609 | -0.0378 | [-0.0628, -0.0137] | 0.0217 | 41/56 | worse |
| weighted 0.5/0.5 | 0.7758 | -0.0228 | [-0.0466, +0.0013] | 0.2464 | 29/67 | no detectable difference |
| weighted 0.7/0.3 | 0.8040 | +0.0054 | [-0.0098, +0.0213] | 0.8122 | 31/41 | no detectable difference |
| weighted 0.9/0.1 | 0.8078 | +0.0091 | [+0.0008, +0.0185] | 0.2045 | 19/16 | no detectable difference |
| repaired: weighted RRF (tuned) | 0.8006 | +0.0019 | [-0.0004, +0.0050] | 0.5109 | 5/4 | no detectable difference |
| repaired: weighted fusion (tuned) | 0.8007 | +0.0020 | [-0.0026, +0.0072] | 0.8122 | 12/7 | no detectable difference |

### s07_reranking (control: dense (no rerank), 0.7987)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 (no rerank) | 0.6652 | -0.1334 | [-0.1735, -0.0930] | 0.0013 | 31/109 | worse |
| dense -> bge-reranker-base (replace) | 0.7364 | -0.0623 | [-0.0969, -0.0287] | 0.0048 | 38/89 | worse |
| dense -> bge-reranker-base (interpolate, tuned) | 0.7992 | +0.0006 | [-0.0081, +0.0097] | 0.9062 | 14/16 | no detectable difference |
| bm25 -> bge-reranker-base (replace) | 0.7058 | -0.0929 | [-0.1314, -0.0546] | 0.0013 | 35/96 | worse |
| bm25 -> bge-reranker-base (interpolate, tuned) | 0.7038 | -0.0948 | [-0.1330, -0.0572] | 0.0013 | 35/93 | worse |
| dense -> bge-reranker-v2-m3 (replace) | 0.7630 | -0.0357 | [-0.0675, -0.0035] | 0.1876 | 38/73 | no detectable difference |
| dense -> bge-reranker-v2-m3 (interpolate, tuned) | 0.8034 | +0.0048 | [-0.0034, +0.0138] | 0.8328 | 17/14 | no detectable difference |
| bm25 -> bge-reranker-v2-m3 (replace) | 0.7322 | -0.0665 | [-0.1034, -0.0299] | 0.0036 | 36/81 | worse |
| bm25 -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7286 | -0.0701 | [-0.1071, -0.0328] | 0.0030 | 37/79 | worse |
| dense -> qwen3-reranker-4b (replace) | 0.8120 | +0.0134 | [-0.0141, +0.0406] | 0.8328 | 55/51 | no detectable difference |
| dense -> qwen3-reranker-4b (interpolate, tuned) | 0.8087 | +0.0100 | [+0.0014, +0.0195] | 0.1876 | 24/9 | no detectable difference |
| bm25 -> qwen3-reranker-4b (replace) | 0.7757 | -0.0230 | [-0.0570, +0.0103] | 0.7156 | 54/58 | no detectable difference |
| bm25 -> qwen3-reranker-4b (interpolate, tuned) | 0.7720 | -0.0267 | [-0.0610, +0.0069] | 0.5875 | 49/56 | no detectable difference |

### s07_reranking_vs_bm25 (control: bm25 (no rerank), 0.6652)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 -> bge-reranker-base (replace) | 0.7058 | +0.0406 | [+0.0159, +0.0665] | 0.0014 | 63/30 | better |
| bm25 -> bge-reranker-base (interpolate, tuned) | 0.7038 | +0.0386 | [+0.0201, +0.0586] | 0.0006 | 54/22 | better |
| bm25 -> bge-reranker-v2-m3 (replace) | 0.7322 | +0.0670 | [+0.0406, +0.0941] | 0.0006 | 69/31 | better |
| bm25 -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7286 | +0.0634 | [+0.0420, +0.0856] | 0.0006 | 64/19 | better |
| bm25 -> qwen3-reranker-4b (replace) | 0.7757 | +0.1105 | [+0.0818, +0.1402] | 0.0006 | 90/22 | better |
| bm25 -> qwen3-reranker-4b (interpolate, tuned) | 0.7720 | +0.1067 | [+0.0794, +0.1353] | 0.0006 | 86/18 | better |

### s08_query_rewriting (control: dense, 0.7987)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| multi_query (v1 form) | 0.7969 | -0.0018 | [-0.0161, +0.0128] | 1.0000 | 29/28 | no detectable difference |
| hyde (v1 form) | 0.7283 | -0.0704 | [-0.0986, -0.0418] | 0.0007 | 36/77 | worse |
| hyde (repaired, tuned) | 0.8041 | +0.0054 | [-0.0037, +0.0151] | 0.7936 | 19/19 | no detectable difference |
| step_back (v1 form) | 0.7277 | -0.0710 | [-0.0971, -0.0459] | 0.0007 | 25/75 | worse |
| step_back (repaired, tuned) | 0.7967 | -0.0020 | [-0.0079, +0.0038] | 1.0000 | 11/15 | no detectable difference |
| decompose (v1 form) | 0.7471 | -0.0516 | [-0.0810, -0.0223] | 0.0045 | 39/67 | worse |
| decompose (repaired, tuned) | 0.8010 | +0.0024 | [-0.0009, +0.0063] | 0.7936 | 9/4 | no detectable difference |

### s09_contextual (control: whole (600-doc subset), 0.9206)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| chunked 100/20 | 0.9149 | -0.0057 | [-0.0148, +0.0031] | 0.4140 | 9/19 | no detectable difference |
| contextualised (gemma4:12b blurbs) | 0.9193 | -0.0013 | [-0.0113, +0.0088] | 0.8074 | 13/19 | no detectable difference |

### s10_late_interaction (control: dense, 0.7987)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| colbert (full corpus) | 0.7608 | -0.0379 | [-0.0703, -0.0054] | 0.0474 | 40/75 | worse |
| dense -> colbert rerank (top 100) | 0.7640 | -0.0346 | [-0.0665, -0.0029] | 0.0474 | 40/73 | worse |


## bge-base

### s03_benchmark (control: dense baseline, 0.7588)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| query format: alt | 0.7449 | -0.0139 | [-0.0311, +0.0030] | 0.1144 | 33/42 | no detectable difference |
| no titles | 0.7416 | -0.0172 | [-0.0309, -0.0039] | 0.0266 | 22/45 | worse |

### s04_chunking (control: whole documents, 0.7588)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| fixed 200/40 | 0.7518 | -0.0070 | [-0.0209, +0.0065] | 0.9100 | 28/40 | no detectable difference |
| sentence 200/1 | 0.7422 | -0.0166 | [-0.0316, -0.0024] | 0.1554 | 25/48 | no detectable difference |
| recursive 200/40 | 0.7500 | -0.0088 | [-0.0233, +0.0054] | 0.9100 | 26/42 | no detectable difference |
| fixed 100/20 | 0.7490 | -0.0098 | [-0.0263, +0.0062] | 0.9100 | 38/42 | no detectable difference |
| recursive 100/20 | 0.7555 | -0.0033 | [-0.0205, +0.0137] | 0.9100 | 38/41 | no detectable difference |
| sentence 100/1 | 0.7448 | -0.0140 | [-0.0319, +0.0032] | 0.5955 | 37/48 | no detectable difference |

### s05_bm25 (control: bm25 default (k1=1.2, b=0.75), 0.6652)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 tuned | 0.6624 | -0.0028 | [-0.0100, +0.0040] | 0.4257 | 11/21 | no detectable difference |

### s06_hybrid (control: dense, 0.7588)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 | 0.6652 | -0.0936 | [-0.1295, -0.0587] | 0.0009 | 40/93 | worse |
| RRF k=60 | 0.7347 | -0.0241 | [-0.0501, +0.0010] | 0.3810 | 46/51 | no detectable difference |
| RRF k=10 | 0.7499 | -0.0089 | [-0.0312, +0.0129] | 0.8808 | 42/55 | no detectable difference |
| RRF k=60 w=2:1 | 0.7464 | -0.0123 | [-0.0352, +0.0090] | 0.8808 | 43/43 | no detectable difference |
| weighted 0.5/0.5 | 0.7528 | -0.0060 | [-0.0278, +0.0157] | 0.8808 | 46/58 | no detectable difference |
| weighted 0.7/0.3 | 0.7689 | +0.0101 | [-0.0058, +0.0266] | 0.8808 | 40/38 | no detectable difference |
| weighted 0.9/0.1 | 0.7663 | +0.0075 | [+0.0013, +0.0143] | 0.2032 | 25/9 | no detectable difference |
| repaired: weighted RRF (tuned) | 0.7698 | +0.0110 | [-0.0007, +0.0232] | 0.3810 | 35/31 | no detectable difference |
| repaired: weighted fusion (tuned) | 0.7725 | +0.0137 | [+0.0004, +0.0276] | 0.3388 | 38/32 | no detectable difference |

### s07_reranking (control: dense (no rerank), 0.7588)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 (no rerank) | 0.6652 | -0.0936 | [-0.1295, -0.0587] | 0.0013 | 40/93 | worse |
| dense -> bge-reranker-base (replace) | 0.7256 | -0.0332 | [-0.0622, -0.0035] | 0.2192 | 47/73 | no detectable difference |
| dense -> bge-reranker-base (interpolate, tuned) | 0.7677 | +0.0089 | [-0.0087, +0.0272] | 1.0000 | 42/45 | no detectable difference |
| bm25 -> bge-reranker-base (replace) | 0.7058 | -0.0530 | [-0.0879, -0.0187] | 0.0360 | 43/77 | worse |
| bm25 -> bge-reranker-base (interpolate, tuned) | 0.7038 | -0.0549 | [-0.0884, -0.0219] | 0.0130 | 45/81 | worse |
| dense -> bge-reranker-v2-m3 (replace) | 0.7519 | -0.0068 | [-0.0351, +0.0213] | 1.0000 | 54/57 | no detectable difference |
| dense -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7703 | +0.0115 | [-0.0036, +0.0267] | 0.7890 | 43/30 | no detectable difference |
| bm25 -> bge-reranker-v2-m3 (replace) | 0.7322 | -0.0266 | [-0.0609, +0.0079] | 0.7890 | 50/69 | no detectable difference |
| bm25 -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7286 | -0.0302 | [-0.0640, +0.0036] | 0.5635 | 50/67 | no detectable difference |
| dense -> qwen3-reranker-4b (replace) | 0.8028 | +0.0440 | [+0.0191, +0.0698] | 0.0099 | 61/38 | better |
| dense -> qwen3-reranker-4b (interpolate, tuned) | 0.8012 | +0.0424 | [+0.0271, +0.0586] | 0.0013 | 54/12 | better |
| bm25 -> qwen3-reranker-4b (replace) | 0.7757 | +0.0169 | [-0.0133, +0.0467] | 1.0000 | 61/45 | no detectable difference |
| bm25 -> qwen3-reranker-4b (interpolate, tuned) | 0.7720 | +0.0132 | [-0.0172, +0.0434] | 1.0000 | 59/48 | no detectable difference |

### s07_reranking_vs_bm25 (control: bm25 (no rerank), 0.6652)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 -> bge-reranker-base (replace) | 0.7058 | +0.0406 | [+0.0159, +0.0665] | 0.0014 | 63/30 | better |
| bm25 -> bge-reranker-base (interpolate, tuned) | 0.7038 | +0.0386 | [+0.0201, +0.0586] | 0.0006 | 54/22 | better |
| bm25 -> bge-reranker-v2-m3 (replace) | 0.7322 | +0.0670 | [+0.0406, +0.0941] | 0.0006 | 69/31 | better |
| bm25 -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7286 | +0.0634 | [+0.0420, +0.0856] | 0.0006 | 64/19 | better |
| bm25 -> qwen3-reranker-4b (replace) | 0.7757 | +0.1105 | [+0.0818, +0.1402] | 0.0006 | 90/22 | better |
| bm25 -> qwen3-reranker-4b (interpolate, tuned) | 0.7720 | +0.1067 | [+0.0794, +0.1353] | 0.0006 | 86/18 | better |

### s08_query_rewriting (control: dense, 0.7588)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| multi_query (v1 form) | 0.7679 | +0.0091 | [-0.0096, +0.0281] | 1.0000 | 46/35 | no detectable difference |
| hyde (v1 form) | 0.7185 | -0.0403 | [-0.0692, -0.0121] | 0.0280 | 42/71 | worse |
| hyde (repaired, tuned) | 0.7729 | +0.0141 | [-0.0005, +0.0289] | 0.2416 | 37/27 | no detectable difference |
| step_back (v1 form) | 0.7200 | -0.0388 | [-0.0640, -0.0149] | 0.0066 | 39/61 | worse |
| step_back (repaired, tuned) | 0.7628 | +0.0040 | [-0.0100, +0.0177] | 1.0000 | 36/34 | no detectable difference |
| decompose (v1 form) | 0.6719 | -0.0869 | [-0.1197, -0.0547] | 0.0007 | 39/89 | worse |
| decompose (repaired, tuned) | 0.7611 | +0.0024 | [-0.0114, +0.0163] | 1.0000 | 37/34 | no detectable difference |

### s09_contextual (control: whole (600-doc subset), 0.8961)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| chunked 100/20 | 0.9042 | +0.0081 | [-0.0023, +0.0194] | 0.2882 | 19/15 | no detectable difference |
| contextualised (gemma4:12b blurbs) | 0.8970 | +0.0009 | [-0.0086, +0.0105] | 0.8567 | 16/20 | no detectable difference |

### s10_late_interaction (control: dense, 0.7588)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| colbert (full corpus) | 0.7608 | +0.0020 | [-0.0268, +0.0308] | 1.0000 | 52/58 | no detectable difference |
| dense -> colbert rerank (top 100) | 0.7563 | -0.0024 | [-0.0297, +0.0253] | 1.0000 | 50/56 | no detectable difference |


## qwen3-embedding-4b

### s03_benchmark (control: dense baseline, 0.7699)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| query format: alt | 0.7659 | -0.0039 | [-0.0139, +0.0059] | 0.8656 | 24/33 | no detectable difference |
| no titles | 0.7682 | -0.0016 | [-0.0130, +0.0098] | 0.8656 | 27/34 | no detectable difference |

### s04_chunking (control: whole documents, 0.7699)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| fixed 200/40 | 0.7664 | -0.0034 | [-0.0145, +0.0080] | 1.0000 | 20/35 | no detectable difference |
| sentence 200/1 | 0.7706 | +0.0007 | [-0.0078, +0.0096] | 1.0000 | 22/27 | no detectable difference |
| recursive 200/40 | 0.7733 | +0.0034 | [-0.0068, +0.0139] | 1.0000 | 22/32 | no detectable difference |
| fixed 100/20 | 0.7800 | +0.0101 | [-0.0027, +0.0239] | 0.7896 | 32/32 | no detectable difference |
| recursive 100/20 | 0.7775 | +0.0076 | [-0.0042, +0.0205] | 1.0000 | 32/39 | no detectable difference |
| sentence 100/1 | 0.7708 | +0.0009 | [-0.0109, +0.0134] | 1.0000 | 22/49 | no detectable difference |

### s05_bm25 (control: bm25 default (k1=1.2, b=0.75), 0.6652)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 tuned | 0.6624 | -0.0028 | [-0.0100, +0.0040] | 0.4257 | 11/21 | no detectable difference |

### s06_hybrid (control: dense, 0.7699)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 | 0.6652 | -0.1046 | [-0.1396, -0.0696] | 0.0009 | 34/98 | worse |
| RRF k=60 | 0.7398 | -0.0300 | [-0.0573, -0.0028] | 0.2205 | 47/58 | no detectable difference |
| RRF k=10 | 0.7595 | -0.0103 | [-0.0314, +0.0117] | 1.0000 | 45/57 | no detectable difference |
| RRF k=60 w=2:1 | 0.7508 | -0.0191 | [-0.0403, +0.0022] | 0.4560 | 41/52 | no detectable difference |
| weighted 0.5/0.5 | 0.7588 | -0.0111 | [-0.0326, +0.0112] | 1.0000 | 40/63 | no detectable difference |
| weighted 0.7/0.3 | 0.7780 | +0.0081 | [-0.0067, +0.0233] | 1.0000 | 38/45 | no detectable difference |
| weighted 0.9/0.1 | 0.7786 | +0.0088 | [+0.0016, +0.0171] | 0.2176 | 20/18 | no detectable difference |
| repaired: weighted RRF (tuned) | 0.7710 | +0.0012 | [-0.0094, +0.0125] | 1.0000 | 28/39 | no detectable difference |
| repaired: weighted fusion (tuned) | 0.7780 | +0.0081 | [-0.0067, +0.0233] | 1.0000 | 38/45 | no detectable difference |

### s07_reranking (control: dense (no rerank), 0.7699)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 (no rerank) | 0.6652 | -0.1046 | [-0.1396, -0.0696] | 0.0013 | 34/98 | worse |
| dense -> bge-reranker-base (replace) | 0.7258 | -0.0441 | [-0.0703, -0.0176] | 0.0090 | 41/79 | worse |
| dense -> bge-reranker-base (interpolate, tuned) | 0.7776 | +0.0077 | [-0.0077, +0.0240] | 1.0000 | 34/49 | no detectable difference |
| bm25 -> bge-reranker-base (replace) | 0.7058 | -0.0641 | [-0.0970, -0.0311] | 0.0013 | 40/87 | worse |
| bm25 -> bge-reranker-base (interpolate, tuned) | 0.7038 | -0.0660 | [-0.0988, -0.0334] | 0.0013 | 39/83 | worse |
| dense -> bge-reranker-v2-m3 (replace) | 0.7520 | -0.0178 | [-0.0437, +0.0089] | 0.7044 | 42/67 | no detectable difference |
| dense -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7837 | +0.0138 | [+0.0037, +0.0248] | 0.0602 | 25/24 | no detectable difference |
| bm25 -> bge-reranker-v2-m3 (replace) | 0.7322 | -0.0377 | [-0.0718, -0.0030] | 0.1590 | 42/78 | no detectable difference |
| bm25 -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7286 | -0.0413 | [-0.0751, -0.0073] | 0.0972 | 44/74 | no detectable difference |
| dense -> qwen3-reranker-4b (replace) | 0.8025 | +0.0327 | [+0.0120, +0.0538] | 0.0208 | 56/42 | better |
| dense -> qwen3-reranker-4b (interpolate, tuned) | 0.8048 | +0.0350 | [+0.0206, +0.0500] | 0.0013 | 49/21 | better |
| bm25 -> qwen3-reranker-4b (replace) | 0.7757 | +0.0058 | [-0.0231, +0.0350] | 1.0000 | 51/54 | no detectable difference |
| bm25 -> qwen3-reranker-4b (interpolate, tuned) | 0.7720 | +0.0021 | [-0.0272, +0.0317] | 1.0000 | 52/52 | no detectable difference |

### s07_reranking_vs_bm25 (control: bm25 (no rerank), 0.6652)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| bm25 -> bge-reranker-base (replace) | 0.7058 | +0.0406 | [+0.0159, +0.0665] | 0.0014 | 63/30 | better |
| bm25 -> bge-reranker-base (interpolate, tuned) | 0.7038 | +0.0386 | [+0.0201, +0.0586] | 0.0006 | 54/22 | better |
| bm25 -> bge-reranker-v2-m3 (replace) | 0.7322 | +0.0670 | [+0.0406, +0.0941] | 0.0006 | 69/31 | better |
| bm25 -> bge-reranker-v2-m3 (interpolate, tuned) | 0.7286 | +0.0634 | [+0.0420, +0.0856] | 0.0006 | 64/19 | better |
| bm25 -> qwen3-reranker-4b (replace) | 0.7757 | +0.1105 | [+0.0818, +0.1402] | 0.0006 | 90/22 | better |
| bm25 -> qwen3-reranker-4b (interpolate, tuned) | 0.7720 | +0.1067 | [+0.0794, +0.1353] | 0.0006 | 86/18 | better |

### s08_query_rewriting (control: dense, 0.7699)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| multi_query (v1 form) | 0.7723 | +0.0024 | [-0.0081, +0.0134] | 0.8900 | 31/32 | no detectable difference |
| hyde (v1 form) | 0.7513 | -0.0185 | [-0.0409, +0.0035] | 0.5125 | 41/56 | no detectable difference |
| hyde (repaired, tuned) | 0.7730 | +0.0032 | [-0.0020, +0.0088] | 0.8900 | 18/11 | no detectable difference |
| step_back (v1 form) | 0.7099 | -0.0599 | [-0.0827, -0.0388] | 0.0007 | 33/68 | worse |
| step_back (repaired, tuned) | 0.7726 | +0.0027 | [-0.0015, +0.0074] | 0.8900 | 19/14 | no detectable difference |
| decompose (v1 form) | 0.7202 | -0.0496 | [-0.0771, -0.0230] | 0.0042 | 37/71 | worse |
| decompose (repaired, tuned) | 0.7728 | +0.0029 | [-0.0023, +0.0086] | 0.8900 | 11/8 | no detectable difference |

### s09_contextual (control: whole (600-doc subset), 0.9210)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| chunked 100/20 | 0.9243 | +0.0033 | [-0.0073, +0.0137] | 0.7464 | 15/13 | no detectable difference |
| contextualised (gemma4:12b blurbs) | 0.9265 | +0.0054 | [-0.0065, +0.0175] | 0.7464 | 19/14 | no detectable difference |

### s10_late_interaction (control: dense, 0.7699)

| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |
|---|---|---|---|---|---|---|
| colbert (full corpus) | 0.7608 | -0.0091 | [-0.0361, +0.0185] | 0.5530 | 48/59 | no detectable difference |
| dense -> colbert rerank (top 100) | 0.7553 | -0.0146 | [-0.0401, +0.0119] | 0.5530 | 45/59 | no detectable difference |
