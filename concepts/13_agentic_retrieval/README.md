# Stage 13 — Agentic / multi-hop retrieval

**Tier: Pro** · prerequisite: Stage 12 · status: code written, not yet run

Some questions need chaining: find A, use A to find B. **One retrieval pass
structurally cannot answer them**, no matter how good the retriever is. This is
not a quality problem you can fix with a better embedding model.

```
loop:
  retrieve(current_query)
  ask the model: do I have enough? if not, what is the next query?
  stop when satisfied, or when the hop budget runs out
```

## What to actually look at

The **trace** is the output worth studying, more than the metric. It shows the
query the system *invented* at each hop — which is where you see whether it is
reasoning or drifting. A common failure is hop 2 restating hop 1 in different
words, burning a call to retrieve the same documents.

## Honest expectation for SciFact

SciFact claims are single-hop by construction: one abstract supports or refutes
one claim. **Multi-hop should not help here**, and if it appears to, be
suspicious of the eval rather than pleased with the technique.

The right corpora for this are HotpotQA, MuSiQue, or 2WikiMultiHopQA — built
specifically so that no single document contains the answer. Running it here
demonstrates the mechanism and its cost, not its benefit.

## Cost

One LLM call per hop plus a final answer, all on the latency path. A 3-hop query
against a local model is tens of seconds. Agentic retrieval is the most
expensive system in this repo per query, and the most frequently adopted before
anyone measured whether the questions are actually multi-hop.

Code: [`../../rag/adaptive.py`](../../rag/adaptive.py) → `multi_hop_retrieve`
