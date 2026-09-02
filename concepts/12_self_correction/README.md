# Stage 12 — Self-correction (CRAG, Self-RAG)

**Tier: Pro** · prerequisite: Stage 11 · status: code written, not yet run

Everything so far is `retrieve → stuff → generate`, run identically for every
query. **That pipeline has no way to notice it has failed.** Garbage retrieval
produces a confident answer from garbage.

These add a feedback edge.

## Self-RAG — decide whether to retrieve at all

Not every query needs retrieval. *"Rewrite this in bullet points"* does not.
Retrieving anyway injects irrelevant context, which measurably degrades answers
(see Stage 11's distractor condition). A cheap classifier up front is one of the
highest value-per-line changes available.

## CRAG — grade what came back, then correct

Score retrieved documents for actual relevance. Three outcomes:

| Verdict | Meaning | Action |
|---|---|---|
| `CORRECT` | good documents | proceed |
| `AMBIGUOUS` | mixed | keep the good ones, widen the search |
| `INCORRECT` | all bad | discard; rewrite query, try another index, or abstain |

**The point:** "every retrieved document is irrelevant" becomes a state the
system can *detect and act on*, instead of silently answering from noise.

## Cost

Naive RAG is one LLM call. CRAG is two or three. Every one lands on the query
latency path. This is why control flow belongs *after* the cheap retrieval wins,
not before — and Stages 4-8 showed how often those "cheap wins" are negative.

Code: [`../../rag/adaptive.py`](../../rag/adaptive.py)
