# Stage 11, Grounded generation

**Tier: Advanced** · prerequisite: Stage 10 · status: code written, not yet run

Every stage so far measured **retrieval**. This measures what the model does
with what retrieval handed it, and it needs its own scoreboard, because the two
fail independently. An answer can be perfectly faithful to the wrong documents.

## The two guardrails Stage 1 lacked

**Citations.** Number the context, require `[1]`, `[2]` per claim. Not
decoration: an uncited sentence is one you cannot check, and citations make
hallucination *mechanically* detectable, you can verify a span against its
cited source without a human reading everything.

**Abstention.** Give the model an explicit, blessed way to say the context does
not answer the question. Without one, the strongest force in the prompt is the
instruction to answer, and a model handed irrelevant context will answer anyway.

## The test that matters

Three context conditions per query:

| Condition | Context | Correct behaviour |
|---|---|---|
| **gold** | the right document | answer, and cite it |
| **distractor** | top-ranked docs, gold *removed* | **abstain** |
| **empty** | nothing | **abstain** |

The distractor row is the one to watch. It is topically plausible and does not
contain the answer. A low abstention rate there means the system answers
confidently from irrelevant context.

**That is the most dangerous RAG failure in production** — not a miss, but a
fluent wrong answer with no signal that anything went wrong. It is invisible to
every retrieval metric in this repo.

## Faithfulness

For each claim, is it supported by its cited source? LLM-as-judge, which is
cheap and useful, but it is a model grading a model. Report it *alongside*
retrieval metrics, never instead of them.
