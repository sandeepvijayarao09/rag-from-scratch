# Stage 9 — Context enrichment

**Tier: Advanced** · prerequisite: Stage 8 · status: code written, not yet run

Stage 4 showed chunking costs you context. This stage buys it back.

## Contextual retrieval (Anthropic, 2024)

Split a document and each piece stops being self-describing:

> "The company's revenue grew by 3% over the previous quarter."

Which company? Which quarter? Unfindable by any query naming either, because
neither appears in the chunk. **No retrieval technique can fix this** — the
information is not in the indexed text.

The fix: before embedding, ask an LLM to write a short situating blurb using the
*whole* document, and prepend it.

> "This chunk is from ACME Corp's Q2 2023 filing; prior-quarter revenue was
> $314M. The company's revenue grew by 3%…"

**The cost is the whole catch:** one LLM call *per chunk*, at index time. For
SciFact's 5,183 abstracts at ~8s per local call that is over 11 hours. Normally
run against a fast hosted model with prompt caching, since the same long
document is re-sent for every chunk inside it.

Two things worth internalising:
- It is an **index-time** cost, paid once. Compare Stage 8, which pays on every
  query forever.
- It is the technique most improved by prompt caching, for the reason above.

## Small-to-big (parent-document retrieval)

Retrieve on small precise units, return the larger parent for generation. Small
chunks match better; large chunks answer better. You stop having to choose.

Note this repo's doc-level `max` pooling already implements the retrieval half —
see [`../../rag/retrieve.py`](../../rag/retrieve.py).

## Code

[`../../rag/contextual.py`](../../rag/contextual.py) — disk-cached and resumable,
because you will not want to pay for it twice.

**Expectation, stated before running:** Stage 4 showed SciFact abstracts are
short and self-contained, so there is little lost context to restore. Expect a
small effect here and a large one on a corpus of long multi-topic documents.
