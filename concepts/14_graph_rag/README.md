# Stage 14 — GraphRAG

**Tier: Pro** · prerequisite: Stage 13 · status: code written, not yet run

Genuinely different from everything else in this repo.

Vector retrieval answers **local** questions: *"what does document X say about
Y?"* It finds the passage best matching the query. It structurally cannot answer
**global** questions:

> "What are the main themes in this corpus?"
> "Which topics does this collection cover most?"

No single passage contains that answer, so no top-k over passages can retrieve
it. The answer is a property of the *collection*, not of any member of it.

## Pipeline

1. **Extract** — LLM pulls (entity, relation, entity) triples per document
2. **Build** — entities become nodes, relations edges. Entities appearing across
   many documents become hubs, encoding cross-document structure a flat chunk
   list throws away
3. **Cluster** — detect communities of densely connected entities. Topics get
   *discovered*, not declared
4. **Summarise** — LLM writes a summary per community. **These** are what get
   retrieved for global questions
5. **Query** — local: match entities, walk the neighbourhood. Global: retrieve
   community summaries

Community detection here is **label propagation**, implemented directly (~20
lines): every node starts in its own community, then repeatedly adopts whichever
community is most common among its neighbours. Microsoft's GraphRAG uses Leiden,
which is better but needs a dependency and obscures the idea.

## Cost

At least one LLM call per document for extraction, plus one per community for
summarisation. **The most expensive system here.** Runs on a corpus subset, and
the README says so rather than implying a full-corpus run.

## Honest expectation

SciFact's eval is claim→abstract, a purely local task with document-level
qrels. **GraphRAG's strength does not show up in nDCG@10 at all.** The
demonstration is the graph statistics and the community summaries, not a
retrieval score — which is itself worth knowing: some techniques are not
comparable on your benchmark, and pretending otherwise produces meaningless
numbers.

Code: [`../../rag/graph.py`](../../rag/graph.py)
