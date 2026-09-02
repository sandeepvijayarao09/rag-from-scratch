"""Layer 4: let the model decide, instead of a fixed pipeline.

Everything so far is `retrieve -> stuff -> generate`, run identically for every
query. That pipeline has no way to notice it has failed. If retrieval returns
garbage, generation confidently uses the garbage.

These add a feedback edge.

SELF-RAG (decide whether to retrieve at all)
    Not every query needs retrieval. "Rewrite this in bullet points" does not.
    Retrieving anyway injects irrelevant context, which measurably degrades
    answers. A cheap classifier up front is one of the highest
    value-per-line changes available.

CRAG (grade what came back, then correct)
    Score retrieved documents for actual relevance. Three outcomes:
      CORRECT     -- good documents, proceed
      AMBIGUOUS   -- mixed, keep the good ones and widen the search
      INCORRECT   -- all bad, discard and fall back (rewrite query, other
                     index, web, or abstain)
    The point is that "all retrieved documents are irrelevant" is a state the
    system can DETECT and act on, instead of silently answering from noise.

MULTI-HOP / AGENTIC (iterate)
    Some questions need chaining: find A, use A to find B. One retrieval pass
    structurally cannot answer them, no matter how good the retriever is.
    Loop: retrieve, ask what is still missing, retrieve again, stop when the
    model says it has enough or the hop budget runs out.

COST, and it is the real constraint: every one of these adds LLM calls to the
query path. Naive RAG is one call. CRAG is two or three. Multi-hop is one per
hop plus a final answer. Latency and cost scale with the loop, which is why
these belong after the cheap retrieval wins, not before.
"""

import ollama
from pydantic import BaseModel

DECIDER_MODEL = "gemma4:e4b"


class RetrievalDecision(BaseModel):
    needs_retrieval: bool
    reason: str


class RelevanceGrade(BaseModel):
    relevant: bool
    reason: str


class HopDecision(BaseModel):
    have_enough: bool
    next_query: str


def _structured(prompt: str, schema, model: str = DECIDER_MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema.model_json_schema(),
        options={"temperature": 0.0},
    )
    return schema.model_validate_json(response["message"]["content"])


def should_retrieve(query: str) -> RetrievalDecision:
    """Self-RAG step 1: does this query need external knowledge at all?"""
    return _structured(f"""Does answering this require looking up external facts
in a knowledge base, or can it be answered from general ability alone
(reasoning, rewriting, formatting, arithmetic, opinion)?

Query: "{query}"

Return JSON only.""", RetrievalDecision)


def grade_relevance(query: str, document: str) -> RelevanceGrade:
    """CRAG step: is this retrieved document actually usable for this query?"""
    return _structured(f"""Query: "{query}"

Document:
\"\"\"{document[:1500]}\"\"\"

Is this document relevant and useful for answering the query? Be strict:
topically adjacent is NOT relevant. Return JSON only.""", RelevanceGrade)


def crag_triage(query: str, chunks, threshold_good: float = 0.5):
    """Grade every candidate, then classify the retrieval as a whole.

    Returns (action, kept_chunks) where action is one of
    'correct' | 'ambiguous' | 'incorrect'.
    """
    grades = [grade_relevance(query, c.text).relevant for c in chunks]
    kept = [c for c, ok in zip(chunks, grades) if ok]
    ratio = len(kept) / len(chunks) if chunks else 0.0

    if not kept:
        return "incorrect", []
    if ratio >= threshold_good:
        return "correct", kept
    return "ambiguous", kept


def next_hop(original: str, gathered: list[str], hop: int) -> HopDecision:
    """Agentic step: given what we have, do we have enough, or what next?"""
    context = "\n\n".join(f"- {g[:400]}" for g in gathered) or "(nothing yet)"
    return _structured(f"""Original question: "{original}"

Evidence gathered so far (hop {hop}):
{context}

Do you have enough to answer the original question completely? If not, write
the single most useful NEXT search query -- it should target what is still
missing, not repeat what you already have. Return JSON only.""", HopDecision)


def multi_hop_retrieve(query: str, retrieve_fn, max_hops: int = 3, k: int = 5):
    """Loop retrieve -> assess -> retrieve. Returns (chunks, trace).

    `retrieve_fn(query, k) -> list[Chunk]`. The trace is the interesting output
    for a learning repo: it shows the query the system INVENTED at each hop.
    """
    gathered, seen, trace = [], set(), []
    current = query

    for hop in range(1, max_hops + 1):
        chunks = [c for c in retrieve_fn(current, k) if c.id not in seen]
        seen.update(c.id for c in chunks)
        gathered.extend(chunks)
        trace.append({"hop": hop, "query": current, "new_chunks": len(chunks)})

        if hop == max_hops:
            break
        decision = next_hop(query, [c.text for c in gathered], hop)
        if decision.have_enough or not decision.next_query.strip():
            trace[-1]["stopped"] = "model satisfied"
            break
        current = decision.next_query

    return gathered, trace
