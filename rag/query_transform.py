"""Layer 3: fix the QUERY, not the index.

Everything in layers 1-2 assumed the query was fine and only the matching
needed work. These techniques assume the opposite. Stage 2 gave us direct
evidence that this is a real failure mode: paraphrasing the questions dropped
recall@1 from 0.927 to 0.680 without touching the corpus at all. The documents
were always findable. The words used to look for them were the problem.

Four transformations, each attacking a different mismatch:

  MULTI-QUERY   Generate N paraphrases, retrieve for each, fuse the rankings.
                Brute-force coverage of vocabulary space. Costs N retrievals.
                Directly targets the Stage 2 leakage gap.

  HyDE          Have the LLM write a hypothetical ANSWER to the query, then
                embed that instead of the query. The insight: you are searching
                a corpus of answers, so a fake answer is a better probe than a
                real question. Fixes the asymmetry measured in Stage 3 --
                12-word claims searching 250-word abstracts. The hypothetical
                document does not need to be factually correct; it only needs
                to look like the thing you are trying to find.

  STEP-BACK     Ask a more general question first, retrieve for both. Helps
                when a narrow question's context lives in a broader passage.

  DECOMPOSITION Split a compound question into sub-questions and retrieve for
                each. "Compare X and Y" cannot be served by one retrieval,
                because no single document is about both.

All of these cost at least one LLM call per query, which moves retrieval from
milliseconds to seconds. That is the tradeoff, and it is why they belong after
hybrid and reranking rather than before.
"""

import ollama
from pydantic import BaseModel

GENERATOR_MODEL = "gemma4:e4b"


class Paraphrases(BaseModel):
    queries: list[str]


class SubQuestions(BaseModel):
    sub_questions: list[str]


def _chat(prompt: str, schema=None, temperature: float = 0.3) -> str:
    kwargs = {"format": schema.model_json_schema()} if schema else {}
    response = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
        **kwargs,
    )
    return response["message"]["content"]


MULTI_QUERY_PROMPT = """Rewrite this search query {n} different ways.

Query: "{query}"

Each rewrite must preserve the exact information need but use DIFFERENT
vocabulary -- synonyms, more general terms, more specific terms, different
phrasing. The goal is to cover wording that relevant documents might use but
the original query does not. Return JSON only."""


def multi_query(query: str, n: int = 3) -> list[str]:
    """Return the original query plus n paraphrases."""
    try:
        out = Paraphrases.model_validate_json(
            _chat(MULTI_QUERY_PROMPT.format(query=query, n=n), Paraphrases, 0.7))
        return [query] + out.queries[:n]
    except Exception:
        return [query]


HYDE_PROMPT = """Write a short passage from a scientific paper abstract that
would directly support or refute this claim:

"{query}"

Write it in the voice and vocabulary of a real abstract: technical terms,
methods, results. Two or three sentences. Do not hedge, do not say you are
uncertain, do not mention the claim itself. Just write the passage."""


def hyde(query: str) -> str:
    """A hypothetical document to embed INSTEAD of the query.

    Factual correctness is irrelevant. What matters is that the text lives in
    the same region of embedding space as real answers, which a bare question
    does not.
    """
    try:
        return _chat(HYDE_PROMPT.format(query=query), temperature=0.5).strip() or query
    except Exception:
        return query


STEP_BACK_PROMPT = """Here is a specific claim:

"{query}"

Write ONE broader, more general question about the underlying topic -- the
question you would ask to find background material that gives context for
judging this claim. Return only the question."""


def step_back(query: str) -> list[str]:
    try:
        broader = _chat(STEP_BACK_PROMPT.format(query=query)).strip()
        return [query, broader] if broader else [query]
    except Exception:
        return [query]


DECOMPOSE_PROMPT = """Break this claim into the smallest set of independent
sub-questions that together would let you verify it:

"{query}"

If it is already a single simple claim, return it unchanged as one item.
Return JSON only."""


def decompose(query: str) -> list[str]:
    try:
        out = SubQuestions.model_validate_json(
            _chat(DECOMPOSE_PROMPT.format(query=query), SubQuestions))
        return out.sub_questions or [query]
    except Exception:
        return [query]
