"""The generation half, with the two guardrails naive RAG lacks.

Stage 1's `naive_rag.py` said "Don't make up any new information" in the system prompt and
provided no mechanism to enforce it. That is a wish, not a design. Two things
turn it into something you can actually verify:

CITATIONS
    Number the context and require the model to cite [1], [2] per claim. This
    is not decoration. An uncited sentence is a sentence you cannot check, and
    citations make hallucination MECHANICALLY detectable -- you can verify a
    span against its cited source without a human reading the whole thing.

ABSTENTION
    Give the model an explicit, blessed way to say "the context does not
    answer this". Without one, the strongest force in the prompt is the
    instruction to answer, and a model with irrelevant context will answer
    anyway. This is the single most common RAG failure in production, and it
    is worse than a retrieval miss because it is confident and invisible.

FAITHFULNESS is then measurable: for each claim in the answer, is it supported
by the cited source? That is a separate scoreboard from retrieval, and it has
to be, because an answer can be perfectly faithful to documents that were the
wrong ones.
"""

import ollama
from pydantic import BaseModel

ANSWER_MODEL = "gemma4:e4b"
NO_ANSWER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer strictly from the numbered context provided.

Rules:
1. Every factual sentence must end with a citation like [1] or [2][3].
2. Use ONLY the context. Do not add outside knowledge, even if you are certain.
3. If the context does not contain the answer, reply with exactly
   {NO_ANSWER} and nothing else. This is the correct answer when the context
   is insufficient -- it is not a failure.
4. Do not speculate, hedge, or pad. Short answers are good."""


def format_context(chunks) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


def answer(question: str, chunks, model: str = ANSWER_MODEL, stream: bool = False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]
    if stream:
        return ollama.chat(model=model, messages=messages, stream=True)
    return ollama.chat(model=model, messages=messages,
                       options={"temperature": 0.0})["message"]["content"].strip()


class Faithfulness(BaseModel):
    supported: bool
    reason: str


JUDGE_PROMPT = """Source passage:
\"\"\"{source}\"\"\"

Claim:
\"\"\"{claim}\"\"\"

Is the claim FULLY supported by the source passage alone? Answer strictly on
whether the source states it. A claim that is true in the world but absent from
the source is NOT supported. Return JSON only."""


def check_faithfulness(claim: str, source: str, model: str = ANSWER_MODEL) -> Faithfulness:
    """LLM-as-judge. Cheap and useful, but remember it is a model grading a
    model -- report it alongside retrieval metrics, never instead of them."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(source=source, claim=claim)}],
        format=Faithfulness.model_json_schema(),
        options={"temperature": 0.0},
    )
    return Faithfulness.model_validate_json(response["message"]["content"])
