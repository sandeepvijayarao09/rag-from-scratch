"""Stage 11c: is the low verdict rate the prompt, or the model?

Where this stands:
    11a  question-answering prompt, gold verdict rate 0.33
    11b  three-way claim verification,  gold verdict rate 0.40

Reframing recovered 0.07. Something else is suppressing the other 0.60.

Two candidates:
    PROMPT   11b tells the model "topically related is NOT the same as settling
             the claim", at temperature 0. That may over-suppress verdicts.
    MODEL    SciFact claims are technical inferences from scientific abstracts.
             Connecting "cells lacking clpC have a defect in sporulation
             efficiency" to a high-throughput genetic screen is expert reading.
             gemma4:e4b may simply not be able to do it.

These are distinguishable. Run the identical claims and identical gold documents
under a relaxed prompt. If the verdict rate jumps, it was the prompt. If it
barely moves, the generator is the ceiling for this stage and the honest
write-up says so.

Gold context only, since that is where the disagreement is.
"""

import random
import sys
from pathlib import Path

import ollama
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import load_corpus, load_queries  # noqa: E402
from rag.generate import ANSWER_MODEL, format_context  # noqa: E402

SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = 0


class Verdict(BaseModel):
    verdict: str
    citation: str
    reason: str


STRICT = """You are verifying a scientific claim against retrieved evidence.

Evidence:
{context}

Claim: "{claim}"

Return exactly one verdict:
  SUPPORTED   the evidence states this claim is true
  REFUTED     the evidence states this claim is false
  NOT_ENOUGH  the evidence does not settle the claim either way

Rules:
- Use ONLY the evidence above. Outside knowledge is not permitted, even if you
  are certain the claim is true.
- Topically related is NOT the same as settling the claim. If the evidence is
  about the right field but does not address this specific claim, return
  NOT_ENOUGH.
- Cite the passage number you used, e.g. "[1]". Leave empty for NOT_ENOUGH.

Return JSON only."""

RELAXED = """You are verifying a scientific claim against retrieved evidence.

Evidence:
{context}

Claim: "{claim}"

Return exactly one verdict:
  SUPPORTED   the evidence is consistent with this claim
  REFUTED     the evidence contradicts this claim
  NOT_ENOUGH  the evidence is about a different topic entirely

Reason from the evidence like a domain expert would. The evidence rarely
restates a claim word for word, so draw the reasonable inference from the
methods and results reported. Reserve NOT_ENOUGH for evidence that is genuinely
off-topic, not merely for evidence that requires a step of reasoning.

Cite the passage number you used, e.g. "[1]". Return JSON only."""


def verify(prompt: str, claim: str, chunks) -> Verdict:
    r = ollama.chat(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt.format(
            context=format_context(chunks), claim=claim)}],
        format=Verdict.model_json_schema(),
        options={"temperature": 0.0},
    )
    return Verdict.model_validate_json(r["message"]["content"])


if __name__ == "__main__":
    by_id = {d.id: d for d in load_corpus(with_title=True)}
    queries = load_queries(split="test")
    random.Random(SEED).shuffle(queries)
    queries = [q for q in queries if q.relevant][:SAMPLE_N]

    print(f"n={len(queries)} claims, GOLD context only\n", flush=True)
    results = {}
    for name, prompt in [("strict (11b)", STRICT), ("relaxed", RELAXED)]:
        verdicts = 0
        for i, q in enumerate(queries, 1):
            gold = [by_id[list(q.relevant)[0]]]
            if verify(prompt, q.question, gold).verdict.upper() in ("SUPPORTED", "REFUTED"):
                verdicts += 1
            if i % 15 == 0:
                print(f"  {name}: {i}/{len(queries)}", flush=True)
        results[name] = verdicts
        print(f"  {name}: {verdicts}/{len(queries)} = {verdicts/len(queries):.2f}\n", flush=True)

    n = len(queries)
    print("=" * 62)
    print(f"{'prompt':<16} {'verdict rate on gold':>22}")
    print("-" * 62)
    for name, v in results.items():
        print(f"{name:<16} {v}/{n} = {v/n:>14.2f}")
    print("=" * 62)
    delta = (results["relaxed"] - results["strict (11b)"]) / n
    print(f"\ndelta: {delta:+.2f}")
    print("\nA large positive delta means the strictness clause was suppressing\n"
          "verdicts and the prompt was the problem. A small one means gemma4:e4b\n"
          "cannot make the inference and the generator is this stage's ceiling.")
