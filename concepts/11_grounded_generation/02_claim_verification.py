"""Stage 11b: the same test, with the task framed correctly.

11a asked the model to ANSWER a claim from context, and abstain if the context
was insufficient. It abstained on 20 of 30 gold documents, which looked like
catastrophic over-refusal.

Inspecting the outputs showed the eval was wrong, not the model. SciFact is a
CLAIM VERIFICATION dataset: a gold document may SUPPORT or REFUTE the claim.
Under a question-answering prompt a refuting document does not "answer"
anything, so INSUFFICIENT_CONTEXT was a defensible response and my binary
framing could not represent the dataset.

This is the three-way task the corpus actually encodes:

    SUPPORTED   the context states the claim is true
    REFUTED     the context states the claim is false
    NOT_ENOUGH  the context does not settle it

Correct behaviour per condition:
    gold        -> a VERDICT (supported or refuted). Abstaining is the failure.
    distractor  -> NOT_ENOUGH. A verdict here is a hallucinated judgement.
    empty       -> NOT_ENOUGH.

The distractor row still carries the important number. A system that returns
confident verdicts from irrelevant context fails invisibly: no retrieval metric
in this repo can see it.
"""

import random
import sys
from pathlib import Path

import ollama
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_corpus, load_queries  # noqa: E402
from rag.generate import ANSWER_MODEL, format_context  # noqa: E402

SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED, K = 0, 3


class Verdict(BaseModel):
    verdict: str          # SUPPORTED | REFUTED | NOT_ENOUGH
    citation: str         # e.g. "[1]", or "" when NOT_ENOUGH
    reason: str


PROMPT = """You are verifying a scientific claim against retrieved evidence.

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


def verify(claim: str, chunks) -> Verdict:
    context = format_context(chunks) if chunks else "(no evidence retrieved)"
    r = ollama.chat(
        model=ANSWER_MODEL,
        messages=[{"role": "user",
                   "content": PROMPT.format(context=context, claim=claim)}],
        format=Verdict.model_json_schema(),
        options={"temperature": 0.0},
    )
    return Verdict.model_validate_json(r["message"]["content"])


if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    by_id = {d.id: d for d in docs}
    queries = load_queries(split="test")
    random.Random(SEED).shuffle(queries)
    queries = [q for q in queries if q.relevant][:SAMPLE_N]

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    print(f"n={len(queries)} claims (seed={SEED}), top-{K} context\n", flush=True)

    counts = {c: {"verdict": 0, "not_enough": 0} for c in ("gold", "distractor", "empty")}
    cited = {"with": 0, "total": 0}

    for i, q in enumerate(queries, 1):
        retrieved = [c for c, _ in store.search(embedder.embed([q.question])[0], k=K + 5)]
        conditions = {
            "gold": [by_id[d] for d in list(q.relevant)[:1]],
            "distractor": [c for c in retrieved if c.doc_id not in q.relevant][:K],
            "empty": [],
        }
        for name, chunks in conditions.items():
            v = verify(q.question, chunks)
            gave_verdict = v.verdict.upper() in ("SUPPORTED", "REFUTED")
            counts[name]["verdict" if gave_verdict else "not_enough"] += 1
            if name == "gold" and gave_verdict:
                cited["total"] += 1
                cited["with"] += 1 if v.citation.strip() else 0
        if i % 10 == 0:
            print(f"  {i}/{len(queries)}", flush=True)

    n = len(queries)
    want = {"gold": "verdict", "distractor": "not_enough", "empty": "not_enough"}
    print("\n" + "=" * 72)
    print(f"{'context':<14} {'gave verdict':>13} {'NOT_ENOUGH':>12} {'correct':>18}")
    print("-" * 72)
    for name, c in counts.items():
        good = c[want[name]]
        print(f"{name:<14} {c['verdict']:>13} {c['not_enough']:>12} "
              f"{good}/{n} = {good/n:.2f}  (want {want[name]})")
    print("=" * 72)
    if cited["total"]:
        print(f"\ncitation present on gold verdicts: "
              f"{cited['with']}/{cited['total']} = {cited['with']/cited['total']:.2f}")
    print("\n11a used a question-answering prompt on the same data and scored gold\n"
          "at 0.33. Reframing as claim verification moved it to 0.40, so the framing\n"
          "was PART of the problem and not the whole of it. See 03 for what remains.")
