"""Stage 2b: build a labelled eval set -- and build in its own bias check.

For every fact in the corpus we ask a strong local model (gemma4) for TWO
questions that the fact answers:

  direct       -- phrased the way someone who just read the fact would phrase it
  paraphrased  -- the same information need, deliberately avoiding the fact's
                  distinctive vocabulary; synonyms, everyday words, user voice

The gold label is free: the question was generated FROM chunk i, so chunk i is
the correct retrieval. That is the whole trick that makes synthetic eval cheap.

It is also the trap. A `direct` question inherits the chunk's rare words, so
embedding similarity is being handed the answer -- scores come out inflated and
every technique looks like it works. The `paraphrased` set removes that gift.

We keep both on purpose. The GAP between the two is the measurement that tells
you how much of your recall number is real and how much is leakage. Any single
synthetic number, reported alone, is close to meaningless.

Resumable: re-running only generates what's missing.
"""

import json
import sys
import time
from pathlib import Path

import ollama
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import load_cat_facts  # noqa: E402

GENERATOR_MODEL = "gemma4:e4b"
OUT = Path("data/evalset.json")


class QuestionPair(BaseModel):
    direct: str
    paraphrased: str


PROMPT = """Here is a single fact from a cat trivia database:

"{fact}"

Write two questions that this fact answers.

1. "direct": how someone who had just read this fact would naturally ask it.

2. "paraphrased": the SAME information need, but written by someone who has
   never seen this fact and does not know its wording. Deliberately avoid the
   distinctive or technical words used above -- reach for synonyms and plain
   everyday language. Make it sound like a real person typing into a search box.

Both must be answerable by this fact alone. Do not include the answer.
Return JSON only."""


def generate(fact: str) -> dict:
    response = ollama.chat(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(fact=fact)}],
        format=QuestionPair.model_json_schema(),
        options={"temperature": 0.7},
    )
    return QuestionPair.model_validate_json(response["message"]["content"]).model_dump()


if __name__ == "__main__":
    chunks = load_cat_facts()
    OUT.parent.mkdir(exist_ok=True)

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [c for c in chunks if str(c.id) not in done]
    print(f"{len(done)} cached, {len(todo)} to generate\n", flush=True)

    start = time.perf_counter()
    for i, chunk in enumerate(todo, 1):
        try:
            done[str(chunk.id)] = generate(chunk.text)
        except Exception as e:
            print(f"  [{chunk.id}] failed: {e}", flush=True)
            continue

        if i % 10 == 0 or i == len(todo):
            OUT.write_text(json.dumps(done, indent=2))
            rate = (time.perf_counter() - start) / i
            print(f"  {i}/{len(todo)}  ({rate:.1f}s/fact, ~{rate*(len(todo)-i)/60:.1f}min left)",
                  flush=True)

    OUT.write_text(json.dumps(done, indent=2))
    print(f"\nwrote {len(done)} question pairs -> {OUT}")
