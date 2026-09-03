"""Stage 11: the generation half -- citations, abstention, faithfulness.

Everything up to here measured RETRIEVAL. This measures what the model does
with what retrieval handed it, and it needs its own scoreboard because the two
fail independently: an answer can be perfectly faithful to the wrong documents.

THE ABSTENTION TEST is the interesting one. For each query we build three
context conditions:

  GOLD       the correct document. The model should answer and cite it.
  DISTRACTOR top-ranked documents with the gold one REMOVED. The context is
             topically plausible and does not contain the answer. The model
             should abstain.
  EMPTY      no context at all. The model should abstain.

Naive RAG has no abstention path, so it answers confidently from the distractor
condition. That is the most dangerous RAG failure in production: not a miss, but
a fluent wrong answer with no signal that anything went wrong.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_corpus, load_queries  # noqa: E402
from rag.generate import NO_ANSWER, answer, check_faithfulness  # noqa: E402

SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED, K = 0, 3

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    by_id = {d.id: d for d in docs}
    queries = load_queries(split="test")
    random.Random(SEED).shuffle(queries)
    queries = [q for q in queries if q.relevant][:SAMPLE_N]

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    print(f"n={len(queries)} queries (seed={SEED}), top-{K} context\n")

    counts = {c: {"answered": 0, "abstained": 0} for c in ("gold", "distractor", "empty")}
    faithful = {"supported": 0, "total": 0}

    for i, q in enumerate(queries, 1):
        qv = embedder.embed([q.question])[0]
        retrieved = [c for c, _ in store.search(qv, k=K + 5)]

        conditions = {
            "gold": [by_id[d] for d in list(q.relevant)[:1]],
            "distractor": [c for c in retrieved if c.doc_id not in q.relevant][:K],
            "empty": [],
        }

        for name, chunks in conditions.items():
            out = answer(q.question, chunks)
            abstained = NO_ANSWER in out
            counts[name]["abstained" if abstained else "answered"] += 1

            if name == "gold" and not abstained and chunks:
                faithful["total"] += 1
                if check_faithfulness(out, chunks[0].text).supported:
                    faithful["supported"] += 1

        if i % 10 == 0:
            print(f"  {i}/{len(queries)}", flush=True)

    n = len(queries)
    print("\n" + "=" * 68)
    print(f"{'context condition':<20} {'answered':>10} {'abstained':>11} {'correct?':>16}")
    print("-" * 68)
    want = {"gold": "answer", "distractor": "abstain", "empty": "abstain"}
    for name, c in counts.items():
        good = c["answered"] if want[name] == "answer" else c["abstained"]
        print(f"{name:<20} {c['answered']:>10} {c['abstained']:>11} "
              f"{good}/{n} = {good/n:.2f}  (want {want[name]})")
    print("=" * 68)
    if faithful["total"]:
        print(f"\nfaithfulness on GOLD context: "
              f"{faithful['supported']}/{faithful['total']} = "
              f"{faithful['supported']/faithful['total']:.2f} supported")
    print("\nThe distractor row is the one that matters. A low abstention rate there\n"
          "means the system answers confidently from irrelevant context -- invisible\n"
          "in every retrieval metric in this repo.")
