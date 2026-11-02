"""v2 Stages 1-2: the cat-facts warm-up, rerun on the v2 models.

Stage 1 is naive RAG end to end; Stage 2 builds a 150-question eval set and
finds the leak in it: questions written by looking at a chunk reuse its words,
so "direct" questions are far easier than the same questions paraphrased.

v2 keeps the same eval set (it is data, and regenerating it would move the
goalposts) and asks one new question of it: does the paraphrase gap shrink
with a stronger embedding model? It also regenerates Stage 1's answers with
gemma4:12b for ten fixed questions, saved for the guide.

Scores are hit@1/@3/@5 and MRR@5 against the one gold chunk, with a paired
bootstrap on the direct-vs-paraphrased gap.
"""

import json
from pathlib import Path

import numpy as np

from _common import banner, embeddings_from_args

from rag.corpus import load_cat_facts
from rag.embed import Embedder
from rag.metrics import load_evalset
from rag.store import VectorStore
from rag.v2.bench import paired_bootstrap
from rag.v2.llm import LLM
from rag.v2.models import EMBEDDINGS, GENERATOR

OUT = Path("results/v2/s01_s02_cat_facts.json")
ANSWER_PROMPT = """Use only the following facts to answer. If they do not contain the
answer, say so.

{context}

Question: {question}"""

if __name__ == "__main__":
    chunks = load_cat_facts()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in embeddings_from_args(__doc__):
        banner(f"Stages 1-2, {name}")
        spec = EMBEDDINGS[name]
        emb = Embedder(model=spec.ollama, cache_dir=f"cache/v2/emb/{name}/cat-facts")
        store = VectorStore(chunks, emb.embed([spec.doc.format(title="none", text=c.text)
                                               for c in chunks]))
        row = {}
        for variant in ["direct", "paraphrased"]:
            qs = load_evalset(variant)
            vecs = emb.embed([spec.query.format(q=q.question) for q in qs])
            ranks = []
            for q, v in zip(qs, vecs):
                ids = [c.id for c, _ in store.search(v, k=5)]
                gold = next(iter(q.relevant))
                ranks.append(ids.index(gold) + 1 if gold in ids else None)
            row[variant] = {
                "hit@1": float(np.mean([r == 1 for r in ranks])),
                "hit@3": float(np.mean([r is not None and r <= 3 for r in ranks])),
                "hit@5": float(np.mean([r is not None for r in ranks])),
                "mrr@5": float(np.mean([1 / r if r else 0 for r in ranks])),
                "per_query_rr": [1 / r if r else 0 for r in ranks],
            }
            print(f"  {variant:<12} hit@1 {row[variant]['hit@1']:.3f}  "
                  f"hit@5 {row[variant]['hit@5']:.3f}  MRR@5 {row[variant]['mrr@5']:.3f}", flush=True)
        diff = np.array(row["paraphrased"]["per_query_rr"]) - np.array(row["direct"]["per_query_rr"])
        lo, hi, p = paired_bootstrap(diff)
        row["paraphrase_gap_mrr"] = {"delta": float(diff.mean()), "ci95": [lo, hi], "p": p}
        print(f"  paraphrase gap (MRR@5): {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        results[name] = row
        OUT.write_text(json.dumps(results, indent=1))

    # Stage 1: ten fixed questions answered by the 12B model over bge-base retrieval.
    banner(f"Stage 1, answers from {GENERATOR}")
    spec = EMBEDDINGS["bge-base"]
    emb = Embedder(model=spec.ollama, cache_dir="cache/v2/emb/bge-base/cat-facts")
    store = VectorStore(chunks, emb.embed([c.text for c in chunks]))
    llm = LLM(GENERATOR)
    answers = []
    for q in load_evalset("paraphrased")[:10]:
        ctx = [c.text for c, _ in store.search(emb.embed([q.question])[0], k=3)]
        a = llm(ANSWER_PROMPT.format(context="\n".join(f"- {t}" for t in ctx), question=q.question))
        answers.append({"question": q.question, "context": ctx, "answer": a,
                        "gold_in_context": chunks[int(next(iter(q.relevant)))].text in ctx})
        print(f"  Q: {q.question}\n  A: {a[:200]}\n", flush=True)
    llm.save()
    results["stage1_answers"] = {"model": GENERATOR, "answers": answers}
    OUT.write_text(json.dumps(results, indent=1))
