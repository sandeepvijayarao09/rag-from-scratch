"""v2 Stage 3: the benchmark, the baseline, and the formats.

v1 tested four bge-base configs on the 300 TEST queries and then used the best
one as the baseline for every later stage, which flatters the baseline a
little. v2 makes the same decision on the TRAIN split instead, separately for
each embedding model, and only then scores it on test.

Two questions per model:
  query format  the model card's recommended format vs one alternative
                (bge: no prefix vs its instruction prefix; embeddinggemma:
                "search result" vs "fact checking"; qwen3: a SciFact-specific
                instruction vs its generic web-search one)
  titles        "title. text" vs text alone, the convention every published
                BEIR number uses
"""

import json

from _common import banner, embeddings_from_args

from rag.v2.bench import Bench
from rag.store import VectorStore

STAGE = "s03_benchmark"

if __name__ == "__main__":
    for name in embeddings_from_args(__doc__):
        banner(f"Stage 3, {name}")
        bench = Bench(name)
        formats = {"card": bench.spec.query, "alt": bench.spec.query_alt}

        # 1. Query format, decided on train.
        train_scores = {}
        for label, fmt in formats.items():
            vecs = bench.qvecs([q.question for q in bench.train], fmt)
            ranked = {q.qid: [d for d, _ in bench.dense_vec(v, 10)] for q, v in zip(bench.train, vecs)}
            train_scores[label] = bench.record(STAGE, f"query format: {label}", ranked, split="train",
                                               meta={"format": fmt})
        chosen = max(train_scores, key=train_scores.get)
        (bench.out / "query_format.json").write_text(json.dumps(
            {"format": formats[chosen], "label": chosen, "train_ndcg10": train_scores}, indent=2))
        bench.qfmt = formats[chosen]
        print(f"  chosen on train: {chosen}", flush=True)

        # 2. Both formats on test, so the effect is visible, not just the choice.
        for label, fmt in formats.items():
            vecs = bench.qvecs([q.question for q in bench.test], fmt)
            ranked = {q.qid: [d for d, _ in bench.dense_vec(v, 10)] for q, v in zip(bench.test, vecs)}
            bench.record(STAGE, f"query format: {label}", ranked, meta={"format": fmt})

        # 3. Titles, with the chosen query format.
        no_title = [bench.spec.doc.format(title="none", text=d.meta["raw_text"])
                    if "{title}" in bench.spec.doc and "title:" in bench.spec.doc
                    else d.meta["raw_text"] for d in bench.docs]
        store = VectorStore(bench.docs, bench.embedder.embed(no_title, progress=True))
        vecs = bench.qvecs([q.question for q in bench.test])
        bench.record(STAGE, "no titles", {q.qid: [c.doc_id for c, _ in store.search(v, k=10)]
                                          for q, v in zip(bench.test, vecs)})
        # The baseline every later stage compares against.
        bench.record(STAGE, "dense baseline", {q.qid: [d for d, _ in bench.dense_vec(v, 10)]
                                               for q, v in zip(bench.test, vecs)},
                     meta={"format": bench.qfmt})
