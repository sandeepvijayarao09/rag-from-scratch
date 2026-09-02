"""Stage 8: query rewriting. Fix the query, not the index.

Stage 2 proved this failure mode is real: paraphrasing the questions dropped
recall@1 from 0.927 to 0.680 with the corpus untouched. The documents were
always findable; the words used to look for them were wrong.

SUBSAMPLE. Every technique costs an LLM call per query at ~8s. We use a FIXED,
SEEDED subsample and score the baseline on the SAME queries, so deltas are
paired and comparable. Absolute values are NOT comparable to the full-300
results elsewhere in the repo.

Run one technique at a time:
    python 01_transform_eval.py baseline
    python 01_transform_eval.py multi_query
    python 01_transform_eval.py hyde
    python 01_transform_eval.py step_back
    python 01_transform_eval.py decompose
    python 01_transform_eval.py report
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag.fusion import reciprocal_rank_fusion  # noqa: E402
from rag.query_transform import decompose, hyde, multi_query, step_back  # noqa: E402

SAMPLE_N, SEED = 50, 0
TRANSFORMS = Path("results/llm_transforms.json")
RESULTS = Path("results/query_rewriting.json")
ORDER = ["baseline", "multi_query", "hyde", "step_back", "decompose"]


def sample():
    qs = load_queries(split="test")
    random.Random(SEED).shuffle(qs)
    return sorted(qs[:SAMPLE_N], key=lambda q: int(q.qid))


def report():
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if "baseline" not in saved:
        print("run 'baseline' first")
        return
    base = saved["baseline"]["ndcg10"]
    print("\n" + "=" * 76)
    print(f"{'technique':<16} {'nDCG@10':>9} {'vs baseline':>13} {'R@10':>8} "
          f"{'MRR':>8} {'s/query':>9}")
    print("-" * 76)
    for name in ORDER:
        if name not in saved:
            print(f"{name:<16} {'(not yet run)':>40}")
            continue
        r = saved[name]
        d = "" if name == "baseline" else f"{r['ndcg10'] - base:+.4f}"
        print(f"{name:<16} {r['ndcg10']:>9.4f} {d:>13} {r['recall10']:>8.4f} "
              f"{r['mrr']:>8.4f} {r['s_per_query']:>9.1f}")
    print("=" * 76)
    print(f"n={SAMPLE_N} paired subsample (seed={SEED}). Deltas are comparable;\n"
          f"absolute values are not comparable to the full-300 results elsewhere.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    if which == "report":
        report()
        sys.exit()

    queries = sample()
    docs = load_corpus(with_title=True)
    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))
    TRANSFORMS.parent.mkdir(exist_ok=True)
    cache = json.loads(TRANSFORMS.read_text()) if TRANSFORMS.exists() else {}

    FNS = {"multi_query": lambda t: multi_query(t, n=3), "hyde": lambda t: [hyde(t)],
           "step_back": step_back, "decompose": decompose}

    # Generate transformations up front so the timing is honest and resumable.
    if which != "baseline":
        start = time.perf_counter()
        for i, q in enumerate(queries, 1):
            key = f"{which}:{q.qid}"
            if key not in cache:
                cache[key] = FNS[which](q.question)
                TRANSFORMS.write_text(json.dumps(cache, indent=1))
            if i % 10 == 0:
                print(f"  transformed {i}/{len(queries)} "
                      f"({time.perf_counter()-start:.0f}s)", flush=True)

    def retrieve(question, k):
        q = next(x for x in queries if x.question == question)
        variants = [question] if which == "baseline" else cache[f"{which}:{q.qid}"]
        if len(variants) == 1:
            return [c.doc_id for c, _ in store.search(embedder.embed(variants)[0], k=k)]
        rankings = [[c.doc_id for c, _ in store.search(embedder.embed([v])[0], k=100)]
                    for v in variants]
        return reciprocal_rank_fusion(rankings, k=60)[:k]

    start = time.perf_counter()
    r = evaluate(queries, retrieve, ndcg_ks=(10,))
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    saved[which] = {"ndcg10": r.ndcg[10], "recall10": r.recall[10], "mrr": r.mrr,
                    "s_per_query": (time.perf_counter() - start) / len(queries)}
    RESULTS.write_text(json.dumps(saved, indent=2))
    print(f"\n{which:<14} {r}")

    if which != "baseline":
        print(f"\n--- example: {queries[0].question}")
        for v in cache[f'{which}:{queries[0].qid}'][:3]:
            print(f"    -> {v[:160]}")
    report()
