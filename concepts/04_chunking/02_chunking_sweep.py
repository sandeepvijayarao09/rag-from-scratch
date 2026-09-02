"""Stage 4b: does chunking actually help? Measure, don't assume.

Every strategy is scored on the same 300 queries with the same nDCG@10, and
`whole` is the no-chunking control, so the numbers are directly comparable to
the Stage 3 baseline.

Read `01_truncation_check.py` first. It predicts the answer: only 2.8% of
SciFact abstracts exceed the encoder window, so there is very little truncation
for chunking to rescue. If chunking is a universal good this should still show
gains. If it is a RESPONSE TO A SPECIFIC PROBLEM -- documents longer than the
window, or documents mixing topics -- then on a corpus with neither it should do
nothing, or hurt by fragmenting the context that made an abstract matchable.

Run in slices so no single pass is long:
    python 03_chunking_sweep.py 200        # whole + every size-200 config
    python 03_chunking_sweep.py 100a       # fixed/recursive at size 100
    python 03_chunking_sweep.py 100b       # sentence at size 100
    python 03_chunking_sweep.py report     # full table from cached results
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import BGE_QUERY_PREFIX, Embedder, VectorStore, evaluate, load_corpus, load_queries  # noqa: E402
from rag import chunking  # noqa: E402
from rag.retrieve import doc_level_retriever  # noqa: E402

RESULTS = Path("results/chunking.json")

SLICES = {
    "200": [("whole", 0, 0), ("fixed", 200, 40), ("sentence", 200, 1), ("recursive", 200, 40)],
    "100a": [("fixed", 100, 20), ("recursive", 100, 20)],
    "100b": [("sentence", 100, 1)],
}
ORDER = ["whole", "fixed 200/40", "sentence 200/1", "recursive 200/40",
         "fixed 100/20", "recursive 100/20", "sentence 100/1"]


def label(strategy, size, overlap):
    return strategy if not size else f"{strategy} {size}/{overlap}"


def report():
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if "whole" not in saved:
        print("run the '200' slice first (it contains the control)")
        return
    base = saved["whole"]["ndcg10"]
    print("\n" + "=" * 84)
    print(f"{'strategy':<22} {'chunks':>8} {'nDCG@10':>9} {'vs whole':>10} {'R@10':>8} {'MRR':>8}")
    print("-" * 84)
    for name in ORDER:
        if name not in saved:
            print(f"{name:<22} {'(not yet run)':>44}")
            continue
        r = saved[name]
        d = "" if name == "whole" else f"{r['ndcg10'] - base:+.4f}"
        print(f"{name:<22} {r['chunks']:>8} {r['ndcg10']:>9.4f} {d:>10} "
              f"{r['recall10']:>8.4f} {r['mrr']:>8.4f}")
    print("=" * 84)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    if which == "report":
        report()
        sys.exit()

    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")
    embedder = Embedder()
    RESULTS.parent.mkdir(exist_ok=True)
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

    for strategy, size, overlap in SLICES[which]:
        name = label(strategy, size, overlap)
        chunks = chunking.apply(docs, strategy, size, overlap)
        print(f"[{name}]  {len(chunks)} chunks from {len(docs)} docs "
              f"({len(chunks)/len(docs):.2f}x)", flush=True)

        store = VectorStore(chunks, embedder.embed([c.text for c in chunks], progress=True))
        retrieve = doc_level_retriever(store, embedder, prefix="", pool="max")
        r = evaluate(queries, retrieve, ndcg_ks=(10,))
        print(f"  {r}\n", flush=True)

        saved[name] = {"chunks": len(chunks), "ndcg10": r.ndcg[10],
                       "recall10": r.recall[10], "mrr": r.mrr}
        RESULTS.write_text(json.dumps(saved, indent=2))

    report()
