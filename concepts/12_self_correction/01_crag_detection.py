"""Stage 12: does self-correction actually detect its own failures?

Every pipeline up to here runs identically for every query and cannot notice
when it has failed. CRAG adds a grading step; Self-RAG adds a decision about
whether to retrieve at all. Both are only worth their latency if the judgements
are accurate, so accuracy is what this measures.

CRAG AS A DETECTION PROBLEM. The useful question is not "can an LLM grade a
document" but "does grading catch the cases where retrieval missed?" Split the
queries using ground truth:

    retrieval SUCCEEDED   gold document is in the top-k
    retrieval FAILED      gold document is not

then run the grader on both and see whether it can tell them apart.

    on SUCCEEDED we want 'correct'                (keep good documents)
    on FAILED    we want 'incorrect'/'ambiguous'  (detect, then fall back)

Both error directions cost something. Missing a failure means answering from
noise. Flagging a success means discarding documents that were fine and paying
for a needless fallback.

Watch for the degenerate case: a grader that always says "bad" scores 1.00 on
failure detection for free. Failure detection is only meaningful read next to
success detection.

SELF-RAG is a cheaper, separate test. Given a mix of queries that need lookup
and queries that do not, how accurately does the router separate them?

Resumable: every judgement is checkpointed, so interrupting costs one item.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import Embedder, VectorStore, load_corpus, load_queries  # noqa: E402
from rag.adaptive import crag_triage, should_retrieve  # noqa: E402
from rag.checkpoint import Checkpoint  # noqa: E402

SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED, K = 0, 3

# Queries needing no corpus lookup. Hand-written, because the router has to be
# tested on the kind of input that should bypass retrieval entirely.
NO_LOOKUP_NEEDED = [
    "Rewrite this as three bullet points: the study found no significant effect.",
    "What is 17 percent of 4,200?",
    "Translate 'the results were inconclusive' into formal academic phrasing.",
    "Summarise the previous answer in one sentence.",
    "Convert this list into a markdown table.",
    "Is 'affect' or 'effect' correct in 'the treatment had a positive ___'?",
    "Write a one-line git commit message for a bug fix in a parser.",
    "Explain the difference between precision and recall.",
]

if __name__ == "__main__":
    docs = load_corpus(with_title=True)
    queries = load_queries(split="test")
    random.Random(SEED).shuffle(queries)

    embedder = Embedder()
    store = VectorStore(docs, embedder.embed([d.text for d in docs]))

    succeeded, failed = [], []
    for q in queries:
        hits = [c for c, _ in store.search(embedder.embed([q.question])[0], k=K)]
        group = succeeded if {c.doc_id for c in hits} & set(q.relevant) else failed
        group.append((q, hits))
        if len(succeeded) >= SAMPLE_N and len(failed) >= SAMPLE_N // 2:
            break

    n_ok, n_bad = min(SAMPLE_N, len(succeeded)), min(SAMPLE_N // 2, len(failed))
    ck = Checkpoint("crag")
    print(f"CRAG detection: {n_ok} succeeded, {n_bad} failed retrievals "
          f"(top-{K}) [{ck.resumed()}]\n", flush=True)

    results = {}
    for label, group, count in [("retrieval SUCCEEDED", succeeded, n_ok),
                                ("retrieval FAILED", failed, n_bad)]:
        verdicts = {"correct": 0, "ambiguous": 0, "incorrect": 0}
        for i, (q, hits) in enumerate(group[:count], 1):
            key = f"{label}:{q.qid}"
            action = ck.get(key)
            if action is None:
                action, _ = crag_triage(q.question, hits)
                ck.set(key, action)
            verdicts[action] += 1
            if i % 4 == 0:
                print(f"  {label}: {i}/{count}", flush=True)
        results[label] = verdicts
        print(f"  {label}: {verdicts}\n", flush=True)
    ck.save()

    ok, bad = results["retrieval SUCCEEDED"], results["retrieval FAILED"]
    hit_ok = ok["correct"] / max(1, sum(ok.values()))
    hit_bad = (bad["incorrect"] + bad["ambiguous"]) / max(1, sum(bad.values()))

    print("=" * 78)
    print(f"{'ground truth':<22} {'correct':>9} {'ambiguous':>11} {'incorrect':>11} "
          f"{'detected':>12}")
    print("-" * 78)
    print(f"{'retrieval SUCCEEDED':<22} {ok['correct']:>9} {ok['ambiguous']:>11} "
          f"{ok['incorrect']:>11} {hit_ok:>11.2f}  (want correct)")
    print(f"{'retrieval FAILED':<22} {bad['correct']:>9} {bad['ambiguous']:>11} "
          f"{bad['incorrect']:>11} {hit_bad:>11.2f}  (want flagged)")
    print("=" * 78)
    # The decisive number. A grader with no signal flags the same fraction of
    # good and bad retrievals; balanced accuracy then lands on 0.50 no matter
    # how impressive the raw failure-detection rate looks.
    flag_given_ok = 1 - hit_ok
    flag_given_bad = hit_bad
    balanced = (hit_ok + hit_bad) / 2
    print(f"\n{'flag rate | retrieval SUCCEEDED':<34} {flag_given_ok:.3f}")
    print(f"{'flag rate | retrieval FAILED':<34} {flag_given_bad:.3f}")
    print(f"{'separation':<34} {flag_given_bad - flag_given_ok:+.3f}")
    print(f"{'balanced accuracy':<34} {balanced:.3f}")
    print("\nBalanced accuracy near 0.50 means the grader carries no signal: it\n"
          "flags good and bad retrievals at the same rate, and its headline\n"
          "failure-detection score is an artifact of mostly saying bad.")

    print("\n--- Self-RAG: should we retrieve at all? ---", flush=True)
    rck = Checkpoint("selfrag")
    n_probe = min(8, len(succeeded))
    lookup_ok = 0
    for q, _ in succeeded[:n_probe]:
        v = rck.get(f"claim:{q.qid}")
        if v is None:
            v = should_retrieve(q.question).needs_retrieval
            rck.set(f"claim:{q.qid}", v)
        lookup_ok += bool(v)
    skip_ok = 0
    for j, text in enumerate(NO_LOOKUP_NEEDED):
        v = rck.get(f"nolookup:{j}")
        if v is None:
            v = should_retrieve(text).needs_retrieval
            rck.set(f"nolookup:{j}", v)
        skip_ok += (not v)
    rck.save()

    print(f"\n{'query type':<24} {'routed correctly':>18}")
    print("-" * 46)
    print(f"{'SciFact claims':<24} {lookup_ok}/{n_probe} = "
          f"{lookup_ok/n_probe:>9.2f}  (want retrieve)")
    print(f"{'no-lookup prompts':<24} {skip_ok}/{len(NO_LOOKUP_NEEDED)} = "
          f"{skip_ok/len(NO_LOOKUP_NEEDED):>9.2f}  (want skip)")
