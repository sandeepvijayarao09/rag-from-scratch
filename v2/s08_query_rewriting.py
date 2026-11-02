"""v2 Stage 8: query rewriting with a 12B rewriter, on all 300 queries.

What v1 did, and what changes:
  - v1 rewrote queries with gemma4:e4b at temperature 0.3-0.7, on a 50-query
    subsample. v2 uses gemma4:12b at temperature 0, on all 300 test queries.
  - v1 compared four techniques in their "replace or fuse by rank" form. v2 keeps
    those arms, and adds one repaired arm per technique that keeps the original
    query and adds the rewrite at a weight tuned on a 150-query train subsample.
  - Optional: --small also rewrites with gemma4:e4b on the same 300 queries, so
    the effect of rewriter size can be read off directly.

Two phases, because generation is the slow part and is shared by all three
embedding models:
    python v2/s08_query_rewriting.py generate [--small]
    python v2/s08_query_rewriting.py score    [--small] [--embedding NAME]
"""

import json
import sys
import time

from _common import banner, embeddings_from_args
from pydantic import BaseModel

from rag.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.beir import load_queries
from rag.query_transform import DECOMPOSE_PROMPT, HYDE_PROMPT, MULTI_QUERY_PROMPT, STEP_BACK_PROMPT
from rag.store import normalize
from rag.v2.bench import Bench, subsample
from rag.v2.llm import LLM
from rag.v2.models import GENERATORS

STAGE = "s08_query_rewriting"
TRAIN_N = 150
TECHNIQUES = ["multi_query", "hyde", "step_back", "decompose"]


class Paraphrases(BaseModel):
    queries: list[str]


class SubQuestions(BaseModel):
    sub_questions: list[str]


def rewrite(llm: LLM, technique: str, claim: str) -> list[str]:
    """Return the rewritten queries only (the original is added by the caller)."""
    try:
        if technique == "multi_query":
            out = Paraphrases.model_validate_json(llm(MULTI_QUERY_PROMPT.format(query=claim, n=3),
                                                      Paraphrases))
            return [x for x in out.queries[:3] if x.strip()]
        if technique == "hyde":
            return [llm(HYDE_PROMPT.format(query=claim))]
        if technique == "step_back":
            return [llm(STEP_BACK_PROMPT.format(query=claim))]
        out = SubQuestions.model_validate_json(llm(DECOMPOSE_PROMPT.format(query=claim), SubQuestions))
        return [x for x in out.sub_questions if x.strip()]
    except Exception as e:
        print(f"    rewrite failed ({technique}): {e}", flush=True)
        return []


def query_sets():
    return load_queries(split="test"), subsample(load_queries(split="train"), TRAIN_N)


def generate(model_key: str) -> None:
    llm = LLM(GENERATORS[model_key])
    test, train = query_sets()
    jobs = [(t, q) for t in TECHNIQUES for q in test]
    if model_key == "large":
        jobs += [(t, q) for t in TECHNIQUES if t != "multi_query" for q in train]
    start = time.perf_counter()
    store = {}
    for i, (t, q) in enumerate(jobs, 1):
        store[f"{t}:{q.qid}"] = rewrite(llm, t, q.question)
        if i % 25 == 0 or i == len(jobs):
            llm.save()
            el = time.perf_counter() - start
            print(f"  {i}/{len(jobs)}  {el / 60:.0f} min elapsed", flush=True)
    llm.save()
    out = f"results/v2/rewrites_{GENERATORS[model_key].replace(':', '-')}.json"
    with open(out, "w") as f:
        json.dump(store, f, indent=1)
    print(f"  wrote {out}", flush=True)


def score(model_key: str, names: list[str]) -> None:
    gen = GENERATORS[model_key]
    rewrites = json.loads(open(f"results/v2/rewrites_{gen.replace(':', '-')}.json").read())
    tag = "" if model_key == "large" else f" [{gen}]"
    for name in names:
        banner(f"Stage 8, {name}, rewriter {gen}")
        bench = Bench(name)
        test, train = query_sets()
        # Embed every rewrite in one batch first. The embedder rewrites its whole
        # cache file on each call that adds something, so one call per rewrite
        # would be quadratic.
        bench.embedder.embed([bench.query_text(r) for rws in rewrites.values() for r in rws],
                             progress=True)

        def vec(text):
            return normalize(bench.embedder.embed([bench.query_text(text)])[0])

        def originals(qs):
            vs = bench.qvecs([q.question for q in qs])
            return {q.qid: (v, bench.dense_vec(v, 100)) for q, v in zip(qs, vs)}

        orig = {"test": originals(test)}
        if model_key == "large":
            orig["train"] = originals(train)
        bench.record(STAGE, "dense", {q: [d for d, _ in o[1]] for q, o in orig["test"].items()})

        for t in TECHNIQUES:
            # The v1 form: HyDE embeds the fake answer alone; the others fuse every
            # query (step-back and multi-query keep the original, decompose drops it).
            def v1_form(qs):
                out = {}
                for q in qs:
                    rw = rewrites.get(f"{t}:{q.qid}", [])
                    if not rw:
                        out[q.qid] = [d for d, _ in orig["test"][q.qid][1]]
                    elif t == "hyde":
                        out[q.qid] = [d for d, _ in bench.dense_vec(vec(rw[0]), 10)]
                    else:
                        lists = ([] if t == "decompose" else [[d for d, _ in orig["test"][q.qid][1]]])
                        lists += [[d for d, _ in bench.dense_vec(vec(r), 100)] for r in rw]
                        out[q.qid] = reciprocal_rank_fusion(lists, k=60)
                return out
            bench.record(STAGE, f"{t} (v1 form){tag}", v1_form(test))

            if model_key != "large" or t == "multi_query":
                continue

            # The repair: keep the original, add the rewrite at a train-tuned weight.
            def pre(qs, split):
                return {q.qid: [vec(r) for r in rewrites.get(f"{t}:{q.qid}", [])] for q in qs}
            rv = {"test": pre(test, "test"), "train": pre(train, "train")}

            def repaired(w, qs, split):
                out = {}
                for q in qs:
                    ov, olist = orig[split][q.qid]
                    vs = rv[split][q.qid]
                    if not vs:
                        out[q.qid] = [d for d, _ in olist[:10]]
                    elif t == "hyde":
                        out[q.qid] = [d for d, _ in bench.dense_vec(normalize(ov + w * vs[0]), 10)]
                    else:
                        extra = [bench.dense_vec(v, 100) for v in vs]
                        out[q.qid] = weighted_fusion([olist] + extra,
                                                     [1 - w] + [w / len(extra)] * len(extra))[:10]
                return out
            grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] if t == "hyde" else [0.1, 0.2, 0.3, 0.4, 0.5]
            w, tr = bench.tune(lambda p, qs: repaired(p, qs, "train"), grid, queries=train)
            bench.record(STAGE, f"{t} (repaired, tuned)", repaired(w, test, "test"),
                         meta={"weight": w, "train_ndcg10": tr, "train_n": len(train)})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "score"
    keys = ["large"] + (["small"] if "--small" in sys.argv else [])
    for key in keys:
        if mode == "generate":
            generate(key)
        else:
            score(key, embeddings_from_args(__doc__))
