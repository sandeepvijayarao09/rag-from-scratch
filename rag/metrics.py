"""Retrieval metrics.

Stage 2 had exactly one correct chunk per query. Real benchmarks don't:
SciFact has up to 5 relevant documents for a single claim, and other BEIR
datasets use graded relevance (0/1/2). So the metrics generalise to a
{doc_id: grade} judgement set.

  recall@k -- |retrieved ∩ relevant| / |relevant|
              "did the generator even get a chance?" If this is 0.6, no amount
              of prompt engineering helps -- 40% of the time the answer simply
              is not in the context window.

  MRR@k    -- 1 / (rank of the FIRST relevant doc), 0 if none in top k.
              Cheap, intuitive, ignores everything after the first hit.

  nDCG@k   -- discounted gain over ALL relevant docs, normalised by the best
              possible ranking. This is BEIR's headline metric and the one to
              quote if you want to compare against published leaderboards.
              Unlike recall it is sensitive to ORDER, and unlike MRR it cares
              about the 2nd and 3rd relevant doc too.

Why three? Because they fail differently, and Stage 2 already showed why that
matters: paraphrasing the queries barely moved recall@5 (0.98 -> 0.96) while
MRR fell hard (0.98 -> 0.80). The right chunk was still being retrieved, just
ranked lower. Recall alone would have reported "no problem".
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuery:
    question: str
    relevant: dict[str, int] = field(default_factory=dict)  # doc_id -> grade
    qid: str = ""
    variant: str = ""

    @classmethod
    def single(cls, question: str, gold_id, **kw) -> "EvalQuery":
        """Convenience for the one-correct-answer case (Stage 2)."""
        return cls(question=question, relevant={str(gold_id): 1}, **kw)


@dataclass
class Report:
    n: int
    recall: dict[int, float]
    mrr: float
    ndcg: dict[int, float]
    k_max: int

    def __str__(self) -> str:
        r = "  ".join(f"r@{k}={self.recall[k]:.3f}" for k in sorted(self.recall))
        n = "  ".join(f"nDCG@{k}={self.ndcg[k]:.3f}" for k in sorted(self.ndcg))
        return f"n={self.n:<5} {r}  MRR@{self.k_max}={self.mrr:.3f}  {n}"


def _dcg(grades: list[int]) -> float:
    # rank i (1-based) is discounted by log2(i+1): position 1 undiscounted,
    # position 2 worth 0.63, position 10 worth 0.29.
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def evaluate(queries: list[EvalQuery], retrieve_fn, ks=(1, 3, 5, 10), ndcg_ks=(10,)) -> Report:
    """`retrieve_fn(question, k) -> list of doc ids, best first`."""
    k_max = max(max(ks), max(ndcg_ks))
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ndcg_ks}
    reciprocal_ranks = []

    for q in queries:
        ranked = [str(d) for d in retrieve_fn(q.question, k_max)]
        grades = [q.relevant.get(d, 0) for d in ranked]

        for k in ks:
            found = sum(1 for g in grades[:k] if g > 0)
            recalls[k].append(found / len(q.relevant) if q.relevant else 0.0)

        rank = next((i + 1 for i, g in enumerate(grades) if g > 0), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in ndcg_ks:
            ideal = sorted(q.relevant.values(), reverse=True)[:k]
            idcg = _dcg(ideal)
            ndcgs[k].append(_dcg(grades[:k]) / idcg if idcg else 0.0)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return Report(
        n=len(queries),
        recall={k: mean(v) for k, v in recalls.items()},
        mrr=mean(reciprocal_ranks),
        ndcg={k: mean(v) for k, v in ndcgs.items()},
        k_max=k_max,
    )


def load_evalset(variant: str = "direct", path: str = "data/evalset.json") -> list[EvalQuery]:
    """Load the Stage 2 synthetic cat-facts eval set."""
    data = json.loads(Path(path).read_text())
    return [
        EvalQuery.single(pair[variant], chunk_id, variant=variant, qid=chunk_id)
        for chunk_id, pair in sorted(data.items(), key=lambda kv: int(kv[0]))
        if pair.get(variant)
    ]
