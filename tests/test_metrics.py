import math

import pytest

from rag.metrics import EvalQuery, evaluate


def fixed(ranking):
    """A retriever that always returns the same ranking."""
    return lambda question, k: ranking[:k]


def test_perfect_ranking_scores_one():
    q = EvalQuery("q", relevant={"a": 1, "b": 1})
    r = evaluate([q], fixed(["a", "b", "c"]), ks=(1, 2), ndcg_ks=(10,))
    assert r.recall[1] == 0.5
    assert r.recall[2] == 1.0
    assert r.mrr == 1.0
    assert r.ndcg[10] == pytest.approx(1.0)


def test_mrr_uses_first_relevant_rank():
    q = EvalQuery("q", relevant={"c": 1})
    r = evaluate([q], fixed(["a", "b", "c"]), ks=(1, 3), ndcg_ks=(3,))
    assert r.mrr == pytest.approx(1 / 3)
    assert r.recall[1] == 0.0
    assert r.recall[3] == 1.0


def test_ndcg_matches_hand_computation():
    # One relevant doc at rank 2: DCG = 1/log2(3), ideal DCG = 1.
    q = EvalQuery("q", relevant={"b": 1})
    r = evaluate([q], fixed(["a", "b"]), ks=(1,), ndcg_ks=(10,))
    assert r.ndcg[10] == pytest.approx(1 / math.log2(3))


def test_graded_relevance_rewards_putting_the_better_doc_first():
    q = EvalQuery("q", relevant={"x": 2, "y": 1})
    best = evaluate([q], fixed(["x", "y"]), ks=(1,), ndcg_ks=(2,))
    swapped = evaluate([q], fixed(["y", "x"]), ks=(1,), ndcg_ks=(2,))
    assert best.ndcg[2] == pytest.approx(1.0)
    assert swapped.ndcg[2] < best.ndcg[2]


def test_miss_scores_zero_and_ids_are_compared_as_strings():
    q = EvalQuery.single("q", 7)
    hit = evaluate([q], fixed([7]), ks=(1,), ndcg_ks=(1,))
    miss = evaluate([q], fixed([1, 2]), ks=(1,), ndcg_ks=(1,))
    assert hit.recall[1] == 1.0
    assert (miss.recall[1], miss.mrr, miss.ndcg[1]) == (0.0, 0.0, 0.0)


def test_means_are_over_queries():
    qs = [EvalQuery("hit", relevant={"a": 1}), EvalQuery("miss", relevant={"z": 1})]
    r = evaluate(qs, fixed(["a"]), ks=(1,), ndcg_ks=(1,))
    assert r.n == 2
    assert r.recall[1] == 0.5
    assert r.mrr == 0.5
