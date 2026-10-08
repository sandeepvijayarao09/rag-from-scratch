import pytest

from rag.fusion import _minmax, reciprocal_rank_fusion, weighted_fusion


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"}
    assert fused[2:] == ["c", "d"] or fused[2:] == ["d", "c"]


def test_rrf_score_is_one_over_k_plus_rank():
    # With k=0, rank 1 in one list (1.0) beats rank 2 in both (0.5 + 0.5 = 1.0 tie
    # broken by insertion order), and rank 3 alone (0.33) loses.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=0)
    assert fused[-1] not in {"b"}
    assert "b" in fused[:3]


def test_rrf_weights_shift_the_winner():
    rankings = [["dense"], ["sparse"]]
    assert reciprocal_rank_fusion(rankings, weights=[1.0, 2.0])[0] == "sparse"
    assert reciprocal_rank_fusion(rankings, weights=[2.0, 1.0])[0] == "dense"


def test_minmax_handles_flat_and_empty():
    assert _minmax([]) == []
    assert _minmax([3.0, 3.0]) == [0.5, 0.5]
    assert _minmax([1.0, 2.0, 3.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_weighted_fusion_is_scale_free():
    dense = [("a", 0.91), ("b", 0.90)]
    bm25 = [("b", 40.0), ("a", 10.0)]
    # Equal weights: a and b each get 1.0, so neither scale dominates.
    assert set(weighted_fusion([dense, bm25])) == {"a", "b"}
    assert weighted_fusion([dense, bm25], weights=[0.7, 0.3])[0] == "a"
    assert weighted_fusion([dense, bm25], weights=[0.3, 0.7])[0] == "b"
