"""The v2 statistics: every verdict in results/v2/RESULTS.md goes through these."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rag.v2.bench import compare, holm, mrr10, ndcg10, paired_bootstrap, recall10, verdict


def test_ranking_metrics():
    rel = {"a": 1, "b": 1}
    assert ndcg10(["a", "b", "c"], rel) == pytest.approx(1.0)
    assert ndcg10(["c", "d"], rel) == 0.0
    assert recall10(["a", "x"], rel) == 0.5
    assert mrr10(["x", "x", "b"], rel) == pytest.approx(1 / 3)


def test_graded_ndcg_rewards_the_better_document_first():
    rel = {"a": 2, "b": 1}
    assert ndcg10(["a", "b"], rel) > ndcg10(["b", "a"], rel)


def test_bootstrap_on_no_difference():
    lo, hi, p = paired_bootstrap(np.zeros(300))
    assert lo == hi == 0.0
    assert p == 1.0


def test_bootstrap_detects_a_real_shift():
    diff = np.random.default_rng(1).normal(0.05, 0.1, size=300)
    lo, hi, p = paired_bootstrap(diff)
    assert 0 < lo < diff.mean() < hi
    assert p < 0.01


def test_bootstrap_does_not_call_noise_significant():
    diff = np.random.default_rng(2).normal(0.0, 0.1, size=300)
    lo, hi, p = paired_bootstrap(diff)
    assert lo < 0 < hi
    assert p > 0.05


def test_bootstrap_is_reproducible():
    diff = np.random.default_rng(3).normal(0.01, 0.1, size=300)
    assert paired_bootstrap(diff) == paired_bootstrap(diff)


def test_holm_scales_and_stays_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)          # 3 x 0.01
    assert adj["c"] == pytest.approx(0.06)          # 2 x 0.03
    assert adj["b"] == pytest.approx(0.06)          # max(0.06, 1 x 0.04)
    assert adj["a"] <= adj["c"] <= adj["b"]


def test_holm_caps_at_one():
    assert holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_verdict_needs_both_p_and_an_interval_off_zero():
    assert verdict(-0.04, -0.07, -0.01, 0.01) == "worse"
    assert verdict(0.04, 0.01, 0.07, 0.01) == "better"
    assert verdict(-0.03, -0.06, -0.004, 0.22) == "no detectable difference"   # raw CI off zero, Holm p not
    assert verdict(0.01, -0.01, 0.03, 0.30) == "no detectable difference"


def test_compare_is_paired_and_counts_wins():
    stage = {"ctrl": {"ndcg10": {"1": 0.5, "2": 0.5, "3": 0.5}},
             "treat": {"ndcg10": {"1": 0.6, "2": 0.5, "3": 0.4}}}
    r = compare(stage, "treat", "ctrl")
    assert r["n"] == 3
    assert r["delta"] == pytest.approx(0.0)
    assert (r["better"], r["worse"]) == (1, 1)


def _report():
    path = Path(__file__).resolve().parents[1] / "v2" / "report.py"
    spec = importlib.util.spec_from_file_location("v2_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_labels_identical_and_missing_cells():
    report = _report()
    row = {"verdict": "worse", "delta": -0.04}
    summary = {"m": {"s": {"rows": {"cfg": row}, "identical": ["same"]}}}
    assert report.cell(summary, "m", "s", "cfg") == "▼ -0.040"
    assert report.cell(summary, "m", "s", "same") == "= 0"
    assert report.cell(summary, "m", "s", "absent") == "n/a"


def test_stage_comparisons_skip_train_rows_and_identical_configs():
    report = _report()
    q = {str(i): 0.5 for i in range(50)}
    worse = {k: 0.3 for k in q}
    data = {"ctrl": {"n": 50, "ndcg10": q}, "same": {"n": 50, "ndcg10": dict(q)},
            "bad": {"n": 50, "ndcg10": worse}, "bad [train]": {"n": 809, "ndcg10": worse}}
    out = report.stage_comparisons(data, "s", control_override="ctrl")
    assert set(out["rows"]) == {"bad"}
    assert out["identical"] == ["same"]
    assert out["rows"]["bad"]["verdict"] == "worse"
