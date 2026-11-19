import numpy as np
import pytest

from defects import evaluation as ev


def test_rates():
    s = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.7])
    y = np.array([0, 0, 0, 1, 1, 1])
    r = ev.rates(s, y, 0.5)
    assert r["catch_rate"] == pytest.approx(2 / 3)
    assert r["false_alarm_rate"] == pytest.approx(1 / 3)


def test_cost_per_unit_by_hand():
    # 2% defective, catch 90%, false alarms 5%: inspect 0.02*0.9 + 0.98*0.05 = 0.067
    # escapes 0.02*0.1 = 0.002 -> 0.067 * $1 + 0.002 * $100 = $0.267
    assert ev.flagged_share(0.9, 0.05, 0.02) == pytest.approx(0.067)
    assert ev.cost_per_unit(0.9, 0.05, 0.02, 1.0, 100.0) == pytest.approx(0.267)
    assert ev.blanket_costs(0.02, 1.0, 100.0) == {"inspect_all": 1.0, "inspect_none": 2.0}


def test_threshold_for_catch_rate():
    bad = np.arange(1.0, 11.0)                   # ten defective scores 1..10
    s = np.concatenate([bad, [0.5, 0.2]])
    y = np.concatenate([np.ones(10), np.zeros(2)]).astype(int)
    assert ev.threshold_for_catch_rate(s, y, 0.9) == 2.0   # one defect may pass
    assert ev.threshold_for_catch_rate(s, y, 1.0) == 1.0
    assert ev.rates(s, y, ev.threshold_for_catch_rate(s, y, 0.9))["catch_rate"] >= 0.9


def test_cost_optimal_threshold_extremes():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 50)])
    y = np.concatenate([np.zeros(100), np.ones(50)]).astype(int)
    # inspections free -> flag everything; escapes free -> flag nothing
    assert ev.rates(s, y, ev.cost_optimal_threshold(s, y, 0.02, 0.0, 100.0))["catch_rate"] == 1.0
    assert ev.cost_optimal_threshold(s, y, 0.02, 1.0, 0.0) == np.inf
    # in between, the chosen threshold beats both blanket policies on these images
    thr = ev.cost_optimal_threshold(s, y, 0.02, 1.0, 100.0)
    r = ev.rates(s, y, thr)
    c = ev.cost_per_unit(r["catch_rate"], r["false_alarm_rate"], 0.02, 1.0, 100.0)
    assert c <= min(ev.blanket_costs(0.02, 1.0, 100.0).values())


def test_bootstrap_auroc_is_paired_and_reproducible():
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 20)
    s = rng.normal(size=50) + y
    a = ev.bootstrap_auroc(y, {"m": s, "same": s.copy()}, 50, np.random.default_rng(1))
    b = ev.bootstrap_auroc(y, {"m": s, "same": s.copy()}, 50, np.random.default_rng(1))
    np.testing.assert_array_equal(a["m"], b["m"])
    np.testing.assert_array_equal(a["m"], a["same"])
    lo, hi = ev.interval(a["m"])
    assert lo <= ev.auroc(y, s) <= hi
