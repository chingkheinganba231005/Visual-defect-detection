"""Detection metrics with bootstrap intervals, and the inspection decisions built on them."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- detection quality

def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def bootstrap_auroc(labels: np.ndarray, scores: dict[str, np.ndarray], n_boot: int, rng: np.random.Generator):
    """Stratified bootstrap of AUROC, paired across methods (same resample for each)."""
    labels = np.asarray(labels)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    out = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            out[k][b] = roc_auc_score(labels[idx], np.asarray(s)[idx])
    return out


def interval(samples: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(x) for x in np.quantile(samples, [a, 1 - a])]


# ---------------------------------------------------------------- inspection decisions
#
# A unit is flagged when its score is at or above the threshold. Flagged units go
# to a human inspector (cost per unit inspected); defective units that are not
# flagged escape to the customer (cost per escape). The test images have far
# more defects than a real production line, so rates are measured separately on
# good and defective images and then weighted by an assumed defect rate.

def rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = np.asarray(scores) >= threshold
    labels = np.asarray(labels)
    catch = float(flagged[labels == 1].mean())
    false_alarm = float(flagged[labels == 0].mean())
    return {"catch_rate": catch, "false_alarm_rate": false_alarm}


def flagged_share(catch_rate: float, false_alarm_rate: float, defect_rate: float) -> float:
    return defect_rate * catch_rate + (1 - defect_rate) * false_alarm_rate


def cost_per_unit(catch_rate: float, false_alarm_rate: float, defect_rate: float,
                  cost_inspection: float, cost_escape: float) -> float:
    inspected = flagged_share(catch_rate, false_alarm_rate, defect_rate)
    escaped = defect_rate * (1 - catch_rate)
    return cost_inspection * inspected + cost_escape * escaped


def blanket_costs(defect_rate: float, cost_inspection: float, cost_escape: float) -> dict[str, float]:
    """Inspect every unit by hand (assumed to catch every defect), or inspect nothing."""
    return {"inspect_all": cost_inspection, "inspect_none": cost_escape * defect_rate}


def threshold_for_catch_rate(scores: np.ndarray, labels: np.ndarray, target: float) -> float:
    """Highest threshold that still flags at least ``target`` of the defective images."""
    bad = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    # number of defects allowed to pass; the small tolerance stops 0.1 * 10 = 0.999... rounding down
    k = int(np.floor((1 - target) * len(bad) + 1e-9))
    return float(bad[k])


def cost_optimal_threshold(scores: np.ndarray, labels: np.ndarray, defect_rate: float,
                           cost_inspection: float, cost_escape: float) -> float:
    """Threshold with the lowest expected cost per unit on the given images."""
    candidates = np.unique(np.concatenate([np.asarray(scores), [np.inf]]))
    best, best_cost = np.inf, np.inf
    for thr in candidates:
        r = rates(scores, labels, thr)
        c = cost_per_unit(r["catch_rate"], r["false_alarm_rate"], defect_rate, cost_inspection, cost_escape)
        if c < best_cost - 1e-12:
            best, best_cost = float(thr), c
    return best
