"""Stage 4: turn anomaly scores into inspection decisions.

Every threshold is chosen on the calibration images (held-back good training
images plus half of the test set) and judged on the held-out half of the test
set, which no earlier step used for any choice.

Two policies per product category:
  triage     - flag enough units to catch CATCH_TARGET of the calibration
               defects; flagged units go to a human inspector, the rest pass.
  cost-based - the threshold with the lowest expected cost per unit, given the
               cost of an inspection, the cost of a defect escaping and the
               defect rate.
Rates are measured on good and defective images separately and weighted by the
assumed production defect rate, because the test set has far more defects
than a real line. Categories are averaged with equal weight (equal volumes).
Writes results/decisions.json, models/thresholds.json and figures.
"""

import csv
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from _common import write_json  # noqa: E402
from defects import config, evaluation as ev, plots  # noqa: E402

log = logging.getLogger("decisions")
METHODS = ["patchcore", "gaussian"]
DEFECT_RATES = [0.005, 0.01, 0.02, 0.05, 0.10]
COST_RATIOS = [10, 25, 50, 100, 250, 1000]   # cost of an escape / cost of an inspection


def load(category):
    with open(config.RESULTS_DIR / "scores" / f"{category}.csv") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for role in ("calibration", "holdout"):
        r = [x for x in rows if x["role"] == role]
        out[role] = {"y": np.array([int(x["label"]) for x in r]),
                     **{m: np.array([float(x[m]) for x in r]) for m in METHODS}}
    return out


def holdout_outcome(s, y, thr, rate, c_insp, c_esc):
    r = ev.rates(s, y, thr)
    flagged = ev.flagged_share(r["catch_rate"], r["false_alarm_rate"], rate)
    return {**r, "flagged_share": flagged, "workload_saved": 1 - flagged,
            "cost_per_unit": ev.cost_per_unit(r["catch_rate"], r["false_alarm_rate"], rate, c_insp, c_esc)}


def bootstrap(s, y, thr, rate, c_insp, c_esc, rng, n):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keys = ["catch_rate", "false_alarm_rate", "workload_saved", "cost_per_unit"]
    out = {k: np.empty(n) for k in keys}
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        o = holdout_outcome(s[idx], y[idx], thr, rate, c_insp, c_esc)
        for k in keys:
            out[k][b] = o[k]
    return out


def main():
    rate, c_insp, c_esc = config.DEFECT_RATE, config.COST_INSPECTION, config.COST_ESCAPE
    rng = np.random.default_rng(config.SEED)
    per_cat, boots, thresholds = {}, {m: {p: [] for p in ("triage", "cost")} for m in METHODS}, {}
    data = {c: load(c) for c in config.CATEGORIES}
    for c in config.CATEGORIES:
        cal, hold = data[c]["calibration"], data[c]["holdout"]
        per_cat[c] = {"n_calibration": len(cal["y"]), "n_calibration_defective": int(cal["y"].sum()),
                      "n_holdout": len(hold["y"]), "n_holdout_defective": int(hold["y"].sum())}
        for m in METHODS:
            t_tri = ev.threshold_for_catch_rate(cal[m], cal["y"], config.CATCH_TARGET)
            t_cost = ev.cost_optimal_threshold(cal[m], cal["y"], rate, c_insp, c_esc)
            for name, thr in (("triage", t_tri), ("cost", t_cost)):
                o = holdout_outcome(hold[m], hold["y"], thr, rate, c_insp, c_esc)
                b = bootstrap(hold[m], hold["y"], thr, rate, c_insp, c_esc, rng, config.N_BOOTSTRAP)
                per_cat[c][f"{m}_{name}"] = {**o, "threshold": thr,
                                             **{f"{k}_ci": ev.interval(v) for k, v in b.items()}}
                boots[m][name].append(b)
            if m == "patchcore":
                thresholds[c] = {"triage": t_tri, "cost_based": t_cost}

    blanket = ev.blanket_costs(rate, c_insp, c_esc)
    summary = {}
    for m in METHODS:
        for name in ("triage", "cost"):
            key = f"{m}_{name}"
            agg = {k: float(np.mean([per_cat[c][key][k] for c in config.CATEGORIES]))
                   for k in ("catch_rate", "false_alarm_rate", "workload_saved", "cost_per_unit", "flagged_share")}
            for k in ("catch_rate", "false_alarm_rate", "workload_saved", "cost_per_unit"):
                agg[f"{k}_ci"] = ev.interval(np.mean(np.stack([b[k] for b in boots[m][name]]), axis=0))
            agg["categories_below_catch_target"] = int(sum(per_cat[c][key]["catch_rate"] < config.CATCH_TARGET
                                                           for c in config.CATEGORIES))
            summary[key] = agg
    best_blanket = min(blanket.values())
    summary["savings_vs_best_blanket"] = 1 - summary["patchcore_cost"]["cost_per_unit"] / best_blanket

    # sensitivity: re-choose the cost-based threshold on calibration for every setting
    grid = []
    for r in DEFECT_RATES:
        for ratio in COST_RATIOS:
            esc = ratio * c_insp
            costs = []
            for c in config.CATEGORIES:
                cal, hold = data[c]["calibration"], data[c]["holdout"]
                thr = ev.cost_optimal_threshold(cal["patchcore"], cal["y"], r, c_insp, esc)
                costs.append(holdout_outcome(hold["patchcore"], hold["y"], thr, r, c_insp, esc)["cost_per_unit"])
            bl = ev.blanket_costs(r, c_insp, esc)
            best = min(bl.values())
            grid.append({"defect_rate": r, "cost_ratio": ratio, "model_cost": float(np.mean(costs)),
                         "inspect_all": bl["inspect_all"], "inspect_none": bl["inspect_none"],
                         "saving_vs_best_blanket": 1 - float(np.mean(costs)) / best})

    write_json("decisions.json", {
        "assumptions": {"defect_rate": rate, "cost_inspection": c_insp, "cost_escape": c_esc,
                        "catch_target": config.CATCH_TARGET},
        "blanket_policies": blanket, "summary": summary, "categories": per_cat,
        "sensitivity": grid, "n_bootstrap": config.N_BOOTSTRAP,
    })
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "thresholds.json").write_text(json.dumps(
        {"catch_target": config.CATCH_TARGET, "defect_rate": rate, "cost_inspection": c_insp,
         "cost_escape": c_esc, "categories": thresholds}, indent=2) + "\n")
    figures(per_cat, grid)


def figures(per_cat, grid):
    plots.style()
    cats = sorted(config.CATEGORIES, key=lambda c: per_cat[c]["patchcore_triage"]["workload_saved"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    y = np.arange(len(cats))
    for ax, key, title, target in (
        (axes[0], "catch_rate", "Defects caught on held-out parts", config.CATCH_TARGET),
        (axes[1], "workload_saved", "Manual inspections saved", None),
    ):
        v = np.array([per_cat[c]["patchcore_triage"][key] for c in cats])
        ci = np.array([per_cat[c]["patchcore_triage"][f"{key}_ci"] for c in cats])
        ax.barh(y, v, color=plots.SERIES[0], height=0.62)
        ax.errorbar(v, y, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", ecolor=plots.INK_2, lw=1, capsize=2)
        if target:
            ax.axvline(target, color=plots.SERIES[7], lw=1, ls="--")
            ax.text(target, len(cats) - 0.3, " target", color=plots.SERIES[7], fontsize=8, va="bottom", ha="right")
        ax.set_xlim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_yticks(y, [c.replace("_", " ") for c in cats])
    fig.suptitle(f"Triage with PatchCore: threshold set on calibration parts to catch {config.CATCH_TARGET:.0%} of defects",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "triage_by_category.png")

    mat = np.array([[next(g["saving_vs_best_blanket"] for g in grid if g["defect_rate"] == r and g["cost_ratio"] == k)
                     for k in COST_RATIOS] for r in DEFECT_RATES])
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    # diverging: blue where the model saves money, red where a blanket policy is cheaper
    cmap = LinearSegmentedColormap.from_list("div", plots.DIVERGING[::-1])
    ax.imshow(np.clip(mat, -1, 1), cmap=cmap, vmin=-1, vmax=1, origin="lower", aspect="auto")
    for i in range(len(DEFECT_RATES)):
        for j in range(len(COST_RATIOS)):
            ax.text(j, i, f"{mat[i, j]:+.0%}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(mat[i, j]) > 0.6 else plots.INK)
    ax.set_xticks(range(len(COST_RATIOS)), [f"{k}x" for k in COST_RATIOS])
    ax.set_yticks(range(len(DEFECT_RATES)), [f"{r:.1%}" for r in DEFECT_RATES])
    ax.set_xlabel("Cost of a defect reaching the customer / cost of one inspection")
    ax.set_ylabel("Defect rate in production")
    ax.grid(False)
    ax.set_title("Cost-based threshold: saving vs the cheaper of checking every unit or none\n"
                 "(negative: checking every unit is cheaper)", fontsize=10)
    plots.save(fig, config.FIGURES_DIR / "cost_sensitivity.png")


if __name__ == "__main__":
    main()
