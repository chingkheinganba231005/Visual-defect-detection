"""Fill the README from results/*.json, so every number in it comes from a script.

  python scripts/render_readme.py          # write README.md from docs/README.template.md
  python scripts/render_readme.py --check  # fail if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"



def load(name):
    return json.loads((RESULTS / name).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def auc(x):
    return f"{x:.3f}"


def ci(v, f=auc):
    return f"{f(v[0])}–{f(v[1])}"


def money(x):
    return f"${x:,.2f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def context() -> dict:
    det, dec, fit, lat = load("detection.json"), load("decisions.json"), load("fit.json"), load("latency.json")
    s, cats = det["summary"], det["categories"]
    c = {
        "n_categories": str(len(cats)),
        "n_train_images": f"{sum(v['n_fit_images'] + v['n_calibration_good_from_train'] for v in fit['categories'].values()):,}",
        "n_test_images": f"{sum(v['n_test'] for v in cats.values()):,}",
        "n_test_defective": f"{sum(v['n_defective'] for v in cats.values()):,}",
        "n_boot": f"{det['n_bootstrap']:,}",
        "coreset_pct": pct(fit["coreset_ratio"], 0),
        "fit_minutes": f"{sum(v['seconds_feature_extraction'] + v['seconds_coreset'] for v in fit['categories'].values()) / 60:.0f}",
    }
    for m in ("patchcore", "gaussian", "mean_image"):
        c[f"auroc_{m}"] = auc(s[m]["mean_auroc"])
        c[f"auroc_{m}_ci"] = ci(s[m]["mean_auroc_ci"])
        c[f"n99_{m}"] = str(s[m]["categories_at_least_0_99"])
    for m in ("gaussian", "mean_image"):
        d = s[f"patchcore_minus_{m}"]
        c[f"gain_vs_{m}"] = f"{d['value']:.3f}"
        c[f"gain_vs_{m}_ci"] = ci(d["ci"], lambda x: f"{x:.3f}")
    c["pixel_patchcore"] = auc(s["pixel_auroc_patchcore_mean"])
    c["pixel_mean_image"] = auc(s["pixel_auroc_mean_image_mean"])
    w = s["worst_category_patchcore"]
    c["worst_cat"], c["worst_auroc"], c["worst_ci"] = w["category"].replace("_", " "), auc(w["auroc"]), ci(w["ci"])

    order = sorted(cats, key=lambda k: -cats[k]["auroc_patchcore"])
    c["auroc_table"] = table(
        ["Category", "Test images (defective)", "PatchCore", "Gaussian on CNN features", "Mean-image difference",
         "PatchCore pixel AUROC"],
        [[k.replace("_", " "), f"{cats[k]['n_test']} ({cats[k]['n_defective']})",
          f"{auc(cats[k]['auroc_patchcore'])} ({ci(cats[k]['auroc_patchcore_ci'])})",
          auc(cats[k]["auroc_gaussian"]), auc(cats[k]["auroc_mean_image"]),
          auc(cats[k]["pixel_auroc_patchcore"])] for k in order])

    a = dec["assumptions"]
    c["defect_rate"] = pct(a["defect_rate"], 0)
    c["cost_inspection"] = money(a["cost_inspection"])
    c["cost_escape"] = f"${a['cost_escape']:,.0f}"
    c["catch_target"] = pct(a["catch_target"], 0)
    t = dec["summary"]["patchcore_triage"]
    c["tri_catch"], c["tri_catch_ci"] = pct(t["catch_rate"]), ci(t["catch_rate_ci"], pct)
    c["tri_false_alarm"], c["tri_false_alarm_ci"] = pct(t["false_alarm_rate"]), ci(t["false_alarm_rate_ci"], pct)
    c["tri_saved"], c["tri_saved_ci"] = pct(t["workload_saved"], 0), ci(t["workload_saved_ci"], lambda x: pct(x, 0))
    c["tri_flagged"] = pct(t["flagged_share"])
    c["tri_below"] = str(t["categories_below_catch_target"])
    g = dec["summary"]["gaussian_triage"]
    c["tri_g_catch"], c["tri_g_saved"] = pct(g["catch_rate"]), pct(g["workload_saved"], 0)
    cb = dec["summary"]["patchcore_cost"]
    bl = dec["blanket_policies"]
    c["cost_model"], c["cost_model_ci"] = money(cb["cost_per_unit"]), ci(cb["cost_per_unit_ci"], money)
    c["cost_all"], c["cost_none"] = money(bl["inspect_all"]), money(bl["inspect_none"])
    c["cost_catch"] = pct(cb["catch_rate"])
    c["cost_saving"] = pct(dec["summary"]["savings_vs_best_blanket"], 0)
    c["cost_per_10k_saved"] = f"${(min(bl.values()) - cb['cost_per_unit']) * 10_000:,.0f}"
    grid = dec["sensitivity"]
    wins = [x for x in grid if x["saving_vs_best_blanket"] > 0]
    c["sens_wins"], c["sens_total"] = str(len(wins)), str(len(grid))
    c["sens_best"] = pct(max(x["saving_vs_best_blanket"] for x in grid), 0)
    losses = {}
    for x in grid:
        if x["saving_vs_best_blanket"] <= 0:
            losses.setdefault(x["defect_rate"], []).append(x["cost_ratio"])
    c["sens_losses"] = "; ".join(f"{pct(r, 0 if r >= 0.01 else 1)} defective with an escape costing "
                                 f"{min(v):,}x a check or more" for r, v in sorted(losses.items())) or "none"
    cal_def = [v["n_calibration_defective"] for v in dec["categories"].values()]
    c["cal_def_min"], c["cal_def_max"] = str(min(cal_def)), str(max(cal_def))
    worst_tri = min(dec["categories"], key=lambda k: dec["categories"][k]["patchcore_triage"]["catch_rate"])
    c["tri_worst_cat"] = worst_tri.replace("_", " ")
    c["tri_worst_catch"] = pct(dec["categories"][worst_tri]["patchcore_triage"]["catch_rate"], 0)
    c["tri_worst_n"] = str(dec["categories"][worst_tri]["n_holdout_defective"])

    pc = dec["categories"]
    order = sorted(pc, key=lambda k: pc[k]["patchcore_triage"]["catch_rate"])
    c["triage_table"] = table(
        ["Category", "Held-out images (defective)", "Defects caught", "Good parts flagged", "Inspections saved"],
        [[k.replace("_", " "), f"{pc[k]['n_holdout']} ({pc[k]['n_holdout_defective']})",
          pct(pc[k]["patchcore_triage"]["catch_rate"]), pct(pc[k]["patchcore_triage"]["false_alarm_rate"]),
          pct(pc[k]["patchcore_triage"]["workload_saved"], 0)] for k in order])

    c["lat_p50"], c["lat_p95"] = f"{lat['model_ms_p50']:.0f} ms", f"{lat['model_ms_p95']:.0f} ms"
    c["req_p50"], c["req_p95"] = f"{lat['request_ms_p50']:.0f} ms", f"{lat['request_ms_p95']:.0f} ms"
    c["lat_ips"] = f"{lat['images_per_second']:.1f}"
    c["lat_threads"], c["lat_n"], c["lat_cat"] = str(lat["torch_threads"]), str(lat["n_requests"]), lat["category"]
    return c


def render(template: str, ctx: dict) -> str:
    missing = set(re.findall(r"\{\{(\w+)\}\}", template)) - set(ctx)
    if missing:
        raise KeyError(f"no value for: {sorted(missing)}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(ctx[m.group(1)]), template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    text = render((ROOT / "docs" / "README.template.md").read_text(), context())
    readme = ROOT / "README.md"
    if ap.parse_args().check:
        if not readme.exists() or readme.read_text() != text:
            sys.exit("README.md is out of date: run python scripts/render_readme.py")
        print("README.md matches results")
        return
    readme.write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    main()
