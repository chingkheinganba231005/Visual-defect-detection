"""Stage 3: detection quality on the test images, with bootstrap intervals.

Image-level AUROC is computed on each category's full test set, the number
reported in the MVTec AD literature (no threshold is involved, so using the
whole test set is fair). Intervals come from a stratified bootstrap, paired
across methods; the average over categories is bootstrapped by averaging the
per-category replicates.
Writes results/detection.json and figures/auroc_by_category.png, figures/heatmaps.png.
"""

import csv
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from _common import read_json, write_json  # noqa: E402
from defects import config, data, evaluation, plots  # noqa: E402

log = logging.getLogger("evaluate")
METHODS = ["patchcore", "gaussian", "mean_image"]
LABELS = {"patchcore": "PatchCore", "gaussian": "Gaussian on CNN features", "mean_image": "Mean-image difference"}


def load_scores(category: str) -> list[dict]:
    with open(config.RESULTS_DIR / "scores" / f"{category}.csv") as f:
        return list(csv.DictReader(f))


def main():
    fit = read_json("fit.json")["categories"]
    rng = np.random.default_rng(config.SEED)
    per_cat, boot = {}, {m: [] for m in METHODS}
    for c in config.CATEGORIES:
        rows = [r for r in load_scores(c) if "/test/" in r["file"]]
        y = np.array([int(r["label"]) for r in rows])
        s = {m: np.array([float(r[m]) for r in rows]) for m in METHODS}
        b = evaluation.bootstrap_auroc(y, s, config.N_BOOTSTRAP, rng)
        per_cat[c] = {
            "n_test": len(y), "n_defective": int(y.sum()),
            **{f"auroc_{m}": evaluation.auroc(y, s[m]) for m in METHODS},
            **{f"auroc_{m}_ci": evaluation.interval(b[m]) for m in METHODS},
            "pixel_auroc_patchcore": fit[c]["pixel_auroc"]["patchcore"],
            "pixel_auroc_mean_image": fit[c]["pixel_auroc"]["mean_image"],
            "texture": c in config.TEXTURES,
        }
        for m in METHODS:
            boot[m].append(b[m])
    mean_boot = {m: np.mean(np.stack(boot[m]), axis=0) for m in METHODS}
    summary = {}
    for m in METHODS:
        summary[m] = {"mean_auroc": float(np.mean([per_cat[c][f"auroc_{m}"] for c in config.CATEGORIES])),
                      "mean_auroc_ci": evaluation.interval(mean_boot[m]),
                      "categories_at_least_0_99": int(sum(per_cat[c][f"auroc_{m}"] >= 0.99 for c in config.CATEGORIES))}
    for m in ("gaussian", "mean_image"):
        d = mean_boot["patchcore"] - mean_boot[m]
        summary[f"patchcore_minus_{m}"] = {"value": summary["patchcore"]["mean_auroc"] - summary[m]["mean_auroc"],
                                           "ci": evaluation.interval(d)}
    summary["pixel_auroc_patchcore_mean"] = float(np.mean([per_cat[c]["pixel_auroc_patchcore"] for c in config.CATEGORIES]))
    summary["pixel_auroc_mean_image_mean"] = float(np.mean([per_cat[c]["pixel_auroc_mean_image"] for c in config.CATEGORIES]))
    worst = min(config.CATEGORIES, key=lambda c: per_cat[c]["auroc_patchcore"])
    summary["worst_category_patchcore"] = {"category": worst, "auroc": per_cat[worst]["auroc_patchcore"],
                                           "ci": per_cat[worst]["auroc_patchcore_ci"]}
    write_json("detection.json", {"summary": summary, "categories": per_cat, "n_bootstrap": config.N_BOOTSTRAP})
    figures(per_cat)


def figures(per_cat):
    plots.style()
    cats = sorted(config.CATEGORIES, key=lambda c: per_cat[c]["auroc_patchcore"])
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    y = np.arange(len(cats))
    for k, (m, color) in enumerate(zip(METHODS, (plots.SERIES[0], plots.SERIES[1], plots.MUTED))):
        v = np.array([per_cat[c][f"auroc_{m}"] for c in cats])
        ci = np.array([per_cat[c][f"auroc_{m}_ci"] for c in cats])
        off = (k - 1) * 0.22
        ax.errorbar(v, y + off, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o", ms=5, color=color,
                    ecolor=color, elinewidth=1, capsize=0, label=LABELS[m])
    ax.set_yticks(y, [c.replace("_", " ") for c in cats])
    ax.set_xlabel("Image-level AUROC on the test set (95% bootstrap interval)")
    ax.set_xlim(0.3, 1.01)
    ax.set_title("Detecting defective parts, by product category")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, frameon=False)
    plots.save(fig, config.FIGURES_DIR / "auroc_by_category.png")

    show = ["bottle", "screw", "hazelnut", "carpet", "transistor", "pill"]
    fig, axes = plt.subplots(2, len(show), figsize=(2.0 * len(show), 4.3))
    for j, c in enumerate(show):
        z = np.load(config.ROOT / "data" / "examples" / f"{c}.npz")
        img = data.load_image(z["files"][0]) * data.STD + data.MEAN
        img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
        m = z["maps"][0].astype(np.float32)
        axes[0, j].imshow(img)
        axes[0, j].contour(z["masks"][0], levels=[0.5], colors=[plots.SERIES[7]], linewidths=1)
        axes[0, j].set_title(c.replace("_", " "), fontsize=9)
        axes[1, j].imshow(img)
        axes[1, j].imshow(m, cmap="magma", alpha=0.55)
        for a in axes[:, j]:
            a.set_axis_off()
    axes[0, 0].text(-12, 112, "image +\ntrue defect", rotation=90, va="center", ha="right", fontsize=8.5)
    axes[1, 0].text(-12, 112, "PatchCore\nheatmap", rotation=90, va="center", ha="right", fontsize=8.5)
    fig.suptitle("Held-out defective parts: where the model looks", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "heatmaps.png")


if __name__ == "__main__":
    main()
