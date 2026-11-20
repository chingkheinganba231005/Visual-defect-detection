"""Stage 2: fit PatchCore and the two baselines per category, score every evaluation image.

For each category:
  - 80% of the good training images build the models;
  - the other 20% are kept back as extra good images for choosing thresholds;
  - the test images are split 50/50 by defect type into calibration and holdout.
Writes results/scores/<category>.csv, results/fit.json (timings, pixel AUROC),
models/<category>.npz (PatchCore memory banks for the service) and example
heatmaps for the figures.
"""

import argparse
import csv
import logging
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from _common import read_json, write_json
from defects import config, data, models

log = logging.getLogger("fit")
TRAIN_HOLDOUT_SHARE = 0.2
BATCH = 16


def extract(fx, items):
    patches, pooled, images = [], [], []
    for _, x in data.batches(items, BATCH):
        p = fx.patches(x)
        patches.append(p)
        pooled.append(fx.pooled(x))
        images.append(x)
    return torch.cat(patches), torch.cat(pooled).numpy(), torch.cat(images)


def run_category(fx, category: str) -> dict:
    train_all = data.list_items(category, "train")
    fit_items, cal_good = data.split_train(train_all, TRAIN_HOLDOUT_SHARE, config.SEED)
    test_items = data.list_items(category, "test")
    test_role = data.split_test(test_items, config.CALIBRATION_SHARE, config.SEED)

    t0 = time.perf_counter()
    patches, pooled, images = extract(fx, fit_items)
    t_extract = time.perf_counter() - t0
    t0 = time.perf_counter()
    patchcore = models.PatchCore().fit(patches)
    t_coreset = time.perf_counter() - t0
    gaussian = models.GaussianModel().fit(pooled)
    mean_img = models.MeanImageModel().fit(images)
    del patches, images
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    patchcore.save(config.MODELS_DIR / f"{category}.npz")

    eval_items = cal_good + test_items
    roles = ["calibration"] * len(cal_good) + [test_role[i.path] for i in test_items]
    rows, maps_pc, maps_mi, masks = [], [], [], []
    t0 = time.perf_counter()
    for chunk, x in data.batches(eval_items, BATCH):
        p = fx.patches(x)
        s_pc, m_pc = patchcore.score(p)
        s_ga = gaussian.score(fx.pooled(x).numpy())
        s_mi, m_mi = mean_img.score(x)
        for k, it in enumerate(chunk):
            rows.append([it.path.relative_to(config.DATA_DIR), it.label, it.defect_type,
                         float(s_pc[k]), float(s_ga[k]), float(s_mi[k])])
            if it.path in test_role:
                maps_pc.append(m_pc[k].astype(np.float16))
                maps_mi.append(m_mi[k].astype(np.float16))
                masks.append(data.load_mask(it))
    t_score = time.perf_counter() - t0
    for r, role in zip(rows, roles):
        r.insert(3, role)

    out = config.RESULTS_DIR / "scores" / f"{category}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "defect_type", "role", "patchcore", "gaussian", "mean_image"])
        w.writerows(rows)

    # pixel-level AUROC on the whole test set (localisation), as in the MVTec literature
    gt = np.stack(masks).ravel()
    pixel = {"patchcore": roc_auc_score(gt, np.stack(maps_pc).astype(np.float32).ravel()),
             "mean_image": roc_auc_score(gt, np.stack(maps_mi).astype(np.float32).ravel())}

    # keep a few defective holdout maps for the figures
    ex_dir = config.ROOT / "data" / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    picks = [k for k, it in enumerate(test_items) if it.label == 1 and test_role[it.path] == "holdout"][:3]
    np.savez_compressed(ex_dir / f"{category}.npz",
                        files=np.array([str(test_items[k].path) for k in picks]),
                        maps=np.stack([maps_pc[k] for k in picks]),
                        masks=np.stack([masks[k] for k in picks]))

    n_eval = len(eval_items)
    info = {
        "n_fit_images": len(fit_items), "n_calibration_good_from_train": len(cal_good),
        "n_test": len(test_items), "n_test_defective": sum(i.label for i in test_items),
        "memory_bank_patches": int(patchcore.memory.shape[0]),
        "fit_patches": len(fit_items) * patchcore.grid ** 2,
        "seconds_feature_extraction": t_extract, "seconds_coreset": t_coreset,
        "seconds_per_image_scoring_all_models": t_score / n_eval,
        "pixel_auroc": pixel,
    }
    log.info("%s: %s", category, info)
    return info


def main(categories):
    torch.set_num_threads(4)
    torch.manual_seed(config.SEED)
    fx = models.FeatureExtractor()
    path = config.RESULTS_DIR / "fit.json"
    out = read_json("fit.json")["categories"] if path.exists() else {}
    for c in categories:
        out[c] = run_category(fx, c)
        # written after every category so a partial run keeps what it finished
        write_json("fit.json", {"categories": {k: out[k] for k in config.CATEGORIES if k in out},
                                "coreset_ratio": config.CORESET_RATIO,
                                "train_holdout_share": TRAIN_HOLDOUT_SHARE, "torch_threads": 4})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("categories", nargs="*", default=config.CATEGORIES)
    main(ap.parse_args().categories)
