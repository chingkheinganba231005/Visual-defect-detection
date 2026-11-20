"""Fit a small model straight from a folder of good images (used by CI to smoke-test Docker).

Usage: python scripts/make_demo_model.py --data tests/data/mvtec --category bottle --out ci_models

The threshold is simply the largest score among the training images, so this
model is only a plumbing check. Real models come from scripts/02 and scripts/04.
"""

import argparse
import json
from pathlib import Path

import torch

import _common  # noqa: F401
from defects import data, models

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fx = models.FeatureExtractor()
    items = data.list_items(a.category, "train", a.data)
    x = torch.stack([data.load_image(i.path) for i in items])
    feats = fx.patches(x)
    pc = models.PatchCore(coreset_ratio=0.1).fit(feats)
    thr = float(pc.score(feats)[0].max())
    a.out.mkdir(parents=True, exist_ok=True)
    pc.save(a.out / f"{a.category}.npz")
    (a.out / "thresholds.json").write_text(json.dumps(
        {"categories": {a.category: {"triage": thr, "cost_based": thr}}}, indent=2))
    print(f"wrote {a.out}")
