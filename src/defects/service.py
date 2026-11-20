"""HTTP service: send a product photo, get back a score, a decision and a heatmap.

Run locally:   uvicorn defects.service:app --app-dir src
Endpoints:
  GET  /health                     status and the categories that have a model
  POST /inspect/{category}         multipart image upload
       ?policy=triage|cost_based   which threshold to apply (default triage)
       &heatmap=true               also return the anomaly heatmap as a base64 PNG
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from matplotlib import colormaps
from PIL import Image, UnidentifiedImageError

from . import config, data, models

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.environ.get("DEFECTS_THREADS", "4")))
    STATE["extractor"] = models.FeatureExtractor()
    yield
    STATE.clear()


app = FastAPI(title="Visual defect inspection", version="1.0.0", lifespan=lifespan)


def models_dir() -> Path:
    return config.MODELS_DIR


@lru_cache(maxsize=32)
def load_model(path: str) -> models.PatchCore:
    return models.PatchCore.load(path)


def thresholds() -> dict:
    p = models_dir() / "thresholds.json"
    return json.loads(p.read_text())["categories"] if p.exists() else {}


def available() -> list[str]:
    th = thresholds()
    return sorted(p.stem for p in models_dir().glob("*.npz") if p.stem in th)


def heatmap_png(image: torch.Tensor, amap: np.ndarray, threshold: float) -> str:
    """Overlay of the anomaly map on the (cropped) input; full colour at 1.5 x the threshold."""
    rgb = np.clip((image * data.STD + data.MEAN).permute(1, 2, 0).numpy(), 0, 1)
    heat = colormaps["magma"](np.clip(amap / (1.5 * threshold), 0, 1))[..., :3]
    alpha = np.clip(amap / threshold, 0, 1)[..., None] * 0.6
    out = (1 - alpha) * rgb + alpha * heat
    buf = io.BytesIO()
    Image.fromarray((out * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "categories": available()}


@app.post("/inspect/{category}")
async def inspect(category: str, file: UploadFile = File(...),
                  policy: str = Query("triage", pattern="^(triage|cost_based)$"),
                  heatmap: bool = Query(False)) -> dict:
    th = thresholds()
    path = models_dir() / f"{category}.npz"
    if category not in th or not path.exists():
        raise HTTPException(404, f"no model for category '{category}'; available: {available()}")
    raw = await file.read()
    try:
        with Image.open(io.BytesIO(raw)) as im:
            x = data.preprocess(im)[None]
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "the upload is not a readable image")
    t0 = time.perf_counter()
    model = load_model(str(path))
    scores, maps = model.score(STATE["extractor"].patches(x))
    elapsed = (time.perf_counter() - t0) * 1000
    threshold = float(th[category][policy])
    score = float(scores[0])
    out = {
        "category": category,
        "score": score,
        "threshold": threshold,
        "policy": policy,
        "decision": "send to inspector" if score >= threshold else "pass",
        "inference_ms": round(elapsed, 1),
    }
    if heatmap:
        out["heatmap_png_base64"] = heatmap_png(x[0], maps[0], threshold)
    return out
