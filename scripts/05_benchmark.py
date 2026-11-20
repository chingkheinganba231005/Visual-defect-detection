"""Stage 5: latency of the HTTP service on this machine (CPU only).

Sends every held-out test image of one category through POST /inspect, one at
a time, and records the model time reported by the service and the full
request time. Writes results/latency.json.
"""

import argparse
import csv
import os
import platform
import time

import numpy as np
from fastapi.testclient import TestClient

from _common import write_json
from defects import config, service


def main(category: str, n_warmup: int = 3):
    with open(config.RESULTS_DIR / "scores" / f"{category}.csv") as f:
        files = [config.DATA_DIR / r["file"] for r in csv.DictReader(f) if r["role"] == "holdout"]
    model_ms, request_ms, decisions = [], [], []
    with TestClient(service.app) as client:
        for path in files[:n_warmup]:
            client.post(f"/inspect/{category}", files={"file": (path.name, path.read_bytes(), "image/png")})
        for path in files:
            body = path.read_bytes()
            t0 = time.perf_counter()
            r = client.post(f"/inspect/{category}?heatmap=true", files={"file": (path.name, body, "image/png")})
            request_ms.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            model_ms.append(r.json()["inference_ms"])
            decisions.append(r.json()["decision"])
    q = lambda v, p: float(np.percentile(v, p))
    write_json("latency.json", {
        "category": category, "n_requests": len(files), "torch_threads": int(os.environ.get("DEFECTS_THREADS", "4")),
        "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "model_ms_p50": q(model_ms, 50), "model_ms_p95": q(model_ms, 95),
        "request_ms_p50": q(request_ms, 50), "request_ms_p95": q(request_ms, 95),
        "images_per_second": 1000 / float(np.mean(request_ms)),
        "share_sent_to_inspector": decisions.count("send to inspector") / len(decisions),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    main(ap.parse_args().category)
