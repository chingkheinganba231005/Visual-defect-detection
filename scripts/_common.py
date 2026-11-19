"""Helpers shared by the stage scripts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defects import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(name: str, payload) -> Path:
    path = config.RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2) + "\n")
    logging.info("wrote %s", path)
    return path


def read_json(name: str):
    return json.loads((config.RESULTS_DIR / name).read_text())
