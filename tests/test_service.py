import base64
import io
import json

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FIXTURE
from defects import config, data, models, service


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    fx = models.FeatureExtractor()
    train = data.list_items("bottle", "train", FIXTURE)
    x = torch.stack([data.load_image(i.path) for i in train])
    models.PatchCore(coreset_ratio=0.1, seed=0).fit(fx.patches(x)).save(d / "bottle.npz")
    (d / "thresholds.json").write_text(json.dumps({"categories": {"bottle": {"triage": 0.0, "cost_based": 1e9}}}))
    old = config.MODELS_DIR
    config.MODELS_DIR = d
    with TestClient(service.app) as c:
        yield c
    config.MODELS_DIR = old


def _png(path):
    return (path.name, path.read_bytes(), "image/png")


def test_health_lists_categories(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "categories": ["bottle"]}


def test_inspect_returns_decision_and_heatmap(client):
    img = FIXTURE / "bottle" / "test" / "broken_large" / "000.png"
    r = client.post("/inspect/bottle?heatmap=true", files={"file": _png(img)})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "send to inspector"          # triage threshold 0 flags everything
    assert body["policy"] == "triage" and body["score"] > 0 and body["inference_ms"] > 0
    heat = Image.open(io.BytesIO(base64.b64decode(body["heatmap_png_base64"])))
    assert heat.size == (224, 224)
    r2 = client.post("/inspect/bottle?policy=cost_based", files={"file": _png(img)})
    assert r2.json()["decision"] == "pass" and "heatmap_png_base64" not in r2.json()


def test_unknown_category_and_bad_upload(client):
    img = FIXTURE / "bottle" / "test" / "good" / "000.png"
    assert client.post("/inspect/teapot", files={"file": _png(img)}).status_code == 404
    bad = client.post("/inspect/bottle", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
    assert client.post("/inspect/bottle?policy=yolo", files={"file": _png(img)}).status_code == 422
