import numpy as np
import pytest
import torch
from PIL import Image

from conftest import FIXTURE
from defects import config, data


def test_list_items_reads_mvtec_layout():
    train = data.list_items("bottle", "train", FIXTURE)
    test = data.list_items("bottle", "test", FIXTURE)
    assert len(train) == 8 and all(i.label == 0 for i in train)
    bad = [i for i in test if i.label == 1]
    assert len(bad) == 2 and all(i.defect_type == "broken_large" and i.mask_path.exists() for i in bad)


def test_missing_category_says_how_to_fix():
    with pytest.raises(FileNotFoundError, match="01_download"):
        data.list_items("bottle", "train", FIXTURE / "nowhere")


def test_test_split_is_stratified_and_deterministic():
    items = [data.Item(config.ROOT / f"{d}_{k}.png", int(d != "good"), d, None)
             for d, n in (("good", 20), ("crack", 11), ("scratch", 6)) for k in range(n)]
    a = data.split_test(items, 0.5, seed=1)
    b = data.split_test(items, 0.5, seed=1)
    assert a == b
    for d, n in (("good", 20), ("crack", 11), ("scratch", 6)):
        cal = sum(1 for i in items if i.defect_type == d and a[i.path] == "calibration")
        assert cal == n // 2
    assert data.split_test(items, 0.5, seed=2) != a


def test_train_split_keeps_every_image_once():
    items = data.list_items("bottle", "train", FIXTURE)
    fit, hold = data.split_train(items, 0.25, seed=0)
    assert len(hold) == 2 and len(fit) == 6
    assert {i.path for i in fit} | {i.path for i in hold} == {i.path for i in items}


def test_preprocess_shape_and_normalisation():
    img = Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8))
    x = data.preprocess(img)
    assert x.shape == (3, 224, 224)
    expected = (128 / 255 - data.MEAN.flatten()) / data.STD.flatten()
    torch.testing.assert_close(x[:, 0, 0], expected, atol=1e-5, rtol=0)


def test_mask_is_aligned_with_the_image():
    bad = [i for i in data.list_items("bottle", "test", FIXTURE) if i.label == 1][0]
    m = data.load_mask(bad)
    assert m.shape == (224, 224) and m.dtype == bool and 0 < m.mean() < 0.5
    good = data.list_items("bottle", "test", FIXTURE)[-1]
    assert not data.load_mask(good).any()
