import numpy as np
import pytest
import torch

from conftest import FIXTURE
from defects import data, models


@pytest.fixture(scope="module")
def extractor():
    torch.manual_seed(0)
    return models.FeatureExtractor()


def test_feature_shapes(extractor):
    x = torch.randn(2, 3, 224, 224)
    assert extractor.patches(x).shape == (2, 28 * 28, 1536)
    assert extractor.pooled(x).shape == (2, 1536)


def test_coreset_covers_every_cluster():
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    pts = torch.cat([c + 0.1 * torch.randn(200, 2, generator=g) for c in centres])
    idx = models.greedy_coreset(pts, 4, proj_dim=2, seed=0)
    picked = {int(i) // 200 for i in idx}
    assert picked == {0, 1, 2, 3}


def test_patchcore_scores_training_patches_as_normal():
    g = torch.Generator().manual_seed(1)
    train = torch.randn(3, 16, 8, generator=g)
    pc = models.PatchCore(coreset_ratio=1.0, seed=0).fit(train)
    s, maps = pc.score(train[:1])
    assert s[0] == pytest.approx(0.0, abs=1e-2)   # cdist's fast formula rounds at ~1e-3
    odd = train[:1].clone()
    odd[0, 5] += 10.0                      # one patch far from everything seen
    s2, m2 = pc.score(odd)
    assert s2[0] > 9.0
    assert maps.shape == (1, 224, 224)
    # the hot spot is in the right place: patch 5 is row 1, column 1 of a 4 x 4 grid
    r, c = np.unravel_index(np.argmax(m2[0]), m2[0].shape)
    assert 56 <= r < 112 and 56 <= c < 112


def test_patchcore_save_and_load(tmp_path):
    pc = models.PatchCore(coreset_ratio=0.5, seed=0).fit(torch.randn(2, 16, 8))
    pc.save(tmp_path / "m.npz")
    back = models.PatchCore.load(tmp_path / "m.npz")
    assert back.grid == 4 and back.memory.shape == pc.memory.shape
    torch.testing.assert_close(back.memory, pc.memory, atol=1e-2, rtol=1e-2)  # stored as float16


def test_gaussian_model_ranks_an_outlier_higher():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    m = models.GaussianModel().fit(x)
    assert m.score(np.full((1, 5), 6.0))[0] > m.score(np.zeros((1, 5)))[0]


def test_end_to_end_on_real_images(extractor):
    train = data.list_items("bottle", "train", FIXTURE)
    x = torch.stack([data.load_image(i.path) for i in train])
    pc = models.PatchCore(coreset_ratio=0.1, seed=0).fit(extractor.patches(x))
    test = data.list_items("bottle", "test", FIXTURE)
    xt = torch.stack([data.load_image(i.path) for i in test])
    s, maps = pc.score(extractor.patches(xt))
    assert s.shape == (4,) and np.isfinite(s).all() and maps.shape == (4, 224, 224)
    mi, mmaps = models.MeanImageModel().fit(x).score(xt)
    assert mi.shape == (4,) and mmaps.shape == (4, 224, 224)
