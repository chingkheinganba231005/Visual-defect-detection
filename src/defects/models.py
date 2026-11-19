"""Feature extractor, PatchCore and the two baselines.

PatchCore (Roth et al., 2022): mid-level features of a CNN pretrained on
ImageNet describe small patches of the image. The patches of good training
images go into a memory bank, thinned out by greedy coreset selection. A test
patch's anomaly score is its distance to the nearest patch in the bank; the
image score is the largest patch score, and the patch scores, upsampled and
smoothed, form the heatmap.

Baselines:
  GaussianModel  - a single Gaussian on globally pooled CNN features, scored
                   by Mahalanobis distance (Rippel et al., 2020).
  MeanImageModel - the per-pixel difference from the average good image. No
                   learned features: it shows what the CNN features add.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf

from . import config


def load_backbone() -> torch.nn.Module:
    """WideResNet-50-2 with ImageNet weights.

    DEFECTS_BACKBONE_WEIGHTS=none gives random weights, used by the tests so
    they run offline and fast; the pipeline and the service always use the
    pretrained weights.
    """
    use_pretrained = os.environ.get("DEFECTS_BACKBONE_WEIGHTS", "imagenet") != "none"
    weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if use_pretrained else None
    net = torchvision.models.wide_resnet50_2(weights=weights)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class FeatureExtractor(torch.nn.Module):
    """Layer-2 and layer-3 features of WideResNet-50-2 (the layers PatchCore uses)."""

    def __init__(self, backbone: torch.nn.Module | None = None):
        super().__init__()
        self.net = backbone or load_backbone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        return l2, l3

    @torch.no_grad()
    def patches(self, x: torch.Tensor) -> torch.Tensor:
        """Locally aware patch features, shape (B, H*W, 1536) on the 28 x 28 grid."""
        l2, l3 = self(x)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3], dim=1)
        return f.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Global average of layer-2 and layer-3 features, shape (B, 1536)."""
        l2, l3 = self(x)
        return torch.cat([l2.mean(dim=(2, 3)), l3.mean(dim=(2, 3))], dim=1)


def greedy_coreset(features: torch.Tensor, n_select: int, proj_dim: int, seed: int) -> torch.Tensor:
    """Indices of a subset that covers the feature set (k-centre greedy).

    Each step adds the point farthest from everything chosen so far. Distances
    are computed after a random projection to ``proj_dim`` dimensions, which
    preserves them approximately (Johnson-Lindenstrauss) at a fraction of the cost.
    """
    n = features.shape[0]
    n_select = max(1, min(n_select, n))
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, generator=g) / proj_dim ** 0.5
    z = features @ proj
    chosen = torch.empty(n_select, dtype=torch.long)
    chosen[0] = int(torch.randint(n, (1,), generator=g))
    min_dist = torch.cdist(z, z[chosen[0]][None]).squeeze(1)
    for i in range(1, n_select):
        nxt = int(torch.argmax(min_dist))
        chosen[i] = nxt
        min_dist = torch.minimum(min_dist, torch.cdist(z, z[nxt][None]).squeeze(1))
    return chosen


def smooth_map(patch_scores: np.ndarray, size: int = config.IMAGE_SIZE, sigma: float = config.BLUR_SIGMA) -> np.ndarray:
    """Upsample a patch-score grid to image size and smooth it."""
    t = torch.from_numpy(np.asarray(patch_scores, dtype=np.float32))[None, None]
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(up, sigma=sigma)


@dataclass
class PatchCore:
    coreset_ratio: float = config.CORESET_RATIO
    seed: int = config.SEED
    memory: torch.Tensor | None = None
    grid: int = 28

    def fit(self, patch_features: torch.Tensor) -> "PatchCore":
        """``patch_features``: (N_images, H*W, D) from good training images."""
        n_img, n_patch, d = patch_features.shape
        self.grid = int(round(n_patch ** 0.5))
        flat = patch_features.reshape(-1, d)
        k = int(np.ceil(self.coreset_ratio * flat.shape[0]))
        idx = greedy_coreset(flat, k, config.PROJECTION_DIM, self.seed)
        self.memory = flat[idx].clone()
        return self

    def patch_scores(self, patch_features: torch.Tensor, chunk: int = 4096) -> np.ndarray:
        """Distance of every test patch to its nearest memory patch, shape (B, grid, grid)."""
        b, n_patch, d = patch_features.shape
        flat = patch_features.reshape(-1, d)
        out = torch.empty(flat.shape[0])
        for i in range(0, flat.shape[0], chunk):
            out[i:i + chunk] = torch.cdist(flat[i:i + chunk], self.memory).min(dim=1).values
        return out.reshape(b, self.grid, self.grid).numpy()

    def score(self, patch_features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Image scores (B,) and smoothed anomaly maps (B, 224, 224)."""
        ps = self.patch_scores(patch_features)
        maps = np.stack([smooth_map(p) for p in ps])
        return ps.reshape(len(ps), -1).max(axis=1), maps

    def save(self, path) -> None:
        np.savez_compressed(path, memory=self.memory.numpy().astype(np.float16), grid=self.grid,
                            coreset_ratio=self.coreset_ratio, seed=self.seed)

    @classmethod
    def load(cls, path) -> "PatchCore":
        z = np.load(path)
        return cls(coreset_ratio=float(z["coreset_ratio"]), seed=int(z["seed"]),
                   memory=torch.from_numpy(z["memory"].astype(np.float32)), grid=int(z["grid"]))


@dataclass
class GaussianModel:
    mean: np.ndarray | None = None
    precision: np.ndarray | None = None

    def fit(self, pooled: np.ndarray) -> "GaussianModel":
        lw = LedoitWolf().fit(pooled)
        self.mean, self.precision = lw.location_, lw.precision_
        return self

    def score(self, pooled: np.ndarray) -> np.ndarray:
        d = pooled - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d))


@dataclass
class MeanImageModel:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, images: torch.Tensor) -> "MeanImageModel":
        x = images.numpy()
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-3
        return self

    def score(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        z = np.abs((images.numpy() - self.mean) / self.std).mean(axis=1)
        maps = np.stack([gaussian_filter(m, sigma=config.BLUR_SIGMA) for m in z])
        return maps.reshape(len(maps), -1).max(axis=1), maps
