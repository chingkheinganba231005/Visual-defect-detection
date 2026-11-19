"""MVTec AD on disk: download, listing, splits and image loading."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config

log = logging.getLogger(__name__)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(category: str, data_dir: Path = config.DATA_DIR) -> Path:
    """Download and unpack one category, checking the archive's SHA-256 when it is known."""
    target = data_dir / category
    if (target / "train").exists():
        return target
    url, expected = config.DOWNLOADS[category]
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{category}.tar.xz"
    log.info("downloading %s", category)
    with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f)
    digest = sha256_of(archive)
    if expected and digest != expected:
        archive.unlink()
        raise RuntimeError(f"{category}: checksum mismatch ({digest})")
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    archive.unlink()
    return target


@dataclass(frozen=True)
class Item:
    path: Path
    label: int            # 0 = good, 1 = defective
    defect_type: str      # "good" or the MVTec defect folder name
    mask_path: Path | None


def list_items(category: str, split: str, data_dir: Path = config.DATA_DIR) -> list[Item]:
    root = data_dir / category / split
    if not root.exists():
        raise FileNotFoundError(f"{root} not found: run scripts/01_download.py")
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(folder.glob("*.png")):
            if folder.name == "good":
                items.append(Item(img, 0, "good", None))
            else:
                mask = data_dir / category / "ground_truth" / folder.name / f"{img.stem}_mask.png"
                items.append(Item(img, 1, folder.name, mask if mask.exists() else None))
    return items


def split_train(items: list[Item], holdout_share: float, seed: int) -> tuple[list[Item], list[Item]]:
    """Training images used to build the model, and good images kept back for calibration."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_hold = int(round(holdout_share * len(items)))
    hold = sorted(idx[:n_hold])
    keep = sorted(idx[n_hold:])
    return [items[i] for i in keep], [items[i] for i in hold]


def split_test(items: list[Item], calibration_share: float, seed: int) -> dict[Path, str]:
    """Assign each test image to 'calibration' or 'holdout', stratified by defect type.

    Deterministic for a given seed and file list, so every script sees the same split.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for defect in sorted({i.defect_type for i in items}):
        group = [i for i in items if i.defect_type == defect]
        order = rng.permutation(len(group))
        n_cal = int(np.floor(calibration_share * len(group)))
        for rank, j in enumerate(order):
            out[group[j].path] = "calibration" if rank < n_cal else "holdout"
    return out


def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize the short side to 256, centre-crop 224, normalise with ImageNet statistics."""
    img = img.convert("RGB").resize((config.RESIZE, config.RESIZE), Image.BILINEAR)
    off = (config.RESIZE - config.IMAGE_SIZE) // 2
    img = img.crop((off, off, off + config.IMAGE_SIZE, off + config.IMAGE_SIZE))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - MEAN) / STD


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        return preprocess(im)


def load_mask(item: Item) -> np.ndarray:
    """Ground-truth defect mask aligned with the preprocessed image (zeros for good images)."""
    if item.mask_path is None:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=bool)
    with Image.open(item.mask_path) as m:
        m = m.convert("L").resize((config.RESIZE, config.RESIZE), Image.NEAREST)
        off = (config.RESIZE - config.IMAGE_SIZE) // 2
        m = m.crop((off, off, off + config.IMAGE_SIZE, off + config.IMAGE_SIZE))
        return np.asarray(m) > 127


def batches(items: list[Item], size: int):
    for i in range(0, len(items), size):
        chunk = items[i:i + size]
        yield chunk, torch.stack([load_image(it.path) for it in chunk])
