"""
Paired dataset loader for aerial -> map translation.

The same class serves both the Sat2Maps pretraining data and a future InnovX
drone fine-tune set, as long as the on-disk layout matches:

    <root>/<split>/<aerial_dir>/NNNNNN.png
    <root>/<split>/map/NNNNNN.png

For Sat2Maps  ``aerial_dir = "satellite"``.
For the drone fine-tune  ``aerial_dir = "aerial"`` (see FUTURE section of the
package README).

Augmentations are applied identically to both images of a pair so the input and
target never stop describing the same coordinates. Only transforms that keep
that property are allowed: 90/180/270 rotation, flips, and photometric changes
on the *aerial* image only (brightness/contrast/noise/blur/JPEG) - the map
target is left photometrically untouched.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:  # torch is optional at import time for the rest of the package
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


@dataclass
class AugmentConfig:
    rot90: bool = True
    flip: bool = True
    small_rotation_deg: float = 5.0
    brightness: float = 0.15
    contrast: float = 0.15
    gaussian_noise_std: float = 6.0
    blur_prob: float = 0.2
    jpeg_prob: float = 0.2
    jpeg_quality_min: int = 55


def _photometric(aerial: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    out = aerial.astype(np.float32)
    if cfg.brightness:
        out += 255.0 * rng.uniform(-cfg.brightness, cfg.brightness)
    if cfg.contrast:
        factor = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        out = (out - 128.0) * factor + 128.0
    out = np.clip(out, 0, 255)
    if cfg.gaussian_noise_std:
        out += rng.normal(0, cfg.gaussian_noise_std, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < cfg.blur_prob:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < cfg.jpeg_prob:
        q = int(rng.integers(cfg.jpeg_quality_min, 96))
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def _geometric(aerial: np.ndarray, mp: np.ndarray, rng: np.random.Generator,
               cfg: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.rot90:
        k = int(rng.integers(0, 4))
        aerial, mp = np.rot90(aerial, k), np.rot90(mp, k)
    if cfg.flip and rng.random() < 0.5:
        aerial, mp = aerial[:, ::-1], mp[:, ::-1]
    if cfg.small_rotation_deg:
        h, w = aerial.shape[:2]
        ang = rng.uniform(-cfg.small_rotation_deg, cfg.small_rotation_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        aerial = cv2.warpAffine(aerial, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        mp = cv2.warpAffine(mp, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.ascontiguousarray(aerial), np.ascontiguousarray(mp)


class PairedTranslationDataset(Dataset):
    """Returns ``(aerial_tensor, map_tensor)`` in [0, 1], CHW float32."""

    def __init__(self, root: Path, split: str = "train", aerial_dir: str = "satellite",
                 size: int = 256, augment: bool = True, seed: int = 0):
        self.aerial_root = Path(root) / split / aerial_dir
        self.map_root = Path(root) / split / "map"
        self.size = size
        self.augment = augment and split == "train"
        self.cfg = AugmentConfig()
        self._seed = seed
        self.names: List[str] = sorted(
            p.name for p in self.aerial_root.glob("*.png")
            if (self.map_root / p.name).exists())
        if not self.names:
            raise FileNotFoundError(
                f"No paired images under {self.aerial_root} / {self.map_root}")

    def __len__(self) -> int:
        return len(self.names)

    def _read(self, path: Path) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Corrupt image: {path}")
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int):
        name = self.names[idx]
        aerial = self._read(self.aerial_root / name)
        mp = self._read(self.map_root / name)

        if self.augment:
            rng = np.random.default_rng(self._seed + idx * 100003)
            aerial, mp = _geometric(aerial, mp, rng, self.cfg)
            aerial = _photometric(aerial, rng, self.cfg)

        aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mp = cv2.cvtColor(mp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        a = torch.from_numpy(np.transpose(aerial, (2, 0, 1)).copy())
        m = torch.from_numpy(np.transpose(mp, (2, 0, 1)).copy())
        return a, m
