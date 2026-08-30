"""
Global descriptors for candidate retrieval (spec section 8).

Primary backend: a frozen, pretrained DINOv2 ViT.  Its CLS token is a strong
appearance descriptor for aerial scenes and needs no training.

Fallback backend: a hand-rolled classical descriptor (gradient-orientation
histograms over a spatial grid + coarse colour statistics).  It exists so the
prototype still retrieves sensible candidates on a machine without PyTorch.

Retrieval is *only* a shortlisting step.  Nothing here decides the final
position - that is the job of geometric verification.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.models.loader import load_dinov2, resolve_device, try_import_torch

log = get_logger(__name__)

DINO_INPUT = 224                     # multiple of the ViT patch size (14)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-8)


# --------------------------------------------------------------------------
# Classical fallback descriptor
# --------------------------------------------------------------------------
def classical_embedding(img: np.ndarray, grid: int = 4, bins: int = 9) -> np.ndarray:
    """
    Gradient-orientation histogram per grid cell + per-cell colour means.

    Rotation is *not* handled here on purpose: retrieval only has to surface
    the right neighbourhood, and the pipeline separately evaluates rotated
    query variants when the upright match is weak.
    """
    resized = cv2.resize(img, (DINO_INPUT, DINO_INPUT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0   # unsigned orientation

    cell = DINO_INPUT // grid
    feats: List[float] = []
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    for gy_i in range(grid):
        for gx_i in range(grid):
            ys = slice(gy_i * cell, (gy_i + 1) * cell)
            xs = slice(gx_i * cell, (gx_i + 1) * cell)
            hist, _ = np.histogram(ang[ys, xs], bins=bins, range=(0, 180),
                                   weights=mag[ys, xs])
            hist = hist / max(float(hist.sum()), 1e-6)
            feats.extend(hist.tolist())
            patch = hsv[ys, xs]
            feats.extend([
                float(patch[..., 0].mean()) / 180.0,
                float(patch[..., 1].mean()) / 255.0,
                float(patch[..., 2].mean()) / 255.0,
                float(patch[..., 2].std()) / 255.0,
            ])
    return l2_normalize(np.asarray(feats, dtype=np.float32))[0]


# --------------------------------------------------------------------------
# DINOv2 descriptor
# --------------------------------------------------------------------------
def _to_tensor_batch(images: Sequence[np.ndarray], torch):
    batch = []
    for img in images:
        resized = cv2.resize(img, (DINO_INPUT, DINO_INPUT), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(np.transpose(rgb, (2, 0, 1)))
    return torch.from_numpy(np.stack(batch)).float()


class EmbeddingEngine:
    """Uniform interface over the DINOv2 and classical descriptor backends."""

    def __init__(self, prefer_dino: bool = True):
        self.model = load_dinov2() if prefer_dino else None
        self.torch = try_import_torch() if self.model is not None else None
        self.device = resolve_device() if self.model is not None else "cpu"

    @property
    def backend(self) -> str:
        return "dinov2" if self.model is not None else "classical-embedding"

    def embed_batch(self, images: Sequence[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not len(images):
            return np.zeros((0, 1), dtype=np.float32)
        if self.model is None:
            return l2_normalize(np.stack([classical_embedding(i) for i in images]))

        torch = self.torch
        outputs: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                tensor = _to_tensor_batch(chunk, torch).to(self.device)
                feats = self.model(tensor)                     # CLS token, (B, D)
                outputs.append(feats.detach().float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs, axis=0))

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_batch([image])[0]


_ENGINE: Optional[EmbeddingEngine] = None


def get_engine() -> EmbeddingEngine:
    """Process-wide singleton so the backbone is loaded at most once."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EmbeddingEngine()
        log.info("Embedding backend: %s", _ENGINE.backend)
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    _ENGINE = None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def cosine_similarity(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Both sides are L2-normalised, so cosine similarity is a dot product."""
    query = l2_normalize(query)[0]
    if bank.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return bank.astype(np.float32) @ query.astype(np.float32)


def top_k(query: np.ndarray, bank: np.ndarray, k: Optional[int] = None) -> List[dict]:
    """Return the k highest-similarity bank rows as ``{index, similarity}``."""
    k = k or settings.top_k_candidates
    sims = cosine_similarity(query, bank)
    if sims.size == 0:
        return []
    k = int(min(k, sims.size))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [{"index": int(i), "similarity": float(sims[int(i)])} for i in idx]
