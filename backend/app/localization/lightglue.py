"""
LightGlue matching of SuperPoint features (spec section 12), plus the matcher
dispatcher used by the pipeline.

``match_features`` is the single entry point: it honours the configured
matcher, silently degrades to SIFT when the torch stack is missing, and also
retries with SIFT when LightGlue returns too few correspondences to be
geometrically useful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.localization import sift as sift_backend
from app.localization import superpoint as sp_backend
from app.localization.superpoint import FeatureSet
from app.models.loader import load_lightglue, try_import_torch

log = get_logger(__name__)

# Below this, RANSAC has nothing to work with, so it is worth a second opinion.
MIN_USEFUL_MATCHES = 8


@dataclass
class MatchResult:
    query_pts: np.ndarray        # (N, 2) in query image pixels
    target_pts: np.ndarray       # (N, 2) in target image pixels
    confidence: np.ndarray       # (N,) 0..1
    backend: str

    @property
    def count(self) -> int:
        return int(len(self.query_pts))

    @classmethod
    def empty(cls, backend: str = "none") -> "MatchResult":
        return cls(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                   np.zeros((0,), np.float32), backend)


def match_superpoint(query: FeatureSet, target: FeatureSet) -> Optional[MatchResult]:
    """LightGlue on two SuperPoint feature sets; None when unavailable."""
    matcher = load_lightglue()
    if matcher is None:
        return None
    if "lightglue_feats" not in query.raw or "lightglue_feats" not in target.raw:
        return None
    torch = try_import_torch()
    try:
        with torch.inference_mode():
            out = matcher({"image0": query.raw["lightglue_feats"],
                           "image1": target.raw["lightglue_feats"]})
        matches = out["matches"][0].detach().cpu().numpy()          # (M, 2) indices
        scores = out.get("scores")
        if scores is None:
            scores = out.get("matching_scores0")
        conf = (scores[0].detach().float().cpu().numpy()
                if scores is not None else np.ones(len(matches), np.float32))
    except Exception as exc:  # pragma: no cover - runtime dependent
        log.warning("LightGlue matching failed (%s) - falling back to SIFT.", exc)
        return None

    if len(matches) == 0:
        return MatchResult.empty("superpoint+lightglue")
    q = query.keypoints[matches[:, 0]].astype(np.float32)
    t = target.keypoints[matches[:, 1]].astype(np.float32)
    conf = np.asarray(conf, np.float32).reshape(-1)[:len(q)]
    if len(conf) != len(q):
        conf = np.ones(len(q), np.float32)
    return MatchResult(q, t, conf, "superpoint+lightglue")


def extract_features(img: np.ndarray, prefer: Optional[str] = None) -> FeatureSet:
    """
    Extract local features with the configured backend.

    SuperPoint when available and requested; SIFT otherwise.  SuperPoint
    feature sets keep their native tensors so LightGlue can reuse them.
    """
    prefer = (prefer or settings.matcher).lower()
    if prefer != "sift":
        fs = sp_backend.extract(img)
        if fs is not None:
            return fs
    return sift_backend.extract(img)


def match_features(query: FeatureSet, target: FeatureSet,
                   query_img: Optional[np.ndarray] = None,
                   target_img: Optional[np.ndarray] = None) -> MatchResult:
    """
    Match two feature sets, dispatching on backend and degrading gracefully.

    ``query_img`` / ``target_img`` are only needed for the SIFT retry path when
    the primary features were SuperPoint.
    """
    if query.backend == "superpoint" and target.backend == "superpoint":
        result = match_superpoint(query, target)
        if result is not None and result.count >= MIN_USEFUL_MATCHES:
            return result
        if query_img is not None and target_img is not None:
            log.debug("LightGlue produced %s matches - retrying with SIFT.",
                      "no" if result is None else result.count)
            q_sift = sift_backend.extract(query_img)
            t_sift = sift_backend.extract(target_img)
            q, t, c = sift_backend.match(q_sift, t_sift)
            retry = MatchResult(q, t, c, "sift-fallback")
            if result is None or retry.count > result.count:
                return retry
        return result if result is not None else MatchResult.empty("superpoint+lightglue")

    q, t, c = sift_backend.match(query, target)
    return MatchResult(q, t, c, "sift")


def rotate_image(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate by ``k * 90`` degrees counter-clockwise (numpy convention)."""
    return np.ascontiguousarray(np.rot90(img, k % 4))


def unrotate_points(pts: np.ndarray, k: int, width: int, height: int) -> np.ndarray:
    """
    Map points measured on ``rotate_image(img, k)`` back into the original
    frame, where ``width``/``height`` describe the *original* image.

    Derived directly from the numpy rotation identity
    ``rot90(A, 1)[i, j] == A[j, W - 1 - i]`` and its higher powers, so the
    rotation search (spec section 50) never leaks a coordinate-frame bug into
    the homography.
    """
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    k = k % 4
    if k == 0:
        return pts.copy()
    out = np.empty_like(pts)
    x_rot, y_rot = pts[:, 0], pts[:, 1]
    if k == 1:                                  # 90 CCW
        out[:, 0] = (width - 1) - y_rot
        out[:, 1] = x_rot
    elif k == 2:                                # 180
        out[:, 0] = (width - 1) - x_rot
        out[:, 1] = (height - 1) - y_rot
    else:                                       # 270 CCW == 90 CW
        out[:, 0] = y_rot
        out[:, 1] = (height - 1) - x_rot
    return out


def rotated_size(k: int, width: int, height: int) -> Tuple[int, int]:
    """Size of ``rotate_image(img, k)`` given the original width/height."""
    return (height, width) if k % 2 else (width, height)
