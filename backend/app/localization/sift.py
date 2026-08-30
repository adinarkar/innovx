"""
SIFT + FLANN + Lowe ratio test (spec section 13).

Used as the default matcher when the torch stack is absent, as an explicit
alternative via ``MATCHER=sift``, and as an automatic fallback whenever
LightGlue produces too few correspondences.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.localization.superpoint import FeatureSet

log = get_logger(__name__)

LOWE_RATIO = 0.78
_FLANN_KDTREE = 1


def _detector(max_keypoints: int):
    return cv2.SIFT_create(nfeatures=max_keypoints)


def extract(img: np.ndarray, max_keypoints: Optional[int] = None) -> FeatureSet:
    """Detect SIFT keypoints/descriptors on a BGR image."""
    max_keypoints = max_keypoints or settings.max_keypoints
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # A light equalisation makes SIFT far more stable across exposure changes
    # between a satellite basemap and a drone frame.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    kps, desc = _detector(max_keypoints).detectAndCompute(gray, None)
    h, w = gray.shape[:2]
    if not kps:
        return FeatureSet(keypoints=np.zeros((0, 2), np.float32), descriptors=None,
                          scores=None, size=(w, h), backend="sift", detected=0)

    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    scores = np.array([kp.response for kp in kps], dtype=np.float32)
    return FeatureSet(keypoints=pts, descriptors=desc.astype(np.float32), scores=scores,
                      size=(w, h), backend="sift", detected=len(kps),
                      raw={"cv_keypoints": kps})


def match(query: FeatureSet, target: FeatureSet,
          ratio: float = LOWE_RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FLANN kNN match + Lowe ratio test.

    Returns ``(query_pts, target_pts, confidence)`` with one row per surviving
    correspondence.
    """
    empty = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
             np.zeros((0,), np.float32))
    if query.descriptors is None or target.descriptors is None:
        return empty
    if len(query.descriptors) < 2 or len(target.descriptors) < 2:
        return empty

    try:
        flann = cv2.FlannBasedMatcher({"algorithm": _FLANN_KDTREE, "trees": 5},
                                      {"checks": 64})
        knn = flann.knnMatch(query.descriptors, target.descriptors, k=2)
    except Exception as exc:  # FLANN is picky about tiny/degenerate inputs
        log.debug("FLANN failed (%s) - using brute force.", exc)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(query.descriptors, target.descriptors, k=2)

    q_pts, t_pts, conf = [], [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            q_pts.append(query.keypoints[best.queryIdx])
            t_pts.append(target.keypoints[best.trainIdx])
            # Map the ratio into a 0..1 confidence: lower ratio == more distinctive.
            conf.append(1.0 - (best.distance / max(second.distance, 1e-6)))

    if not q_pts:
        return empty
    return (np.asarray(q_pts, np.float32), np.asarray(t_pts, np.float32),
            np.asarray(conf, np.float32))
