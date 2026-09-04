"""
SuperPoint local feature extraction (spec section 11).

Returns a backend-agnostic :class:`FeatureSet` so LightGlue, SIFT/FLANN and
the visualisation layer can all consume the same structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.models.loader import load_superpoint, resolve_device, try_import_torch

log = get_logger(__name__)


@dataclass
class FeatureSet:
    """Keypoints in *image pixel* coordinates plus their descriptors."""
    keypoints: np.ndarray                     # (N, 2) float32, x/y
    descriptors: Optional[np.ndarray] = None  # (N, D) float32
    scores: Optional[np.ndarray] = None       # (N,) float32
    size: tuple = (0, 0)                      # (width, height)
    backend: str = "unknown"
    detected: int = 0                         # before the max-keypoint cap
    raw: Dict[str, Any] = field(default_factory=dict)  # backend-native payload

    @property
    def count(self) -> int:
        return int(len(self.keypoints)) if self.keypoints is not None else 0

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "detected_keypoints": int(self.detected),
            "selected_keypoints": self.count,
            "width": int(self.size[0]),
            "height": int(self.size[1]),
        }


def extract(img: np.ndarray, max_keypoints: Optional[int] = None,
           detection_threshold: Optional[float] = None,
           nms_radius: Optional[int] = None) -> Optional[FeatureSet]:
    """
    Run SuperPoint on a BGR image.  Returns ``None`` when the model is not
    installed so callers can fall back to SIFT.

    ``max_keypoints``/``detection_threshold``/``nms_radius`` (default from
    ``settings``) select which cached extractor instance is used - see
    ``load_superpoint`` for why this has to be config-keyed rather than a
    single global instance.
    """
    max_keypoints = max_keypoints or settings.max_keypoints
    detection_threshold = (settings.superpoint_detection_threshold
                           if detection_threshold is None else detection_threshold)
    nms_radius = settings.superpoint_nms_radius if nms_radius is None else nms_radius
    extractor = load_superpoint(max_keypoints, detection_threshold, nms_radius)
    if extractor is None:
        return None
    torch = try_import_torch()
    device = resolve_device()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray)[None, None].to(device)
    try:
        with torch.inference_mode():
            out = extractor.extract(tensor)
    except Exception as exc:  # pragma: no cover - runtime/model dependent
        log.warning("SuperPoint extraction failed (%s) - falling back.", exc)
        return None

    kpts = out["keypoints"][0].detach().float().cpu().numpy()
    desc = out["descriptors"][0].detach().float().cpu().numpy()
    scores = out.get("keypoint_scores")
    scores = scores[0].detach().float().cpu().numpy() if scores is not None else None

    h, w = gray.shape[:2]
    fs = FeatureSet(keypoints=kpts.astype(np.float32),
                    descriptors=desc.astype(np.float32),
                    scores=None if scores is None else scores.astype(np.float32),
                    size=(w, h), backend="superpoint",
                    detected=int(len(kpts)),
                    raw={"lightglue_feats": out})
    log.debug("SuperPoint: %d keypoints on %dx%d.", fs.count, w, h)
    return fs
