"""
Structural representation of a drone frame (spec Phase 3).

The reference image may be a road map, terrain map or simplified map screenshot
rather than a photograph. A real RGB drone frame and a road map share very
little photometric texture, but they share *structure*: road centrelines,
intersection positions, compound boundaries and building-block geometry. This
module distils that structure from a single RGB frame using classical OpenCV
operations only - no large segmentation model is required for Prototype V1.

The output is an auxiliary representation. It is never substituted for the
photographic ``matching_input`` in the primary geometric branch; it feeds the
optional structural matching / retrieval branch and the "Structural" debug
view, and it is protected by the same geometric verification gates as every
other representation.

The interface is intentionally small so a learned backend (SegFormer, DeepLab,
YOLO-seg, ...) can later be dropped in behind :func:`build_structural_representation`
without touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.logging_config import get_logger
from app.localization.imaging import to_gray

log = get_logger(__name__)


@dataclass
class StructuralRepresentation:
    """
    Structural views of one frame, all the same H x W as the input.

    ``grayscale``  - CLAHE-equalised single channel (uint8).
    ``edges``      - binary edge response (uint8, 0/255).
    ``structural`` - 3-channel road/structure render suitable for feature
                     matching against a map-style reference. Deterministic,
                     geometry preserving, no invented detail.
    ``debug_overlay`` - structural lines drawn over a dimmed frame, or ``None``.
    ``backend``    - which implementation produced this ("opencv" for V1).
    """
    grayscale: np.ndarray
    edges: np.ndarray
    structural: np.ndarray
    debug_overlay: Optional[np.ndarray] = None
    backend: str = "opencv"

    def stats(self) -> dict:
        return {
            "structural_backend": self.backend,
            "structural_edge_density": round(float((self.edges > 0).mean()), 4),
            "structural_line_density": round(
                float((to_gray(self.structural) < 128).mean()), 4),
        }


def _clahe_gray(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)


def _auto_canny(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    med = float(np.median(blurred))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    return cv2.Canny(blurred, lo or 50, hi or 150, L2gradient=True)


def _road_like(gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Extract elongated linear structure (roads, walls, field boundaries).

    Roads in an aerial frame are locally straight ribbons. Closing the edge
    map along horizontal and vertical structuring elements keeps components
    that are long in one direction and suppresses isolated texture speckle.
    A morphological black-hat on the grayscale image adds dark linear valleys
    (asphalt against lighter surroundings) that Canny alone can miss.
    """
    h, w = gray.shape[:2]
    k_len = max(9, (int(round(min(h, w) * 0.04)) | 1))

    horiz = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1)))
    vert = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len)))
    linear = cv2.max(horiz, vert)

    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, k_len)))
    _, dark_lines = cv2.threshold(blackhat, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    roads = cv2.max(linear, dark_lines)
    roads = cv2.morphologyEx(roads, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return roads


def _blocks(edges: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Large closed contours -> simplified building / block outlines."""
    h, w = shape
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros((h, w), np.uint8)
    min_area = max(120.0, 0.0006 * h * w)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(canvas, [approx], -1, 255, 1, cv2.LINE_AA)
    return canvas


def build_structural_representation(img: np.ndarray,
                                    debug: bool = True) -> StructuralRepresentation:
    """
    Deterministic OpenCV structural representation of a BGR frame.

    Cheap enough to run once per localisation request. The ``structural``
    channel renders road-like structure and block outlines on a white ground
    so it can be matched directly against a road/terrain-style reference
    image with the same local feature pipeline.
    """
    if img is None or img.size == 0:
        raise ValueError("build_structural_representation received an empty image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    gray = _clahe_gray(img)
    edges = _auto_canny(gray)
    roads = _road_like(gray, edges)
    blocks = _blocks(edges, (h, w))

    # Map-style render: white ground, dark roads, mid-grey block outlines.
    structural = np.full((h, w, 3), 255, np.uint8)
    structural[blocks > 0] = (150, 150, 150)
    structural[roads > 0] = (40, 40, 40)

    overlay = None
    if debug:
        overlay = np.clip(img.astype(np.float32) * 0.4 + 130.0, 0, 255).astype(np.uint8)
        overlay[blocks > 0] = (115, 115, 229)
        overlay[roads > 0] = (43, 43, 43)

    return StructuralRepresentation(grayscale=gray, edges=edges,
                                    structural=structural, debug_overlay=overlay,
                                    backend="opencv")
