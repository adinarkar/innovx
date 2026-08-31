"""
Drone image preprocessing (spec sections 9 and 10).

Two deliberately separate branches:

    MATCHING BRANCH        -> mild, geometry preserving normalisation that
                              feeds DINOv2 / SuperPoint / SIFT.
    VISUALISATION BRANCH   -> grayscale, edges and the Structural Terrain View
                              used only for explaining the result to a human.

The structural render must never replace the photographic frame in the
matching branch; heavy stylisation destroys the very texture the descriptors
rely on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from app.logging_config import get_logger
from app.localization.imaging import to_gray

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Camera correction
# --------------------------------------------------------------------------
@dataclass
class CameraCalibration:
    """Pinhole intrinsics + Brown-Conrady distortion coefficients."""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["CameraCalibration"]:
        if not data:
            return None
        try:
            fields = set(cls.__annotations__)
            return cls(**{k: float(v) for k, v in data.items() if k in fields})
        except Exception as exc:
            log.warning("Ignoring invalid camera calibration: %s", exc)
            return None

    def matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)
        return K, dist


def undistort(img: np.ndarray, calib: Optional[CameraCalibration]) -> Tuple[np.ndarray, bool]:
    """Remove lens distortion when intrinsics are supplied, else pass through."""
    if calib is None:
        return img.copy(), False
    h, w = img.shape[:2]
    K, dist = calib.matrices()
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
    return cv2.undistort(img, K, dist, None, new_K), True


# --------------------------------------------------------------------------
# Matching branch - mild enhancement only
# --------------------------------------------------------------------------
def enhance(img: np.ndarray,
            clip_limit: float = 2.0,
            tile_grid: int = 8,
            sharpen: bool = True,
            denoise: bool = True) -> np.ndarray:
    """
    CLAHE on the luminance channel + brightness normalisation, plus a light
    unsharp mask.  Kept deliberately moderate: over-processing invents texture
    and inflates false matches.
    """
    out = img.copy()
    if denoise:
        out = cv2.bilateralFilter(out, 5, 35, 5)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lum, chan_a, chan_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lum = clahe.apply(lum)

    # Brightness normalisation: re-centre luminance on mid-grey without
    # clipping detail away.
    mean = float(lum.mean())
    if mean > 1:
        lum = np.clip(lum.astype(np.float32) * (128.0 / mean), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge((lum, chan_a, chan_b)), cv2.COLOR_LAB2BGR)

    if sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
    return out


def normalize_for_matching(img: np.ndarray) -> np.ndarray:
    """
    The image actually handed to the feature extractors.

    Only illumination is touched - no edge maps, no morphology, no geometry
    changes - so keypoint coordinates stay valid in the original frame.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum, chan_a, chan_b = cv2.split(lab)
    lum = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lum)
    return cv2.cvtColor(cv2.merge((lum, chan_a, chan_b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Visualisation branch
# --------------------------------------------------------------------------
def edge_map(img: np.ndarray, low: int = 60, high: int = 160) -> np.ndarray:
    """Canny edges with thresholds auto-tuned around the image median."""
    gray = to_gray(img)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med)) or low
    hi = int(min(255, 1.33 * med)) or high
    return cv2.Canny(gray, lo, hi, L2gradient=True)


# BGR tuples matching the innovX pastel palette.
PASTEL = {
    "bg": (245, 245, 255),
    "open": (232, 238, 255),
    "road": (67, 67, 43),
    "structure": (115, 115, 229),
    "boundary": (69, 69, 69),
}


def structural_terrain(img: np.ndarray) -> np.ndarray:
    """
    Structural Terrain View (spec sections 9, 25 and 53).

    A map-like render built from a single RGB frame: strong linear structure
    (roads, walls, compound boundaries) is separated from blob-like structure
    (buildings, canopies) using morphology on the edge response, then painted
    in the innovX palette.  This is a visualisation only: it is never fed to
    the localisation pipeline, and it is not an elevation model.
    """
    h, w = img.shape[:2]
    edges = edge_map(img)

    # Long thin components -> road / boundary network.
    k_len = max(9, int(round(min(h, w) * 0.035)) | 1)
    horiz = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1)))
    vert = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len)))
    linear = cv2.max(horiz, vert)
    linear = cv2.morphologyEx(linear, cv2.MORPH_DILATE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Region simplification -> open space vs built-up mass.
    small = cv2.resize(img, (max(1, w // 2), max(1, h // 2)))
    flat = cv2.pyrMeanShiftFiltering(small, 12, 24)
    flat = cv2.resize(flat, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mass = cv2.threshold(to_gray(flat), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mass = cv2.morphologyEx(mass, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    canvas = np.full((h, w, 3), PASTEL["bg"], dtype=np.uint8)
    canvas[mass > 0] = PASTEL["open"]

    # Structural blobs: closed contours of a meaningful size, simplified.
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120.0, 0.0006 * h * w)
    kept = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(canvas, [approx], -1, PASTEL["structure"], 1, cv2.LINE_AA)
        kept += 1

    canvas[linear > 0] = PASTEL["road"]
    log.debug("Structural view: %d contours kept, %d linear pixels.",
              kept, int((linear > 0).sum()))
    return canvas


def contour_overlay(img: np.ndarray) -> np.ndarray:
    """Contours drawn over a dimmed copy of the frame (intermediate view)."""
    base = np.clip(img.astype(np.float32) * 0.45 + 120.0, 0, 255).astype(np.uint8)
    edges = edge_map(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 40:
            continue
        cv2.drawContours(base, [cnt], -1, PASTEL["structure"], 1, cv2.LINE_AA)
    return base


# --------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    original: np.ndarray
    corrected: np.ndarray
    enhanced: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray
    structural: np.ndarray
    contours: np.ndarray
    matching_input: np.ndarray
    calibration_applied: bool = False
    stats: Dict[str, float] = field(default_factory=dict)


def prepare_for_matching(img: np.ndarray, calib: Optional[CameraCalibration] = None
                         ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Fast branch: only what the feature extractors actually consume.

    Returns ``(corrected, matching_input, calibration_applied)``.  This is the
    single piece of preprocessing that sits on the latency-critical path; the
    explanatory renders are produced separately by :func:`build_visualisations`.
    """
    corrected, applied = undistort(img, calib)
    matching_input = normalize_for_matching(corrected)
    return corrected, matching_input, applied


def build_visualisations(original: np.ndarray, corrected: np.ndarray,
                         matching_input: np.ndarray,
                         calibration_applied: bool) -> PreprocessResult:
    """
    Slow branch: every explanatory render (enhanced frame, edges, Structural
    Terrain View, contours).  None of it feeds the pipeline, so it is safe to
    run off the critical path / on a worker thread.  Never raises - on any
    failure it degrades to plain stand-ins so a bad render can't fail a good
    localisation.
    """
    try:
        enhanced = enhance(corrected)
        gray = to_gray(corrected)
        edges = edge_map(corrected)
        structural = structural_terrain(corrected)
        contours = contour_overlay(corrected)
        stats = {
            "mean_brightness_original": round(float(to_gray(original).mean()), 2),
            "mean_brightness_enhanced": round(float(to_gray(enhanced).mean()), 2),
            "contrast_original": round(float(to_gray(original).std()), 2),
            "contrast_enhanced": round(float(to_gray(enhanced).std()), 2),
            "edge_density": round(float((edges > 0).mean()), 4),
        }
    except Exception as exc:                       # pragma: no cover - defensive
        log.warning("Visualisation branch failed (%s) - using plain stand-ins.", exc)
        gray = to_gray(corrected)
        enhanced, structural, contours = corrected, corrected, corrected
        edges = gray
        stats = {"visualisation_error": 1.0}

    return PreprocessResult(
        original=original, corrected=corrected, enhanced=enhanced, grayscale=gray,
        edges=edges, structural=structural, contours=contours,
        matching_input=matching_input, calibration_applied=calibration_applied,
        stats=stats,
    )


def run_preprocessing(img: np.ndarray,
                      calib: Optional[CameraCalibration] = None) -> PreprocessResult:
    """Both branches, synchronously - kept for tests and offline callers."""
    corrected, matching_input, applied = prepare_for_matching(img, calib)
    return build_visualisations(img, corrected, matching_input, applied)
