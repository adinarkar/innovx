"""
RANSAC + homography estimation and geometric verification
(spec sections 14, 15, 16 and 18).

This module is the arbiter of the whole pipeline: retrieval only proposes,
geometry disposes.  A candidate is accepted only when a homography exists that
is numerically well conditioned, projects the drone frame to a sane convex
quadrilateral, and is supported by inliers spread across the frame rather than
clustered on one building.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class HomographyResult:
    ok: bool
    H: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    raw_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = float("inf")
    spatial_coverage: float = 0.0
    coverage_cells: int = 0
    coverage_grid: int = settings.coverage_grid
    plausible: bool = False
    rejection: Optional[str] = None
    corners: Optional[np.ndarray] = None      # (4, 2) projected frame outline
    center: Optional[np.ndarray] = None       # (2,) projected frame centre
    scale_ratio: float = 0.0                  # projected area / query area
    rotation_deg: float = 0.0
    shear: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "homography_valid": bool(self.ok and self.plausible),
            "raw_matches": int(self.raw_matches),
            "ransac_inliers": int(self.inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "reprojection_error": (round(float(self.reprojection_error), 3)
                                   if np.isfinite(self.reprojection_error) else None),
            "spatial_coverage": round(float(self.spatial_coverage), 4),
            "coverage_cells": int(self.coverage_cells),
            "coverage_grid": int(self.coverage_grid),
            "scale_ratio": round(float(self.scale_ratio), 4),
            "rotation_deg": round(float(self.rotation_deg), 2),
            "shear": round(float(self.shear), 4),
            "rejection": self.rejection,
        }


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points, returning (N, 2)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2)


def frame_corners(width: int, height: int) -> np.ndarray:
    """The four drone-image corners, clockwise from the top-left."""
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area (always positive)."""
    poly = np.asarray(poly, np.float64).reshape(-1, 2)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_convex(poly: np.ndarray) -> bool:
    """A valid perspective view of a rectangle stays convex and non-degenerate."""
    poly = np.asarray(poly, np.float64).reshape(-1, 2)
    if len(poly) < 4:
        return False
    crosses = []
    for i in range(len(poly)):
        a, b, c = poly[i], poly[(i + 1) % len(poly)], poly[(i + 2) % len(poly)]
        u, v = b - a, c - b
        # 2-D cross product (numpy 2 removed the 2-vector form of np.cross).
        crosses.append(float(u[0] * v[1] - u[1] * v[0]))
    crosses = np.asarray(crosses, dtype=np.float64)
    return bool(np.all(crosses > 0) or np.all(crosses < 0))


def decompose_affine(H: np.ndarray) -> Tuple[float, float, float]:
    """
    Rough (scale, rotation_deg, shear) from the affine part of the homography.

    Used only for plausibility checks - a drone frame that maps onto the map
    with wild shear or a near-zero scale is not a real match.
    """
    a = np.asarray(H, np.float64)[:2, :2]
    sx = float(np.linalg.norm(a[:, 0]))
    sy = float(np.linalg.norm(a[:, 1]))
    if sx < 1e-9 or sy < 1e-9:
        return 0.0, 0.0, 1.0
    rot = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    shear = float(abs(np.dot(a[:, 0] / sx, a[:, 1] / sy)))
    return float(np.sqrt(sx * sy)), rot, shear


def spatial_coverage(points: np.ndarray, width: int, height: int,
                     grid: Optional[int] = None) -> Tuple[float, int]:
    """
    Fraction of an NxN grid over the drone image that contains at least one
    inlier (spec section 18).  Guards against a homography supported entirely
    by one repetitive rooftop.
    """
    grid = grid or settings.coverage_grid
    if points is None or len(points) == 0 or width <= 0 or height <= 0:
        return 0.0, 0
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    gx = np.clip((pts[:, 0] / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((pts[:, 1] / height * grid).astype(int), 0, grid - 1)
    cells = len(set(zip(gx.tolist(), gy.tolist())))
    return cells / float(grid * grid), cells


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
def estimate(query_pts: np.ndarray, target_pts: np.ndarray,
             query_size: Tuple[int, int],
             ransac_threshold: Optional[float] = None) -> HomographyResult:
    """
    Estimate the query -> target homography with RANSAC and verify it.

    ``query_size`` is ``(width, height)`` of the drone image the query points
    were measured in.
    """
    thr = ransac_threshold if ransac_threshold is not None else settings.ransac_threshold
    q = np.asarray(query_pts, np.float64).reshape(-1, 2)
    t = np.asarray(target_pts, np.float64).reshape(-1, 2)
    res = HomographyResult(ok=False, raw_matches=int(len(q)))

    # A homography needs 4 correspondences; below MIN_INLIERS it can never pass.
    if len(q) < 4:
        res.rejection = "too_few_matches"
        return res

    try:
        H, mask = cv2.findHomography(q.reshape(-1, 1, 2), t.reshape(-1, 1, 2),
                                     cv2.USAC_MAGSAC, thr,
                                     maxIters=10000, confidence=0.9999)
    except Exception as exc:
        log.debug("MAGSAC failed (%s) - retrying with plain RANSAC.", exc)
        H, mask = None, None
    if H is None:
        try:
            H, mask = cv2.findHomography(q.reshape(-1, 1, 2), t.reshape(-1, 1, 2),
                                         cv2.RANSAC, thr, maxIters=5000)
        except Exception as exc:
            log.warning("Homography estimation failed: %s", exc)
            H, mask = None, None
    if H is None or mask is None:
        res.rejection = "homography_failed"
        return res

    mask = mask.ravel().astype(bool)
    res.H = np.asarray(H, np.float64)
    res.inlier_mask = mask
    res.inliers = int(mask.sum())
    res.inlier_ratio = res.inliers / float(max(len(q), 1))

    if res.inliers < 4:
        res.rejection = "insufficient_inliers"
        return res

    # Symmetric-free forward reprojection error over the inlier set only.
    projected = transform_points(res.H, q[mask])
    errors = np.linalg.norm(projected - t[mask], axis=1)
    res.reprojection_error = float(np.mean(errors))
    res.details["reprojection_median"] = float(np.median(errors))
    res.details["reprojection_p90"] = float(np.percentile(errors, 90))

    qw, qh = query_size
    res.spatial_coverage, res.coverage_cells = spatial_coverage(q[mask], qw, qh)

    corners = transform_points(res.H, frame_corners(qw, qh))
    res.corners = corners
    res.center = transform_points(res.H, np.array([[qw / 2.0, qh / 2.0]]))[0]

    scale, rot, shear = decompose_affine(res.H)
    res.rotation_deg = rot
    res.shear = shear
    query_area = float(qw * qh)
    res.scale_ratio = polygon_area(corners) / max(query_area, 1e-6)

    res.ok = True
    res.plausible, res.rejection = _plausibility(res, corners, scale, shear)
    return res


def _plausibility(res: HomographyResult, corners: np.ndarray,
                  scale: float, shear: float) -> Tuple[bool, Optional[str]]:
    """
    Hard geometric sanity gates.  These are structural, not confidence-based:
    failing any one of them means the transform cannot describe a planar
    aerial view, whatever the score says.
    """
    if not np.all(np.isfinite(corners)):
        return False, "non_finite_projection"
    if not is_convex(corners):
        return False, "non_convex_projection"
    if scale <= 1e-3 or not np.isfinite(scale):
        return False, "degenerate_scale"
    # The drone frame is rendered at working resolution against a tile of
    # comparable size, so an order-of-magnitude area change is implausible.
    if not (0.04 <= res.scale_ratio <= 25.0):
        return False, "implausible_scale_ratio"
    if shear > 0.7:
        return False, "excessive_shear"

    # Aspect distortion: opposite edges of a planar view stay comparable.
    e = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    if min(e) < 1e-6:
        return False, "degenerate_edges"
    if max(e[0], e[2]) / max(min(e[0], e[2]), 1e-6) > 4.0:
        return False, "extreme_perspective"
    if max(e[1], e[3]) / max(min(e[1], e[3]), 1e-6) > 4.0:
        return False, "extreme_perspective"

    if res.inliers < settings.min_inliers:
        return False, "below_min_inliers"
    if res.inlier_ratio < settings.min_inlier_ratio:
        return False, "below_min_inlier_ratio"
    if res.reprojection_error > settings.max_reprojection_error:
        return False, "reprojection_error_too_high"
    if res.spatial_coverage < settings.min_spatial_coverage:
        return False, "features_too_concentrated"
    return True, None


def scale_homography(H: np.ndarray, query_scale: float, target_scale: float) -> np.ndarray:
    """
    Rebase a homography solved between *resized* images onto the full
    resolution frames.

        H_full = S_target^-1 @ H_resized @ S_query

    where each S is the pure scaling applied before feature extraction.
    """
    Sq = np.diag([query_scale, query_scale, 1.0])
    St_inv = np.diag([1.0 / max(target_scale, 1e-9), 1.0 / max(target_scale, 1e-9), 1.0])
    return St_inv @ np.asarray(H, np.float64) @ Sq


def translate_homography(H: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compose a homography with a translation applied to its *output* space."""
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    return T @ np.asarray(H, np.float64)


def clamp_polygon(poly: np.ndarray, width: int, height: int) -> List[List[int]]:
    """Round a projected polygon to integer map pixels for the API/UI."""
    poly = np.asarray(poly, np.float64).reshape(-1, 2)
    out = []
    for x, y in poly:
        out.append([int(round(float(np.clip(x, -width, 2 * width)))),
                    int(round(float(np.clip(y, -height, 2 * height))))])
    return out
