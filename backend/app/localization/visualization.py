"""
Explainability renders (spec sections 15, 26, 27, 28, 30 and 43).

Everything the presenter shows on the Processing page is produced here so the
frontend never has to reimplement geometry.  All colours follow the innovX
pastel-red / white / dark-grey palette (OpenCV wants BGR).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.logging_config import get_logger
from app.localization.imaging import fit_long_edge, imwrite

log = get_logger(__name__)

# --- palette (BGR) --------------------------------------------------------
PASTEL_RED = (115, 115, 229)      # #E57373
LIGHT_RED = (202, 202, 246)       # #F6CACA
DARK_GREY = (43, 43, 43)          # #2B2B2B
MID_GREY = (119, 119, 119)        # #777777
WHITE = (255, 255, 255)
INLIER_GREEN = (120, 170, 90)     # accepted correspondences
OUTLIER_GREY = (200, 200, 200)    # rejected correspondences

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
def _label(img: np.ndarray, text: str, org: Tuple[int, int],
           color=DARK_GREY, scale: float = 0.5, thickness: int = 1) -> None:
    """Draw text on a white plate so it stays readable over any imagery."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 5, y - th - 6), (x + tw + 5, y + base + 2), WHITE, cv2.FILLED)
    cv2.rectangle(img, (x - 5, y - th - 6), (x + tw + 5, y + base + 2), LIGHT_RED, 1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_keypoints(img: np.ndarray, keypoints: np.ndarray,
                   scores: Optional[np.ndarray] = None,
                   max_draw: int = 1200) -> np.ndarray:
    """Small dots, brighter for stronger responses (spec section 26)."""
    out = img.copy()
    if keypoints is None or len(keypoints) == 0:
        return out
    pts = np.asarray(keypoints, np.float64).reshape(-1, 2)
    order = np.arange(len(pts))
    if scores is not None and len(scores) == len(pts):
        order = np.argsort(-np.asarray(scores).reshape(-1))
    order = order[:max_draw]
    radius = max(1, int(round(min(out.shape[:2]) / 400)))
    for i in order:
        x, y = pts[i]
        cv2.circle(out, (int(round(x)), int(round(y))), radius + 1, WHITE, -1, cv2.LINE_AA)
        cv2.circle(out, (int(round(x)), int(round(y))), radius, PASTEL_RED, -1, cv2.LINE_AA)
    return out


def draw_polygon(img: np.ndarray, polygon: Sequence[Sequence[float]],
                 label: Optional[str] = None,
                 color=PASTEL_RED, fill_alpha: float = 0.22,
                 thickness: Optional[int] = None) -> np.ndarray:
    """Semi-transparent pastel-red fill + solid outline (spec section 15)."""
    out = img.copy()
    if polygon is None or len(polygon) < 3:
        return out
    poly = np.asarray(polygon, np.float64).round().astype(np.int32).reshape(-1, 1, 2)
    thickness = thickness or max(2, int(round(min(img.shape[:2]) / 300)))

    if fill_alpha > 0:
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0)
    cv2.polylines(out, [poly], True, color, thickness, cv2.LINE_AA)

    if label:
        x, y = poly.reshape(-1, 2).min(axis=0)
        _label(out, label, (int(x) + 8, max(24, int(y) - 10)),
               scale=max(0.5, min(img.shape[:2]) / 1200))
    return out


def draw_marker(img: np.ndarray, point: Sequence[float],
                label: Optional[str] = None, color=PASTEL_RED) -> np.ndarray:
    """Crosshair marker for the estimated drone centre (spec section 16)."""
    out = img.copy()
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    r = max(8, int(round(min(img.shape[:2]) / 90)))
    t = max(2, r // 5)
    cv2.circle(out, (x, y), r, WHITE, t + 2, cv2.LINE_AA)
    cv2.circle(out, (x, y), r, color, t, cv2.LINE_AA)
    cv2.line(out, (x - r * 2, y), (x - r, y), color, t, cv2.LINE_AA)
    cv2.line(out, (x + r, y), (x + r * 2, y), color, t, cv2.LINE_AA)
    cv2.line(out, (x, y - r * 2), (x, y - r), color, t, cv2.LINE_AA)
    cv2.line(out, (x, y + r), (x, y + r * 2), color, t, cv2.LINE_AA)
    cv2.circle(out, (x, y), max(2, t), color, cv2.FILLED, cv2.LINE_AA)
    if label:
        _label(out, label, (x + r * 2 + 6, y + 6),
               scale=max(0.5, min(img.shape[:2]) / 1200))
    return out


# --------------------------------------------------------------------------
def side_by_side(left: np.ndarray, right: np.ndarray,
                 gap: int = 24) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Compose two images on one canvas for correspondence drawing.

    Returns the canvas plus the (dx, dy) offset applied to each side so callers
    can translate their point sets into canvas space.
    """
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    height = max(lh, rh)
    canvas = np.full((height, lw + gap + rw, 3), 250, dtype=np.uint8)
    l_off = (0, (height - lh) // 2)
    r_off = (lw + gap, (height - rh) // 2)
    canvas[l_off[1]:l_off[1] + lh, l_off[0]:l_off[0] + lw] = left
    canvas[r_off[1]:r_off[1] + rh, r_off[0]:r_off[0] + rw] = right
    return canvas, l_off, r_off


def draw_matches(query_img: np.ndarray, target_img: np.ndarray,
                 query_pts: np.ndarray, target_pts: np.ndarray,
                 inlier_mask: Optional[np.ndarray] = None,
                 inliers_only: bool = False,
                 max_lines: int = 220) -> np.ndarray:
    """
    Correspondence visualisation (spec section 28).

    Rejected matches are drawn thin and grey, accepted RANSAC inliers thicker
    and in the accent colour, so the audience can see the geometric filter do
    its job.
    """
    canvas, l_off, r_off = side_by_side(query_img, target_img)
    q = np.asarray(query_pts, np.float64).reshape(-1, 2)
    t = np.asarray(target_pts, np.float64).reshape(-1, 2)
    n = min(len(q), len(t))
    if n == 0:
        _label(canvas, "No correspondences", (16, 30), color=PASTEL_RED, scale=0.7)
        return canvas

    mask = (np.ones(n, bool) if inlier_mask is None
            else np.asarray(inlier_mask, bool).reshape(-1)[:n])
    order = list(np.flatnonzero(~mask)) + list(np.flatnonzero(mask))  # inliers on top
    if inliers_only:
        order = list(np.flatnonzero(mask))
    if len(order) > max_lines:
        step = len(order) / float(max_lines)
        order = [order[int(i * step)] for i in range(max_lines)]

    for i in order:
        p0 = (int(round(q[i, 0])) + l_off[0], int(round(q[i, 1])) + l_off[1])
        p1 = (int(round(t[i, 0])) + r_off[0], int(round(t[i, 1])) + r_off[1])
        if mask[i]:
            cv2.line(canvas, p0, p1, INLIER_GREEN, 1, cv2.LINE_AA)
            cv2.circle(canvas, p0, 2, PASTEL_RED, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, PASTEL_RED, -1, cv2.LINE_AA)
        else:
            cv2.line(canvas, p0, p1, OUTLIER_GREY, 1, cv2.LINE_AA)

    _label(canvas, "DRONE CAPTURE", (12, 26))
    _label(canvas, "CANDIDATE TILE", (r_off[0] + 12, 26))
    _label(canvas, f"{int(mask.sum())} inliers / {n} matches",
           (12, canvas.shape[0] - 14), color=PASTEL_RED)
    return canvas


def draw_candidate_boxes(map_img: np.ndarray, boxes: Sequence[dict],
                         max_dim: int = 1400) -> np.ndarray:
    """Top-K candidate windows on a downscaled map (spec section 27)."""
    small, scale = fit_long_edge(map_img, max_dim)
    out = small.copy()
    for b in boxes:
        x, y = int(b["x"] * scale), int(b["y"] * scale)
        w, h = int(b["width"] * scale), int(b["height"] * scale)
        rank = int(b.get("rank", 0))
        color = PASTEL_RED if rank == 1 else MID_GREY
        thick = 3 if rank == 1 else 1
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick, cv2.LINE_AA)
        _label(out, f"#{rank} tile {b.get('tile_id')}", (x + 6, y + 22), scale=0.45)
    return out


def render_result_map(map_img: np.ndarray,
                      polygon: Optional[Sequence[Sequence[float]]],
                      center: Optional[Sequence[float]],
                      candidate_box: Optional[dict] = None,
                      max_dim: int = 1800) -> np.ndarray:
    """The headline render: matched region + estimated centre on the full map."""
    small, scale = fit_long_edge(map_img, max_dim)
    out = small.copy()

    if candidate_box:
        x, y = int(candidate_box["x"] * scale), int(candidate_box["y"] * scale)
        w, h = int(candidate_box["width"] * scale), int(candidate_box["height"] * scale)
        cv2.rectangle(out, (x, y), (x + w, y + h), MID_GREY, 1, cv2.LINE_AA)

    if polygon is not None and len(polygon) >= 3:
        poly = np.asarray(polygon, np.float64) * scale
        out = draw_polygon(out, poly, label="Predicted Drone View")
    if center is not None:
        pt = (float(center[0]) * scale, float(center[1]) * scale)
        out = draw_marker(out, pt, label="Estimated Drone Position")
    return out


def crop_around(map_img: np.ndarray, polygon: Sequence[Sequence[float]],
                margin: float = 0.45, max_dim: int = 900) -> np.ndarray:
    """Zoomed 'LOCALIZED AREA' view beside the full map (spec section 30)."""
    h, w = map_img.shape[:2]
    poly = np.asarray(polygon, np.float64).reshape(-1, 2)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    pad_x = (x1 - x0) * margin
    pad_y = (y1 - y0) * margin
    x0 = int(max(0, x0 - pad_x)); y0 = int(max(0, y0 - pad_y))
    x1 = int(min(w, x1 + pad_x)); y1 = int(min(h, y1 + pad_y))
    if x1 <= x0 or y1 <= y0:
        return np.full((200, 200, 3), 245, np.uint8)

    crop = map_img[y0:y1, x0:x1].copy()
    local = poly - np.array([x0, y0], np.float64)
    crop = draw_polygon(crop, local, fill_alpha=0.18)
    crop, _ = fit_long_edge(crop, max_dim)
    return crop


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray


def save_all(job_dir: Path, images: Dict[str, np.ndarray]) -> Dict[str, str]:
    """
    Persist every render for a job and return ``{name: relative_url_path}``.

    Paths are relative so nothing in the API depends on the machine layout.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for name, img in images.items():
        if img is None:
            continue
        ext = ".png" if name in ("structural_map", "edges") else ".jpg"
        path = job_dir / f"{name}{ext}"
        try:
            imwrite(path, gray_to_bgr(img))
            written[name] = f"{job_dir.name}/{path.name}"
        except Exception as exc:  # a failed render must not fail the request
            log.warning("Could not write %s: %s", path.name, exc)
    return written
