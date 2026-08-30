"""Small shared image helpers (I/O, resizing, encoding)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from app.logging_config import get_logger

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class ImageError(ValueError):
    """Raised for unreadable / unsupported imagery."""


def imread(path: Path) -> np.ndarray:
    """Read an image as BGR, tolerant of unicode paths (cv2.imread is not)."""
    path = Path(path)
    if not path.exists():
        raise ImageError(f"Image not found: {path.name}")
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:  # pragma: no cover
        raise ImageError(f"Could not decode {path.name}: {exc}") from exc
    if img is None or img.size == 0:
        raise ImageError(f"Corrupted or unsupported image: {path.name}")
    return img


def imwrite(path: Path, img: np.ndarray, quality: int = 90) -> Path:
    """Write an image, creating parent dirs; unicode-safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:  # pragma: no cover
        raise ImageError(f"Failed to encode {path.name}")
    buf.tofile(str(path))
    return path


def fit_long_edge(img: np.ndarray, long_edge: int) -> Tuple[np.ndarray, float]:
    """
    Downscale so the longest side is ``long_edge``.

    Returns the resized image and the scale factor applied, so that
    coordinates computed on the resized image can be mapped back:
        original_xy = resized_xy / scale
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= long_edge:
        return img, 1.0
    scale = long_edge / float(longest)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_info(path: Path, img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    size = Path(path).stat().st_size if Path(path).exists() else 0
    return {
        "filename": Path(path).name,
        "width": int(w),
        "height": int(h),
        "channels": int(img.shape[2]) if img.ndim == 3 else 1,
        "file_size": int(size),
        "aspect_ratio": round(w / h, 4) if h else 0.0,
    }
