"""
Shared pytest fixtures for the innovX VisualNav backend smoke suite.

The suite is deliberately small and fast: it locks down the pieces that a
refactor could silently break - geometric verification, the confidence maths,
coordinate rebasing and one full end-to-end ``localize`` call - without needing
any network access, GPU or the optional torch stack.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import path + isolated storage.  Both must be set BEFORE app.config is
# imported, because Settings is a module-level singleton built at import time.
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="visualnav-tests-"))
os.environ.setdefault("UPLOAD_DIR", str(_TMP / "uploads"))
os.environ.setdefault("CACHE_DIR", str(_TMP / "cache"))
os.environ.setdefault("PROCESSED_DIR", str(_TMP / "processed"))
os.environ.setdefault("APP_MODE", "real")

import cv2  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic reference map
# ---------------------------------------------------------------------------
def _build_map(width: int = 1400, height: int = 1400, seed: int = 7) -> np.ndarray:
    """A small procedural 'aerial' scene with plenty of corner structure."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((height // 30, width // 30)).astype(np.float32)
    terrain = cv2.GaussianBlur(cv2.resize(coarse, (width, height),
                                          interpolation=cv2.INTER_CUBIC), (0, 0), 8)
    img = np.dstack([70 + terrain * 55, 95 + terrain * 60,
                     85 + terrain * 50]).astype(np.uint8)

    for i in range(1, 7):                       # road grid
        y = int(height * i / 7 + rng.integers(-20, 20))
        x = int(width * i / 7 + rng.integers(-20, 20))
        cv2.line(img, (0, y), (width, y), (58, 58, 58), int(rng.integers(6, 13)), cv2.LINE_AA)
        cv2.line(img, (x, 0), (x, height), (58, 58, 58), int(rng.integers(6, 13)), cv2.LINE_AA)

    for _ in range(320):                        # buildings
        w = int(rng.integers(16, 70)); h = int(rng.integers(16, 70))
        cx = int(rng.integers(w, width - w)); cy = int(rng.integers(h, height - h))
        box = cv2.boxPoints(((cx, cy), (w, h), float(rng.integers(0, 90)))).astype(np.int32)
        roof = tuple(int(v) for v in rng.integers(120, 225, size=3))
        cv2.fillPoly(img, [box + np.array([5, 6])], (45, 45, 45), cv2.LINE_AA)
        cv2.fillPoly(img, [box], roof, cv2.LINE_AA)
        cv2.polylines(img, [box], True, (70, 70, 70), 1, cv2.LINE_AA)

    noise = rng.normal(0, 3.5, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _crop_fraction(img: np.ndarray, fraction: float, cx: float, cy: float):
    h, w = img.shape[:2]
    side = min(int(round(math.sqrt(fraction * w * h))), h, w)
    x = int(np.clip(cx - side / 2, 0, w - side))
    y = int(np.clip(cy - side / 2, 0, h - side))
    return img[y:y + side, x:x + side].copy(), (x + side // 2, y + side // 2)


@pytest.fixture(scope="session")
def tmp_storage() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def synthetic_map(tmp_storage: Path) -> dict:
    """Write a synthetic reference map once and return its path + dimensions."""
    img = _build_map()
    path = tmp_storage / "ref_map.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    h, w = img.shape[:2]
    return {"path": path, "image": img, "width": w, "height": h}


@pytest.fixture(scope="session")
def indexed_map(synthetic_map: dict):
    """A MapRecord whose tile index has been built (classical backend)."""
    from app.localization.pipeline import build_map_index
    from app.store import MapRecord, new_id

    rec = MapRecord(
        map_id=new_id("map"), path=synthetic_map["path"],
        width=synthetic_map["width"], height=synthetic_map["height"],
        filename="ref_map.jpg", file_size=synthetic_map["path"].stat().st_size,
    )
    build_map_index(rec)
    assert rec.embedding_status == "ready", rec.error
    return rec


@pytest.fixture
def drone_crop(synthetic_map: dict, tmp_storage: Path):
    """A direct 15% crop of the map and its true centre in map pixels."""
    img = synthetic_map["image"]
    cx, cy = int(synthetic_map["width"] * 0.60), int(synthetic_map["height"] * 0.42)
    crop, truth = _crop_fraction(img, 0.15, cx, cy)
    path = tmp_storage / "drone_crop.jpg"
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"path": path, "truth": truth}


@pytest.fixture
def unrelated_frame(tmp_storage: Path):
    """Structured noise that does not appear anywhere in the map."""
    rng = np.random.default_rng(999)
    size = 520
    frame = np.full((size, size, 3), 200, np.uint8)
    for _ in range(70):
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        col = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.rectangle(frame, p1, p2, col, -1, cv2.LINE_AA)
    path = tmp_storage / "unrelated.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"path": path}
