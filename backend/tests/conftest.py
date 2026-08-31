"""Shared fixtures. Ensures ``backend/`` is importable as the package root."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """A deterministic BGR frame with roads, blocks and texture."""
    rng = np.random.default_rng(0)
    img = (rng.integers(90, 140, (480, 640, 3))).astype(np.uint8)
    img[200:230, :, :] = 40          # horizontal road
    img[:, 300:330, :] = 40          # vertical road
    img[60:150, 60:180, :] = 200     # building block
    img[300:420, 400:560, :] = 210   # building block
    return img
