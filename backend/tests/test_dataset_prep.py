"""Phase 1 - dataset splitting and paired-image correctness."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from training.sat2map.prepare_dataset import prepare, _split_pair  # noqa: E402


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", arr)[1].tofile(str(path))


def test_split_pair_geometry():
    tile = np.random.default_rng(0).integers(0, 255, (128, 256, 3)).astype(np.uint8)
    sat, mp = _split_pair(tile)
    assert sat.shape == (128, 128, 3)
    assert mp.shape == (128, 128, 3)
    assert _split_pair(np.zeros((128, 100, 3), np.uint8)) is None


def test_prepare_produces_paired_split(tmp_path):
    src = tmp_path / "raw"
    rng = np.random.default_rng(1)
    for i in range(40):
        left = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
        right = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
        _write(src / f"{i:03d}.png", np.concatenate([left, right], axis=1))

    out = tmp_path / "dataset"
    summary = prepare(src, out, val_split=0.25, size=32, seed=7, already_split=False)

    assert summary["pairs_written"] == 40
    for part in ("train", "val"):
        sat = sorted((out / part / "satellite").glob("*.png"))
        mp = sorted((out / part / "map").glob("*.png"))
        assert [p.name for p in sat] == [p.name for p in mp]  # exact correspondence
        for p in sat:
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (32, 32, 3)
    n_val = len(list((out / "val" / "satellite").glob("*.png")))
    n_train = len(list((out / "train" / "satellite").glob("*.png")))
    assert n_val + n_train == 40 and n_val > 0


def test_corrupt_tiles_are_skipped(tmp_path):
    src = tmp_path / "raw"
    _write(src / "flat.png", np.full((64, 128, 3), 127, np.uint8))     # flat -> skip
    good = np.random.default_rng(2).integers(0, 255, (64, 128, 3)).astype(np.uint8)
    _write(src / "good.png", good)

    summary = prepare(src, tmp_path / "out", val_split=0.0, size=16, seed=1,
                      already_split=False)
    assert summary["pairs_written"] == 1
    assert summary["skipped"] == 1
