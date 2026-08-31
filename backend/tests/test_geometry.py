"""Geometric verification and coordinate-rebasing invariants."""
from __future__ import annotations

import numpy as np
import pytest

from app.localization import homography as hg
from app.localization.tiling import plan_tiles


# ---------------------------------------------------------------------------
# Convexity / coverage helpers
# ---------------------------------------------------------------------------
def test_is_convex_accepts_a_rectangle():
    assert hg.is_convex(np.array([[0, 0], [10, 0], [10, 8], [0, 8]]))


def test_is_convex_rejects_a_bowtie():
    assert not hg.is_convex(np.array([[0, 0], [10, 0], [0, 8], [10, 8]]))


def test_spatial_coverage_full_grid():
    grid = 4
    pts = [[x * 25 + 5, y * 25 + 5] for x in range(grid) for y in range(grid)]
    frac, cells = hg.spatial_coverage(np.array(pts, float), 100, 100, grid=grid)
    assert cells == grid * grid
    assert frac == pytest.approx(1.0)


def test_spatial_coverage_single_corner():
    frac, cells = hg.spatial_coverage(np.array([[1, 1], [2, 2], [3, 3]], float),
                                      100, 100, grid=4)
    assert cells == 1
    assert frac == pytest.approx(1 / 16)


# ---------------------------------------------------------------------------
# Homography rebasing
# ---------------------------------------------------------------------------
def test_scale_homography_identity_is_identity():
    out = hg.scale_homography(np.eye(3), 1.0, 1.0)
    assert np.allclose(out, np.eye(3))


def test_scale_homography_composes_scales():
    # A pure 2x zoom in the resized frame, undone/redone by the scale factors.
    H = np.diag([2.0, 2.0, 1.0])
    out = hg.scale_homography(H, query_scale=0.5, target_scale=0.25)
    pt = hg.transform_points(out, np.array([[10.0, 20.0]]))[0]
    # out = diag(1/0.25) @ diag(2) @ diag(0.5) = diag(4) -> (40, 80)
    assert pt == pytest.approx([40.0, 80.0])


def test_translate_homography_shifts_output():
    out = hg.translate_homography(np.eye(3), 100, 250)
    pt = hg.transform_points(out, np.array([[5.0, 7.0]]))[0]
    assert pt == pytest.approx([105.0, 257.0])


# ---------------------------------------------------------------------------
# RANSAC estimation
# ---------------------------------------------------------------------------
def _apply(H, pts):
    return hg.transform_points(H, pts)


def test_estimate_recovers_a_known_homography():
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 400, size=(80, 2))
    true_H = np.array([[1.02, 0.01, 12.0],
                       [-0.015, 0.99, -7.0],
                       [1e-5, -2e-5, 1.0]])
    t = _apply(true_H, q)
    res = hg.estimate(q, t, (400, 400))
    assert res.ok and res.plausible
    assert res.inliers >= 60
    assert res.reprojection_error < 1.0
    # H is defined up to scale; compare after normalising the bottom-right term.
    got = res.H / res.H[2, 2]
    assert np.allclose(got, true_H / true_H[2, 2], atol=5e-2)


def test_estimate_rejects_random_correspondences():
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 400, size=(60, 2))
    t = rng.uniform(0, 400, size=(60, 2))
    res = hg.estimate(q, t, (400, 400))
    assert not (res.ok and res.plausible)


def test_estimate_needs_four_points():
    res = hg.estimate(np.zeros((3, 2)), np.zeros((3, 2)), (100, 100))
    assert res.rejection == "too_few_matches"
    assert not res.plausible


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def test_plan_tiles_respects_the_budget():
    tiles = plan_tiles(4000, 4000)
    assert 0 < len(tiles) <= 600
    ids = [t.tile_id for t in tiles]
    assert ids == list(range(len(tiles)))          # contiguous ids after subsample


def test_plan_tiles_area_tracks_scale():
    tiles = plan_tiles(3000, 3000, scales=[0.10], overlap=0.25, max_tiles=600)
    map_area = 3000 * 3000
    for t in tiles:
        assert t.width == t.height
        assert t.width * t.height == pytest.approx(0.10 * map_area, rel=0.05)


def test_plan_tiles_tiny_map_still_produces_one_tile():
    tiles = plan_tiles(300, 300, scales=[0.25], overlap=0.25, max_tiles=50)
    assert len(tiles) >= 1
