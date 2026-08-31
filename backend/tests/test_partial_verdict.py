"""
A homography that projects the drone frame onto a sane, convex quadrilateral
but is supported by too few inliers is reported as ``partial`` (with the
reason), not flatly ``rejected``. Structural failures stay hard rejections.
"""
import numpy as np

from app.config import settings
from app.localization import homography as hg


def _square(n: int, jitter: float, seed: int = 0):
    """n correspondences on a unit-ish grid, target = query + small noise."""
    rng = np.random.default_rng(seed)
    g = int(np.ceil(np.sqrt(n)))
    xs, ys = np.meshgrid(np.linspace(20, 300, g), np.linspace(20, 220, g))
    q = np.column_stack([xs.ravel(), ys.ravel()])[:n].astype(float)
    t = q + 100.0 + rng.normal(0, jitter, q.shape)
    return q, t


def test_weak_but_sane_match_is_partial():
    q, t = _square(max(settings.min_inliers - 6, 5), jitter=0.3)
    res = hg.estimate(q, t, query_size=(320, 240))
    d = res.to_dict()

    assert not d["homography_valid"]
    assert d["verdict"] == "partial"
    assert d["partial"] is True
    assert d["rejection"] in {
        "below_min_inliers", "below_min_inlier_ratio",
        "reprojection_error_too_high", "features_too_concentrated",
    }


def test_strong_clean_match_is_verified():
    q, t = _square(settings.min_inliers + 25, jitter=0.15)
    res = hg.estimate(q, t, query_size=(320, 240))
    d = res.to_dict()
    assert d["homography_valid"] is True
    assert d["verdict"] == "verified"
    assert d["partial"] is False


def test_structural_failure_is_hard_rejection():
    # A wild point cloud yields a degenerate / non-convex projection, which is
    # a structural gate: it must never be softened to "partial".
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 300, (40, 2))
    t = rng.uniform(0, 3000, (40, 2))
    res = hg.estimate(q, t, query_size=(320, 240))
    d = res.to_dict()
    assert d["homography_valid"] is False
    if d["verdict"] != "verified":  # almost certainly rejected
        assert d["verdict"] == "rejected"
        assert d["partial"] is False
