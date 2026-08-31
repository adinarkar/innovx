"""Preprocessing split: fast matching branch vs the deferred visualisation branch."""
from __future__ import annotations

import numpy as np

from app.localization import preprocessing as pp


def _frame(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(240, 240, 3), dtype=np.uint8)


def test_prepare_for_matching_is_geometry_preserving():
    img = _frame()
    corrected, matching_input, applied = pp.prepare_for_matching(img, None)
    assert corrected.shape == img.shape
    assert matching_input.shape == img.shape        # no crop / warp
    assert applied is False


def test_run_preprocessing_still_returns_every_render():
    pre = pp.run_preprocessing(_frame())
    for field in ("original", "corrected", "enhanced", "grayscale", "edges",
                  "structural", "contours", "matching_input"):
        assert getattr(pre, field) is not None
    assert "edge_density" in pre.stats


def test_build_visualisations_never_raises_on_bad_input(monkeypatch):
    # Force the heavy branch to blow up; the result must still be well-formed.
    monkeypatch.setattr(pp, "structural_terrain",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    img = _frame()
    corrected, matching_input, applied = pp.prepare_for_matching(img, None)
    pre = pp.build_visualisations(img, corrected, matching_input, applied)
    assert pre.structural is not None
    assert pre.matching_input is matching_input
    assert pre.stats.get("visualisation_error") == 1.0
