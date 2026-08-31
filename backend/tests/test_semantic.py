"""Phase 3 - structural representation generation."""
from __future__ import annotations

import numpy as np
import pytest

from app.localization.semantic import (StructuralRepresentation,
                                       build_structural_representation)


def test_shapes_and_types(synthetic_frame):
    rep = build_structural_representation(synthetic_frame)
    assert isinstance(rep, StructuralRepresentation)
    h, w = synthetic_frame.shape[:2]
    assert rep.grayscale.shape == (h, w)
    assert rep.edges.shape == (h, w)
    assert rep.structural.shape == (h, w, 3)
    assert rep.debug_overlay is not None and rep.debug_overlay.shape == (h, w, 3)
    assert rep.edges.dtype == np.uint8
    assert set(np.unique(rep.edges)).issubset({0, 255})


def test_detects_linear_structure(synthetic_frame):
    rep = build_structural_representation(synthetic_frame)
    # The rendered structural view must contain dark road pixels.
    dark = (rep.structural < 80).all(axis=2)
    assert dark.mean() > 0.01
    assert 0.0 <= rep.stats()["structural_edge_density"] <= 1.0


def test_debug_can_be_disabled(synthetic_frame):
    rep = build_structural_representation(synthetic_frame, debug=False)
    assert rep.debug_overlay is None


def test_grayscale_input_is_accepted(synthetic_frame):
    gray = synthetic_frame[..., 0]
    rep = build_structural_representation(gray)
    assert rep.structural.shape == (*gray.shape, 3)


def test_empty_image_raises():
    with pytest.raises(ValueError):
        build_structural_representation(np.zeros((0, 0, 3), np.uint8))
