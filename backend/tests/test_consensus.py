"""Phase 6/8 - representation-level scoring and cross-representation consensus."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.localization import pipeline
from app.localization.confidence import (RepresentationScore, normalised_weights,
                                         representation_consensus,
                                         representation_geometric_score)
from app.localization.homography import HomographyResult


def test_consensus_true_when_representations_agree():
    positions = {"rgb": (1000.0, 500.0), "structural": (1008.0, 494.0),
                 "map": (995.0, 505.0)}
    res = representation_consensus(positions, tolerance_px=40.0)
    assert res.agree is True
    assert res.reference == "rgb"
    assert set(res.offsets_px) == {"structural", "map"}
    assert res.max_disagreement_px < 40.0


def test_consensus_false_when_translated_map_points_elsewhere():
    positions = {"rgb": (1000.0, 500.0), "structural": (1005.0, 503.0),
                 "map": (1600.0, 900.0)}
    res = representation_consensus(positions, tolerance_px=40.0)
    assert res.agree is False
    assert res.offsets_px["map"] > 40.0


def test_consensus_single_branch_cannot_disagree():
    res = representation_consensus({"rgb": (10.0, 10.0)}, tolerance_px=5.0)
    assert res.agree is True
    assert res.max_disagreement_px == 0.0


def test_normalised_weights_sum_to_one_over_present_subset(monkeypatch):
    monkeypatch.setattr(settings, "rgb_weight", 0.40)
    monkeypatch.setattr(settings, "structural_weight", 0.25)
    monkeypatch.setattr(settings, "sat2map_weight", 0.15)
    monkeypatch.setattr(settings, "retrieval_weight", 0.20)

    w = normalised_weights(["rgb", "structural", "retrieval"])
    assert set(w) == {"rgb", "structural", "retrieval"}
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["rgb"] > w["structural"] > w["retrieval"]  # ordering preserved
    assert w["rgb"] == pytest.approx(0.40 / 0.85)


def test_representation_geometric_score_zero_for_invalid_homography():
    bad = HomographyResult(ok=False)
    assert representation_geometric_score(bad) == 0.0

    good = HomographyResult(ok=True, plausible=True, inliers=120, inlier_ratio=0.6,
                            reprojection_error=2.0, spatial_coverage=0.7)
    assert representation_geometric_score(good) > 0.5


def test_cross_representation_skips_without_a_verified_location():
    events = []
    scores, consensus = pipeline._cross_representation(
        representations={"structural": np.zeros((8, 8, 3), np.uint8)},
        structural_obj=object(), cluster_scores=[], best_cluster=None, best=None,
        by_id={}, agg_center=None, same_place_px=100.0,
        work_w=960, work_h=720, dw=1280, dh=960,
        stage=lambda *a, **k: events.append(a))
    assert scores == [] and consensus is None
    assert any(e[0] == "consensus" and e[2] == "skipped" for e in events)


def test_representation_estimates_empty_when_no_aux_images():
    out = pipeline._representation_estimates({}, None, [], 960, 720, 1280, 960)
    assert out == {}


def test_representation_score_serialises():
    s = RepresentationScore(representation="structural", inliers=40, inlier_ratio=0.5,
                            homography_plausible=True, geometric_score=0.6,
                            weight=0.25, map_center=(12.3456, 7.8))
    d = s.to_dict()
    assert d["representation"] == "structural"
    assert d["map_center"] == [12.3, 7.8]
    assert d["weight"] == 0.25
