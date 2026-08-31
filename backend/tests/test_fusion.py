"""Phase 8 - cross-domain confidence fusion and NO_MATCH hardening."""
from __future__ import annotations

import pytest

from app.config import settings
from app.localization.confidence import (ConsensusResult, FusedConfidence, MatchStatus,
                                         RepresentationScore, fuse_confidence,
                                         refine_status)


def _rgb(score=0.7, valid=True):
    return RepresentationScore(representation="rgb", geometric_score=score,
                               homography_plausible=valid, weight=0.47,
                               map_center=(1000.0, 500.0))


def _aux(name, *, valid, geo=0.6, weight=0.29, center=(1000.0, 500.0)):
    return RepresentationScore(representation=name, geometric_score=geo,
                               homography_plausible=valid, weight=weight,
                               map_center=center)


def _consensus(offsets, tol=50.0, agree=None):
    agree = all(v <= tol for v in offsets.values()) if agree is None else agree
    return ConsensusResult(reference="rgb", agree=agree, tolerance_px=tol,
                           max_disagreement_px=max(offsets.values(), default=0.0),
                           offsets_px=dict(offsets),
                           participating=["rgb", *offsets])


def test_no_aux_leaves_base_untouched():
    fused = fuse_confidence(0.62, True, [_rgb(0.62)], None)
    assert fused.overall == pytest.approx(0.62)
    assert fused.applied_bonus == 0.0 and fused.applied_penalty == 0.0


def test_agreeing_representation_adds_bounded_bonus():
    reps = [_rgb(0.55), _aux("structural", valid=True, geo=0.8)]
    cons = _consensus({"structural": 5.0}, tol=50.0)
    fused = fuse_confidence(0.55, True, reps, cons)
    assert fused.overall > 0.55
    assert fused.overall <= 0.55 * (1 + settings.consensus_bonus_cap) + 1e-9
    assert fused.corroborating == ["structural"]
    assert fused.consensus is True


def test_no_bonus_when_rgb_homography_failed():
    reps = [_rgb(0.5, valid=False), _aux("map", valid=True, geo=0.9)]
    cons = _consensus({"map": 2.0}, tol=50.0)
    fused = fuse_confidence(0.5, False, reps, cons)
    assert fused.overall == pytest.approx(0.5)          # aux cannot lift a failed RGB
    assert fused.applied_bonus == 0.0


def test_dissenting_representation_penalises():
    reps = [_rgb(0.8), _aux("map", valid=True, geo=0.7, center=(1700.0, 900.0))]
    cons = _consensus({"map": 460.0}, tol=50.0)
    fused = fuse_confidence(0.8, True, reps, cons)
    assert fused.overall < 0.8
    assert fused.dissenting == ["map"]
    assert fused.consensus is False


def test_unverified_aux_contributes_nothing():
    reps = [_rgb(0.7), _aux("structural", valid=False, geo=0.05, center=None)]
    cons = _consensus({}, tol=50.0)
    fused = fuse_confidence(0.7, True, reps, cons)
    assert fused.overall == pytest.approx(0.7)


# ---- refine_status --------------------------------------------------------
def test_match_downgraded_to_no_match_when_fusion_collapses(monkeypatch):
    monkeypatch.setattr(settings, "low_confidence", 0.40)
    fused = FusedConfidence(overall=0.30, base=0.62)
    st, ex = refine_status(MatchStatus.MATCH_FOUND, "base.", fused, None, True, 0)
    assert st == MatchStatus.NO_MATCH


def test_match_downgraded_to_low_when_representations_disagree(monkeypatch):
    monkeypatch.setattr(settings, "low_confidence", 0.40)
    monkeypatch.setattr(settings, "match_confidence", 0.60)
    fused = FusedConfidence(overall=0.66, base=0.72, dissenting=["map"])
    cons = _consensus({"map": 500.0}, tol=50.0, agree=False)
    st, ex = refine_status(MatchStatus.MATCH_FOUND, "base.", fused, cons, True, 1)
    assert st == MatchStatus.LOW_CONFIDENCE


def test_low_promoted_to_match_when_independently_corroborated(monkeypatch):
    monkeypatch.setattr(settings, "match_confidence", 0.60)
    monkeypatch.setattr(settings, "low_confidence", 0.40)
    fused = FusedConfidence(overall=0.63, base=0.55, corroborating=["structural", "map"])
    cons = _consensus({"structural": 8.0, "map": 6.0}, tol=50.0)
    st, ex = refine_status(MatchStatus.LOW_CONFIDENCE, "base.", fused, cons, True, 2)
    assert st == MatchStatus.MATCH_FOUND


def test_failed_rgb_is_never_promoted():
    fused = FusedConfidence(overall=0.9, base=0.2, corroborating=["map"])
    cons = _consensus({"map": 1.0}, tol=50.0)
    st, ex = refine_status(MatchStatus.NO_MATCH, "no valid homography.", fused,
                           cons, False, 1)
    assert st == MatchStatus.NO_MATCH


def test_ambiguous_is_left_alone():
    fused = FusedConfidence(overall=0.7, base=0.7)
    st, ex = refine_status(MatchStatus.AMBIGUOUS, "two regions.", fused, None, True, 0)
    assert st == MatchStatus.AMBIGUOUS
