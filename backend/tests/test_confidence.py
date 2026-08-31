"""Confidence decomposition and the MATCH / NO_MATCH / AMBIGUOUS decision."""
from __future__ import annotations

import numpy as np
import pytest

from app.localization import confidence as cf
from app.localization.homography import HomographyResult


def _good_hom(inliers: int = 200, ratio: float = 0.55, err: float = 2.0,
              coverage: float = 0.75) -> HomographyResult:
    return HomographyResult(
        ok=True, plausible=True, raw_matches=int(inliers / max(ratio, 1e-6)),
        inliers=inliers, inlier_ratio=ratio, reprojection_error=err,
        spatial_coverage=coverage, coverage_cells=int(coverage * 16),
    )


def _bad_hom() -> HomographyResult:
    return HomographyResult(ok=True, plausible=False, raw_matches=40, inliers=6,
                            inlier_ratio=0.15, reprojection_error=float("inf"),
                            spatial_coverage=0.05, rejection="below_min_inliers")


# ---------------------------------------------------------------------------
def test_weights_sum_to_one():
    assert sum(cf.WEIGHTS.values()) == pytest.approx(1.0)


def test_retrieval_score_is_clamped():
    assert cf.retrieval_score(-1.0) == 0.0
    assert cf.retrieval_score(5.0) == 1.0
    assert 0.0 < cf.retrieval_score(0.6) < 1.0


def test_inlier_score_monotonic_in_count():
    lo = cf.inlier_score(_good_hom(inliers=20))
    hi = cf.inlier_score(_good_hom(inliers=400))
    assert hi > lo


def test_geometry_score_zero_for_implausible():
    assert cf.geometry_score(_bad_hom()) == 0.0


# ---------------------------------------------------------------------------
def test_invalid_homography_is_capped():
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.9, _bad_hom()),
        cf.evaluate_candidate(2, 11, 0.4, _bad_hom()),
    ])
    for s in scores:
        assert s.final_score <= 0.35


def test_decide_no_valid_homography_is_no_match():
    scores = cf.finalize_scores([cf.evaluate_candidate(1, 10, 0.8, _bad_hom())])
    status, msg = cf.decide(scores)
    assert status is cf.MatchStatus.NO_MATCH
    assert "geometric verification" in msg


def test_decide_strong_single_candidate_is_match_found():
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.85, _good_hom()),
        cf.evaluate_candidate(2, 11, 0.35, _bad_hom()),
    ])
    status, _ = cf.decide(scores, positions={1: (500, 500)}, same_place_px=300)
    assert status is cf.MatchStatus.MATCH_FOUND


def _valid_score(cid: int, tid: int, final: float) -> cf.CandidateScore:
    """A hand-built, geometrically-valid CandidateScore at a chosen final score."""
    return cf.CandidateScore(
        candidate_id=cid, tile_id=tid, rank=cid, dino_similarity=0.7,
        raw_matches=300, inliers=180, inlier_ratio=0.6, spatial_coverage=0.7,
        reprojection_error=2.0, homography_valid=True, geometric_score=final,
        final_score=final,
    )


def test_decide_near_tie_far_apart_is_ambiguous():
    # scores already ranked (best first); gap 0.02 < ambiguity_gap 0.06.
    scores = [_valid_score(1, 10, 0.70), _valid_score(2, 20, 0.68)]
    status, msg = cf.decide(scores, positions={1: (200.0, 200.0), 2: (1800.0, 1800.0)},
                            same_place_px=300.0)
    assert status is cf.MatchStatus.AMBIGUOUS
    assert "different" in msg


def test_decide_near_tie_same_place_is_match_found():
    scores = [_valid_score(1, 10, 0.70), _valid_score(2, 20, 0.68)]
    status, _ = cf.decide(scores, positions={1: (1000.0, 1000.0), 2: (1010.0, 1015.0)},
                          same_place_px=300.0)
    assert status is cf.MatchStatus.MATCH_FOUND


def test_finalize_scores_preserves_a_genuine_tie():
    """Two geometrically-tied candidates must stay tied on final score, so the
    decision logic can see the ambiguity (Phase 3.1)."""
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.80, _good_hom(inliers=180)),
        cf.evaluate_candidate(2, 20, 0.80, _good_hom(inliers=178)),
    ])
    top, second = scores[0], scores[1]
    assert abs(top.final_score - second.final_score) < cf.settings.ambiguity_gap


def test_finalize_scores_single_candidate_is_unchallenged():
    scores = cf.finalize_scores([cf.evaluate_candidate(1, 10, 0.8, _good_hom())])
    assert scores[0].components["ambiguity"] == 1.0


def test_finalize_scores_clear_winner_keeps_its_lead():
    """A clearly-better candidate must not lose its margin to a runner-up that
    picked up a spurious ambiguity bonus."""
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.85, _good_hom(inliers=260, coverage=0.85)),
        cf.evaluate_candidate(2, 20, 0.55, _good_hom(inliers=40, ratio=0.25,
                                                     err=6.0, coverage=0.35)),
    ])
    winner = next(s for s in scores if s.candidate_id == 1)
    runner = next(s for s in scores if s.candidate_id == 2)
    assert winner.rank == 1
    assert winner.final_score - runner.final_score >= cf.settings.ambiguity_gap


def test_two_tied_valid_candidates_far_apart_decide_ambiguous_end_to_end():
    """Full path: evaluate_candidate -> finalize_scores -> decide."""
    positions = {1: (300.0, 300.0), 2: (2200.0, 2000.0)}
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.80, _good_hom(inliers=180)),
        cf.evaluate_candidate(2, 20, 0.80, _good_hom(inliers=179)),
    ], positions=positions, same_place_px=350.0)
    status, _ = cf.decide(scores, positions=positions, same_place_px=350.0)
    assert status is cf.MatchStatus.AMBIGUOUS


def test_overlapping_same_place_candidates_do_not_deflate_confidence():
    """Two strong candidates at the *same* spot are corroboration - the winner
    must not be penalised as if it were a genuine tie."""
    positions = {1: (1000.0, 1000.0), 2: (1020.0, 1010.0)}
    solo = cf.finalize_scores([cf.evaluate_candidate(1, 10, 0.85, _good_hom())])
    pair = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.85, _good_hom()),
        cf.evaluate_candidate(2, 20, 0.85, _good_hom(inliers=190)),
    ], positions=positions, same_place_px=400.0)

    winner = next(s for s in pair if s.rank == 1)
    assert winner.components["ambiguity"] == pytest.approx(1.0)
    assert winner.final_score == pytest.approx(solo[0].final_score, abs=0.03)
    status, _ = cf.decide(pair, positions=positions, same_place_px=400.0)
    assert status is cf.MatchStatus.MATCH_FOUND


def test_decide_empty_is_no_match():
    status, _ = cf.decide([])
    assert status is cf.MatchStatus.NO_MATCH


def test_runner_up_margin_ignores_same_place_rivals():
    positions = {1: (1000.0, 1000.0), 2: (1015.0, 1005.0)}   # overlapping tiles
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.85, _good_hom(inliers=200)),
        cf.evaluate_candidate(2, 20, 0.85, _good_hom(inliers=190)),
    ], positions=positions, same_place_px=400.0)
    margin, rival = cf.runner_up_margin(scores, positions, same_place_px=400.0)
    assert margin is None and rival is None          # unchallenged


def test_runner_up_margin_reports_a_different_place_rival():
    positions = {1: (300.0, 300.0), 2: (2000.0, 1800.0)}
    scores = cf.finalize_scores([
        cf.evaluate_candidate(1, 10, 0.85, _good_hom(inliers=200)),
        cf.evaluate_candidate(2, 20, 0.80, _good_hom(inliers=120)),
    ], positions=positions, same_place_px=400.0)
    margin, rival = cf.runner_up_margin(scores, positions, same_place_px=400.0)
    assert margin is not None and margin > 0
    assert rival.tile_id == 20


def test_runner_up_margin_no_verified_candidate():
    scores = cf.finalize_scores([cf.evaluate_candidate(1, 10, 0.8, _bad_hom())])
    assert cf.runner_up_margin(scores) == (None, None)
