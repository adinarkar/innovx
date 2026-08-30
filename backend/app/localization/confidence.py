"""
Confidence engine and match/no-match decision logic
(spec sections 17, 19, 20 and 51).

No single metric decides anything.  Every candidate gets a decomposed score
whose parts are visible in the UI, and the final status is derived from the
winner's score *and* its margin over the runner-up, so the system is allowed
to answer NO_MATCH or AMBIGUOUS instead of always naming a best tile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.localization.homography import HomographyResult

log = get_logger(__name__)


class MatchStatus(str, Enum):
    MATCH_FOUND = "MATCH_FOUND"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"


# Weights of the confidence components.  They sum to 1.0 so the result is a
# genuine 0..1 quantity rather than an arbitrary total.
WEIGHTS: Dict[str, float] = {
    "retrieval": 0.15,     # DINOv2 / classical similarity
    "inliers": 0.30,       # absolute RANSAC support
    "geometry": 0.25,      # reprojection error + homography plausibility
    "coverage": 0.15,      # spatial spread of inliers
    "ambiguity": 0.15,     # margin over the runner-up candidate
}


def _saturating(value: float, half: float) -> float:
    """Monotone 0..1 curve reaching 0.5 at ``half``; no hard ceiling artefacts."""
    v = max(0.0, float(value))
    return v / (v + max(half, 1e-6))


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def retrieval_score(similarity: float) -> float:
    """
    Map a cosine similarity onto 0..1.

    The floor is deliberately generous: retrieval only shortlists, and a modest
    similarity followed by 200 geometric inliers is a better match than a high
    similarity with none.
    """
    return _clamp01((float(similarity) - 0.30) / 0.60)


def inlier_score(hom: HomographyResult) -> float:
    """Blend absolute inlier count with inlier ratio."""
    count_part = _saturating(hom.inliers, half=45.0)
    ratio_part = _clamp01(hom.inlier_ratio / 0.60)
    return _clamp01(0.6 * count_part + 0.4 * ratio_part)


def geometry_score(hom: HomographyResult) -> float:
    """Reprojection accuracy, zeroed when the homography is implausible."""
    if not hom.ok or not hom.plausible:
        return 0.0
    err = hom.reprojection_error
    if not np.isfinite(err):
        return 0.0
    # Exponential decay: at the configured max error the score is ~0.37.
    accuracy = float(np.exp(-err / max(settings.max_reprojection_error, 1e-6)))
    shear_penalty = _clamp01(1.0 - hom.shear / 0.7)
    return _clamp01(0.75 * accuracy + 0.25 * shear_penalty)


def coverage_score(hom: HomographyResult) -> float:
    """Full credit once inliers touch ~60% of the grid cells."""
    return _clamp01(hom.spatial_coverage / 0.60)


def ambiguity_score(best_geo: float, runner_up_geo: Optional[float]) -> float:
    """
    1.0 when the winner is unchallenged, falling towards 0 as the runner-up
    closes in.  A tiny margin means the map contains repeated structure.
    """
    if runner_up_geo is None:
        return 1.0
    gap = float(best_geo) - float(runner_up_geo)
    return _clamp01(gap / 0.25)


@dataclass
class CandidateScore:
    """Full diagnostic record for one candidate (spec section 51)."""
    candidate_id: int
    tile_id: int
    rank: int = 0
    dino_similarity: float = 0.0
    raw_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    spatial_coverage: float = 0.0
    reprojection_error: Optional[float] = None
    homography_valid: bool = False
    rejection: Optional[str] = None
    rotation_applied: int = 0
    components: Dict[str, float] = field(default_factory=dict)
    geometric_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "tile_id": self.tile_id,
            "rank": self.rank,
            "dino_similarity": round(float(self.dino_similarity), 4),
            "raw_matches": int(self.raw_matches),
            "inliers": int(self.inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "spatial_coverage": round(float(self.spatial_coverage), 4),
            "reprojection_error": (round(float(self.reprojection_error), 3)
                                   if self.reprojection_error is not None else None),
            "homography_valid": bool(self.homography_valid),
            "rejection": self.rejection,
            "rotation_applied": int(self.rotation_applied),
            "geometric_score": round(float(self.geometric_score), 4),
            "final_score": round(float(self.final_score), 4),
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
        }


def evaluate_candidate(candidate_id: int, tile_id: int, similarity: float,
                       hom: HomographyResult, rotation_applied: int = 0) -> CandidateScore:
    """
    Score one candidate *without* the ambiguity term - that needs the whole
    ranked field and is applied afterwards by :func:`finalize_scores`.
    """
    comp = {
        "retrieval": retrieval_score(similarity),
        "inliers": inlier_score(hom),
        "geometry": geometry_score(hom),
        "coverage": coverage_score(hom),
    }
    # Geometric score = the purely verification-driven part, used for ranking
    # and for measuring ambiguity between candidates.
    geo = (0.45 * comp["inliers"] + 0.35 * comp["geometry"] + 0.20 * comp["coverage"])
    if not (hom.ok and hom.plausible):
        geo *= 0.25          # keep the ordering informative, kill the credit

    return CandidateScore(
        candidate_id=candidate_id, tile_id=tile_id, dino_similarity=float(similarity),
        raw_matches=hom.raw_matches, inliers=hom.inliers, inlier_ratio=hom.inlier_ratio,
        spatial_coverage=hom.spatial_coverage,
        reprojection_error=(float(hom.reprojection_error)
                            if np.isfinite(hom.reprojection_error) else None),
        homography_valid=bool(hom.ok and hom.plausible), rejection=hom.rejection,
        rotation_applied=rotation_applied, components=comp,
        geometric_score=_clamp01(geo),
    )


def _same_place(id_a: int, id_b: int, positions: Dict[int, tuple],
                same_place_px: float) -> bool:
    """Whether two candidates' projected centres describe one physical spot."""
    if same_place_px <= 0:
        return False
    pa, pb = positions.get(id_a), positions.get(id_b)
    if pa is None or pb is None:
        return False
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) <= same_place_px


def _distinct_runner_up(candidate: CandidateScore, pool: List[CandidateScore],
                        positions: Dict[int, tuple],
                        same_place_px: float) -> Optional[CandidateScore]:
    """
    First entry in ``pool`` (already sorted by score, descending) whose
    projected position is NOT the same physical location as ``candidate``.

    Overlapping map tiles routinely produce two independently-strong
    candidates for one true location; that agreement must not count as a
    rival when measuring how uniquely identifiable the location is, in either
    the confidence score's ambiguity term or the AMBIGUOUS/MATCH_FOUND
    decision.
    """
    for other in pool:
        if _same_place(candidate.candidate_id, other.candidate_id, positions, same_place_px):
            continue
        return other
    return None


def finalize_scores(scores: List[CandidateScore],
                    positions: Optional[Dict[int, tuple]] = None,
                    same_place_px: float = 0.0) -> List[CandidateScore]:
    """
    Apply the ambiguity term, compute final confidence and rank.

    ``positions``/``same_place_px`` let the ambiguity term skip a runner-up
    that is really the same location seen through an overlapping tile, so two
    genuinely correct candidates never suppress each other's confidence.
    """
    if not scores:
        return []
    positions = positions or {}
    ordered = sorted(scores, key=lambda s: s.geometric_score, reverse=True)
    for i, s in enumerate(ordered):
        runner = _distinct_runner_up(s, ordered[i + 1:], positions, same_place_px)
        runner_up = runner.geometric_score if runner else None
        s.components["ambiguity"] = ambiguity_score(s.geometric_score, runner_up)
        s.final_score = _clamp01(sum(WEIGHTS[k] * s.components.get(k, 0.0) for k in WEIGHTS))
        if not s.homography_valid:
            # An invalid homography can never present as a confident match.
            s.final_score = min(s.final_score, 0.35)

    ordered = sorted(ordered, key=lambda s: s.final_score, reverse=True)
    for rank, s in enumerate(ordered, start=1):
        s.rank = rank
    return ordered


def decide(scores: List[CandidateScore],
           positions: Optional[Dict[int, tuple]] = None,
           same_place_px: float = 0.0) -> tuple[MatchStatus, str]:
    """
    Turn the ranked field into a status plus a human-readable explanation.

    The order of the checks matters: structural failure (no valid homography)
    outranks ambiguity, which outranks a merely low score.

    ``positions`` maps candidate ids to their projected map centre.  Tiles
    overlap by design, so the same physical location routinely appears as two
    strong candidates; a near-tie only means AMBIGUOUS when the two candidates
    also disagree about *where* the drone is, by more than ``same_place_px``.
    """
    if not scores:
        return MatchStatus.NO_MATCH, "No candidate regions could be evaluated."

    best = scores[0]
    valid = [s for s in scores if s.homography_valid]

    if not valid:
        reason = best.rejection or "no_valid_homography"
        return (MatchStatus.NO_MATCH,
                f"No candidate passed geometric verification (best rejection: {reason}). "
                "The drone frame does not appear to overlap this reference map.")

    if not best.homography_valid:
        best = valid[0]

    if best.final_score < settings.low_confidence:
        return (MatchStatus.NO_MATCH,
                f"Best verified candidate scored {best.final_score:.2f}, below the "
                f"no-match threshold of {settings.low_confidence:.2f}.")

    # Ambiguity: a near-tie between two valid geometries that place the drone
    # in genuinely different parts of the map. Gaps only grow further down a
    # score-sorted list, so the first distinct-location runner-up is the only
    # one that can possibly trigger this.
    positions = positions or {}
    others = [s for s in valid if s.candidate_id != best.candidate_id]
    runner = _distinct_runner_up(best, others, positions, same_place_px)
    if runner is not None:
        gap = best.final_score - runner.final_score
        if gap < settings.ambiguity_gap and runner.final_score >= settings.low_confidence:
            return (MatchStatus.AMBIGUOUS,
                    f"Candidates {best.tile_id} ({best.final_score:.0%}) and "
                    f"{runner.tile_id} ({runner.final_score:.0%}) score within "
                    f"{gap:.0%} of each other but place the drone in different "
                    "map regions - the scene is not uniquely identifiable.")

    if best.final_score >= settings.match_confidence:
        return (MatchStatus.MATCH_FOUND,
                f"Verified with {best.inliers} RANSAC inliers "
                f"({best.inlier_ratio:.0%} of {best.raw_matches} matches) and "
                f"{best.spatial_coverage:.0%} spatial coverage.")

    return (MatchStatus.LOW_CONFIDENCE,
            f"A geometrically valid match exists at {best.final_score:.0%} confidence, "
            "below the reporting threshold. Treat the position as indicative only.")


def status_message(status: MatchStatus) -> str:
    """Presenter-facing wording (spec section 45): estimates, never guarantees."""
    return {
        MatchStatus.MATCH_FOUND: "Estimated visual position recovered.",
        MatchStatus.LOW_CONFIDENCE: "VISUAL LOCALIZATION UNRELIABLE - low confidence.",
        MatchStatus.AMBIGUOUS: "Ambiguous - multiple map regions explain this frame.",
        MatchStatus.NO_MATCH: "No corresponding region found in the reference map.",
    }[status]
