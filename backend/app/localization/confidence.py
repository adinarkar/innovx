"""
Confidence engine and match/no-match decision logic
(spec sections 17, 19, 20 and 51).

No single metric decides anything.  Every candidate gets a decomposed score
whose parts are visible in the UI, and the final status is derived from the
winner's score *and* its margin over the runner-up, so the system is allowed
to answer NO_MATCH or AMBIGUOUS instead of always naming a best tile.

The overlapping tiles used for retrieval (spec section 7) mean the true
location routinely produces *several* independently-verified candidates
rather than one.  Scoring those candidates individually and then comparing
the top two treats that agreement as a threat ("these two are suspiciously
close"), when it is actually the strongest evidence available.  This module
therefore works in two layers:

    per-tile diagnostics  -> CandidateScore   (unchanged, for the UI)
    per-location decision -> ClusterScore     (drives status + confidence)

Candidates are grouped by the physical map location they project to
(:func:`cluster_candidates`), and the location with the strongest *combined*
evidence - not the single best tile - is what the pipeline reports.  A
location corroborated by several overlapping tiles scores higher than one
supported by a single tile, which is exactly the additional signal a lone
tile score is blind to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

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


# Weights of the per-tile diagnostic score.  They sum to 1.0 so the result is
# a genuine 0..1 quantity rather than an arbitrary total.
WEIGHTS: Dict[str, float] = {
    "retrieval": 0.15,     # DINOv2 / classical similarity
    "inliers": 0.30,       # absolute RANSAC support
    "geometry": 0.25,      # reprojection error + homography plausibility
    "coverage": 0.15,      # spatial spread of inliers
    "ambiguity": 0.15,     # margin over the runner-up candidate
}

# Weights of the location-level score that actually drives the verdict.
# "consensus" is new here: it rewards a location being independently
# corroborated by more than one overlapping tile.
CLUSTER_WEIGHTS: Dict[str, float] = {
    "retrieval": 0.12,
    "inliers": 0.28,
    "geometry": 0.20,
    "coverage": 0.12,
    "consensus": 0.18,     # more supporting tiles for one location = stronger evidence
    "ambiguity": 0.10,     # margin over the nearest *distinct* location
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


def inlier_score(inliers: int, inlier_ratio: float) -> float:
    """
    Blend absolute inlier count with inlier ratio.

    Takes raw numbers rather than a :class:`HomographyResult` so the same
    formula scores both a single tile and a location's *combined* evidence
    (summed inliers across every tile that corroborates it).
    """
    count_part = _saturating(inliers, half=45.0)
    ratio_part = _clamp01(inlier_ratio / 0.60)
    return _clamp01(0.6 * count_part + 0.4 * ratio_part)


def geometry_score(reprojection_error: Optional[float], shear: float, valid: bool) -> float:
    """Reprojection accuracy, zeroed when the homography is implausible."""
    if not valid or reprojection_error is None or not np.isfinite(reprojection_error):
        return 0.0
    # Exponential decay: at the configured max error the score is ~0.37.
    accuracy = float(np.exp(-reprojection_error / max(settings.max_reprojection_error, 1e-6)))
    shear_penalty = _clamp01(1.0 - shear / 0.7)
    return _clamp01(0.75 * accuracy + 0.25 * shear_penalty)


def coverage_score(spatial_coverage: float) -> float:
    """Full credit once inliers touch ~60% of the grid cells."""
    return _clamp01(spatial_coverage / 0.60)


def consensus_score(support: int) -> float:
    """
    Reward a location being independently corroborated by more than one
    overlapping tile.  Zero for a single supporting tile (no corroboration
    to speak of); a saturating curve beyond that so a fourth or fifth
    agreeing tile adds progressively less than the second did.
    """
    return _saturating(max(0, support - 1), half=1.0)


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
    partial: bool = False        # cleared every structural gate, missed only a strength threshold
    rejection: Optional[str] = None
    rotation_applied: int = 0
    shear: float = 0.0
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
            "verdict": ("verified" if self.homography_valid
                        else "partial" if self.partial else "rejected"),
            "partial": bool(self.partial),
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
    valid = bool(hom.ok and hom.plausible)
    err = float(hom.reprojection_error) if np.isfinite(hom.reprojection_error) else None
    comp = {
        "retrieval": retrieval_score(similarity),
        "inliers": inlier_score(hom.inliers, hom.inlier_ratio),
        "geometry": geometry_score(err, hom.shear, valid),
        "coverage": coverage_score(hom.spatial_coverage),
    }
    # Geometric score = the purely verification-driven part, used for ranking
    # and for measuring ambiguity between candidates.
    geo = (0.45 * comp["inliers"] + 0.35 * comp["geometry"] + 0.20 * comp["coverage"])
    if not valid:
        geo *= 0.25          # keep the ordering informative, kill the credit

    return CandidateScore(
        candidate_id=candidate_id, tile_id=tile_id, dino_similarity=float(similarity),
        raw_matches=hom.raw_matches, inliers=hom.inliers, inlier_ratio=hom.inlier_ratio,
        spatial_coverage=hom.spatial_coverage, reprojection_error=err,
        homography_valid=valid, partial=bool(hom.partial and not valid),
        rejection=hom.rejection,
        rotation_applied=rotation_applied, shear=float(hom.shear), components=comp,
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


@dataclass
class ClusterScore:
    """
    Aggregated evidence for one physical map location, built from every
    candidate tile whose projection points there (spec: decide from
    differentiated parts of the map, not from raw overlapping tiles).
    """
    cluster_id: int
    tile_id: int                          # representative tile, for display
    representative_id: int                # candidate_id of that representative
    member_candidate_ids: List[int]
    support: int                          # member tiles with a valid homography
    total_members: int
    dino_similarity: float = 0.0
    raw_matches: int = 0                  # summed across supporting tiles
    inliers: int = 0                      # summed across supporting tiles
    inlier_ratio: float = 0.0
    spatial_coverage: float = 0.0         # best observed among supporting tiles
    reprojection_error: Optional[float] = None   # inlier-weighted average
    homography_valid: bool = False
    partial: bool = False                 # a member cleared every structural gate, just too weak
    rejection: Optional[str] = None
    rank: int = 0
    components: Dict[str, float] = field(default_factory=dict)
    geometric_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "tile_id": self.tile_id,
            "rank": self.rank,
            "member_tile_count": self.total_members,
            "support": self.support,
            "member_candidate_ids": list(self.member_candidate_ids),
            "dino_similarity": round(float(self.dino_similarity), 4),
            "raw_matches": int(self.raw_matches),
            "inliers": int(self.inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "spatial_coverage": round(float(self.spatial_coverage), 4),
            "reprojection_error": (round(float(self.reprojection_error), 3)
                                   if self.reprojection_error is not None else None),
            "homography_valid": bool(self.homography_valid),
            "verdict": ("verified" if self.homography_valid
                        else "partial" if self.partial else "rejected"),
            "partial": bool(self.partial),
            "rejection": self.rejection,
            "geometric_score": round(float(self.geometric_score), 4),
            "final_score": round(float(self.final_score), 4),
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
        }


def cluster_candidates(scores: List[CandidateScore], positions: Dict[int, tuple],
                       same_place_px: float) -> List[List[CandidateScore]]:
    """
    Group candidates by the physical map location they project to.

    Greedy nearest-centroid assignment, seeded by the strongest evidence
    first: each candidate joins the nearest existing cluster if its projected
    centre is within ``same_place_px`` of that cluster's running centroid,
    otherwise it starts a new one. Candidates with no projected position
    (too few matches to attempt a homography) each form their own singleton
    cluster - they carry no locational evidence to merge on.
    """
    ordered = sorted(scores, key=lambda s: s.geometric_score, reverse=True)
    clusters: List[dict] = []
    singletons: List[CandidateScore] = []

    for s in ordered:
        pos = positions.get(s.candidate_id)
        if pos is None:
            singletons.append(s)
            continue
        best_cluster, best_dist = None, None
        for c in clusters:
            d = float(np.hypot(pos[0] - c["centroid"][0], pos[1] - c["centroid"][1]))
            if d <= same_place_px and (best_dist is None or d < best_dist):
                best_cluster, best_dist = c, d
        if best_cluster is not None:
            best_cluster["members"].append(s)
            n = len(best_cluster["members"])
            sx = best_cluster["sum_x"] + pos[0]
            sy = best_cluster["sum_y"] + pos[1]
            best_cluster["sum_x"], best_cluster["sum_y"] = sx, sy
            best_cluster["centroid"] = (sx / n, sy / n)
        else:
            clusters.append({"members": [s], "sum_x": pos[0], "sum_y": pos[1],
                             "centroid": pos})

    groups = [c["members"] for c in clusters] + [[s] for s in singletons]
    return groups


def score_clusters(clusters: List[List[CandidateScore]]) -> List[ClusterScore]:
    """
    Turn each location cluster into a :class:`ClusterScore` using its
    *combined* evidence, then rank by final confidence.

    Inliers and raw matches are summed across every supporting tile - each
    overlapping tile is an independent look at the same ground truth, so more
    of them agreeing is itself additional evidence, captured by the
    ``consensus`` component. A location with one 150-inlier tile and a
    location with three tiles totalling 400 inliers are not equally certain.
    """
    if not clusters:
        return []

    raw: List[ClusterScore] = []
    for idx, members in enumerate(clusters):
        valid = [m for m in members if m.homography_valid]
        # Representative: a verified member if any, else a partial member
        # (near-miss), else the strongest of whatever is left. This is the
        # tile the UI shows for the cluster.
        pool = valid or [m for m in members if m.partial] or members
        representative = max(pool, key=lambda m: m.final_score or m.geometric_score)
        support = len(valid)

        if valid:
            raw_matches = sum(m.raw_matches for m in valid)
            inliers = sum(m.inliers for m in valid)
            inlier_ratio = inliers / max(raw_matches, 1)
            spatial_coverage = max(m.spatial_coverage for m in valid)
            weighted = [(m.reprojection_error, m.inliers) for m in valid
                       if m.reprojection_error is not None]
            total_w = sum(w for _, w in weighted)
            reprojection_error = (sum(e * w for e, w in weighted) / total_w
                                  if total_w > 0 else None)
            homography_valid, rejection, partial = True, None, False
        else:
            raw_matches = max((m.raw_matches for m in members), default=0)
            inliers, inlier_ratio, reprojection_error = 0, 0.0, None
            spatial_coverage = max((m.spatial_coverage for m in members), default=0.0)
            homography_valid, rejection = False, representative.rejection
            # Prefer a partial member's reason: it cleared every structural
            # gate and only missed a strength threshold - the most informative
            # thing we can say about a location we could not accept.
            partial_member = next((m for m in members if m.partial), None)
            partial = partial_member is not None
            if partial_member is not None:
                rejection = partial_member.rejection

        dino_similarity = max(m.dino_similarity for m in members)
        comp = {
            "retrieval": retrieval_score(dino_similarity),
            "inliers": inlier_score(inliers, inlier_ratio),
            "geometry": geometry_score(reprojection_error, representative.shear, homography_valid),
            "coverage": coverage_score(spatial_coverage),
            "consensus": consensus_score(support),
        }
        geo = (0.38 * comp["inliers"] + 0.28 * comp["geometry"]
              + 0.14 * comp["coverage"] + 0.20 * comp["consensus"])
        if not homography_valid:
            geo *= 0.25

        raw.append(ClusterScore(
            cluster_id=idx, tile_id=representative.tile_id,
            representative_id=representative.candidate_id,
            member_candidate_ids=[m.candidate_id for m in members],
            support=support, total_members=len(members),
            dino_similarity=dino_similarity, raw_matches=raw_matches, inliers=inliers,
            inlier_ratio=inlier_ratio, spatial_coverage=spatial_coverage,
            reprojection_error=reprojection_error, homography_valid=homography_valid,
            partial=partial, rejection=rejection, components=comp,
            geometric_score=_clamp01(geo),
        ))

    # Clusters are already spatially distinct by construction, so the
    # immediate neighbour in score order is always a genuine rival location -
    # no same-place exception needed here, unlike the per-tile score above.
    ordered = sorted(raw, key=lambda c: c.geometric_score, reverse=True)
    for i, c in enumerate(ordered):
        runner_geo = ordered[i + 1].geometric_score if i + 1 < len(ordered) else None
        c.components["ambiguity"] = ambiguity_score(c.geometric_score, runner_geo)
        c.final_score = _clamp01(sum(CLUSTER_WEIGHTS[k] * c.components.get(k, 0.0)
                                     for k in CLUSTER_WEIGHTS))
        if not c.homography_valid:
            c.final_score = min(c.final_score, 0.35)

    ordered = sorted(ordered, key=lambda c: c.final_score, reverse=True)
    for rank, c in enumerate(ordered, start=1):
        c.rank = rank
    return ordered


# --------------------------------------------------------------------------
# Cross-representation evidence (spec Phases 6 & 8)
# --------------------------------------------------------------------------
@dataclass
class RepresentationScore:
    """
    Measurable, per-representation verification evidence for one map location.

    Every field is a real geometric statistic - nothing here is a fabricated
    "AI confidence". ``weight`` is the fusion weight this representation carries
    (from config, normalised across the representations actually present).
    """
    representation: str
    retrieval_similarity: float = 0.0
    total_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: Optional[float] = None
    homography_plausible: bool = False
    geometric_score: float = 0.0
    weight: float = 0.0
    map_center: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        return {
            "representation": self.representation,
            "retrieval_similarity": round(float(self.retrieval_similarity), 4),
            "total_matches": int(self.total_matches),
            "inliers": int(self.inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "reprojection_error": (round(float(self.reprojection_error), 3)
                                   if self.reprojection_error is not None else None),
            "homography_plausible": bool(self.homography_plausible),
            "geometric_score": round(float(self.geometric_score), 4),
            "weight": round(float(self.weight), 4),
            "map_center": ([round(float(self.map_center[0]), 1),
                            round(float(self.map_center[1]), 1)]
                           if self.map_center is not None else None),
        }


def representation_geometric_score(hom: HomographyResult, similarity: float = 0.0) -> float:
    """Same measurable components as a candidate score, condensed to one number."""
    valid = bool(hom.ok and hom.plausible)
    err = float(hom.reprojection_error) if np.isfinite(hom.reprojection_error) else None
    comp = (0.45 * inlier_score(hom.inliers, hom.inlier_ratio)
            + 0.35 * geometry_score(err, hom.shear, valid)
            + 0.20 * coverage_score(hom.spatial_coverage))
    if not valid:
        comp *= 0.25
    return _clamp01(comp)


def normalised_weights(present: List[str]) -> Dict[str, float]:
    """
    Config fusion weights (RGB/structural/map/retrieval), renormalised over the
    representations actually available so they always sum to 1.0.
    """
    raw = {
        "rgb": max(0.0, settings.rgb_weight),
        "structural": max(0.0, settings.structural_weight),
        "map": max(0.0, settings.sat2map_weight),
        "retrieval": max(0.0, settings.retrieval_weight),
    }
    active = {k: raw.get(k, 0.0) for k in present}
    total = sum(active.values())
    if total <= 0:
        return {k: 1.0 / len(active) for k in active} if active else {}
    return {k: v / total for k, v in active.items()}


@dataclass
class ConsensusResult:
    """
    Do the independent representations place the drone in the same spot?

    ``reference`` is always the RGB/geometric branch - it has the highest
    safety priority, so agreement is measured as every other representation's
    distance from the RGB estimate, never a free-floating average.
    """
    reference: str = "rgb"
    agree: bool = True
    tolerance_px: float = 0.0
    max_disagreement_px: float = 0.0
    offsets_px: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    participating: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "agree": bool(self.agree),
            "tolerance_px": round(float(self.tolerance_px), 1),
            "max_disagreement_px": round(float(self.max_disagreement_px), 1),
            "offsets_px": {k: round(float(v), 1) for k, v in self.offsets_px.items()},
            "positions": {k: [round(float(p[0]), 1), round(float(p[1]), 1)]
                          for k, p in self.positions.items()},
            "participating": list(self.participating),
        }


def representation_consensus(positions: Dict[str, Tuple[float, float]],
                             tolerance_px: float,
                             reference: str = "rgb") -> ConsensusResult:
    """
    Compare every representation's projected map centre against the reference.

    ``agree`` is True when every other representation lands within
    ``tolerance_px`` of the reference (or when only the reference is present -
    a single branch cannot disagree with itself). Independent agreement is
    strong evidence; a translated-map branch that points somewhere else is a
    warning, not a vote to move the fix.
    """
    res = ConsensusResult(reference=reference, tolerance_px=float(tolerance_px),
                          positions=dict(positions),
                          participating=sorted(positions))
    ref = positions.get(reference)
    if ref is None or len(positions) < 2:
        return res
    max_off = 0.0
    for name, p in positions.items():
        if name == reference:
            continue
        d = float(np.hypot(p[0] - ref[0], p[1] - ref[1]))
        res.offsets_px[name] = d
        max_off = max(max_off, d)
    res.max_disagreement_px = max_off
    res.agree = max_off <= float(tolerance_px)
    return res


@dataclass
class FusedConfidence:
    """
    The reported ``overall`` confidence and its cross-representation breakdown
    (spec Phase 8).

    ``base`` is the RGB/geometric cluster score - the anchor. ``overall`` is
    ``base`` adjusted only by *verified* auxiliary representations: agreement
    adds up to ``consensus_bonus_cap`` (and only when the RGB homography
    passed), disagreement subtracts up to ``consensus_penalty_cap``. A branch
    that failed its own verification contributes nothing either way, so a
    missing or hallucinating Sat2Map output can never by itself move the fix.
    """
    overall: float
    base: float
    components: Dict[str, float] = field(default_factory=dict)   # per-representation geo score
    weights: Dict[str, float] = field(default_factory=dict)
    applied_bonus: float = 0.0
    applied_penalty: float = 0.0
    consensus: Optional[bool] = None
    corroborating: List[str] = field(default_factory=list)
    dissenting: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": round(float(self.overall), 4),
            "base_rgb": round(float(self.base), 4),
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "applied_bonus": round(float(self.applied_bonus), 4),
            "applied_penalty": round(float(self.applied_penalty), 4),
            "consensus": self.consensus,
            "corroborating": list(self.corroborating),
            "dissenting": list(self.dissenting),
        }


def fuse_confidence(base: float, rgb_homography_valid: bool,
                    rep_scores: List["RepresentationScore"],
                    consensus: Optional["ConsensusResult"]) -> FusedConfidence:
    """Blend measurable auxiliary evidence into the RGB-anchored confidence."""
    base = _clamp01(base)
    fused = FusedConfidence(overall=base, base=base)
    for s in rep_scores:
        fused.components[s.representation] = s.geometric_score
        fused.weights[s.representation] = s.weight

    verified = [s for s in rep_scores
                if s.representation != "rgb" and s.homography_plausible
                and s.map_center is not None]
    if not verified or consensus is None:
        fused.consensus = (consensus.agree if consensus is not None else None)
        return fused

    tol = max(float(consensus.tolerance_px), 1.0)
    bonus = penalty = 0.0
    for s in verified:
        off = consensus.offsets_px.get(s.representation)
        if off is None:
            continue
        if off <= tol:
            closeness = _clamp01(1.0 - off / tol)
            bonus += closeness * s.weight * s.geometric_score
            fused.corroborating.append(s.representation)
        else:
            over = _clamp01((off - tol) / tol)
            penalty += over * s.weight
            fused.dissenting.append(s.representation)

    # RGB is the safety anchor - no bonus can be claimed when its own
    # homography failed verification.
    fused.applied_bonus = (0.0 if not rgb_homography_valid
                           else _clamp01(bonus) * settings.consensus_bonus_cap)
    fused.applied_penalty = _clamp01(penalty) * settings.consensus_penalty_cap
    fused.overall = _clamp01(base * (1.0 + fused.applied_bonus - fused.applied_penalty))
    fused.consensus = consensus.agree
    return fused


def refine_status(status: MatchStatus, explanation: str, fused: FusedConfidence,
                  consensus: Optional["ConsensusResult"], rgb_homography_valid: bool,
                  aux_verified: int) -> tuple[MatchStatus, str]:
    """
    Re-grade the verdict against the fused confidence (spec Phase 8: NO_MATCH
    when evidence is insufficient; agreement can promote, disagreement demotes).

    The RGB branch keeps priority: if its homography never verified, nothing
    here can manufacture a match. Promotion from LOW to MATCH needs the RGB
    homography valid *and* an independent representation actually corroborating
    the location - never an auxiliary branch acting alone.
    """
    if not rgb_homography_valid or status in (MatchStatus.NO_MATCH, MatchStatus.AMBIGUOUS):
        return status, explanation

    x = fused.overall
    if x < settings.low_confidence:
        return (MatchStatus.NO_MATCH,
                f"{explanation} Cross-representation fusion lowered confidence to "
                f"{x:.0%}, below the {settings.low_confidence:.0%} no-match threshold.")

    if status == MatchStatus.MATCH_FOUND:
        if x < settings.match_confidence:
            return (MatchStatus.LOW_CONFIDENCE,
                    f"{explanation} Fused confidence {x:.0%} is below the "
                    f"{settings.match_confidence:.0%} reporting threshold.")
        if consensus is not None and not consensus.agree:
            return (MatchStatus.LOW_CONFIDENCE,
                    f"{explanation} Independent representations disagree on the "
                    f"location (max {consensus.max_disagreement_px:.0f}px apart) - "
                    "treat the position as indicative only.")
        if fused.corroborating:
            return (status, f"{explanation} Corroborated by "
                    f"{', '.join(fused.corroborating)} ({x:.0%} fused confidence).")
        return status, explanation

    if status == MatchStatus.LOW_CONFIDENCE:
        if (x >= settings.match_confidence and aux_verified >= 1
                and consensus is not None and consensus.agree and fused.corroborating):
            return (MatchStatus.MATCH_FOUND,
                    f"Verified geometry corroborated by {', '.join(fused.corroborating)} "
                    f"at {x:.0%} fused confidence.")
        return status, explanation

    return status, explanation


def decide(scores: List[ClusterScore]) -> tuple[MatchStatus, str]:
    """
    Turn the ranked locations into a status plus a human-readable explanation.

    The order of the checks matters: structural failure (no valid homography)
    outranks ambiguity, which outranks a merely low score. Unlike the old
    per-tile version, no same-place exception is needed - clustering has
    already merged overlapping tiles that agree, so any runner-up here is a
    genuinely different part of the map.
    """
    if not scores:
        return MatchStatus.NO_MATCH, "No candidate regions could be evaluated."

    best = scores[0]
    valid = [s for s in scores if s.homography_valid]

    if not valid:
        reason = best.rejection or "no_valid_homography"
        partial = next((c for c in scores if c.partial), None)
        if partial is not None:
            return (MatchStatus.NO_MATCH,
                    f"Partial match on candidate tile {partial.tile_id}: the drone frame "
                    f"projects onto a plausible location but the evidence is too weak to "
                    f"accept ({partial.rejection}). Not reported as a fix.")
        return (MatchStatus.NO_MATCH,
                f"No candidate passed geometric verification (best rejection: {reason}). "
                "The drone frame does not appear to overlap this reference map.")

    if not best.homography_valid:
        best = valid[0]

    if best.final_score < settings.low_confidence:
        return (MatchStatus.NO_MATCH,
                f"Best verified location scored {best.final_score:.2f}, below the "
                f"no-match threshold of {settings.low_confidence:.2f}.")

    others = [s for s in valid if s.cluster_id != best.cluster_id]
    if others:
        runner = others[0]
        gap = best.final_score - runner.final_score
        if gap < settings.ambiguity_gap and runner.final_score >= settings.low_confidence:
            return (MatchStatus.AMBIGUOUS,
                    f"A region near tile {best.tile_id} ({best.final_score:.0%}) and a "
                    f"separate region near tile {runner.tile_id} ({runner.final_score:.0%}) "
                    f"score within {gap:.0%} of each other - the scene is not uniquely "
                    "identifiable.")

    support_note = f", corroborated by {best.support} overlapping tiles" if best.support > 1 else ""
    if best.final_score >= settings.match_confidence:
        return (MatchStatus.MATCH_FOUND,
                f"Verified with {best.inliers} combined RANSAC inliers "
                f"({best.inlier_ratio:.0%} of {best.raw_matches} matches){support_note} "
                f"and {best.spatial_coverage:.0%} spatial coverage.")

    return (MatchStatus.LOW_CONFIDENCE,
            f"A geometrically valid match exists at {best.final_score:.0%} confidence"
            f"{support_note}, below the reporting threshold. Treat the position as "
            "indicative only.")


def status_message(status: MatchStatus) -> str:
    """Presenter-facing wording (spec section 45): estimates, never guarantees."""
    return {
        MatchStatus.MATCH_FOUND: "Estimated visual position recovered.",
        MatchStatus.LOW_CONFIDENCE: "VISUAL LOCALIZATION UNRELIABLE - low confidence.",
        MatchStatus.AMBIGUOUS: "Ambiguous - multiple map regions explain this frame.",
        MatchStatus.NO_MATCH: "No corresponding region found in the reference map.",
    }[status]
