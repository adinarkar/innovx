"""
Localisation orchestration (spec sections 6, 35 and 59).

    Reference map -> multi-scale tiling -> DINOv2 embeddings
    Drone frame   -> preprocessing -> embedding -> top-K retrieval
                  -> SuperPoint -> LightGlue -> RANSAC -> homography
                  -> geometric verification -> confidence -> position

Coordinate bookkeeping is the subtle part.  Features are extracted at a fixed
working resolution, so every homography is solved in *working* pixels and then
rebased onto full-resolution drone pixels and absolute map pixels before any
result leaves this module.
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.localization import dino, homography as hg, visualization as viz
from app.localization import semantic, tiling
from app.localization.domain_translation import get_translation_engine
from app.localization.imaging import fit_long_edge, imread
from app.localization.confidence import (CandidateScore, ClusterScore, ConsensusResult,
                                         MatchStatus, RepresentationScore,
                                         cluster_candidates, decide, evaluate_candidate,
                                         finalize_scores, fuse_confidence, normalised_weights,
                                         refine_status, representation_consensus,
                                         representation_geometric_score,
                                         score_clusters, status_message)
from app.localization.lightglue import (MatchResult, extract_features, match_features,
                                        rotate_image, unrotate_points)
from app.localization.preprocessing import (CameraCalibration, PreprocessResult,
                                            run_preprocessing)
from app.localization.superpoint import FeatureSet
from app.localization.tiling import Tile, plan_tiles
from app.models.loader import probe_capabilities
from app.store import MapRecord

log = get_logger(__name__)

ProgressFn = Callable[[str, str, str, Optional[str]], None]

# Stage keys mirrored by the frontend progress animation (spec section 35).
STAGES = [
    ("prepare", "Preparing reference map..."),
    ("preprocess", "Processing drone frame..."),
    ("embed", "Computing AI embeddings..."),
    ("retrieve", "Searching map for candidate regions..."),
    ("features", "Extracting local features..."),
    ("match", "Matching structural features..."),
    ("verify", "Running geometric verification..."),
    ("position", "Estimating position..."),
]


def _noop(*_args, **_kwargs) -> None:
    return None


# --------------------------------------------------------------------------
# Reference map indexing
# --------------------------------------------------------------------------
def build_map_index(record: MapRecord, force: bool = False) -> MapRecord:
    """
    Tile the reference map and embed every tile once (spec section 7).

    Results are cached on disk so repeated localisation requests against the
    same map skip this entirely.
    """
    if not force and record.load_cache():
        return record

    started = time.time()
    try:
        map_img = imread(record.path)
        h, w = map_img.shape[:2]
        record.width, record.height = w, h

        tiles = plan_tiles(w, h)
        engine = dino.get_engine()

        crops: List[np.ndarray] = []
        for t in tiles:
            crop = t.crop(map_img)
            if crop.size == 0:
                crop = np.zeros((8, 8, 3), np.uint8)
            crops.append(cv2.resize(crop, (dino.DINO_INPUT, dino.DINO_INPUT),
                                    interpolation=cv2.INTER_AREA))

        record.tiles = tiles
        record.embeddings = engine.embed_batch(crops)
        record.embedding_backend = engine.backend
        record.embedding_status = "ready"
        record.index_seconds = time.time() - started
        record.save_cache()
        log.info("Indexed map %s: %d tiles in %.1fs via %s.",
                 record.map_id, len(tiles), record.index_seconds, engine.backend)
    except Exception as exc:
        record.embedding_status = "failed"
        record.error = str(exc)
        log.exception("Map indexing failed for %s", record.map_id)
    return record


# --------------------------------------------------------------------------
@dataclass
class CandidateEvaluation:
    """One verified (or rejected) candidate, in map coordinates."""
    candidate_id: int
    tile: Optional[Tile]
    similarity: float
    score: CandidateScore
    hom: hg.HomographyResult
    match: MatchResult
    rotation: int = 0
    H_map: Optional[np.ndarray] = None          # drone(original px) -> map px
    polygon: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    tile_image: Optional[np.ndarray] = None
    tile_scale: float = 1.0
    query_pts_work: Optional[np.ndarray] = None
    target_pts_work: Optional[np.ndarray] = None
    inlier_map_points: Optional[np.ndarray] = None   # inliers in map pixels
    source: str = "tile"                        # tile | global
    retrieval_sources: List[str] = field(default_factory=lambda: ["rgb"])
    retrieval_similarity_by_source: Dict[str, float] = field(default_factory=dict)


def _prepare_work_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Resize to the configured working long edge, returning the scale used."""
    return fit_long_edge(img, settings.work_size)


def _match_pair(query_img: np.ndarray, query_feats: FeatureSet,
                target_img: np.ndarray,
                extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
                ) -> Tuple[MatchResult, FeatureSet]:
    target_feats = extract_fn(target_img)
    result = match_features(query_feats, target_feats, query_img, target_img)
    return result, target_feats


def _evaluate(query_img: np.ndarray, query_feats: FeatureSet,
              target_img: np.ndarray, rotation: int = 0,
              canonical_size: Optional[Tuple[int, int]] = None,
              extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
              ) -> Tuple[hg.HomographyResult, MatchResult]:
    """
    Match then verify, always returning query points in the *canonical*
    (unrotated, working-resolution) drone frame.

    ``query_img``/``query_feats`` must describe the same image - for the
    rotation search that is the rotated copy - while ``canonical_size`` is the
    unrotated ``(width, height)`` the points are mapped back into.
    """
    match, _ = _match_pair(query_img, query_feats, target_img, extract_fn)
    cw, ch = canonical_size or (query_img.shape[1], query_img.shape[0])
    if rotation:
        q_pts = unrotate_points(match.query_pts, rotation, cw, ch)
        match = MatchResult(q_pts.astype(np.float32), match.target_pts,
                            match.confidence, match.backend)
    hom = hg.estimate(match.query_pts, match.target_pts, (cw, ch))
    return hom, match


def _best_rotation(work_img: np.ndarray, query_feats: FeatureSet, tile_img: np.ndarray,
                   work_w: int, work_h: int,
                   extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
                   ) -> Tuple[hg.HomographyResult, MatchResult, int]:
    """
    Evaluate a candidate tile at every orientation the query might be in and
    keep the strongest verified result (spec section 50).

    Trying every rotation - not only when the upright attempt fails - avoids
    understating a genuinely correct candidate whose true confidence is
    limited by a few degrees of heading offset rather than a wrong location.
    Ranked by (passed verification, inlier count) so a valid geometry always
    beats an invalid one regardless of raw inlier counts.
    """
    hom0, match0 = _evaluate(work_img, query_feats, tile_img, rotation=0, extract_fn=extract_fn)
    attempts = [(0, hom0, match0)]

    if settings.rotation_search:
        run_all = settings.rotation_search_always
        needs_search = run_all or not (hom0.ok and hom0.plausible)
        if needs_search:
            for k in (1, 2, 3):
                rot_img = rotate_image(work_img, k)
                rot_feats = extract_fn(rot_img)
                cand_hom, cand_match = _evaluate(rot_img, rot_feats, tile_img, rotation=k,
                                                 canonical_size=(work_w, work_h),
                                                 extract_fn=extract_fn)
                attempts.append((k, cand_hom, cand_match))

    def rank(attempt):
        _, hom, _ = attempt
        return (1 if (hom.ok and hom.plausible) else 0, hom.inliers)

    rotation_used, hom, match = max(attempts, key=rank)
    return hom, match, rotation_used


# --------------------------------------------------------------------------
# Cross-domain representations (spec Phases 3-4)
# --------------------------------------------------------------------------
def _build_representations(pre: PreprocessResult, stage: Callable
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], dict]:
    """
    Build the auxiliary representations of the drone frame.

    ``"rgb"`` (the existing ``pre.matching_input``) is always present and is
    the only representation the primary geometric branch depends on. The
    optional ``"structural"`` and ``"map"`` branches are each wrapped so a
    failure is logged and reported as ``skipped`` - never fatal, exactly as
    the spec's failure-safety section requires.
    """
    representations: Dict[str, np.ndarray] = {"rgb": pre.matching_input}
    meta: Dict[str, dict] = {"rgb": {"state": "ready", "backend": "clahe"}}
    stats: dict = {}

    # --- structural branch ---
    if settings.structural_matching_enabled:
        t0 = time.time()
        try:
            struct = semantic.build_structural_representation(pre.corrected)
            representations["structural"] = struct.structural
            representations["_structural_obj"] = struct  # renders only, popped later
            stats.update(struct.stats())
            meta["structural"] = {"state": "ready", "backend": struct.backend,
                                  "seconds": round(time.time() - t0, 3)}
            log.info("Structural representation built in %.3fs (%s).",
                     time.time() - t0, struct.backend)
        except Exception as exc:
            meta["structural"] = {"state": "skipped", "error": str(exc)}
            log.warning("Structural representation failed (%s) - skipping branch.", exc)
        _st = "done" if meta.get("structural", {}).get("state") == "ready" else "skipped"
        stage("structure", "Building structural representation...", _st,
              meta.get("structural", {}).get("error"))
    else:
        meta["structural"] = {"state": "skipped", "error": "disabled"}
        stage("structure", "Building structural representation...", "skipped",
              "STRUCTURAL_MATCHING_ENABLED is false")

    # --- translated-map branch ---
    engine = get_translation_engine()
    if engine.available:
        t0 = time.time()
        try:
            translated = engine.translate(pre.corrected)
            representations["map"] = translated
            meta["map"] = {"state": "ready", "backend": "sat2map-unet",
                           "seconds": round(time.time() - t0, 3)}
        except Exception as exc:
            meta["map"] = {"state": "skipped", "error": str(exc)}
            log.warning("Sat2Map translation failed (%s) - skipping branch.", exc)
        _mt = "done" if meta.get("map", {}).get("state") == "ready" else "skipped"
        stage("translate", "Generating map-style representation...", _mt,
              meta.get("map", {}).get("error"))
    else:
        meta["map"] = {"state": "skipped", "error": engine.status}
        stage("translate", "Generating map-style representation...", "skipped",
              "Map translation model not installed")

    return representations, meta, stats


# Which auxiliary representations may contribute their own retrieval shortlist
# (spec Phase 5). RGB is always the primary; these are unioned in on top.
_RETRIEVAL_AUX_ORDER = ("map", "structural")


def _multi_domain_retrieval(engine, record: MapRecord, rgb_embedding: np.ndarray,
                            representations: Dict[str, np.ndarray],
                            primary_top_k: int,
                            tile_indices: Optional[np.ndarray] = None
                            ) -> Tuple[List[dict], Dict[str, int]]:
    """
    Coarse candidate retrieval, optionally unioned across representations.

    DINOv2 remains the *only* coarse search stage - LightGlue never runs over
    every tile. The RGB representation always drives the primary shortlist; when
    ``MAP_DOMAIN_RETRIEVAL_ENABLED`` is set, the generated map-like image (and
    the structural image, if present) each contribute their own ``MAP_DOMAIN_TOP_K``
    candidates. The lists are unioned by tile, de-duplicated, and capped at
    ``CANDIDATE_UNION_MAX``. Each surviving hit records which representations
    retrieved it and its best similarity, so downstream fusion can reward
    cross-domain agreement.

    ``tile_indices``, when given (an operator-chosen search region), restricts
    every retrieval bank to that subset of ``record.tiles``/``record.embeddings``
    - hit indices are remapped back to the full tile list before returning, so
    callers never need to know retrieval was narrowed.
    """
    bank = record.embeddings if tile_indices is None else record.embeddings[tile_indices]

    def _remap(hits: List[dict]) -> List[dict]:
        if tile_indices is None:
            return hits
        for h in hits:
            h["index"] = int(tile_indices[h["index"]])
        return hits

    primary = _remap(dino.top_k(rgb_embedding, bank, primary_top_k))
    merged: Dict[int, dict] = {}
    for rank, hit in enumerate(primary):
        merged[hit["index"]] = {
            "index": hit["index"], "similarity": hit["similarity"],
            "sources": ["rgb"], "similarity_by_source": {"rgb": hit["similarity"]},
            "_order": (0, rank),
        }
    per_source_counts = {"rgb": len(primary)}

    if settings.map_domain_retrieval_enabled:
        for si, name in enumerate(_RETRIEVAL_AUX_ORDER, start=1):
            img = representations.get(name)
            if img is None:
                continue
            try:
                emb = engine.embed(img)
                hits = _remap(dino.top_k(emb, bank, settings.map_domain_top_k))
            except Exception as exc:  # failure-safe: aux retrieval never blocks
                log.warning("%s-domain retrieval failed (%s) - skipping.", name, exc)
                continue
            per_source_counts[name] = len(hits)
            for rank, hit in enumerate(hits):
                entry = merged.get(hit["index"])
                if entry is None:
                    merged[hit["index"]] = {
                        "index": hit["index"], "similarity": hit["similarity"],
                        "sources": [name],
                        "similarity_by_source": {name: hit["similarity"]},
                        "_order": (si, rank),
                    }
                else:
                    entry["sources"].append(name)
                    entry["similarity_by_source"][name] = hit["similarity"]
                    entry["similarity"] = max(entry["similarity"], hit["similarity"])

    ordered = sorted(merged.values(), key=lambda e: e.pop("_order"))
    ordered = ordered[:max(primary_top_k, settings.candidate_union_max)]
    return ordered, per_source_counts


# --------------------------------------------------------------------------
def localize(record: MapRecord, drone_path: Path, job_dir: Path,
             calibration: Optional[CameraCalibration] = None,
             progress: Optional[ProgressFn] = None,
             top_k: Optional[int] = None,
             search_region: Optional[Tuple[float, float, float, float]] = None,
             feature_params: Optional[Dict[str, float]] = None) -> dict:
    """
    Full localisation for one drone frame.  Always returns a structured result
    - including an explicit NO_MATCH - rather than raising for a poor match.

    ``search_region``, when given, is a ``(x, y, width, height)`` map-pixel
    rectangle that narrows retrieval to the tiles it intersects - see
    ``tiling.tiles_in_region``. It never touches map indexing/caching, only
    which already-embedded tiles this request considers.

    ``feature_params``, when given, resolves the keypoint-strength config
    (``max_keypoints``, ``superpoint_detection_threshold``,
    ``superpoint_nms_radius``, ``sift_contrast_threshold``,
    ``sift_edge_threshold``) for this job *once*, up front, and every
    extraction call in this run uses that fixed snapshot via ``extract_fn``
    - never the shared ``settings`` singleton, which a different,
    concurrently-running request could mutate mid-flight (this is what backs
    the frontend's Efficient Matching toggle).
    """
    progress = progress or _noop
    top_k = top_k or settings.top_k_candidates
    t_start = time.time()
    timings: Dict[str, float] = {}
    fp = feature_params or {}
    extract_fn = functools.partial(
        extract_features,
        max_keypoints=fp.get("max_keypoints"),
        superpoint_detection_threshold=fp.get("superpoint_detection_threshold"),
        superpoint_nms_radius=fp.get("superpoint_nms_radius"),
        sift_contrast_threshold=fp.get("sift_contrast_threshold"),
        sift_edge_threshold=fp.get("sift_edge_threshold"),
    )

    def stage(key: str, label: str, state: str = "running", detail: Optional[str] = None):
        progress(key, label, state, detail)

    # ---- 1. reference map ------------------------------------------------
    stage(*STAGES[0])
    t0 = time.time()
    if record.embedding_status != "ready":
        build_map_index(record)
    if record.embedding_status != "ready":
        raise RuntimeError(record.error or "Reference map could not be indexed.")
    map_img = imread(record.path)
    map_h, map_w = map_img.shape[:2]
    timings["prepare"] = time.time() - t0
    stage("prepare", STAGES[0][1], "done", f"{len(record.tiles)} candidate regions ready")

    # ---- 2. drone preprocessing -----------------------------------------
    stage(*STAGES[1])
    t0 = time.time()
    drone_img = imread(drone_path)
    dh, dw = drone_img.shape[:2]
    if dw * dh > map_w * map_h:
        raise ValueError("The drone image is larger than the reference map. "
                         "Upload a wider-area reference map.")
    pre: PreprocessResult = run_preprocessing(drone_img, calibration)
    work_img, work_scale = _prepare_work_image(pre.matching_input)
    work_h, work_w = work_img.shape[:2]
    timings["preprocess"] = time.time() - t0
    stage("preprocess", STAGES[1][1], "done",
          f"{dw}x{dh} -> {work_w}x{work_h} working resolution")

    # ---- 2b. auxiliary cross-domain representations --------------------
    t0 = time.time()
    representations, rep_meta, rep_stats = _build_representations(pre, stage)
    structural_obj = representations.pop("_structural_obj", None)
    timings["representations"] = time.time() - t0

    # ---- 3. query embedding ---------------------------------------------
    stage(*STAGES[2])
    t0 = time.time()
    engine = dino.get_engine()
    query_embedding = engine.embed(pre.matching_input)
    timings["embed"] = time.time() - t0
    stage("embed", STAGES[2][1], "done", f"backend: {engine.backend}")

    # ---- 4. retrieval (optionally multi-domain / region-limited) --------
    stage(*STAGES[3])
    t0 = time.time()
    tile_indices = None
    region_note = ""
    if search_region is not None:
        tile_indices = tiling.tiles_in_region(record.tiles, search_region)
        if tile_indices.size == 0:
            log.warning("search_region matched no tiles for map %s - searching the full map.",
                       record.map_id)
            tile_indices = None
        else:
            region_note = f" (region-limited from {len(record.tiles)})"
    retrieved, retrieval_counts = _multi_domain_retrieval(
        engine, record, query_embedding, representations, top_k, tile_indices)
    timings["retrieve"] = time.time() - t0
    searched = len(record.tiles) if tile_indices is None else len(tile_indices)
    if len(retrieval_counts) > 1:
        detail = (f"{len(retrieved)} unioned of {searched} tiles{region_note} ("
                  + ", ".join(f"{k}:{v}" for k, v in retrieval_counts.items()) + ")")
    else:
        detail = f"top {len(retrieved)} of {searched} tiles{region_note}"
    stage("retrieve", STAGES[3][1], "done", detail)

    # ---- 5. query features ----------------------------------------------
    stage(*STAGES[4])
    t0 = time.time()
    query_feats = extract_fn(work_img)
    timings["features"] = time.time() - t0
    stage("features", STAGES[4][1], "done",
          f"{query_feats.count} keypoints ({query_feats.backend})")

    # ---- 6/7. match + verify each candidate ------------------------------
    stage(*STAGES[5])
    t0 = time.time()
    evaluations: List[CandidateEvaluation] = []
    for cand_idx, hit in enumerate(retrieved):
        tile = record.tiles[hit["index"]]
        crop = tile.crop(map_img)
        if crop.size == 0:
            continue
        tile_img, tile_scale = _prepare_work_image(crop)

        hom, match, rotation_used = _best_rotation(work_img, query_feats, tile_img,
                                                   work_w, work_h, extract_fn)

        score = evaluate_candidate(cand_idx + 1, tile.tile_id, hit["similarity"],
                                   hom, rotation_used)
        ev = CandidateEvaluation(cand_idx + 1, tile, hit["similarity"], score, hom,
                                 match, rotation_used, tile_image=tile_img,
                                 tile_scale=tile_scale,
                                 query_pts_work=match.query_pts,
                                 target_pts_work=match.target_pts,
                                 retrieval_sources=list(hit.get("sources", ["rgb"])),
                                 retrieval_similarity_by_source=dict(
                                     hit.get("similarity_by_source", {})))

        if hom.ok and hom.H is not None:
            # working-drone -> working-tile  =>  original-drone -> map pixels
            H_full = hg.scale_homography(hom.H, work_scale, tile_scale)
            ev.H_map = hg.translate_homography(H_full, tile.x, tile.y)
            ev.polygon = hg.transform_points(ev.H_map, hg.frame_corners(dw, dh))
            ev.center = hg.transform_points(ev.H_map, np.array([[dw / 2.0, dh / 2.0]]))[0]
            ev.inlier_map_points = _project_inliers(ev, work_scale)
        evaluations.append(ev)
    timings["match"] = time.time() - t0
    stage("match", STAGES[5][1], "done", f"{len(evaluations)} candidates matched")

    # ---- 8. global fallback ---------------------------------------------
    stage(*STAGES[6])
    t0 = time.time()
    verified = [e for e in evaluations if e.hom.ok and e.hom.plausible]
    if settings.global_fallback and not verified:
        fallback = _global_fallback(map_img, work_img, query_feats, dw, dh, work_scale,
                                    extract_fn)
        if fallback is not None:
            evaluations.append(fallback)
            log.info("Global fallback recovered a match with %d inliers.",
                     fallback.hom.inliers)

    by_id = {e.score.candidate_id: e for e in evaluations}

    # Overlapping tiles routinely describe the same place. Rather than score
    # tiles individually and then worry two of them are "too close", group
    # them by the map location they actually project to and let the location
    # with the strongest *combined* evidence win (spec: decide from
    # differentiated parts of the map, not from raw overlapping tiles).
    positions = {e.score.candidate_id: (float(e.center[0]), float(e.center[1]))
                 for e in evaluations if e.center is not None}
    same_place_px = _same_place_radius(evaluations, dw, dh)

    # Per-tile diagnostics for the Candidate Analysis UI - unaffected by
    # clustering, still ranks every individual tile on its own merits.
    diagnostic_scores = finalize_scores([e.score for e in evaluations], positions, same_place_px)

    clusters = cluster_candidates([e.score for e in evaluations], positions, same_place_px)
    cluster_scores = score_clusters(clusters)
    status, explanation = decide(cluster_scores)
    timings["verify"] = time.time() - t0
    stage("verify", STAGES[6][1], "done", explanation)

    # ---- 9. position + renders ------------------------------------------
    stage(*STAGES[7])
    t0 = time.time()
    # Prefer a verified cluster; if none verified, fall back to the best
    # *partial* one (structural gates passed, only the strength thresholds
    # missed) so the "Matched features" view and result map show the actual
    # near-miss tile rather than an arbitrary rejected one. Still NO_MATCH.
    best_cluster = (
        next((c for c in cluster_scores if c.homography_valid), None)
        or next((c for c in cluster_scores if c.partial), None)
        or (cluster_scores[0] if cluster_scores else None))
    best = by_id.get(best_cluster.representative_id) if best_cluster else None
    accepted = status in (MatchStatus.MATCH_FOUND, MatchStatus.LOW_CONFIDENCE)

    # The reported position and outline are a confidence-weighted average
    # across every tile supporting the winning location, not just the single
    # best tile - corroborating tiles refine the estimate as well as the score.
    agg_center, agg_polygon = (
        _aggregate_geometry(best_cluster.member_candidate_ids, by_id)
        if best_cluster is not None else (None, None))
    timings["position"] = time.time() - t0

    # ---- 9b. per-representation matching + consensus (spec Phases 6 & 8) --
    t0 = time.time()
    rep_scores, consensus = _cross_representation(
        representations, structural_obj, cluster_scores, best_cluster, best,
        by_id, agg_center, same_place_px, work_w, work_h, dw, dh, stage, extract_fn)

    # ---- cross-domain confidence fusion + verdict re-grade (spec Phase 8) --
    rgb_valid = bool(best_cluster.homography_valid) if best_cluster else False
    aux_verified = sum(1 for s in rep_scores
                       if s.representation != "rgb" and s.homography_plausible)
    base_conf = float(best_cluster.final_score) if best_cluster else 0.0
    fused = fuse_confidence(base_conf, rgb_valid, rep_scores, consensus)
    status, explanation = refine_status(status, explanation, fused, consensus,
                                        rgb_valid, aux_verified)
    accepted = status in (MatchStatus.MATCH_FOUND, MatchStatus.LOW_CONFIDENCE)
    timings["consensus"] = time.time() - t0

    t0 = time.time()
    renders = _render_all(job_dir, record, map_img, drone_img, pre, work_img,
                          query_feats, evaluations, best, accepted,
                          agg_polygon, agg_center, structural_obj, representations)
    result = _build_result(record, status, explanation, best, best_cluster, diagnostic_scores,
                           evaluations, renders, drone_img, pre, query_feats,
                           map_w, map_h, timings, t_start, engine,
                           agg_center, agg_polygon, cluster_scores, fused.overall)
    result["confidence_breakdown"] = fused.to_dict()
    result["representation_scores"] = [s.to_dict() for s in rep_scores]
    result["consensus"] = consensus.to_dict() if consensus is not None else None
    result["cross_representation_agreement"] = (
        bool(consensus.agree) if consensus is not None else None)
    result["representations"] = {
        "available": [k for k in representations if k in ("rgb", "structural", "map")],
        "branches": rep_meta,
        "renders": {
            "original": renders.get("original"),
            "preprocessed": renders.get("keypoints") or renders.get("enhanced"),
            "structural": renders.get("structural_query"),
            "translated_map": renders.get("translated_map"),
            "matches": renders.get("matches_inliers"),
            "final_overlay": renders.get("result_map"),
        },
    }
    result["preprocessing"] = result["preprocessing"] | rep_stats
    timings["render"] = time.time() - t0
    stage("position", STAGES[7][1], "done", status_message(status))
    result["timings"] = {k: round(v, 3) for k, v in timings.items()}
    result["processing_time"] = round(time.time() - t_start, 2)
    return result


# --------------------------------------------------------------------------
def _global_fallback(map_img: np.ndarray, work_img: np.ndarray,
                     query_feats: FeatureSet, dw: int, dh: int,
                     work_scale: float,
                     extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
                     ) -> Optional[CandidateEvaluation]:
    """
    Last resort: match the drone frame against the whole (downscaled) map.

    Tiling can miss a frame that straddles four tiles, so a single direct
    attempt is cheaper than densifying the grid.  The result is verified by
    exactly the same geometric gates as any tile candidate.
    """
    map_small, map_scale = fit_long_edge(map_img, max(settings.work_size * 2, 1280))
    hom, match = _evaluate(work_img, query_feats, map_small, rotation=0, extract_fn=extract_fn)
    if not (hom.ok and hom.plausible):
        return None

    score = evaluate_candidate(999, -1, 0.0, hom, 0)
    ev = CandidateEvaluation(999, None, 0.0, score, hom, match, 0,
                             tile_image=map_small, source="global",
                             query_pts_work=match.query_pts,
                             target_pts_work=match.target_pts,
                             retrieval_sources=["global"])
    ev.H_map = hg.scale_homography(hom.H, work_scale, map_scale)
    ev.polygon = hg.transform_points(ev.H_map, hg.frame_corners(dw, dh))
    ev.center = hg.transform_points(ev.H_map, np.array([[dw / 2.0, dh / 2.0]]))[0]
    ev.inlier_map_points = _project_inliers(ev, work_scale)
    return ev


def _same_place_radius(evaluations: List["CandidateEvaluation"],
                       dw: int, dh: int) -> float:
    """
    How far two candidate centres may sit apart and still be calling the same
    location: a third of the projected frame's diagonal.

    Derived from the winning projection where possible so the radius scales
    with the actual ground footprint rather than a hard-coded pixel count.
    """
    for ev in evaluations:
        if ev.polygon is not None and ev.hom.ok and ev.hom.plausible:
            poly = np.asarray(ev.polygon, np.float64)
            diag = float(np.linalg.norm(poly.max(axis=0) - poly.min(axis=0)))
            if np.isfinite(diag) and diag > 0:
                return diag / 3.0
    return float(np.hypot(dw, dh)) / 3.0


def _aggregate_geometry(member_candidate_ids: List[int],
                        by_id: Dict[int, "CandidateEvaluation"]
                        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Confidence-weighted average of the centre and outline across every tile
    that supports the winning location.

    Each overlapping tile solves its own homography independently, so its
    centre/corner estimates carry independent noise; weighting by inlier
    count and averaging reduces that noise instead of committing to whichever
    single tile happened to be evaluated first. With one supporting tile this
    reduces to that tile's own estimate.
    """
    members = [by_id[cid] for cid in member_candidate_ids
              if cid in by_id and by_id[cid].center is not None
              and by_id[cid].hom.ok and by_id[cid].hom.plausible]
    if not members:
        return None, None

    weights = np.array([max(m.hom.inliers, 1) for m in members], dtype=np.float64)
    weights = weights / weights.sum()

    centers = np.array([m.center for m in members], dtype=np.float64)
    center = (centers * weights[:, None]).sum(axis=0)

    polygons = np.array([m.polygon for m in members], dtype=np.float64)  # (N, 4, 2)
    polygon = (polygons * weights[:, None, None]).sum(axis=0)
    return center, polygon


def _cross_representation(representations: Dict[str, np.ndarray], structural_obj,
                         cluster_scores: List[ClusterScore],
                         best_cluster: Optional[ClusterScore],
                         best: Optional["CandidateEvaluation"],
                         by_id: Dict[int, "CandidateEvaluation"],
                         agg_center: Optional[np.ndarray], same_place_px: float,
                         work_w: int, work_h: int, dw: int, dh: int, stage,
                         extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
                         ) -> Tuple[List[RepresentationScore], Optional[ConsensusResult]]:
    """
    Independently match the structural / map representations against the
    winning location's candidate tiles and check whether they agree with the
    RGB fix (spec Phases 6 & 8).

    This produces *evidence*, not a new decision: the returned scores and
    consensus flag are reported and (in the fusion step) may nudge confidence,
    but a disagreeing translated-map branch can never move or create a fix -
    the RGB geometric branch keeps sole authority over the position.
    """
    if best_cluster is None or best is None:
        stage("consensus", "Checking representation agreement...", "skipped",
              "no verified location to corroborate")
        return [], None

    try:
        refine_ids: List[int] = []
        for c in cluster_scores[:2]:
            refine_ids.extend(c.member_candidate_ids)
        members = [by_id[i] for i in dict.fromkeys(refine_ids) if i in by_id]
        valid = [m for m in members if m.hom.ok and m.hom.plausible]
        members = (valid or members)
        members.sort(key=lambda m: m.score.geometric_score, reverse=True)
        members = members[:max(1, settings.representation_matching_top_n)]

        estimates = _representation_estimates(representations, structural_obj, members,
                                              work_w, work_h, dw, dh, extract_fn)

        present = ["rgb"] + sorted(estimates) + ["retrieval"]
        weights = normalised_weights(present)

        ref_center = agg_center if agg_center is not None else best.center
        positions: Dict[str, Tuple[float, float]] = {}
        if ref_center is not None:
            positions["rgb"] = (float(ref_center[0]), float(ref_center[1]))

        scores: List[RepresentationScore] = [RepresentationScore(
            representation="rgb",
            retrieval_similarity=float(best_cluster.dino_similarity),
            total_matches=int(best_cluster.raw_matches),
            inliers=int(best_cluster.inliers),
            inlier_ratio=float(best_cluster.inlier_ratio),
            reprojection_error=best_cluster.reprojection_error,
            homography_plausible=bool(best_cluster.homography_valid),
            geometric_score=float(best_cluster.geometric_score),
            weight=float(weights.get("rgb", 0.0)),
            map_center=positions.get("rgb"),
        )]

        for name, (hom, center) in estimates.items():
            if center is not None and hom.ok and hom.plausible:
                positions[name] = (float(center[0]), float(center[1]))
            scores.append(RepresentationScore(
                representation=name,
                total_matches=int(hom.raw_matches),
                inliers=int(hom.inliers),
                inlier_ratio=float(hom.inlier_ratio),
                reprojection_error=(float(hom.reprojection_error)
                                    if np.isfinite(hom.reprojection_error) else None),
                homography_plausible=bool(hom.ok and hom.plausible),
                geometric_score=representation_geometric_score(hom),
                weight=float(weights.get(name, 0.0)),
                map_center=(positions.get(name)),
            ))

        tol = (settings.representation_consensus_px
               if settings.representation_consensus_px > 0 else same_place_px)
        consensus = representation_consensus(positions, tol)

        agree_names = [n for n in consensus.participating if n != "rgb"]
        if not agree_names:
            detail = "no auxiliary representation produced a verified estimate"
        elif consensus.agree:
            detail = (f"{', '.join(agree_names)} agree with RGB within "
                      f"{consensus.tolerance_px:.0f}px "
                      f"(max {consensus.max_disagreement_px:.0f}px)")
        else:
            detail = (f"representations disagree - max {consensus.max_disagreement_px:.0f}px "
                      f"> {consensus.tolerance_px:.0f}px tolerance")
        stage("consensus", "Checking representation agreement...", "done", detail)
        return scores, consensus
    except Exception as exc:  # failure-safe
        log.warning("Cross-representation consensus failed (%s) - skipping.", exc)
        stage("consensus", "Checking representation agreement...", "skipped", str(exc))
        return [], None


def _match_one_representation(rep_work: np.ndarray, rep_feats: FeatureSet,
                              rep_scale: float, members: List["CandidateEvaluation"],
                              work_w: int, work_h: int, dw: int, dh: int,
                              extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
                              ) -> Tuple[hg.HomographyResult, Optional[np.ndarray]]:
    """
    Match one representation's working image against each top candidate tile
    independently and keep the strongest verified geometry.

    The representation's own feature points are verified on their own - they
    are never merged with the RGB correspondences before RANSAC (spec Phase 6).
    """
    best_hom = hg.HomographyResult(ok=False)
    best_center: Optional[np.ndarray] = None
    best_rank = (-1, -1)
    for ev in members:
        if ev.tile is None or ev.tile_image is None:
            continue
        k = ev.rotation
        if k:
            q_img = rotate_image(rep_work, k)
            try:
                q_feats = extract_fn(q_img)
            except Exception:
                continue
            hom, _m = _evaluate(q_img, q_feats, ev.tile_image, rotation=k,
                                canonical_size=(work_w, work_h), extract_fn=extract_fn)
        else:
            hom, _m = _evaluate(rep_work, rep_feats, ev.tile_image, rotation=0,
                                canonical_size=(work_w, work_h), extract_fn=extract_fn)
        rank = (1 if (hom.ok and hom.plausible) else 0, hom.inliers)
        if rank <= best_rank:
            continue
        best_rank = rank
        best_hom = hom
        best_center = None
        if hom.ok and hom.H is not None:
            H_full = hg.scale_homography(hom.H, rep_scale, ev.tile_scale)
            H_map = hg.translate_homography(H_full, ev.tile.x, ev.tile.y)
            best_center = hg.transform_points(
                H_map, np.array([[dw / 2.0, dh / 2.0]]))[0]
    return best_hom, best_center


def _representation_estimates(representations: Dict[str, np.ndarray],
                              structural_obj, members: List["CandidateEvaluation"],
                              work_w: int, work_h: int, dw: int, dh: int,
                              extract_fn: Callable[[np.ndarray], FeatureSet] = extract_features
                              ) -> Dict[str, Tuple[hg.HomographyResult, Optional[np.ndarray]]]:
    """Independent structural / map-domain estimates for the winning location."""
    rep_imgs: Dict[str, np.ndarray] = {}
    if structural_obj is not None and "structural" in representations:
        rep_imgs["structural"] = representations["structural"]
    if "map" in representations:
        rep_imgs["map"] = representations["map"]
    if not rep_imgs or not members:
        return {}

    out: Dict[str, Tuple[hg.HomographyResult, Optional[np.ndarray]]] = {}
    for name, img in rep_imgs.items():
        try:
            rep_work, rep_scale = _prepare_work_image(img)
            rep_feats = extract_fn(rep_work)
            hom, center = _match_one_representation(rep_work, rep_feats, rep_scale,
                                                   members, work_w, work_h, dw, dh, extract_fn)
            out[name] = (hom, center)
            log.info("Representation '%s': %d inliers, plausible=%s.",
                     name, hom.inliers, hom.ok and hom.plausible)
        except Exception as exc:
            log.warning("Representation '%s' matching failed (%s) - skipping.", name, exc)
    return out


def _project_inliers(ev: "CandidateEvaluation", work_scale: float,
                     limit: int = 400) -> Optional[np.ndarray]:
    """
    Lift the RANSAC inliers into map pixels so the result map can plot the
    evidence behind the fix, not just the outline it produced.
    """
    if ev.H_map is None or ev.hom.inlier_mask is None or ev.query_pts_work is None:
        return None
    q = np.asarray(ev.query_pts_work, np.float64)[ev.hom.inlier_mask]
    if len(q) == 0:
        return None
    if len(q) > limit:
        q = q[np.linspace(0, len(q) - 1, limit).round().astype(int)]
    return hg.transform_points(ev.H_map, q / max(work_scale, 1e-9))


# --------------------------------------------------------------------------
def _render_all(job_dir: Path, record: MapRecord, map_img: np.ndarray,
                drone_img: np.ndarray, pre: PreprocessResult, work_img: np.ndarray,
                query_feats: FeatureSet, evaluations: List[CandidateEvaluation],
                best: Optional[CandidateEvaluation], accepted: bool,
                agg_polygon: Optional[np.ndarray] = None,
                agg_center: Optional[np.ndarray] = None,
                structural_obj=None,
                representations: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, str]:
    """Write every processing-stage image for this job (spec section 43)."""
    images: Dict[str, np.ndarray] = {
        "original": drone_img,
        "corrected": pre.corrected,
        "enhanced": pre.enhanced,
        "grayscale": pre.grayscale,
        "edges": pre.edges,
        "structural_map": pre.structural,
        "contours": pre.contours,
        "keypoints": viz.draw_keypoints(work_img, query_feats.keypoints, query_feats.scores),
    }
    if structural_obj is not None:
        images["structural_query"] = structural_obj.structural
        if structural_obj.debug_overlay is not None:
            images["structural_query_overlay"] = structural_obj.debug_overlay
    if representations and "map" in representations:
        images["translated_map"] = representations["map"]

    for ev in evaluations:
        if ev.tile_image is not None and ev.source == "tile":
            images[f"candidate_{ev.candidate_id}"] = ev.tile_image

    boxes = [{"tile_id": e.tile.tile_id, "x": e.tile.x, "y": e.tile.y,
              "width": e.tile.width, "height": e.tile.height, "rank": e.score.rank}
             for e in evaluations if e.tile is not None]
    if boxes:
        images["candidate_overview"] = viz.draw_candidate_boxes(map_img, boxes)

    if best is not None and best.tile_image is not None:
        mask = best.hom.inlier_mask
        images["matches_raw"] = viz.draw_matches(
            work_img, best.tile_image, best.query_pts_work, best.target_pts_work,
            mask, inliers_only=False)
        images["matches_inliers"] = viz.draw_matches(
            work_img, best.tile_image, best.query_pts_work, best.target_pts_work,
            mask, inliers_only=True)

    # The winning cluster's confidence-weighted geometry, falling back to the
    # representative tile's own projection if aggregation found nothing.
    polygon = (agg_polygon if agg_polygon is not None
              else best.polygon if best is not None else None) if accepted else None
    center = (agg_center if agg_center is not None
             else best.center if best is not None else None) if accepted else None
    box = None
    if best is not None and best.tile is not None:
        box = {"x": best.tile.x, "y": best.tile.y,
               "width": best.tile.width, "height": best.tile.height}
    images["result_map"] = viz.render_result_map(map_img, polygon, center, box)
    if polygon is not None:
        images["localized_area"] = viz.crop_around(map_img, polygon)

    return viz.save_all(job_dir, images)


def _build_result(record: MapRecord, status: MatchStatus, explanation: str,
                  best: Optional[CandidateEvaluation],
                  best_cluster: Optional[ClusterScore],
                  scores: List[CandidateScore],
                  evaluations: List[CandidateEvaluation],
                  renders: Dict[str, str], drone_img: np.ndarray,
                  pre: PreprocessResult, query_feats: FeatureSet,
                  map_w: int, map_h: int, timings: Dict[str, float],
                  t_start: float, engine,
                  agg_center: Optional[np.ndarray] = None,
                  agg_polygon: Optional[np.ndarray] = None,
                  cluster_scores: Optional[List[ClusterScore]] = None,
                  overall_confidence: Optional[float] = None) -> dict:
    """Assemble the API payload (spec section 41)."""
    dh, dw = drone_img.shape[:2]
    accepted = status in (MatchStatus.MATCH_FOUND, MatchStatus.LOW_CONFIDENCE)
    by_id = {e.score.candidate_id: e for e in evaluations}
    cluster_scores = cluster_scores or ([best_cluster] if best_cluster else [])
    cluster_by_candidate: Dict[int, ClusterScore] = {
        cid: c for c in cluster_scores for cid in c.member_candidate_ids
    }

    map_pixel = None
    polygon = None
    gps = None
    polygon_gps = None
    center = agg_center if agg_center is not None else (best.center if best else None)
    outline = agg_polygon if agg_polygon is not None else (best.polygon if best else None)
    if accepted and center is not None:
        map_pixel = {"x": int(round(float(center[0]))), "y": int(round(float(center[1])))}
        polygon = hg.clamp_polygon(outline, map_w, map_h)
        if record.georeference is not None:
            gps = record.georeference.pixel_to_latlon(center[0], center[1])
            polygon_gps = record.georeference.polygon_to_latlon(polygon)

    # Combine RANSAC inliers projected by every tile supporting the winning
    # location, not just the representative, so the map shows the full body
    # of evidence behind the fix.
    inlier_points: List[List[float]] = []
    if accepted and best_cluster is not None:
        for cid in best_cluster.member_candidate_ids:
            ev = by_id.get(cid)
            if ev is not None and ev.inlier_map_points is not None:
                inlier_points.extend([round(float(x), 1), round(float(y), 1)]
                                     for x, y in ev.inlier_map_points)

    candidates = []
    for s in scores:
        ev = by_id.get(s.candidate_id)
        entry = s.to_dict()
        if ev is not None and ev.tile is not None:
            entry["tile"] = ev.tile.to_dict()
            entry["preview_url"] = renders.get(f"candidate_{s.candidate_id}")
        entry["source"] = ev.source if ev else "tile"
        entry["retrieval_sources"] = list(ev.retrieval_sources) if ev else ["rgb"]
        if ev is not None and ev.retrieval_similarity_by_source:
            entry["retrieval_similarity_by_source"] = {
                k: round(float(v), 4) for k, v in ev.retrieval_similarity_by_source.items()}
        if ev is not None and ev.polygon is not None:
            entry["polygon"] = hg.clamp_polygon(ev.polygon, map_w, map_h)
        cluster = cluster_by_candidate.get(s.candidate_id)
        entry["cluster_id"] = cluster.cluster_id if cluster else None
        entry["cluster_rank"] = cluster.rank if cluster else None
        candidates.append(entry)

    caps = probe_capabilities()
    hom_dict = best.hom.to_dict() if best is not None else hg.HomographyResult(ok=False).to_dict()
    cluster_dict = best_cluster.to_dict() if best_cluster is not None else None

    return {
        "status": status.value,
        "status_message": status_message(status),
        "explanation": explanation,
        "confidence": round(float(
            overall_confidence if overall_confidence is not None
            else (best_cluster.final_score if best_cluster else 0.0)), 4),
        "accepted": bool(accepted),
        "best_candidate": ({
            "candidate_id": best_cluster.representative_id,
            "tile_id": best_cluster.tile_id,
            "dino_similarity": round(float(best_cluster.dino_similarity), 4),
            "rotation_applied_deg": int(best.score.rotation_applied) * 90 if best else 0,
            "source": best.source if best else "tile",
            "tile": best.tile.to_dict() if (best and best.tile) else None,
            "supporting_tiles": best_cluster.support,
            "member_tile_ids": [by_id[cid].tile.tile_id for cid in best_cluster.member_candidate_ids
                                if cid in by_id and by_id[cid].tile is not None],
        } if best_cluster else None),
        "feature_metrics": {
            "raw_matches": cluster_dict["raw_matches"] if cluster_dict else hom_dict["raw_matches"],
            "ransac_inliers": cluster_dict["inliers"] if cluster_dict else hom_dict["ransac_inliers"],
            "inlier_ratio": cluster_dict["inlier_ratio"] if cluster_dict else hom_dict["inlier_ratio"],
            "spatial_coverage": (cluster_dict["spatial_coverage"] if cluster_dict
                                 else hom_dict["spatial_coverage"]),
            "reprojection_error": (cluster_dict["reprojection_error"] if cluster_dict
                                   else hom_dict["reprojection_error"]),
            "homography_valid": hom_dict["homography_valid"],
            "verdict": (cluster_dict["verdict"] if cluster_dict else hom_dict["verdict"]),
            "partial": bool(cluster_dict["partial"] if cluster_dict else hom_dict["partial"]),
            "coverage_cells": hom_dict["coverage_cells"],
            "coverage_grid": hom_dict["coverage_grid"],
            "rejection": cluster_dict["rejection"] if cluster_dict else hom_dict["rejection"],
            "supporting_tiles": best_cluster.support if best_cluster else 0,
        },
        "homography": (best.H_map.tolist() if (best and best.H_map is not None) else None),
        "map_pixel": map_pixel,
        "polygon": polygon,
        "inlier_map_points": inlier_points,
        "gps": gps,
        "polygon_gps": polygon_gps,
        "georeferenced": record.georeference is not None,
        "candidates": candidates,
        "retrieval": {
            "candidate_count": len(evaluations),
            "best_similarity": round(
                max((float(e.similarity) for e in evaluations), default=0.0), 4),
            "sources": sorted({s for e in evaluations for s in e.retrieval_sources}),
            "multi_domain": len({s for e in evaluations
                                 for s in e.retrieval_sources}) > 1,
        },
        "location_clusters": [c.to_dict() for c in cluster_scores],
        "drone_image": {"width": dw, "height": dh,
                        "aspect_ratio": round(dw / dh, 4) if dh else 0.0},
        "map_image": {"width": map_w, "height": map_h,
                      "tiles": len(record.tiles)},
        "keypoints": query_feats.to_dict(),
        "preprocessing": pre.stats | {"calibration_applied": pre.calibration_applied},
        "renders": renders,
        "engine": {
            "retrieval_backend": engine.backend,
            "matcher_backend": query_feats.backend,
            **caps.to_dict(),
        },
        "coverage_note": ("Estimated visual position - not a guaranteed fix. "
                          "Verify against onboard sensors before use."),
    }
