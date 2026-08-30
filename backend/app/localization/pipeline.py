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

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.localization import dino, homography as hg, visualization as viz
from app.localization.imaging import fit_long_edge, imread
from app.localization.confidence import (CandidateScore, MatchStatus, decide,
                                         evaluate_candidate, finalize_scores,
                                         status_message)
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
    query_pts_work: Optional[np.ndarray] = None
    target_pts_work: Optional[np.ndarray] = None
    inlier_map_points: Optional[np.ndarray] = None   # inliers in map pixels
    source: str = "tile"                        # tile | global


def _prepare_work_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Resize to the configured working long edge, returning the scale used."""
    return fit_long_edge(img, settings.work_size)


def _match_pair(query_img: np.ndarray, query_feats: FeatureSet,
                target_img: np.ndarray) -> Tuple[MatchResult, FeatureSet]:
    target_feats = extract_features(target_img)
    result = match_features(query_feats, target_feats, query_img, target_img)
    return result, target_feats


def _evaluate(query_img: np.ndarray, query_feats: FeatureSet,
              target_img: np.ndarray, rotation: int = 0,
              canonical_size: Optional[Tuple[int, int]] = None
              ) -> Tuple[hg.HomographyResult, MatchResult]:
    """
    Match then verify, always returning query points in the *canonical*
    (unrotated, working-resolution) drone frame.

    ``query_img``/``query_feats`` must describe the same image - for the
    rotation search that is the rotated copy - while ``canonical_size`` is the
    unrotated ``(width, height)`` the points are mapped back into.
    """
    match, _ = _match_pair(query_img, query_feats, target_img)
    cw, ch = canonical_size or (query_img.shape[1], query_img.shape[0])
    if rotation:
        q_pts = unrotate_points(match.query_pts, rotation, cw, ch)
        match = MatchResult(q_pts.astype(np.float32), match.target_pts,
                            match.confidence, match.backend)
    hom = hg.estimate(match.query_pts, match.target_pts, (cw, ch))
    return hom, match


# --------------------------------------------------------------------------
def localize(record: MapRecord, drone_path: Path, job_dir: Path,
             calibration: Optional[CameraCalibration] = None,
             progress: Optional[ProgressFn] = None,
             top_k: Optional[int] = None) -> dict:
    """
    Full localisation for one drone frame.  Always returns a structured result
    - including an explicit NO_MATCH - rather than raising for a poor match.
    """
    progress = progress or _noop
    top_k = top_k or settings.top_k_candidates
    t_start = time.time()
    timings: Dict[str, float] = {}

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

    # ---- 3. query embedding ---------------------------------------------
    stage(*STAGES[2])
    t0 = time.time()
    engine = dino.get_engine()
    query_embedding = engine.embed(pre.matching_input)
    timings["embed"] = time.time() - t0
    stage("embed", STAGES[2][1], "done", f"backend: {engine.backend}")

    # ---- 4. retrieval ----------------------------------------------------
    stage(*STAGES[3])
    t0 = time.time()
    retrieved = dino.top_k(query_embedding, record.embeddings, top_k)
    timings["retrieve"] = time.time() - t0
    stage("retrieve", STAGES[3][1], "done", f"top {len(retrieved)} of {len(record.tiles)} tiles")

    # ---- 5. query features ----------------------------------------------
    stage(*STAGES[4])
    t0 = time.time()
    query_feats = extract_features(work_img)
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

        hom, match = _evaluate(work_img, query_feats, tile_img, rotation=0)
        rotation_used = 0

        # Rotation search: only when the upright attempt is weak, so a strong
        # match is never disturbed (spec section 50).
        if settings.rotation_search and not (hom.ok and hom.plausible):
            for k in (1, 2, 3):
                rot_img = rotate_image(work_img, k)
                rot_feats = extract_features(rot_img)
                cand_hom, cand_match = _evaluate(rot_img, rot_feats, tile_img,
                                                 rotation=k,
                                                 canonical_size=(work_w, work_h))
                if cand_hom.ok and cand_hom.plausible and cand_hom.inliers > hom.inliers:
                    hom, match, rotation_used = cand_hom, cand_match, k
                    break

        score = evaluate_candidate(cand_idx + 1, tile.tile_id, hit["similarity"],
                                   hom, rotation_used)
        ev = CandidateEvaluation(cand_idx + 1, tile, hit["similarity"], score, hom,
                                 match, rotation_used, tile_image=tile_img,
                                 query_pts_work=match.query_pts,
                                 target_pts_work=match.target_pts)

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
        fallback = _global_fallback(map_img, work_img, query_feats, dw, dh, work_scale)
        if fallback is not None:
            evaluations.append(fallback)
            log.info("Global fallback recovered a match with %d inliers.",
                     fallback.hom.inliers)

    scores = finalize_scores([e.score for e in evaluations])
    by_id = {e.score.candidate_id: e for e in evaluations}

    # Overlapping tiles routinely describe the same place; give the decision
    # logic the projected centres so it can tell a true ambiguity from a
    # duplicate detection.
    positions = {e.score.candidate_id: (float(e.center[0]), float(e.center[1]))
                 for e in evaluations if e.center is not None}
    same_place_px = _same_place_radius(evaluations, dw, dh)
    status, explanation = decide(scores, positions, same_place_px)
    timings["verify"] = time.time() - t0
    stage("verify", STAGES[6][1], "done", explanation)

    # ---- 9. position + renders ------------------------------------------
    stage(*STAGES[7])
    t0 = time.time()
    best_score = next((s for s in scores if s.homography_valid), scores[0] if scores else None)
    best = by_id.get(best_score.candidate_id) if best_score else None
    accepted = status in (MatchStatus.MATCH_FOUND, MatchStatus.LOW_CONFIDENCE)

    renders = _render_all(job_dir, record, map_img, drone_img, pre, work_img,
                          query_feats, evaluations, best, accepted)
    result = _build_result(record, status, explanation, best, best_score, scores,
                           evaluations, renders, drone_img, pre, query_feats,
                           map_w, map_h, timings, t_start, engine)
    timings["position"] = time.time() - t0
    stage("position", STAGES[7][1], "done", status_message(status))
    result["timings"] = {k: round(v, 3) for k, v in timings.items()}
    result["processing_time"] = round(time.time() - t_start, 2)
    return result


# --------------------------------------------------------------------------
def _global_fallback(map_img: np.ndarray, work_img: np.ndarray,
                     query_feats: FeatureSet, dw: int, dh: int,
                     work_scale: float) -> Optional[CandidateEvaluation]:
    """
    Last resort: match the drone frame against the whole (downscaled) map.

    Tiling can miss a frame that straddles four tiles, so a single direct
    attempt is cheaper than densifying the grid.  The result is verified by
    exactly the same geometric gates as any tile candidate.
    """
    map_small, map_scale = fit_long_edge(map_img, max(settings.work_size * 2, 1280))
    hom, match = _evaluate(work_img, query_feats, map_small, rotation=0)
    if not (hom.ok and hom.plausible):
        return None

    score = evaluate_candidate(999, -1, 0.0, hom, 0)
    ev = CandidateEvaluation(999, None, 0.0, score, hom, match, 0,
                             tile_image=map_small, source="global",
                             query_pts_work=match.query_pts,
                             target_pts_work=match.target_pts)
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
                best: Optional[CandidateEvaluation], accepted: bool) -> Dict[str, str]:
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

    polygon = best.polygon if (best is not None and accepted) else None
    center = best.center if (best is not None and accepted) else None
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
                  best_score: Optional[CandidateScore],
                  scores: List[CandidateScore],
                  evaluations: List[CandidateEvaluation],
                  renders: Dict[str, str], drone_img: np.ndarray,
                  pre: PreprocessResult, query_feats: FeatureSet,
                  map_w: int, map_h: int, timings: Dict[str, float],
                  t_start: float, engine) -> dict:
    """Assemble the API payload (spec section 41)."""
    dh, dw = drone_img.shape[:2]
    accepted = status in (MatchStatus.MATCH_FOUND, MatchStatus.LOW_CONFIDENCE)
    by_id = {e.score.candidate_id: e for e in evaluations}

    map_pixel = None
    polygon = None
    gps = None
    polygon_gps = None
    if best is not None and accepted and best.center is not None:
        map_pixel = {"x": int(round(float(best.center[0]))),
                     "y": int(round(float(best.center[1])))}
        polygon = hg.clamp_polygon(best.polygon, map_w, map_h)
        if record.georeference is not None:
            gps = record.georeference.pixel_to_latlon(best.center[0], best.center[1])
            polygon_gps = record.georeference.polygon_to_latlon(polygon)

    candidates = []
    for s in scores:
        ev = by_id.get(s.candidate_id)
        entry = s.to_dict()
        if ev is not None and ev.tile is not None:
            entry["tile"] = ev.tile.to_dict()
            entry["preview_url"] = renders.get(f"candidate_{s.candidate_id}")
        entry["source"] = ev.source if ev else "tile"
        if ev is not None and ev.polygon is not None:
            entry["polygon"] = hg.clamp_polygon(ev.polygon, map_w, map_h)
        candidates.append(entry)

    caps = probe_capabilities()
    metrics = best.hom.to_dict() if best is not None else hg.HomographyResult(ok=False).to_dict()

    return {
        "status": status.value,
        "status_message": status_message(status),
        "explanation": explanation,
        "confidence": round(float(best_score.final_score), 4) if best_score else 0.0,
        "accepted": bool(accepted),
        "best_candidate": ({
            "candidate_id": best_score.candidate_id,
            "tile_id": best_score.tile_id,
            "dino_similarity": round(float(best_score.dino_similarity), 4),
            "rotation_applied_deg": int(best_score.rotation_applied) * 90,
            "source": best.source if best else "tile",
            "tile": best.tile.to_dict() if (best and best.tile) else None,
        } if best_score else None),
        "feature_metrics": {
            "raw_matches": metrics["raw_matches"],
            "ransac_inliers": metrics["ransac_inliers"],
            "inlier_ratio": metrics["inlier_ratio"],
            "spatial_coverage": metrics["spatial_coverage"],
            "reprojection_error": metrics["reprojection_error"],
            "homography_valid": metrics["homography_valid"],
            "coverage_cells": metrics["coverage_cells"],
            "coverage_grid": metrics["coverage_grid"],
            "rejection": metrics["rejection"],
        },
        "homography": (best.H_map.tolist() if (best and best.H_map is not None) else None),
        "map_pixel": map_pixel,
        "polygon": polygon,
        "inlier_map_points": ([[round(float(x), 1), round(float(y), 1)]
                               for x, y in best.inlier_map_points]
                              if (best is not None and accepted
                                  and best.inlier_map_points is not None) else []),
        "gps": gps,
        "polygon_gps": polygon_gps,
        "georeferenced": record.georeference is not None,
        "candidates": candidates,
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
