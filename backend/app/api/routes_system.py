"""System information and the Developer Mode batch-testing endpoint."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from app import __version__
from app.config import settings
from app.logging_config import get_logger
from app.localization.geolocation import haversine_m
from app.localization.pipeline import localize
from app.localization.confidence import MatchStatus
from app.models.loader import probe_capabilities
from app.schemas import BatchTestRequest, BatchTestResponse, SystemInfoResponse
from app.services import wait_for_index
from app.store import new_id, registry

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system/info", response_model=SystemInfoResponse,
            summary="Device, model availability and active thresholds")
async def system_info(warm: bool = False) -> SystemInfoResponse:
    """Set ``warm=true`` to force model loading and report true availability."""
    caps = probe_capabilities(warm=warm)
    return SystemInfoResponse(
        version=__version__,
        app_mode=settings.app_mode,
        device=caps.device.upper(),
        capabilities=caps.to_dict(),
        settings={
            "top_k_candidates": settings.top_k_candidates,
            "max_keypoints": settings.max_keypoints,
            "min_query_keypoints": settings.min_query_keypoints,
            "matcher": settings.matcher,
            "tile_scales": settings.tile_scales,
            "tile_overlap": settings.tile_overlap,
            "max_tiles": settings.max_tiles,
            "work_size": settings.work_size,
            "ransac_threshold": settings.ransac_threshold,
            "min_inliers": settings.min_inliers,
            "min_inlier_ratio": settings.min_inlier_ratio,
            "max_reprojection_error": settings.max_reprojection_error,
            "min_spatial_coverage": settings.min_spatial_coverage,
            "match_confidence": settings.match_confidence,
            "low_confidence": settings.low_confidence,
            "ambiguity_gap": settings.ambiguity_gap,
            "rotation_search": settings.rotation_search,
            "global_fallback": settings.global_fallback,
            "max_map_size": settings.max_map_size,
        },
        registry=registry.summary(),
    )


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "version": __version__, "mode": settings.app_mode}


@router.post("/dev/batch", response_model=BatchTestResponse,
             summary="Developer Mode: score a batch of drone frames")
async def batch_test(req: BatchTestRequest) -> BatchTestResponse:
    """
    Runs the pipeline over several already-uploaded drone frames and reports
    Top-1 accuracy, Top-5 retrieval accuracy, no-match detection rate and the
    false-localisation rate (spec section 46).
    """
    map_rec = registry.get_map(req.map_id)
    if map_rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown map_id '{req.map_id}'.")
    if not req.items:
        raise HTTPException(status_code=400, detail="No test items supplied.")
    try:
        wait_for_index(map_rec)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    results = []
    top1_hits = top5_hits = positioned = 0
    expected_no_match = detected_no_match = false_localizations = 0

    for item in req.items:
        drone = registry.get_drone(item.drone_id)
        if drone is None:
            results.append({"drone_id": item.drone_id, "error": "unknown drone_id"})
            continue

        started = time.time()
        job_dir = settings.processed_dir / new_id("batch")
        try:
            res = localize(map_rec, drone.path, job_dir)
        except Exception as exc:
            results.append({"drone_id": item.drone_id, "label": item.label,
                            "error": str(exc)})
            continue
        elapsed = round(time.time() - started, 2)

        status = res["status"]
        got_position = res.get("map_pixel") is not None
        error_px = None
        correct = None

        if item.expect_no_match:
            expected_no_match += 1
            correct = status == MatchStatus.NO_MATCH.value
            if correct:
                detected_no_match += 1
            elif status == MatchStatus.MATCH_FOUND.value:
                false_localizations += 1
        elif item.expected_x is not None and item.expected_y is not None and got_position:
            positioned += 1
            dx = res["map_pixel"]["x"] - item.expected_x
            dy = res["map_pixel"]["y"] - item.expected_y
            error_px = round(float((dx * dx + dy * dy) ** 0.5), 1)
            correct = error_px <= req.tolerance_px
            if correct:
                top1_hits += 1
            elif status == MatchStatus.MATCH_FOUND.value:
                false_localizations += 1
            # Top-5 retrieval: did any returned candidate land near the truth?
            for cand in res.get("candidates", []):
                tile = cand.get("tile")
                if not tile:
                    continue
                cx, cy = tile["center"]
                if ((cx - item.expected_x) ** 2 + (cy - item.expected_y) ** 2) ** 0.5 \
                        <= req.tolerance_px + max(tile["width"], tile["height"]) / 2:
                    top5_hits += 1
                    break

        results.append({
            "drone_id": item.drone_id,
            "label": item.label or drone.filename,
            "status": status,
            "confidence": res.get("confidence"),
            "map_pixel": res.get("map_pixel"),
            "expected": (None if item.expected_x is None
                         else {"x": item.expected_x, "y": item.expected_y}),
            "expect_no_match": item.expect_no_match,
            "error_px": error_px,
            "correct": correct,
            "inliers": res["feature_metrics"]["ransac_inliers"],
            "processing_time": elapsed,
            "candidate_tile": (res.get("best_candidate") or {}).get("tile_id"),
        })

    localizable = max(1, len([r for r in results if not r.get("expect_no_match")
                              and r.get("expected")]))
    metrics = {
        "total": len(results),
        "top1_accuracy": round(top1_hits / localizable, 4),
        "top5_retrieval_accuracy": round(top5_hits / localizable, 4),
        "no_match_detection_rate": (round(detected_no_match / expected_no_match, 4)
                                    if expected_no_match else None),
        "false_localization_rate": round(false_localizations / max(1, len(results)), 4),
        "evaluated_with_ground_truth": localizable,
        "expected_no_match": expected_no_match,
    }
    return BatchTestResponse(map_id=req.map_id, tolerance_px=req.tolerance_px,
                             results=results, metrics=metrics)


@router.get("/dev/distance", summary="Great-circle distance between two coordinates")
async def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    """Utility for relating an estimated fix to a .plan waypoint."""
    return {"metres": round(haversine_m(lat1, lon1, lat2, lon2), 2)}
