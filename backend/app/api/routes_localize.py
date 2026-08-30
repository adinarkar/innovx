"""Localisation job endpoints and georeferencing."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.logging_config import get_logger
from app.localization.geolocation import Georeference, GeoreferenceError
from app.schemas import (CandidatesResponse, GeoreferenceRequest, GeoreferenceResponse,
                         JobAccepted, JobStatusResponse, LocalizeRequest)
from app.services import start_job
from app.store import registry

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["localization"])


def _get_job(job_id: str):
    job = registry.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'.")
    return job


@router.post("/localize", response_model=JobAccepted, status_code=202,
             summary="Start a localisation job")
async def localize(req: LocalizeRequest) -> JobAccepted:
    """
    Queues the full pipeline: tiling -> retrieval -> local features ->
    LightGlue/SIFT -> RANSAC -> geometric verification -> position.
    """
    if registry.get_map(req.map_id) is None:
        raise HTTPException(status_code=400,
                            detail="Reference map missing - upload a map first.")
    if registry.get_drone(req.drone_id) is None:
        raise HTTPException(status_code=400,
                            detail="Drone image missing - upload a drone capture first.")
    if req.plan_id and registry.get_plan(req.plan_id) is None:
        raise HTTPException(status_code=400, detail=f"Unknown plan_id '{req.plan_id}'.")

    # Per-request overrides of the runtime matcher / rotation policy.
    if req.matcher:
        if req.matcher.lower() not in ("lightglue", "sift"):
            raise HTTPException(status_code=400,
                                detail="matcher must be 'lightglue' or 'sift'.")
        settings.matcher = req.matcher.lower()
    if req.rotation_search is not None:
        settings.rotation_search = bool(req.rotation_search)

    job = start_job(req.map_id, req.drone_id, req.plan_id, req.top_k, req.calibration)
    return JobAccepted(job_id=job.job_id, state=job.state,
                       poll_url=f"/api/process/{job.job_id}")


@router.get("/process/{job_id}", response_model=JobStatusResponse,
            summary="Pipeline progress for a job")
async def process_status(job_id: str) -> JobStatusResponse:
    return JobStatusResponse(**_get_job(job_id).to_dict())


@router.get("/result/{job_id}", response_model=JobStatusResponse,
            summary="Final localisation result")
async def result(job_id: str) -> JobStatusResponse:
    job = _get_job(job_id)
    if job.state == "error":
        raise HTTPException(status_code=500, detail=job.error or "Localization failed.")
    if job.state != "done":
        raise HTTPException(status_code=409,
                            detail=f"Job is still {job.state}; poll /api/process/{job_id}.")
    return JobStatusResponse(**job.to_dict())


@router.get("/candidates/{job_id}", response_model=CandidatesResponse,
            summary="Ranked candidate diagnostics")
async def candidates(job_id: str) -> CandidatesResponse:
    job = _get_job(job_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail=f"Job is still {job.state}.")
    return CandidatesResponse(job_id=job_id, status=job.result.get("status", "UNKNOWN"),
                              candidates=job.result.get("candidates", []))


@router.post("/georeference", response_model=GeoreferenceResponse,
             summary="Attach an optional lat/lon georeference to a map")
async def georeference(req: GeoreferenceRequest) -> GeoreferenceResponse:
    """
    Without this, results stay in map pixels and ``gps`` is ``null``.
    Coordinates are never inferred from imagery.
    """
    rec = registry.get_map(req.map_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown map_id '{req.map_id}'.")
    try:
        if req.corners:
            geo = Georeference.from_corners(req.corners, rec.width, rec.height)
        else:
            geo = Georeference.from_bbox(req.north, req.south, req.west, req.east,
                                         rec.width, rec.height)
    except GeoreferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rec.georeference = geo
    log.info("Map %s georeferenced (%s).", rec.map_id, geo.kind)
    return GeoreferenceResponse(map_id=rec.map_id, georeference=geo.to_dict())


@router.delete("/georeference/{map_id}", summary="Remove a map georeference")
async def clear_georeference(map_id: str) -> dict:
    rec = registry.get_map(map_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown map_id '{map_id}'.")
    rec.georeference = None
    return {"status": "success", "map_id": map_id, "georeferenced": False}
