"""Upload endpoints: reference map, drone capture and QGroundControl plan."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.logging_config import get_logger
from app.plan.qgc_parser import PlanError, parse_plan
from app.schemas import DroneUploadResponse, MapUploadResponse, PlanUploadResponse
from app.services import UploadError, register_drone, register_map, save_upload
from app.store import PlanRecord, new_id, registry

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/map/upload", response_model=MapUploadResponse,
             summary="Upload the reference satellite / orthomosaic map")
async def upload_map(file: UploadFile = File(...)) -> MapUploadResponse:
    """
    Stores the map, plans multi-scale overlapping tiles and starts embedding
    them in the background.  Poll ``GET /api/map/{map_id}`` until
    ``embedding_status`` is ``ready``.
    """
    try:
        rec = register_map(file)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MapUploadResponse(**rec.to_dict())


@router.get("/map/{map_id}", response_model=MapUploadResponse,
            summary="Reference map indexing status")
async def map_status(map_id: str) -> MapUploadResponse:
    rec = registry.get_map(map_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown map_id '{map_id}'.")
    return MapUploadResponse(**rec.to_dict())


@router.post("/drone/upload", response_model=DroneUploadResponse,
             summary="Upload the downward-facing drone capture")
async def upload_drone(file: UploadFile = File(...)) -> DroneUploadResponse:
    try:
        rec = register_drone(file)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DroneUploadResponse(**rec.to_dict())


@router.post("/plan/upload", response_model=PlanUploadResponse,
             summary="Upload an optional QGroundControl .plan mission file")
async def upload_plan(file: UploadFile = File(...)) -> PlanUploadResponse:
    """
    The .plan file supplies mission coordinates only.  It never contributes
    imagery, and localisation does not require it.
    """
    plan_id = new_id("plan")
    try:
        path = save_upload(file, settings.upload_dir, plan_id, {".plan", ".json"})
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        plan = parse_plan(path)
    except PlanError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid .plan file: {exc}") from exc

    data = plan.to_dict()
    # Report the name the operator uploaded, not the internal storage name.
    data["filename"] = file.filename or path.name
    registry.add_plan(PlanRecord(plan_id=plan_id, path=path, data=data))
    return PlanUploadResponse(plan_id=plan_id, **data)
