"""Typed request/response models so /docs is genuinely useful (spec section 58)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------
class MapUploadResponse(BaseModel):
    status: str = "success"
    map_id: str
    filename: str
    width: int
    height: int
    file_size: int
    tiles_generated: int = 0
    embedding_status: str = Field("pending", description="pending | indexing | ready | failed")
    embedding_backend: str = "pending"
    index_seconds: float = 0.0
    georeferenced: bool = False
    georeference: Optional[Dict[str, Any]] = None
    preview_url: str
    error: Optional[str] = None


class DroneUploadResponse(BaseModel):
    status: str = "success"
    drone_id: str
    filename: str
    width: int
    height: int
    file_size: int
    aspect_ratio: float
    preview_url: str


class PlanUploadResponse(BaseModel):
    status: str = "success"
    plan_id: str
    filename: str
    waypoint_count: int
    planned_home_position: Optional[Dict[str, Optional[float]]] = None
    waypoints: List[Dict[str, Any]] = []
    coordinates: List[List[float]] = []
    bounds: Optional[Dict[str, float]] = None
    suggested_georeference: Optional[Dict[str, float]] = None
    geofence_polygons: int = 0
    rally_points: int = 0
    warnings: List[str] = []
    disclaimer: str
    file_type: Optional[str] = None
    version: Optional[int] = None
    ground_station: Optional[str] = None
    vehicle_type: Optional[int] = None
    cruise_speed: Optional[float] = None
    hover_speed: Optional[float] = None


# --------------------------------------------------------------------------
# Georeferencing
# --------------------------------------------------------------------------
class GeoreferenceRequest(BaseModel):
    map_id: str
    north: Optional[float] = Field(None, description="North latitude of the map top edge")
    south: Optional[float] = None
    west: Optional[float] = None
    east: Optional[float] = None
    corners: Optional[List[List[float]]] = Field(
        None, description="Four [lat, lon] pairs ordered TL, TR, BR, BL")

    model_config = {
        "json_schema_extra": {
            "example": {"map_id": "map_abc123", "north": 12.9721, "south": 12.9605,
                        "west": 77.5904, "east": 77.6062}
        }
    }


class GeoreferenceResponse(BaseModel):
    status: str = "success"
    map_id: str
    georeference: Dict[str, Any]


# --------------------------------------------------------------------------
# Localization
# --------------------------------------------------------------------------
class LocalizeRequest(BaseModel):
    map_id: str
    drone_id: str
    plan_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=25)
    matcher: Optional[str] = Field(None, description="lightglue | sift")
    rotation_search: Optional[bool] = None
    calibration: Optional[Dict[str, float]] = Field(
        None, description="Optional fx, fy, cx, cy, k1, k2, p1, p2, k3")


class JobAccepted(BaseModel):
    status: str = "accepted"
    job_id: str
    state: str
    poll_url: str


class StageInfo(BaseModel):
    key: str
    label: str
    state: str
    detail: Optional[str] = None
    seconds: float = 0.0


class JobStatusResponse(BaseModel):
    job_id: str
    map_id: str
    drone_id: str
    plan_id: Optional[str] = None
    state: str
    stage: str
    stages: List[StageInfo] = []
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    result: Optional[Dict[str, Any]] = None


class CandidatesResponse(BaseModel):
    job_id: str
    status: str
    candidates: List[Dict[str, Any]] = []


# --------------------------------------------------------------------------
# System / developer
# --------------------------------------------------------------------------
class SystemInfoResponse(BaseModel):
    name: str = "innovX VisualNav"
    version: str
    app_mode: str
    device: str
    capabilities: Dict[str, Any]
    settings: Dict[str, Any]
    registry: Dict[str, int]


class BatchTestItem(BaseModel):
    drone_id: str
    expected_x: Optional[int] = None
    expected_y: Optional[int] = None
    expect_no_match: bool = False
    label: Optional[str] = None


class BatchTestRequest(BaseModel):
    map_id: str
    items: List[BatchTestItem]
    tolerance_px: int = Field(150, ge=1, description="Radius counted as a correct fix")


class BatchTestResponse(BaseModel):
    status: str = "success"
    map_id: str
    tolerance_px: int
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None
