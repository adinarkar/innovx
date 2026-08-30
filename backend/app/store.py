"""
In-memory registry for uploaded maps, drone frames, plans and jobs, with a
small on-disk cache for reference-map embeddings.

Recomputing DINOv2 embeddings for every localisation request would dominate
runtime, so a map is indexed once at upload time and the result is cached both
in memory and under ``cache/`` (spec sections 7 and 47).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.localization.geolocation import Georeference
from app.localization.tiling import Tile

log = get_logger(__name__)

_LOCK = threading.RLock()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
@dataclass
class MapRecord:
    map_id: str
    path: Path
    width: int
    height: int
    filename: str
    file_size: int
    tiles: List[Tile] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    embedding_backend: str = "pending"
    embedding_status: str = "pending"        # pending | ready | failed
    index_seconds: float = 0.0
    georeference: Optional[Georeference] = None
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def cache_path(self) -> Path:
        return settings.cache_dir / f"{self.map_id}.npz"

    def to_dict(self) -> dict:
        return {
            "status": "success" if self.embedding_status != "failed" else "error",
            "map_id": self.map_id,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "tiles_generated": len(self.tiles),
            "embedding_status": self.embedding_status,
            "embedding_backend": self.embedding_backend,
            "index_seconds": round(self.index_seconds, 2),
            "georeferenced": self.georeference is not None,
            "georeference": self.georeference.to_dict() if self.georeference else None,
            "preview_url": f"/files/uploads/{self.path.name}",
            "error": self.error,
        }

    # ----- embedding cache ----------------------------------------------
    def save_cache(self) -> None:
        if self.embeddings is None:
            return
        try:
            meta = json.dumps([t.to_dict() for t in self.tiles])
            np.savez_compressed(self.cache_path, embeddings=self.embeddings,
                                tiles=np.array(meta), backend=np.array(self.embedding_backend))
            log.info("Cached %d embeddings for %s.", len(self.embeddings), self.map_id)
        except Exception as exc:
            log.warning("Could not cache embeddings for %s: %s", self.map_id, exc)

    def load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            tiles = json.loads(str(data["tiles"]))
            self.embeddings = data["embeddings"]
            self.embedding_backend = str(data["backend"])
            self.tiles = [Tile(t["tile_id"], t["x"], t["y"], t["width"], t["height"],
                               t["scale"]) for t in tiles]
            self.embedding_status = "ready"
            log.info("Loaded cached index for %s (%d tiles).", self.map_id, len(self.tiles))
            return True
        except Exception as exc:
            log.warning("Ignoring corrupt cache for %s: %s", self.map_id, exc)
            return False


@dataclass
class DroneRecord:
    drone_id: str
    path: Path
    width: int
    height: int
    filename: str
    file_size: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "status": "success",
            "drone_id": self.drone_id,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "aspect_ratio": round(self.width / self.height, 4) if self.height else 0.0,
            "preview_url": f"/files/uploads/{self.path.name}",
        }


@dataclass
class PlanRecord:
    plan_id: str
    path: Path
    data: dict

    def to_dict(self) -> dict:
        return {"status": "success", "plan_id": self.plan_id, **self.data}


@dataclass
class JobRecord:
    job_id: str
    map_id: str
    drone_id: str
    plan_id: Optional[str] = None
    state: str = "queued"                  # queued | running | done | error
    stage: str = "queued"
    stages: List[dict] = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def job_dir(self) -> Path:
        return settings.processed_dir / self.job_id

    def log_stage(self, key: str, label: str, state: str = "running",
                  detail: Optional[str] = None) -> None:
        now = time.time()
        for s in self.stages:
            if s["key"] == key:
                s.update(state=state, detail=detail or s.get("detail"),
                         seconds=round(now - s["_start"], 3))
                break
        else:
            self.stages.append({"key": key, "label": label, "state": state,
                                "detail": detail, "seconds": 0.0, "_start": now})
        self.stage = key

    def public_stages(self) -> List[dict]:
        return [{k: v for k, v in s.items() if not k.startswith("_")} for s in self.stages]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "map_id": self.map_id,
            "drone_id": self.drone_id,
            "plan_id": self.plan_id,
            "state": self.state,
            "stage": self.stage,
            "stages": self.public_stages(),
            "error": self.error,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 2),
            "result": self.result,
        }


# --------------------------------------------------------------------------
class Registry:
    """Process-local store.  A production deployment would swap this for Redis."""

    def __init__(self) -> None:
        self.maps: Dict[str, MapRecord] = {}
        self.drones: Dict[str, DroneRecord] = {}
        self.plans: Dict[str, PlanRecord] = {}
        self.jobs: Dict[str, JobRecord] = {}

    # maps
    def add_map(self, rec: MapRecord) -> MapRecord:
        with _LOCK:
            self.maps[rec.map_id] = rec
        return rec

    def get_map(self, map_id: str) -> Optional[MapRecord]:
        return self.maps.get(map_id)

    # drone frames
    def add_drone(self, rec: DroneRecord) -> DroneRecord:
        with _LOCK:
            self.drones[rec.drone_id] = rec
        return rec

    def get_drone(self, drone_id: str) -> Optional[DroneRecord]:
        return self.drones.get(drone_id)

    # plans
    def add_plan(self, rec: PlanRecord) -> PlanRecord:
        with _LOCK:
            self.plans[rec.plan_id] = rec
        return rec

    def get_plan(self, plan_id: str) -> Optional[PlanRecord]:
        return self.plans.get(plan_id)

    # jobs
    def add_job(self, rec: JobRecord) -> JobRecord:
        with _LOCK:
            self.jobs[rec.job_id] = rec
        return rec

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def summary(self) -> dict:
        return {"maps": len(self.maps), "drone_images": len(self.drones),
                "plans": len(self.plans), "jobs": len(self.jobs)}


registry = Registry()
