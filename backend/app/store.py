"""
In-memory registry for uploaded maps, drone frames, plans and jobs, with a
small on-disk cache for reference-map embeddings.

Recomputing DINOv2 embeddings for every localisation request would dominate
runtime, so a map is indexed once at upload time and the result is cached both
in memory and under ``cache/`` (spec sections 7 and 47).
"""
from __future__ import annotations

import hashlib
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


def file_content_hash(path: Path) -> str:
    """Stable short hash of a file's bytes, used to key the embedding cache."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def index_signature(backend: str) -> str:
    """
    Fingerprint of every setting that changes the tile index.

    A cache entry is only reused when this matches, so editing the tile scales,
    overlap, budget or switching the embedding backend transparently
    invalidates stale ``.npz`` files instead of loading a mismatched index.
    """
    raw = "|".join(str(v) for v in (
        settings.tile_scales_raw, settings.tile_overlap, settings.max_tiles,
        settings.max_map_size, backend,
    ))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
@dataclass
class MapRecord:
    map_id: str
    path: Path
    width: int
    height: int
    filename: str
    file_size: int
    content_hash: str = ""
    tiles: List[Tile] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    embedding_backend: str = "pending"
    embedding_status: str = "pending"        # pending | ready | failed
    index_seconds: float = 0.0
    georeference: Optional[Georeference] = None
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    def _content_key(self) -> str:
        """Content hash of the map file, computed once and memoised."""
        if not self.content_hash:
            try:
                self.content_hash = file_content_hash(self.path)
            except OSError:
                self.content_hash = self.map_id
        return self.content_hash

    @property
    def cache_path(self) -> Path:
        # Keyed by map *content*, not the random map_id, so an identical map
        # (re-upload or a backend restart) reuses the existing index.
        return settings.cache_dir / f"{self._content_key()}.npz"

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
            np.savez_compressed(
                self.cache_path, embeddings=self.embeddings, tiles=np.array(meta),
                backend=np.array(self.embedding_backend),
                signature=np.array(index_signature(self.embedding_backend)),
            )
            log.info("Cached %d embeddings for %s (%s).",
                     len(self.embeddings), self.map_id, self.cache_path.name)
        except Exception as exc:
            log.warning("Could not cache embeddings for %s: %s", self.map_id, exc)

    def load_cache(self, expected_backend: Optional[str] = None) -> bool:
        """
        Restore a cached tile index.  Returns False (so the caller recomputes)
        when the file is absent, corrupt, or was built with different tiling
        settings / a different embedding backend.
        """
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            backend = str(data["backend"])
            if expected_backend is not None and backend != expected_backend:
                log.info("Cache for %s is %s, need %s - rebuilding.",
                         self.map_id, backend, expected_backend)
                return False
            if "signature" in data and str(data["signature"]) != index_signature(backend):
                log.info("Tiling settings changed since %s was cached - rebuilding.",
                         self.map_id)
                return False
            tiles = json.loads(str(data["tiles"]))
            self.embeddings = data["embeddings"]
            self.embedding_backend = backend
            self.tiles = [Tile(t["tile_id"], t["x"], t["y"], t["width"], t["height"],
                               t["scale"]) for t in tiles]
            self.embedding_status = "ready"
            log.info("Loaded cached index for %s (%d tiles).", self.map_id, len(self.tiles))
            return True
        except Exception as exc:
            log.warning("Ignoring corrupt cache for %s: %s", self.map_id, exc)
            return False

    # ----- registry persistence ---------------------------------------
    def to_state(self) -> dict:
        """Lightweight record for the registry sidecar (no tiles/embeddings)."""
        return {
            "map_id": self.map_id, "path": str(self.path), "width": self.width,
            "height": self.height, "filename": self.filename,
            "file_size": self.file_size, "content_hash": self.content_hash,
            "embedding_backend": self.embedding_backend,
            "index_seconds": self.index_seconds, "created_at": self.created_at,
            "georeference": self.georeference.to_dict() if self.georeference else None,
        }

    @classmethod
    def from_state(cls, d: dict) -> "MapRecord":
        rec = cls(
            map_id=d["map_id"], path=Path(d["path"]), width=d["width"],
            height=d["height"], filename=d["filename"], file_size=d["file_size"],
            content_hash=d.get("content_hash", ""),
            embedding_backend=d.get("embedding_backend", "pending"),
            index_seconds=d.get("index_seconds", 0.0),
            created_at=d.get("created_at", time.time()),
        )
        rec.georeference = Georeference.from_dict(d.get("georeference"))
        return rec


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

    def to_state(self) -> dict:
        return {"drone_id": self.drone_id, "path": str(self.path), "width": self.width,
                "height": self.height, "filename": self.filename,
                "file_size": self.file_size, "created_at": self.created_at}

    @classmethod
    def from_state(cls, d: dict) -> "DroneRecord":
        return cls(drone_id=d["drone_id"], path=Path(d["path"]), width=d["width"],
                   height=d["height"], filename=d["filename"],
                   file_size=d["file_size"], created_at=d.get("created_at", time.time()))


@dataclass
class PlanRecord:
    plan_id: str
    path: Path
    data: dict

    def to_dict(self) -> dict:
        return {"status": "success", "plan_id": self.plan_id, **self.data}

    def to_state(self) -> dict:
        return {"plan_id": self.plan_id, "path": str(self.path), "data": self.data}

    @classmethod
    def from_state(cls, d: dict) -> "PlanRecord":
        return cls(plan_id=d["plan_id"], path=Path(d["path"]), data=d["data"])


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
_STATE_VERSION = 1


class Registry:
    """
    Process-local store for uploaded assets and jobs.

    Maps, drone frames and plans are mirrored to a small JSON sidecar so a
    backend restart does not lose already-indexed reference maps (re-indexing a
    large map is the most expensive thing the service does).  Jobs are
    intentionally transient and never persisted.
    """

    def __init__(self, persist: bool = True) -> None:
        self.maps: Dict[str, MapRecord] = {}
        self.drones: Dict[str, DroneRecord] = {}
        self.plans: Dict[str, PlanRecord] = {}
        self.jobs: Dict[str, JobRecord] = {}
        self._persist_enabled = persist
        self._state_path = settings.cache_dir / "registry.json"
        if persist:
            self._restore()

    # ----- persistence -------------------------------------------------
    def _persist(self) -> None:
        if not self._persist_enabled:
            return
        try:
            payload = {
                "version": _STATE_VERSION,
                "maps": [m.to_state() for m in self.maps.values()],
                "drones": [d.to_state() for d in self.drones.values()],
                "plans": [p.to_state() for p in self.plans.values()],
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as exc:                           # pragma: no cover
            log.warning("Could not persist registry state: %s", exc)

    def _restore(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Ignoring unreadable registry state (%s).", exc)
            return
        if payload.get("version") != _STATE_VERSION:
            log.info("Registry state version mismatch - starting fresh.")
            return

        restored_maps = 0
        for d in payload.get("maps", []):
            try:
                rec = MapRecord.from_state(d)
            except Exception as exc:                       # pragma: no cover
                log.warning("Skipping unrestorable map entry: %s", exc)
                continue
            if not rec.path.exists():
                log.info("Map %s file is gone - not restoring.", rec.map_id)
                continue
            if rec.load_cache():                           # only if the index is cached
                self.maps[rec.map_id] = rec
                restored_maps += 1
            else:
                log.info("Map %s has no usable cached index - re-upload to use it.",
                         rec.map_id)

        for d in payload.get("drones", []):
            try:
                rec = DroneRecord.from_state(d)
                if rec.path.exists():
                    self.drones[rec.drone_id] = rec
            except Exception as exc:                       # pragma: no cover
                log.warning("Skipping unrestorable drone entry: %s", exc)

        for d in payload.get("plans", []):
            try:
                self.plans[d["plan_id"]] = PlanRecord.from_state(d)
            except Exception as exc:                       # pragma: no cover
                log.warning("Skipping unrestorable plan entry: %s", exc)

        if restored_maps or self.drones or self.plans:
            log.info("Restored %d map(s), %d drone frame(s), %d plan(s) from disk.",
                     restored_maps, len(self.drones), len(self.plans))

    # ----- maps ------------------------------------------------------------
    def add_map(self, rec: MapRecord) -> MapRecord:
        with _LOCK:
            self.maps[rec.map_id] = rec
            self._persist()
        return rec

    def get_map(self, map_id: str) -> Optional[MapRecord]:
        return self.maps.get(map_id)

    def touch(self) -> None:
        """Re-write the sidecar, e.g. after a map's index or georeference changed."""
        with _LOCK:
            self._persist()

    # ----- drone frames --------------------------------------------------
    def add_drone(self, rec: DroneRecord) -> DroneRecord:
        with _LOCK:
            self.drones[rec.drone_id] = rec
            self._persist()
        return rec

    def get_drone(self, drone_id: str) -> Optional[DroneRecord]:
        return self.drones.get(drone_id)

    # ----- plans -------------------------------------------------------
    def add_plan(self, rec: PlanRecord) -> PlanRecord:
        with _LOCK:
            self.plans[rec.plan_id] = rec
            self._persist()
        return rec

    def get_plan(self, plan_id: str) -> Optional[PlanRecord]:
        return self.plans.get(plan_id)

    # ----- jobs (never persisted) -------------------------------------
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
