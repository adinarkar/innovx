"""
Background execution for map indexing and localisation jobs.

FastAPI request handlers stay thin: they validate, enqueue, and return a job
handle the frontend polls.  Long CV work runs on a small thread pool so a slow
DINOv2 pass on CPU never blocks the event loop.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from app.config import settings
from app.logging_config import get_logger
from app.localization.imaging import SUPPORTED_SUFFIXES, ImageError, imread
from app.localization.pipeline import build_map_index, localize
from app.localization.preprocessing import CameraCalibration
from app.store import DroneRecord, JobRecord, MapRecord, new_id, registry

log = get_logger(__name__)

EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="visualnav")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024      # 200 MB guard for oversized rasters
MIN_MAP_EDGE = 256                        # below this, tiling is meaningless


class UploadError(ValueError):
    """Raised for anything wrong with an uploaded file."""


# --------------------------------------------------------------------------
def save_upload(file: UploadFile, target_dir: Path, stem: str,
                allowed: set[str]) -> Path:
    """Persist an UploadFile, validating extension and size."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise UploadError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(allowed))}.")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / f"{stem}{suffix}"
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise UploadError("File exceeds the 200 MB upload limit.")
                out.write(chunk)
    except UploadError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise UploadError(f"Could not store the upload: {exc}") from exc
    finally:
        file.file.close()
    if size == 0:
        dest.unlink(missing_ok=True)
        raise UploadError("The uploaded file is empty.")
    return dest


def _downscale_if_needed(path: Path) -> None:
    """Cap the reference map at MAX_MAP_SIZE on its long edge, in place."""
    from app.localization.imaging import fit_long_edge, imwrite

    img = imread(path)
    resized, scale = fit_long_edge(img, settings.max_map_size)
    if scale < 1.0:
        imwrite(path, resized)
        log.info("Reference map downscaled by %.3f to fit MAX_MAP_SIZE=%d.",
                 scale, settings.max_map_size)


# --------------------------------------------------------------------------
def register_map(file: UploadFile) -> MapRecord:
    """Store a reference map and kick off indexing in the background."""
    map_id = new_id("map")
    path = save_upload(file, settings.upload_dir, map_id, SUPPORTED_SUFFIXES)
    try:
        _downscale_if_needed(path)
        img = imread(path)
    except ImageError as exc:
        path.unlink(missing_ok=True)
        raise UploadError(str(exc)) from exc

    h, w = img.shape[:2]
    if min(w, h) < MIN_MAP_EDGE:
        path.unlink(missing_ok=True)
        raise UploadError(
            f"Reference map is too small ({w}x{h}). It must be at least "
            f"{MIN_MAP_EDGE}px on the short edge to produce candidate regions.")

    rec = MapRecord(map_id=map_id, path=path, width=w, height=h,
                    filename=file.filename or path.name,
                    file_size=path.stat().st_size,
                    embedding_status="indexing")
    registry.add_map(rec)
    EXECUTOR.submit(_index_map, rec)
    return rec


def _index_map(rec: MapRecord) -> None:
    try:
        build_map_index(rec)
    except Exception as exc:  # pragma: no cover - defensive
        rec.embedding_status = "failed"
        rec.error = str(exc)
        log.exception("Background indexing failed for %s", rec.map_id)


def wait_for_index(rec: MapRecord, timeout: float = 900.0) -> None:
    """Block until a map finishes indexing (used at localisation time)."""
    deadline = time.time() + timeout
    while rec.embedding_status == "indexing" and time.time() < deadline:
        time.sleep(0.25)
    if rec.embedding_status == "failed":
        raise RuntimeError(rec.error or "Reference map indexing failed.")
    if rec.embedding_status != "ready":
        raise RuntimeError("Reference map is still indexing - try again shortly.")


def register_drone(file: UploadFile) -> DroneRecord:
    drone_id = new_id("drone")
    path = save_upload(file, settings.upload_dir, drone_id, SUPPORTED_SUFFIXES)
    try:
        img = imread(path)
    except ImageError as exc:
        path.unlink(missing_ok=True)
        raise UploadError(str(exc)) from exc
    h, w = img.shape[:2]
    rec = DroneRecord(drone_id=drone_id, path=path, width=w, height=h,
                      filename=file.filename or path.name,
                      file_size=path.stat().st_size)
    registry.add_drone(rec)
    return rec


# --------------------------------------------------------------------------
def start_job(map_id: str, drone_id: str, plan_id: Optional[str] = None,
              top_k: Optional[int] = None,
              calibration: Optional[dict] = None,
              search_region: Optional[tuple] = None,
              feature_params: Optional[dict] = None) -> JobRecord:
    """Create and enqueue a localisation job."""
    job = JobRecord(job_id=new_id("job"), map_id=map_id, drone_id=drone_id, plan_id=plan_id)
    registry.add_job(job)
    EXECUTOR.submit(_run_job, job, top_k, calibration, search_region, feature_params)
    return job


def _run_job(job: JobRecord, top_k: Optional[int], calibration: Optional[dict],
            search_region: Optional[tuple] = None,
            feature_params: Optional[dict] = None) -> None:
    job.state = "running"
    map_rec = registry.get_map(job.map_id)
    drone_rec = registry.get_drone(job.drone_id)
    try:
        if map_rec is None:
            raise ValueError("Reference map not found - upload it again.")
        if drone_rec is None:
            raise ValueError("Drone image not found - upload it again.")
        wait_for_index(map_rec)

        def progress(key: str, label: str, state: str, detail: Optional[str]) -> None:
            job.log_stage(key, label, state, detail)

        result = localize(map_rec, drone_rec.path, job.job_dir,
                          calibration=CameraCalibration.from_dict(calibration),
                          progress=progress, top_k=top_k, search_region=search_region,
                          feature_params=feature_params)
        result["mode"] = settings.app_mode
        if settings.app_mode == "demo":
            result["mode_warning"] = ("DEMO MODE - these figures come from the demo "
                                      "dataset and must not be presented as a real "
                                      "localisation result.")
        if job.plan_id and (plan := registry.get_plan(job.plan_id)):
            result["mission"] = plan.data
        job.result = result
        job.state = "done"
    except Exception as exc:
        job.state = "error"
        job.error = str(exc)
        job.log_stage(job.stage or "error", "Localization failed", "error", str(exc))
        log.exception("Job %s failed", job.job_id)
    finally:
        job.finished_at = time.time()


def clear_processed(job_id: str) -> None:
    """Remove a job's render directory (developer utility)."""
    path = settings.processed_dir / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
