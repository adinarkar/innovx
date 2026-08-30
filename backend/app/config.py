"""
Central configuration for the innovX VisualNav backend.

Every tunable lives here so that no module needs hard-coded constants or
absolute paths.  Values are read from the environment (or a local .env file)
and fall back to prototype-friendly defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


def _csv_floats(raw: str) -> List[float]:
    return [float(v) for v in raw.replace(" ", "").split(",") if v]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"),
        extra="ignore",
        case_sensitive=False,
    )

    # --- server ---
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:5173"
    app_mode: str = "real"          # real | demo

    # --- devices / models ---
    model_device: str = "auto"      # auto | cuda | cpu | mps
    dino_model: str = "dinov2_vits14"

    # --- map handling ---
    # Denser tiling (more overlap, more tiles) means whichever window the
    # drone frame lands in is geometrically closer to its true field of view,
    # which directly improves inlier counts and reprojection error downstream.
    max_map_size: int = 8000
    tile_overlap: float = 0.35
    tile_scales_raw: str = "0.08,0.10,0.12,0.15,0.18,0.20,0.25"
    max_tiles: int = 900
    work_size: int = 960             # long edge used for feature extraction

    # --- retrieval / matching ---
    # A wider shortlist and a larger keypoint budget cost more time per
    # request but genuinely raise inlier counts and spatial coverage rather
    # than inflating the confidence score artificially.
    top_k_candidates: int = 8
    max_keypoints: int = 4096
    matcher: str = "lightglue"      # lightglue | sift

    # --- geometric verification ---
    # Thresholds expressed in pixels scale with work_size (they are measured
    # on the working-resolution image), so they are kept at the same ratio
    # as the original 640px baseline (5.0/640 and 8.0/640) when work_size
    # changes, rather than left stale at the old pixel scale.
    ransac_threshold: float = 7.5
    min_inliers: int = 15
    min_inlier_ratio: float = 0.30
    max_reprojection_error: float = 12.0
    min_spatial_coverage: float = 0.25
    coverage_grid: int = 4          # 4x4 grid, see spec section 18

    # --- decision thresholds ---
    match_confidence: float = 0.60
    low_confidence: float = 0.40
    ambiguity_gap: float = 0.06

    # --- search strategy ---
    rotation_search: bool = True
    # Evaluate all four orientations for every shortlisted candidate and keep
    # whichever produces the strongest verified geometry, instead of only
    # trying alternate rotations when the upright attempt already failed.
    # Slower, but it stops a slightly-off rotation from understating a
    # genuinely correct candidate's confidence.
    rotation_search_always: bool = True
    global_fallback: bool = True

    # --- storage ---
    upload_dir: Path = BACKEND_ROOT / "uploads"
    cache_dir: Path = BACKEND_ROOT / "cache"
    processed_dir: Path = BACKEND_ROOT / "processed"

    @property
    def tile_scales(self) -> List[float]:
        return _csv_floats(self.tile_scales_raw)

    @property
    def cors_origins(self) -> List[str]:
        origins = {self.frontend_origin, "http://localhost:5173", "http://127.0.0.1:5173"}
        return sorted(o for o in origins if o)

    def ensure_dirs(self) -> None:
        for d in (self.upload_dir, self.cache_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # TILE_SCALES is exposed with a friendlier env name than the field name.
    if "TILE_SCALES" in os.environ:
        os.environ.setdefault("TILE_SCALES_RAW", os.environ["TILE_SCALES"])
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
