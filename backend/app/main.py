"""
innovX VisualNav - FastAPI application entry point.

    uvicorn app.main:app --reload

Swagger UI is served at /docs and the OpenAPI schema at /openapi.json.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import localize_router, system_router, upload_router
from app.config import settings
from app.localization.pipeline import warmup
from app.logging_config import get_logger, setup_logging
from app.models.loader import probe_capabilities

setup_logging()
log = get_logger(__name__)

DESCRIPTION = """
**GPS-denied drone visual localization.**

Upload a reference satellite/orthomosaic map and a downward-facing drone
capture; the pipeline tiles the map, retrieves candidate regions with global
descriptors, matches local features, verifies the geometry with RANSAC and
returns an *estimated* map position with a decomposed confidence score.

The service is allowed to answer `NO_MATCH` and never fabricates GPS
coordinates: latitude/longitude appear only when a georeference has been
supplied for the reference map.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    caps = probe_capabilities()
    log.info("innovX VisualNav %s starting | mode=%s | device=%s | retrieval=%s | matcher=%s",
             __version__, settings.app_mode, caps.device,
             caps.retrieval_backend, caps.matcher_backend)
    for note in caps.notes:
        log.warning(note)
    warmup()
    yield
    log.info("innovX VisualNav shutting down.")


app = FastAPI(
    title="innovX VisualNav API",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    contact={"name": "innovX"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(localize_router)
app.include_router(system_router)

# Generated renders and original uploads are served read-only to the frontend.
app.mount("/files/processed", StaticFiles(directory=settings.processed_dir),
          name="processed")
app.mount("/files/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")


# --------------------------------------------------------------------------
# Error handling - the server must degrade with a message, never crash.
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
    return JSONResponse(status_code=422, content={
        "status": "error",
        "error": f"Invalid value for '{field}': {first.get('msg', 'validation failed')}",
        "detail": str(exc.errors()),
    })


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"status": "error", "error": str(exc)})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    # Full traceback goes to the server log only - the response stays generic so
    # exception text and stack internals never reach the browser.
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={
        "status": "error",
        "error": "Internal error while processing the request.",
        "detail": None,
    })


@app.get("/", tags=["system"], summary="Service banner")
async def root() -> dict:
    return {
        "name": "innovX VisualNav",
        "subtitle": "GPS-Denied Drone Visual Localization",
        "version": __version__,
        "mode": settings.app_mode,
        "docs": "/docs",
    }
