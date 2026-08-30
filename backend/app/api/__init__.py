"""HTTP API routers."""
from app.api.routes_upload import router as upload_router
from app.api.routes_localize import router as localize_router
from app.api.routes_system import router as system_router

__all__ = ["upload_router", "localize_router", "system_router"]
