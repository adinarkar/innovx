"""
Lazy model loading and device selection.

The prototype is designed to run on a laptop with nothing but OpenCV
installed.  Every heavy dependency (torch, DINOv2, SuperPoint, LightGlue) is
imported lazily and each loader returns ``None`` when unavailable, so callers
can fall back to the classical path instead of crashing the server
(spec section 44: "Model loading failure").
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.config import settings
from app.logging_config import get_logger

log = get_logger(__name__)

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_FAILED: Dict[str, str] = {}


# --------------------------------------------------------------------------
# torch / device
# --------------------------------------------------------------------------
def try_import_torch():
    """Return the torch module, or None if torch is not installed."""
    if "torch" in _CACHE:
        return _CACHE["torch"]
    if "torch" in _FAILED:
        return None
    try:
        import torch  # noqa: WPS433 (deliberate lazy import)

        _CACHE["torch"] = torch
        return torch
    except Exception as exc:  # pragma: no cover - environment dependent
        _FAILED["torch"] = str(exc)
        log.warning("PyTorch unavailable (%s) - running in classical CV mode.", exc)
        return None


def resolve_device() -> str:
    """Pick cuda / mps / cpu honouring MODEL_DEVICE, degrading automatically."""
    wanted = (settings.model_device or "auto").lower()
    torch = try_import_torch()
    if torch is None:
        return "cpu"
    if wanted == "cpu":
        return "cpu"
    if wanted in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if wanted in ("mps", "auto") and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    if wanted not in ("auto", "cpu"):
        log.warning("Requested device '%s' unavailable - falling back to CPU.", wanted)
    return "cpu"


@dataclass
class Capabilities:
    """What the running installation can actually do, surfaced in the UI."""
    device: str = "cpu"
    torch: bool = False
    dinov2: bool = False
    superpoint: bool = False
    lightglue: bool = False
    sift: bool = True
    app_mode: str = "real"
    notes: list = field(default_factory=list)

    @property
    def retrieval_backend(self) -> str:
        return "dinov2" if self.dinov2 else "classical-embedding"

    @property
    def matcher_backend(self) -> str:
        if settings.matcher == "sift":
            return "sift"
        return "superpoint+lightglue" if self.lightglue else "sift"

    def to_dict(self) -> dict:
        return {
            "device": self.device.upper(),
            "app_mode": self.app_mode,
            "torch_available": self.torch,
            "dinov2_available": self.dinov2,
            "superpoint_available": self.superpoint,
            "lightglue_available": self.lightglue,
            "sift_available": self.sift,
            "retrieval_backend": self.retrieval_backend,
            "matcher_backend": self.matcher_backend,
            "dino_model": settings.dino_model,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# DINOv2
# --------------------------------------------------------------------------
def load_dinov2():
    """
    Load a pretrained DINOv2 backbone through torch.hub.

    No training happens here - DINOv2 is used purely as a frozen feature
    extractor for candidate retrieval (spec section 8).
    """
    key = "dinov2"
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if key in _FAILED:
            return None
        torch = try_import_torch()
        if torch is None:
            _FAILED[key] = "torch missing"
            return None
        try:
            log.info("Loading DINOv2 (%s) ...", settings.dino_model)
            model = torch.hub.load("facebookresearch/dinov2", settings.dino_model,
                                   trust_repo=True, verbose=False)
            model.eval().to(resolve_device())
            _CACHE[key] = model
            log.info("DINOv2 ready on %s.", resolve_device())
            return model
        except Exception as exc:  # pragma: no cover
            _FAILED[key] = str(exc)
            log.warning("DINOv2 unavailable (%s) - using classical embeddings.", exc)
            return None


# --------------------------------------------------------------------------
# SuperPoint + LightGlue
# --------------------------------------------------------------------------
def load_superpoint():
    key = "superpoint"
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if key in _FAILED:
            return None
        torch = try_import_torch()
        if torch is None:
            _FAILED[key] = "torch missing"
            return None
        try:
            from lightglue import SuperPoint

            extractor = SuperPoint(max_num_keypoints=settings.max_keypoints)
            extractor = extractor.eval().to(resolve_device())
            _CACHE[key] = extractor
            log.info("SuperPoint ready (max_num_keypoints=%d).", settings.max_keypoints)
            return extractor
        except Exception as exc:
            _FAILED[key] = str(exc)
            log.warning("SuperPoint unavailable (%s) - using SIFT.", exc)
            return None


def load_lightglue():
    key = "lightglue"
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if key in _FAILED:
            return None
        if load_superpoint() is None:
            _FAILED[key] = "superpoint missing"
            return None
        try:
            from lightglue import LightGlue

            matcher = LightGlue(features="superpoint").eval().to(resolve_device())
            _CACHE[key] = matcher
            log.info("LightGlue ready.")
            return matcher
        except Exception as exc:
            _FAILED[key] = str(exc)
            log.warning("LightGlue unavailable (%s) - using SIFT + FLANN.", exc)
            return None


# --------------------------------------------------------------------------
def probe_capabilities(warm: bool = False) -> Capabilities:
    """
    Report what is available.  With ``warm=False`` nothing heavy is imported
    beyond torch itself, keeping server start-up fast.
    """
    torch = try_import_torch()
    caps = Capabilities(device=resolve_device(), torch=torch is not None,
                        app_mode=settings.app_mode)
    if warm:
        caps.dinov2 = load_dinov2() is not None
        caps.superpoint = load_superpoint() is not None
        caps.lightglue = load_lightglue() is not None
    else:
        caps.dinov2 = "dinov2" in _CACHE
        caps.superpoint = "superpoint" in _CACHE
        caps.lightglue = "lightglue" in _CACHE
    if not caps.torch:
        caps.notes.append(
            "PyTorch not installed - DINOv2/SuperPoint/LightGlue are disabled. "
            "Retrieval uses a classical gradient+colour descriptor and matching uses SIFT. "
            "Install backend/requirements-ai.txt to enable the full AI pipeline."
        )
    return caps
