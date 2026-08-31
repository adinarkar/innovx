"""
Satellite/aerial -> map-style domain translation (spec Phase 2 & 3, model
interface section).

This is an *optional* auxiliary representation. The localisation application
must work identically when the checkpoint is absent or ``SAT2MAP_ENABLED=false``:
in that case :attr:`DomainTranslationEngine.available` is ``False``, the
pipeline logs

    "Sat2Map translation unavailable - using standard localization pipeline."

and continues. A hallucinated road or building in the translated image can
never by itself override a failed RGB geometric verification - the fusion
logic downstream treats this branch as supporting evidence only.

The model (a small U-Net, trained separately under ``backend/training/sat2map``)
is loaded once and cached process-wide, mirroring :mod:`app.localization.dino`.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.models.loader import resolve_device, try_import_torch

log = get_logger(__name__)

# Working resolution the U-Net was trained at; input is resized to this and the
# output is resized back to the caller's resolution so coordinates are preserved.
TRANSLATE_SIZE = 256


class DomainTranslationEngine:
    """
    Clean interface over the Sat2Map translator.

    ``translate(image)`` takes a BGR/RGB numpy image and returns a map-like BGR
    numpy image of the same H x W. When the model is unavailable it must not be
    called (guard with :attr:`available`); callers fall back to the structural
    representation instead.
    """

    def __init__(self, checkpoint: Optional[Path] = None, device: Optional[str] = None):
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._reason: Optional[str] = None
        self.checkpoint = Path(checkpoint or settings.sat2map_model_path)

        if not settings.sat2map_enabled:
            self._reason = "SAT2MAP_ENABLED is false"
            return
        if not self.checkpoint.exists():
            self._reason = f"checkpoint not found: {self.checkpoint}"
            return

        torch = try_import_torch()
        if torch is None:
            self._reason = "PyTorch is not installed"
            return

        try:
            from app.localization._sat2map_net import UNetTranslator

            want = (device or settings.sat2map_device or "auto")
            self._device = "cpu" if want == "cpu" else resolve_device()
            ckpt = torch.load(str(self.checkpoint), map_location=self._device)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model = UNetTranslator(**(ckpt.get("model_kwargs", {})
                                      if isinstance(ckpt, dict) else {}))
            model.load_state_dict(state)
            model.eval().to(self._device)
            self._model = model
            self._torch = torch
            log.info("Sat2Map model loaded from %s on %s.", self.checkpoint, self._device)
        except Exception as exc:  # pragma: no cover - runtime/env dependent
            self._reason = f"failed to load checkpoint ({exc})"
            self._model = None

        if self._model is None and self._reason:
            log.warning("Sat2Map translation unavailable - using standard "
                        "localization pipeline. (%s)", self._reason)

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def status(self) -> str:
        return "ready" if self.available else (self._reason or "unavailable")

    def translate(self, image: np.ndarray) -> np.ndarray:
        """BGR/RGB numpy image -> map-like BGR numpy image, same H x W."""
        if not self.available:
            raise RuntimeError(f"Sat2Map translator is not available: {self.status}")
        if image is None or image.size == 0:
            raise ValueError("translate() received an empty image")

        torch = self._torch
        h, w = image.shape[:2]
        started = time.time()
        resized = cv2.resize(image, (TRANSLATE_SIZE, TRANSLATE_SIZE),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))[None].to(self._device)

        with torch.inference_mode():
            out = self._model(tensor)
        out = out.clamp(0.0, 1.0)[0].detach().cpu().numpy()
        out = (np.transpose(out, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        log.info("Sat2Map translation inference took %.3fs.", time.time() - started)
        return out


_ENGINE: Optional[DomainTranslationEngine] = None
_LOCK = threading.Lock()


def get_translation_engine() -> DomainTranslationEngine:
    """Process-wide singleton so the network loads at most once."""
    global _ENGINE
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = DomainTranslationEngine()
                log.info("Domain translation backend: %s", _ENGINE.status)
    return _ENGINE


def reset_translation_engine() -> None:
    global _ENGINE
    _ENGINE = None
