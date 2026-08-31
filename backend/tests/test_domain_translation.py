"""Phase 2 - translation module: missing checkpoint, disabled flag, mock model."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.localization import domain_translation as dt


@pytest.fixture(autouse=True)
def _reset_engine():
    dt.reset_translation_engine()
    yield
    dt.reset_translation_engine()


def test_unavailable_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "sat2map_enabled", False)
    engine = dt.get_translation_engine()
    assert engine.available is False
    assert "false" in engine.status.lower()


def test_unavailable_when_checkpoint_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "sat2map_enabled", True)
    monkeypatch.setattr(settings, "sat2map_model_path", tmp_path / "nope.pt")
    engine = dt.get_translation_engine()
    assert engine.available is False
    assert "not found" in engine.status


def test_translate_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "sat2map_enabled", False)
    engine = dt.get_translation_engine()
    with pytest.raises(RuntimeError):
        engine.translate(np.zeros((32, 32, 3), np.uint8))


class _T:
    """Minimal tensor-like wrapper over a numpy array."""
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def __getitem__(self, i):
        return _T(self.arr[i])

    def to(self, _device):
        return self

    def clamp(self, lo, hi):
        return _T(np.clip(self.arr, lo, hi))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _MockTorch:
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def inference_mode(self):
        return self._NoGrad()

    @staticmethod
    def from_numpy(arr):
        return _T(arr)


def test_mock_model_translate_preserves_shape():
    """A stand-in model exercises the pre/post-processing path without torch."""
    engine = dt.DomainTranslationEngine.__new__(dt.DomainTranslationEngine)
    engine._model = lambda tensor: _T(tensor.numpy())   # identity "network"
    engine._torch = _MockTorch()
    engine._device = "cpu"
    engine._reason = None

    src = (np.random.default_rng(1).integers(0, 255, (64, 96, 3))).astype(np.uint8)
    out = engine.translate(src)
    assert out.shape == src.shape
    assert out.dtype == np.uint8
