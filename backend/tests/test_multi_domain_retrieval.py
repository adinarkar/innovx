"""Phase 5 - multi-domain candidate retrieval: union, dedup, source tracking."""
from __future__ import annotations

import types

import numpy as np
import pytest

from app.config import settings
from app.localization import pipeline


class _FakeEngine:
    """embed() ignores the image and returns a fixed map-domain descriptor."""
    def __init__(self):
        self.calls = 0

    def embed(self, _img):
        self.calls += 1
        return np.array([0, 0, 0, 1, 0], np.float32)


def _record():
    bank = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.1],
    ], np.float32)
    return types.SimpleNamespace(embeddings=bank, tiles=list(range(5)))


RGB_EMB = np.array([1, 0, 0, 0, 0], np.float32)


def test_disabled_matches_plain_top_k(monkeypatch):
    monkeypatch.setattr(settings, "map_domain_retrieval_enabled", False)
    rec = _record()
    hits, counts = pipeline._multi_domain_retrieval(
        _FakeEngine(), rec, RGB_EMB, {"map": np.zeros((8, 8, 3), np.uint8)}, 3)

    assert counts == {"rgb": 3}
    assert [h["index"] for h in hits] == [0, 1, 2]
    assert all(h["sources"] == ["rgb"] for h in hits)


def test_union_adds_map_domain_candidates(monkeypatch):
    monkeypatch.setattr(settings, "map_domain_retrieval_enabled", True)
    monkeypatch.setattr(settings, "map_domain_top_k", 10)
    monkeypatch.setattr(settings, "candidate_union_max", 18)
    rec = _record()
    engine = _FakeEngine()

    hits, counts = pipeline._multi_domain_retrieval(
        engine, rec, RGB_EMB, {"map": np.zeros((8, 8, 3), np.uint8)}, 3)

    by_index = {h["index"]: h for h in hits}
    assert set(by_index) == {0, 1, 2, 3, 4}          # unioned
    assert "rgb" in counts and "map" in counts
    assert by_index[3]["sources"] == ["map"]         # map-only candidate
    assert by_index[0]["sources"] == ["rgb", "map"]  # retrieved by both
    assert "map" in by_index[0]["similarity_by_source"]
    # RGB candidates still lead the ordering (priority preserved).
    assert [h["index"] for h in hits][:3] == [0, 1, 2]


def test_aux_retrieval_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(settings, "map_domain_retrieval_enabled", True)
    rec = _record()

    class _Boom:
        def embed(self, _img):
            raise RuntimeError("descriptor backend exploded")

    hits, counts = pipeline._multi_domain_retrieval(
        _Boom(), rec, RGB_EMB, {"map": np.zeros((8, 8, 3), np.uint8)}, 3)

    assert [h["index"] for h in hits] == [0, 1, 2]   # falls back to RGB only
    assert "map" not in counts


def test_union_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "map_domain_retrieval_enabled", True)
    monkeypatch.setattr(settings, "map_domain_top_k", 10)
    monkeypatch.setattr(settings, "candidate_union_max", 3)
    rec = _record()
    hits, _ = pipeline._multi_domain_retrieval(
        _FakeEngine(), rec, RGB_EMB, {"map": np.zeros((8, 8, 3), np.uint8)}, 3)
    assert len(hits) == 3
