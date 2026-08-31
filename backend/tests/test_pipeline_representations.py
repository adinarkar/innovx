"""Phases 4 & failure-safety - multi-representation build, no regression."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.localization import domain_translation as dt
from app.localization import pipeline
from app.localization.preprocessing import run_preprocessing


@pytest.fixture(autouse=True)
def _reset_engine():
    dt.reset_translation_engine()
    yield
    dt.reset_translation_engine()


def _pre(frame):
    return run_preprocessing(frame, None)


def _capture():
    events = []
    return events, (lambda key, label, state, detail=None: events.append((key, state)))


def test_rgb_always_present_structural_on_by_default(synthetic_frame, monkeypatch):
    monkeypatch.setattr(settings, "structural_matching_enabled", True)
    monkeypatch.setattr(settings, "sat2map_enabled", False)
    events, stage = _capture()

    reps, meta, stats = pipeline._build_representations(_pre(synthetic_frame), stage)

    assert "rgb" in reps
    assert reps["rgb"] is not None
    assert "structural" in reps and reps["structural"].ndim == 3
    assert meta["structural"]["state"] == "ready"
    assert meta["map"]["state"] == "skipped"       # no checkpoint
    assert ("structure", "done") in events
    assert ("translate", "skipped") in events
    assert "structural_edge_density" in stats


def test_structural_disabled_is_skipped_not_failed(synthetic_frame, monkeypatch):
    monkeypatch.setattr(settings, "structural_matching_enabled", False)
    monkeypatch.setattr(settings, "sat2map_enabled", False)
    events, stage = _capture()

    reps, meta, _ = pipeline._build_representations(_pre(synthetic_frame), stage)

    assert list(reps) == ["rgb"]
    assert meta["structural"]["state"] == "skipped"
    assert ("structure", "skipped") in events


def test_structural_failure_is_caught(synthetic_frame, monkeypatch):
    monkeypatch.setattr(settings, "structural_matching_enabled", True)
    monkeypatch.setattr(settings, "sat2map_enabled", False)

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic structural failure")

    monkeypatch.setattr(pipeline.semantic, "build_structural_representation", _boom)
    events, stage = _capture()

    reps, meta, _ = pipeline._build_representations(_pre(synthetic_frame), stage)

    assert "structural" not in reps          # branch dropped, pipeline continues
    assert meta["structural"]["state"] == "skipped"
    assert "synthetic structural failure" in meta["structural"]["error"]


def test_preprocessing_unchanged_contract(synthetic_frame):
    pre = _pre(synthetic_frame)
    assert pre.matching_input.shape == synthetic_frame.shape
    assert pre.structural.shape == synthetic_frame.shape
