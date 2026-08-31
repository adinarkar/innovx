"""End-to-end localisation smoke tests on the synthetic map."""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.localization import pipeline as pl
from app.localization.pipeline import RunConfig, build_map_index, localize
from app.store import MapRecord, new_id


def test_warmup_is_safe_and_primes_the_engine():
    from app.localization import dino
    from app.localization.pipeline import warmup

    dino.reset_engine()
    warmup()                       # must not raise
    assert dino.get_engine().backend in ("classical-embedding", "dinov2")


def test_result_carries_a_decision_block(indexed_map, drone_crop, tmp_storage):
    res = localize(indexed_map, drone_crop["path"], tmp_storage / new_id("job"))
    d = res["decision"]
    assert set(d) >= {"margin", "runner_up_tile_id", "ambiguity_gap", "verified_candidates"}
    assert d["ambiguity_gap"] == 0.06
    assert d["verified_candidates"] >= 1


def test_index_builds_tiles_and_embeddings(indexed_map):
    assert indexed_map.embedding_status == "ready"
    assert len(indexed_map.tiles) > 0
    assert indexed_map.embeddings is not None
    assert indexed_map.embeddings.shape[0] == len(indexed_map.tiles)
    assert indexed_map.embedding_backend == "classical-embedding"


def test_localize_direct_crop_recovers_position(indexed_map, drone_crop, tmp_storage):
    res = localize(indexed_map, drone_crop["path"], tmp_storage / new_id("job"))
    assert res["status"] in ("MATCH_FOUND", "LOW_CONFIDENCE")
    assert res["map_pixel"] is not None
    tx, ty = drone_crop["truth"]
    err = np.hypot(res["map_pixel"]["x"] - tx, res["map_pixel"]["y"] - ty)
    assert err < 150, f"position error {err:.0f}px too large"


def test_localize_unrelated_frame_is_no_match(indexed_map, unrelated_frame, tmp_storage):
    res = localize(indexed_map, unrelated_frame["path"], tmp_storage / new_id("job"))
    assert res["status"] == "NO_MATCH"
    assert res["accepted"] is False
    assert res["map_pixel"] is None


def test_rotation_search_is_skipped_for_rotation_invariant_features(
        indexed_map, unrelated_frame, tmp_storage, monkeypatch):
    """With SIFT (rotation-invariant) the 90/180/270 search must never run,
    even for an all-weak frame with rotation_search explicitly enabled."""
    calls = []
    real_rotate = pl.rotate_image
    monkeypatch.setattr(pl, "rotate_image",
                        lambda img, k: calls.append(k) or real_rotate(img, k))

    cfg = RunConfig.build(matcher="sift", rotation_search=True)
    localize(indexed_map, unrelated_frame["path"], tmp_storage / new_id("job"),
             run_config=cfg)
    assert calls == []


def test_rotated_90_crop_still_localizes(indexed_map, synthetic_map, tmp_storage):
    """SIFT handles a 90-degree rotation without any rotation search."""
    img = synthetic_map["image"]
    w, h = synthetic_map["width"], synthetic_map["height"]
    cx, cy = int(w * 0.55), int(h * 0.5)
    side = min(int(round(math.sqrt(0.15 * w * h))), h, w)
    x = int(np.clip(cx - side / 2, 0, w - side))
    y = int(np.clip(cy - side / 2, 0, h - side))
    crop = img[y:y + side, x:x + side]
    rotated = np.rot90(crop, 1).copy()
    path = tmp_storage / "rot90.jpg"
    cv2.imwrite(str(path), rotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    res = localize(indexed_map, path, tmp_storage / new_id("job"))
    assert res["status"] in ("MATCH_FOUND", "LOW_CONFIDENCE")
    err = np.hypot(res["map_pixel"]["x"] - (x + side // 2),
                   res["map_pixel"]["y"] - (y + side // 2))
    assert err < 150


def test_low_texture_frame_gets_a_clear_message(indexed_map, tmp_storage):
    """A near-featureless frame must say *why* it failed, not imply it's
    simply outside the map."""
    blank = np.full((360, 360, 3), 127, np.uint8)
    path = tmp_storage / "blank.jpg"
    cv2.imwrite(str(path), blank)
    res = localize(indexed_map, path, tmp_storage / new_id("job"))
    assert res["status"] == "NO_MATCH"
    assert "texture" in res["explanation"] and "features detected" in res["explanation"]


def test_duplicated_map_region_is_not_confidently_localised(synthetic_map, tmp_storage):
    """When a distinctive patch appears in two places on the map, a crop of it
    must not be reported as a confident single fix."""
    base = synthetic_map["image"].copy()
    h, w = base.shape[:2]
    patch = base[300:580, 300:580].copy()
    base[h - 620:h - 340, w - 620:w - 340] = patch          # paste far away
    map_path = tmp_storage / "dup_map.jpg"
    cv2.imwrite(str(map_path), base, [cv2.IMWRITE_JPEG_QUALITY, 95])

    rec = MapRecord(map_id=new_id("map"), path=map_path, width=w, height=h,
                    filename="dup_map.jpg", file_size=map_path.stat().st_size)
    build_map_index(rec)

    crop_path = tmp_storage / "dup_crop.jpg"
    cv2.imwrite(str(crop_path), patch[20:240, 20:240], [cv2.IMWRITE_JPEG_QUALITY, 95])
    res = localize(rec, crop_path, tmp_storage / new_id("job"))

    assert res["status"] in ("AMBIGUOUS", "LOW_CONFIDENCE", "NO_MATCH"), res["status"]
    if res["status"] == "MATCH_FOUND":                       # never reached
        raise AssertionError("duplicated region localised as a confident fix")


def test_confidence_degrades_with_image_quality(indexed_map, synthetic_map, tmp_storage):
    """Confidence must track match quality: a clean crop scores clearly higher
    than a heavily blurred one, and both still localise correctly."""
    img = synthetic_map["image"]
    w, hgt = synthetic_map["width"], synthetic_map["height"]
    side = min(int(round(math.sqrt(0.15 * w * hgt))), hgt, w)
    x, y = (w - side) // 2, (hgt - side) // 2
    crop = img[y:y + side, x:x + side]
    truth = (x + side // 2, y + side // 2)

    def conf_for(frame):
        p = tmp_storage / f"q_{new_id('f')}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = localize(indexed_map, p, tmp_storage / new_id("job"))
        assert r["status"] in ("MATCH_FOUND", "LOW_CONFIDENCE")
        err = np.hypot(r["map_pixel"]["x"] - truth[0], r["map_pixel"]["y"] - truth[1])
        assert err < 150
        return r["confidence"]

    clean = conf_for(crop)
    blurred = conf_for(cv2.GaussianBlur(crop, (0, 0), 5))
    assert clean > blurred
    assert clean >= 0.60


def test_localize_never_invents_gps_without_georeference(indexed_map, drone_crop,
                                                         tmp_storage):
    res = localize(indexed_map, drone_crop["path"], tmp_storage / new_id("job"))
    assert res["gps"] is None
    assert res["georeferenced"] is False


def _fresh_record(synthetic_map) -> MapRecord:
    return MapRecord(map_id=new_id("map"), path=synthetic_map["path"],
                     width=synthetic_map["width"], height=synthetic_map["height"],
                     filename="ref_map.jpg",
                     file_size=synthetic_map["path"].stat().st_size)


def test_index_cache_is_keyed_by_content_not_map_id(synthetic_map):
    """Two records over the same file share a cache path and reuse the index."""
    first = _fresh_record(synthetic_map)
    build_map_index(first)
    assert first.embedding_status == "ready"
    assert first.cache_path.exists()

    second = _fresh_record(synthetic_map)
    assert second.map_id != first.map_id
    assert second.cache_path == first.cache_path          # keyed by content hash

    build_map_index(second)
    assert second.embedding_status == "ready"
    assert len(second.tiles) == len(first.tiles)
    assert np.allclose(second.embeddings, first.embeddings)


def test_stale_signature_cache_is_rejected(synthetic_map, monkeypatch):
    """Changing a tiling setting invalidates an existing cache entry."""
    from app.config import settings

    rec = _fresh_record(synthetic_map)
    build_map_index(rec)
    assert rec.cache_path.exists()

    monkeypatch.setattr(settings, "tile_scales_raw", "0.10,0.20", raising=False)
    reloaded = _fresh_record(synthetic_map)
    assert reloaded.load_cache(expected_backend="classical-embedding") is False
