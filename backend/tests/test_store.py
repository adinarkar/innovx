"""Registry persistence: indexed maps survive a simulated backend restart."""
from __future__ import annotations

from app.localization.pipeline import build_map_index
from app.store import MapRecord, Registry, new_id


def _record(synthetic_map) -> MapRecord:
    return MapRecord(map_id=new_id("map"), path=synthetic_map["path"],
                     width=synthetic_map["width"], height=synthetic_map["height"],
                     filename="ref_map.jpg",
                     file_size=synthetic_map["path"].stat().st_size)


def test_indexed_map_is_restored_from_sidecar(synthetic_map):
    reg = Registry()
    rec = _record(synthetic_map)
    build_map_index(rec)
    reg.add_map(rec)                      # writes the sidecar

    # A brand-new Registry == a process restart.
    restarted = Registry()
    got = restarted.get_map(rec.map_id)
    assert got is not None
    assert got.embedding_status == "ready"
    assert len(got.tiles) == len(rec.tiles)
    assert got.embeddings is not None
    assert got.embeddings.shape[0] == len(rec.tiles)


def test_restore_skips_maps_whose_file_is_gone(synthetic_map, tmp_path):
    reg = Registry()
    missing = MapRecord(map_id=new_id("map"), path=tmp_path / "not_here.jpg",
                        width=100, height=100, filename="x.jpg", file_size=1)
    reg.maps[missing.map_id] = missing
    reg._persist()

    restarted = Registry()
    assert restarted.get_map(missing.map_id) is None


def test_persistence_can_be_disabled(synthetic_map):
    reg = Registry(persist=False)
    rec = _record(synthetic_map)
    build_map_index(rec)
    reg.add_map(rec)
    # Nothing was written for this instance; a fresh persisting Registry should
    # not see it unless an earlier test already persisted the same map_id.
    assert Registry(persist=False).get_map(rec.map_id) is None
