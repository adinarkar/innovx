"""HTTP-level smoke tests through FastAPI's TestClient (no live server needed)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_system_info_reports_classical_backend(client):
    r = client.get("/api/system/info")
    assert r.status_code == 200
    caps = r.json()["capabilities"]
    assert caps["retrieval_backend"] == "classical-embedding"
    assert caps["torch_available"] is False


def _upload_and_index(client, path, timeout=60.0):
    with path.open("rb") as fh:
        r = client.post("/api/map/upload", files={"file": ("ref_map.jpg", fh, "image/jpeg")})
    assert r.status_code == 200, r.text
    map_id = r.json()["map_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/map/{map_id}").json()["embedding_status"]
        if status == "ready":
            return map_id
        if status == "failed":
            pytest.fail("map indexing failed")
        time.sleep(0.3)
    pytest.fail("map indexing timed out")


def test_full_localization_flow(client, synthetic_map, drone_crop):
    map_id = _upload_and_index(client, synthetic_map["path"])

    with drone_crop["path"].open("rb") as fh:
        r = client.post("/api/drone/upload",
                        files={"file": ("drone.jpg", fh, "image/jpeg")})
    assert r.status_code == 200, r.text
    drone_id = r.json()["drone_id"]

    r = client.post("/api/localize", json={"map_id": map_id, "drone_id": drone_id})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + 60.0
    state = None
    while time.time() < deadline:
        body = client.get(f"/api/process/{job_id}").json()
        state = body["state"]
        if state in ("done", "error"):
            break
        time.sleep(0.3)
    assert state == "done", f"job ended in state {state}"

    result = client.get(f"/api/result/{job_id}").json()["result"]
    assert result["status"] in ("MATCH_FOUND", "LOW_CONFIDENCE")
    assert result["map_pixel"] is not None
    tx, ty = drone_crop["truth"]
    err = ((result["map_pixel"]["x"] - tx) ** 2 + (result["map_pixel"]["y"] - ty) ** 2) ** 0.5
    assert err < 150


def test_request_overrides_do_not_mutate_global_settings(client, synthetic_map, drone_crop):
    from app.config import settings

    before = (settings.matcher, settings.rotation_search, settings.top_k_candidates)
    map_id = _upload_and_index(client, synthetic_map["path"])
    with drone_crop["path"].open("rb") as fh:
        drone_id = client.post("/api/drone/upload",
                               files={"file": ("d.jpg", fh, "image/jpeg")}).json()["drone_id"]

    r = client.post("/api/localize", json={
        "map_id": map_id, "drone_id": drone_id,
        "matcher": "sift", "rotation_search": False, "top_k": 3,
    })
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if client.get(f"/api/process/{job_id}").json()["state"] in ("done", "error"):
            break
        time.sleep(0.3)

    assert (settings.matcher, settings.rotation_search, settings.top_k_candidates) == before


def test_localize_rejects_unknown_ids(client):
    r = client.post("/api/localize", json={"map_id": "map_missing", "drone_id": "drone_missing"})
    assert r.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/process/job_nope").status_code == 404
