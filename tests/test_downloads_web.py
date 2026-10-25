"""The Downloads endpoints: the page's half of the agent, behind the app's token."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ENTERTAINER_ACQUIRE_SOURCES", "archive")
    lib = tmp_path / "Movies"
    lib.mkdir()
    monkeypatch.setenv("ENTERTAINER_LIBRARY_DIR", str(lib))
    from entertainer import config
    from entertainer.web import app as webapp

    config.PATHS.ensure()
    return lambda **kw: webapp.create_app(live=True, **kw)


def test_get_it_queues_a_job_and_the_list_shows_it(app):
    client = TestClient(app())  # no `with`: the worker thread stays off
    job = client.post("/api/downloads", json={"imdb_id": "tt0013442"}).json()
    assert job["state"] == "queued" and job["via"] == "web"

    listing = client.get("/api/downloads").json()
    assert listing["ready"] and [j["id"] for j in listing["jobs"]] == [job["id"]]
    assert listing["measures"]["target_wrong_rate"] == 0.05

    detail = client.get(f"/api/downloads/{job['id']}").json()
    assert detail["offer"] == [] and detail["log"] == []


def test_asking_twice_returns_the_same_job(app):
    client = TestClient(app())
    a = client.post("/api/downloads", json={"imdb_id": "tt0013442"}).json()
    b = client.post("/api/downloads", json={"imdb_id": "tt0013442"}).json()
    assert a["id"] == b["id"]


def test_a_machine_that_cannot_acquire_says_why(app, monkeypatch):
    monkeypatch.delenv("ENTERTAINER_LIBRARY_DIR")
    client = TestClient(app())
    r = client.post("/api/downloads", json={"imdb_id": "tt0013442"})
    assert r.status_code == 409 and "ENTERTAINER_LIBRARY_DIR" in r.json()["detail"]
    listing = client.get("/api/downloads").json()
    assert not listing["ready"] and "ENTERTAINER_LIBRARY_DIR" in listing["problem"]


def test_a_request_must_name_a_film(app):
    assert TestClient(app()).post("/api/downloads", json={}).status_code == 422


def test_actions_on_the_wrong_state_are_refused_in_words(app):
    client = TestClient(app())
    job = client.post("/api/downloads", json={"imdb_id": "tt0013442"}).json()
    r = client.post(f"/api/downloads/{job['id']}/choose", json={"index": 0})
    assert r.status_code == 409 and "not waiting" in r.json()["detail"]
    r = client.post(f"/api/downloads/{job['id']}/wrong")
    assert r.status_code == 409
    assert client.post(f"/api/downloads/{job['id']}/cancel").json()["state"] == "cancelled"


def test_an_unknown_job_is_refused(app):
    r = TestClient(app()).get("/api/downloads/job-nope")
    assert r.status_code == 409 and "No download job" in r.json()["detail"]


def test_get_it_needs_the_token_off_loopback(app):
    client = TestClient(app(token="s3cret"))
    assert client.post("/api/downloads", json={"imdb_id": "tt1"}).status_code == 401
    assert client.get("/api/downloads").status_code == 401


def test_the_worker_starts_and_stops_with_the_app(app):
    a = app()
    with TestClient(a):
        assert a.state.ctx.worker._thread.is_alive()
    assert not a.state.ctx.worker._thread.is_alive()


def test_leaving_with_nothing_joined_says_so(app):
    client = TestClient(app())
    assert client.get("/api/downloads").json()["joined"] == 0
    r = client.post("/api/downloads/leave-joined").json()
    assert r["message"] == "No joined channels to leave."
