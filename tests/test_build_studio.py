from __future__ import annotations

import json

from fastapi.testclient import TestClient

from entertainer.build_events import STALE_SECONDS, Reporter, list_builds, read_build
from entertainer.web import studio


def test_build_events_are_durable_and_append_only(tmp_path):
    reporter = Reporter(tmp_path, settings={"min_votes": 50})
    with reporter.stage("sources"):
        pass
    reporter.skip("tmdb", "no credentials")
    reporter.complete("build-manifest")

    state = read_build(reporter.build_id, tmp_path)
    assert state["status"] == "complete"
    assert state["manifest_id"] == "build-manifest"
    assert state["stages"][0]["status"] == "complete"
    assert state["stages"][2]["status"] == "skipped"
    events = (tmp_path / reporter.build_id / "events.jsonl").read_text().splitlines()
    assert len(events) == 5
    assert list_builds(tmp_path)[0]["id"] == reporter.build_id
    assert not (tmp_path / "active").exists()


def test_failed_stage_is_visible_to_observer(tmp_path):
    reporter = Reporter(tmp_path)
    try:
        with reporter.stage("catalogue"):
            raise RuntimeError("bad source")
    except RuntimeError:
        pass
    state = read_build(reporter.build_id, tmp_path)
    assert state["status"] == "failed"
    assert state["stages"][1]["status"] == "failed"
    assert state["stages"][1]["error"] == "bad source"


def test_stale_running_build_is_reported_as_interrupted(tmp_path):
    reporter = Reporter(tmp_path)
    state_path = tmp_path / reporter.build_id / "state.json"
    state = json.loads(state_path.read_text())
    state["updated_at"] = "2000-01-01T00:00:00Z"
    state_path.write_text(json.dumps(state))
    observed = read_build(reporter.build_id, tmp_path)
    assert observed["status"] == "interrupted"
    assert observed["interrupted"] is True
    assert STALE_SECONDS > 0


def test_studio_serves_page_and_builds(monkeypatch):
    row = {"id": "build-1", "status": "running", "updated_at": "now", "stages": []}
    monkeypatch.setattr(studio, "list_builds", lambda: [row])
    monkeypatch.setattr(studio, "read_build", lambda build_id: row if build_id == "build-1" else None)
    client = TestClient(studio.create_studio_app())
    assert client.get("/").status_code == 200
    assert client.get("/api/builds").json()["current"]["id"] == "build-1"
    assert client.get("/api/builds/build-1").json()["status"] == "running"
    assert client.get("/api/builds/../state").status_code == 404
