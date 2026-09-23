from __future__ import annotations

import json

from fastapi.testclient import TestClient

from entertainer.build_events import (
    STALE_SECONDS,
    Reporter,
    list_builds,
    read_build,
    read_events,
)
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


def test_progress_inside_a_stage_is_visible_without_growing_the_event_log(tmp_path):
    reporter = Reporter(tmp_path)
    with reporter.stage("tmdb"):
        reporter.progress("tmdb", 10_000, 77_662, note="keywords for 10,000 of 77,662 titles")
        stage = read_build(reporter.build_id, tmp_path)["stages"][2]
        assert stage["done"] == 10_000
        assert stage["total"] == 77_662
        assert stage["note"] == "keywords for 10,000 of 77,662 titles"
        assert stage["started_at"]
    events = (tmp_path / reporter.build_id / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in events] == [
        "build_started", "stage_started", "stage_complete"
    ]
    # Counts belong to work in flight, so a finished stage reports elapsed only.
    finished = read_build(reporter.build_id, tmp_path)["stages"][2]
    assert "done" not in finished and "elapsed_seconds" in finished


def test_progress_for_a_stage_that_is_not_running_is_ignored(tmp_path):
    reporter = Reporter(tmp_path)
    with reporter.stage("sources"):
        reporter.progress("embeddings", 5, 10)
    assert "done" not in read_build(reporter.build_id, tmp_path)["stages"][4]


def test_activity_endpoint_returns_the_event_tail(tmp_path, monkeypatch):
    reporter = Reporter(tmp_path)
    with reporter.stage("sources"):
        pass
    monkeypatch.setattr(studio, "read_build", lambda bid: read_build(bid, tmp_path))
    monkeypatch.setattr(studio, "read_events", lambda bid, limit=60: read_events(bid, limit, tmp_path))
    client = TestClient(studio.create_studio_app())
    kinds = [e["kind"] for e in client.get(f"/api/builds/{reporter.build_id}/activity").json()["events"]]
    assert kinds == ["build_started", "stage_started", "stage_complete"]
    assert client.get("/api/builds/../../etc/activity").status_code == 404


def test_a_passs_last_update_is_never_throttled_away(tmp_path):
    reporter = Reporter(tmp_path)
    with reporter.stage("tmdb"):
        reporter.progress("tmdb", 1, 100)
        # Immediately after, and short of the stage total: only ``final``
        # distinguishes the end of a pass from ordinary chatter.
        reporter.progress("tmdb", 40, 100, note="pass one done", final=True)
        stage = read_build(reporter.build_id, tmp_path)["stages"][2]
        assert stage["done"] == 40
        assert stage["note"] == "pass one done"
