"""Durable, append-only progress records for a full local build.

The build runs in a terminal while Build Studio is a separate read-only
process.  Files are the boundary between them: a browser refresh or Studio
restart must never make a long-running build look as though it disappeared.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from .config import PATHS

STAGES = (
    ("sources", "Verifying source data"),
    ("catalogue", "Building the movie catalogue"),
    ("tmdb", "Enriching titles from TMDB"),
    ("prune", "Pruning by language"),
    ("embeddings", "Turning title text into signals"),
    ("cf", "Learning audience patterns"),
    ("fusion", "Fusing movie signals"),
    ("prior", "Learning a cautious starting point"),
)
HEARTBEAT_SECONDS = 10
STALE_SECONDS = HEARTBEAT_SECONDS * 3


def build_root(root: Path | None = None) -> Path:
    return root or PATHS.reports / "builds"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Reporter:
    """Writer owned only by ``ent setup``; Studio only reads its files."""

    def __init__(self, root: Path | None = None, settings: dict[str, Any] | None = None):
        self.root = build_root(root)
        self.build_id = f"build-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.path = self.root / self.build_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.events_path = self.path / "events.jsonl"
        self._lock = RLock()
        self.state: dict[str, Any] = {
            "id": self.build_id,
            "status": "running",
            "started_at": _now(),
            "updated_at": _now(),
            "stage": None,
            "stages": [
                {"id": key, "label": label, "status": "pending"} for key, label in STAGES
            ],
            "settings": settings or {},
        }
        self._write_state()
        self.event("build_started")

    def _write_state(self) -> None:
        tmp = self.path / ".state.tmp"
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path / "state.json")
        active_tmp = self.root / ".active.tmp"
        active_tmp.write_text(self.build_id, encoding="utf-8")
        os.replace(active_tmp, self.root / "active")

    def event(self, kind: str, **data: Any) -> None:
        with self._lock:
            record = {"at": _now(), "kind": kind, **data}
            with self.events_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
            self.state["updated_at"] = record["at"]
            self._write_state()

    def heartbeat(self, stage_id: str) -> None:
        """Refresh liveness without growing the immutable event log every ten seconds."""
        with self._lock:
            if self.state["status"] != "running" or self.state["stage"] != stage_id:
                return
            now = _now()
            self.state["updated_at"] = now
            self.state["heartbeat_at"] = now
            self._write_state()

    @contextmanager
    def stage(self, stage_id: str, **detail: Any) -> Iterator[None]:
        started = time.monotonic()
        with self._lock:
            self.state["stage"] = stage_id
            for stage in self.state["stages"]:
                if stage["id"] == stage_id:
                    stage["status"] = "running"
                    stage.update(detail)
        self.event("stage_started", stage=stage_id, **detail)
        stop = Event()
        pulse = Thread(
            target=lambda: self._pulse(stage_id, stop), name=f"entertainer-{stage_id}", daemon=True
        )
        pulse.start()
        try:
            yield
        except BaseException as exc:
            with self._lock:
                for stage in self.state["stages"]:
                    if stage["id"] == stage_id:
                        stage["status"] = "failed"
                        stage["error"] = str(exc)
                self.state["status"] = "failed"
            self.event("stage_failed", stage=stage_id, error=str(exc))
            raise
        else:
            elapsed = round(time.monotonic() - started, 3)
            with self._lock:
                for stage in self.state["stages"]:
                    if stage["id"] == stage_id:
                        stage["status"] = "complete"
                        stage["elapsed_seconds"] = elapsed
            self.event("stage_complete", stage=stage_id, elapsed_seconds=elapsed)
        finally:
            stop.set()
            pulse.join(timeout=1)

    def _pulse(self, stage_id: str, stop: Event) -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            self.heartbeat(stage_id)

    def complete(self, manifest_id: str | None = None) -> None:
        self.state["status"] = "complete"
        self.state["finished_at"] = _now()
        self.state["manifest_id"] = manifest_id
        self.event("build_complete", manifest_id=manifest_id)
        active = self.root / "active"
        if active.exists() and active.read_text(encoding="utf-8").strip() == self.build_id:
            active.unlink()

    def skip(self, stage_id: str, reason: str) -> None:
        for stage in self.state["stages"]:
            if stage["id"] == stage_id:
                stage["status"] = "skipped"
                stage["reason"] = reason
        self.event("stage_skipped", stage=stage_id, reason=reason)


def list_builds(root: Path | None = None) -> list[dict[str, Any]]:
    base = build_root(root)
    if not base.exists():
        return []
    records: list[dict[str, Any]] = []
    for state in base.glob("build-*/state.json"):
        try:
            record = json.loads(state.read_text(encoding="utf-8"))
            records.append(_observed_state(record))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda row: row.get("started_at", ""), reverse=True)


def read_build(build_id: str, root: Path | None = None) -> dict[str, Any] | None:
    path = build_root(root) / build_id / "state.json"
    try:
        return _observed_state(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


def _observed_state(record: dict[str, Any]) -> dict[str, Any]:
    """Report a dead writer honestly without mutating its immutable record."""
    if record.get("status") != "running":
        return record
    raw = record.get("heartbeat_at") or record.get("updated_at")
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(str(raw).replace("Z", "+00:00"))).total_seconds()
    except (TypeError, ValueError):
        age = STALE_SECONDS + 1
    if age <= STALE_SECONDS:
        return record
    observed = dict(record)
    observed["status"] = "interrupted"
    observed["interrupted"] = True
    return observed
