"""Every request for a film, kept as an append-only file per job.

Files, not a table, for the reason ``build_events`` uses them: the worker, the
web page and the CLI all touch jobs, a crash must leave a readable record, and
a job is mostly a log. The first line is the request; every later line is one
thing that happened, with its reason. The current state is the fold of the
lines, so nothing is ever rewritten and nothing can disagree with the history.

The lines are also the eval set (``docs/agent/DESIGN.md`` phase 7): each job
keeps its full candidate list, the pick, and the user's verdict on it, so
production outcomes become replay cases without extra work. They live under
``data/``, which is gitignored, because the titles a person asks for are taste
data.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ..config import PATHS
from ..errors import AcquireBusy, EntertainerError

QUEUED = "queued"
SEARCHING = "searching"
PAUSED = "paused"
DOWNLOADING = "downloading"
FILED = "filed"
FAILED = "failed"
CANCELLED = "cancelled"
SEARCHED = "searched"  # a dry run: searched and recorded, never meant to download

TERMINAL = frozenset({FILED, FAILED, CANCELLED, SEARCHED})
RUNNING = frozenset({SEARCHING, DOWNLOADING})


def jobs_root() -> Path:
    return PATHS.root / "acquire" / "jobs"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Job:
    id: str
    request: dict[str, Any]
    created: str
    state: str = QUEUED
    reason: str = ""
    updated: str = ""
    pid: int | None = None
    folder: str | None = None
    imdb_id: str | None = None
    item_id: int | None = None
    target: dict = field(default_factory=dict)  # the film matched against, for replay
    candidates: list[dict] = field(default_factory=list)  # the latest search, best first
    pick: list[dict] = field(default_factory=list)  # the copies it was about to try
    offer: list[dict] = field(default_factory=list)  # what a pause shows
    choice: dict | None = None  # the copy the user chose at a pause
    filed: dict | None = None  # path, label, source of the copy in the library
    wrong: list[str] = field(default_factory=list)  # labels the user called the wrong file
    joins: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    verdict: bool = False  # the `request` event is in the log
    asked: bool = False  # paused for a choice at least once; the pause rate counts these
    confirmed: str | None = None  # the right file, named by the user for a seed case

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def dry_run(self) -> bool:
        return bool(self.request.get("dry_run"))

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "state": self.state, "reason": self.reason,
            "created": self.created, "updated": self.updated, "folder": self.folder,
            "imdb_id": self.imdb_id, "item_id": self.item_id, "title": self.request.get("title"),
            "via": self.request.get("via"), "filed": self.filed, "wrong": len(self.wrong),
        }

    def detail(self) -> dict[str, Any]:
        return {**self.summary(), "offer": self.offer, "choice": self.choice,
                "candidates": self.candidates[:15], "joins": self.joins, "log": self.log}


def fold(lines: list[dict]) -> Job:
    first = lines[0]
    job = Job(id=first["id"], request=first["request"], created=first["at"], updated=first["at"])
    for line in lines[1:]:
        kind = line["kind"]
        job.updated = line["at"]
        if kind == "state":
            job.state, job.reason = line["state"], line.get("reason", "")
            job.pid = line.get("pid")
            if job.state != PAUSED:
                job.offer = []
        elif kind == "target":
            job.folder, job.imdb_id, job.item_id = line["folder"], line["imdb_id"], line["item_id"]
            job.target = line.get("film") or job.target
        elif kind == "plan":
            job.candidates = line["candidates"]
        elif kind == "pick":
            job.pick = line["candidates"]
        elif kind == "offer":
            job.offer, job.asked = line["candidates"], True
        elif kind == "choice":
            job.choice = line["candidate"]
        elif kind == "filed":
            job.filed = {k: line[k] for k in ("path", "label", "source")}
            job.choice = None
        elif kind == "wrong":
            job.wrong.append(line["label"])
            job.filed = None
        elif kind == "join":
            job.joins.append({"link": line["link"], "chat": line.get("chat", ""), "left": False})
        elif kind == "left":
            for j in job.joins:
                if j["chat"] in line["chats"]:
                    j["left"] = True
        elif kind == "confirm":
            job.confirmed = line["label"]
        elif kind == "verdict":
            job.verdict = True
        elif kind == "log":
            job.log.append({"at": line["at"], "msg": line["msg"]})
    return job


class Jobs:
    """Reads and appends job files. Safe to share between threads in one process;
    separate processes only ever append whole lines, which ``O_APPEND`` keeps intact."""

    def __init__(self, root: Path | None = None):
        self._root = root
        self._lock = RLock()

    @property
    def root(self) -> Path:
        return self._root or jobs_root()

    def create(self, request: dict[str, Any], *, run_here: bool = False) -> Job:
        """A new job, queued for the app's worker, or with ``run_here`` already
        marked as this process's, in the same write, so no worker can take it first."""
        job_id = f"job-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.root.mkdir(parents=True, exist_ok=True)
        lines = [{"at": _now(), "kind": "request", "id": job_id, "request": request}]
        if run_here:
            lines.append({"at": _now(), "kind": "state", "state": SEARCHING, "reason": "",
                          "pid": os.getpid()})
        self._write(job_id, *lines)
        return self.get(job_id)

    @contextmanager
    def claim(self, job_id: str) -> Iterator[None]:
        """Run a job in one process only. The app's worker and ``ent acquire run``
        can both see a queued job; whoever takes this lock runs it. ``flock`` dies
        with its process, so a crash never leaves a job claimed."""
        fd = os.open(self.root / f"{job_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise AcquireBusy(f"Job {job_id} is already running in another process.") from None
            yield
        finally:
            os.close(fd)

    def append(self, job_id: str, kind: str, **data: Any) -> None:
        self._write(job_id, {"at": _now(), "kind": kind, **data})

    def state(self, job_id: str, state: str, reason: str = "", **data: Any) -> None:
        self.append(job_id, "state", state=state, reason=reason, **data)

    def note(self, job_id: str, msg: str) -> None:
        self.append(job_id, "log", msg=msg)

    def get(self, job_id: str) -> Job:
        path = self.root / f"{job_id}.jsonl"
        # The id arrives from a URL; it must name a file in this directory and nothing else.
        if path.parent != self.root or not path.is_file():
            raise EntertainerError(f"No download job {job_id}.")
        return fold(_read(path))

    def all(self) -> list[Job]:
        if not self.root.is_dir():
            return []
        found = []
        for path in self.root.glob("job-*.jsonl"):
            try:
                found.append(fold(_read(path)))
            except (ValueError, KeyError, IndexError):
                continue  # a file cut short by a crash mid-create is not a job
        return sorted(found, key=lambda j: j.id, reverse=True)

    def _write(self, job_id: str, *records: dict) -> None:
        # One write per call, so lines written together land together.
        line = "".join(json.dumps(r, default=str) + "\n" for r in records)
        with self._lock:
            fd = os.open(self.root / f"{job_id}.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                         0o600)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)


def _read(path: Path) -> list[dict]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue  # a torn last line from a crash; everything before it stands
    return lines
