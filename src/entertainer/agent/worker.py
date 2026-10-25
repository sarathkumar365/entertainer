"""Running jobs: one at a time, in a thread of its own, inside the app.

One thread with its own event loop, because the web handlers are synchronous,
Telethon is async, and a job blocks on ffprobe, file moves and a Jellyfin call;
on uvicorn's loop, one missed blocking call would freeze the page. Each job
gets a fresh loop through ``asyncio.run``, so the Telegram client is always
made on the loop that uses it.

A pause does not hold the worker. The job records what it offered and ends;
the user's choice puts it back in the queue, and it searches again and finds
the chosen copy by its label. A bot's buttons do not outlive the search that
showed them, and a pause may last a day, so holding the connection open for it
would buy nothing and block every other request behind one question.

``run_one`` is the whole of a job and is what the CLI calls too, so the web
page and the terminal are two front ends over one code path.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import store
from ..acquire import Target
from ..config import AcquireSettings, acquire_settings
from ..errors import AcquireBusy, CannotAcquire, CatalogueBusy, EntertainerError, MissingCatalogue
from . import jobs as J
from . import run
from .target import from_imdb, from_item

BUDGET_S = 45 * 60
PAUSE_S = 24 * 3600

Log = Callable[[str], None]


def resolve(job: J.Job) -> Target:
    """What was asked for, as a ``Target``, by IMDb id in every case."""
    req = job.request
    if req.get("item_id") is not None:
        with store.session(read_only=True) as con:
            return from_item(con, int(req["item_id"]))
    if req.get("imdb_id"):
        return from_imdb(req["imdb_id"])
    if req.get("tmdb_id") is not None:
        from ..data import tmdb

        detail = tmdb.detail(int(req["tmdb_id"]), "movie") or {}
        if not detail.get("imdb_id"):
            raise EntertainerError(
                "TMDB has no IMDb id for that film, so it cannot be matched safely."
            )
        return from_imdb(detail["imdb_id"])
    raise EntertainerError("The request names no film.")


def film(t: Target) -> dict[str, Any]:
    """The target as plain data, so a replay can match against exactly what the job did."""
    return {"title": t.title, "year": t.year, "runtime_min": t.runtime_min,
            "alt_titles": list(t.alt_titles), "imdb_id": t.imdb_id, "language": t.language}


def catalogue_id(imdb_id: str | None) -> int | None:
    if not imdb_id:
        return None
    try:
        with store.session(read_only=True) as con:
            row = con.execute(
                "SELECT item_id FROM titles WHERE imdb_id = ? LIMIT 1", [imdb_id]
            ).fetchone()
    except (CatalogueBusy, MissingCatalogue):
        return None
    return int(row[0]) if row else None


def log_request(jobs: J.Jobs, job: J.Job) -> None:
    """A request is a verdict: the user wanted this film, so stop recommending it.

    Written once per job, and only for a title the catalogue holds; a film from
    outside it has no item to attach to, and the job file still records it.
    """
    if job.verdict or job.item_id is None:
        return
    try:
        with store.session() as con:
            store.log_event(con, job.item_id, "request", source="agent", context={"job": job.id})
    except CatalogueBusy:
        jobs.note(job.id, "the catalogue was busy, so the request was not logged as a verdict")
        return
    jobs.append(job.id, "verdict")


def run_one(
    jobs: J.Jobs, job_id: str, settings: AcquireSettings | None = None, *,
    choose: run.Choose | None = None, progress: Callable[[int, int], None] | None = None,
    log: Log | None = None, budget_s: float = BUDGET_S,
    on_task: Callable[[asyncio.AbstractEventLoop, asyncio.Task], None] | None = None,
) -> J.Job:
    """Take one job from wherever it stands to filed, paused, failed or cancelled.

    ``choose`` answers a pause on the spot (the CLI prompt); without one a pause
    is recorded and the job waits. ``on_task`` hands over the running task so
    another thread can cancel it.
    """
    with jobs.claim(job_id):
        return _run_claimed(jobs, job_id, settings, choose=choose, progress=progress, log=log,
                            budget_s=budget_s, on_task=on_task)


def _run_claimed(
    jobs: J.Jobs, job_id: str, settings: AcquireSettings | None, *,
    choose: run.Choose | None, progress: Callable[[int, int], None] | None,
    log: Log | None, budget_s: float,
    on_task: Callable[[asyncio.AbstractEventLoop, asyncio.Task], None] | None,
) -> J.Job:
    s = settings or acquire_settings()
    job = jobs.get(job_id)
    if job.terminal:
        return job  # cancelled while it waited for the claim
    say = log or (lambda _msg: None)

    def note(msg: str) -> None:
        jobs.note(job_id, msg)
        say(msg)

    jobs.state(job_id, J.SEARCHING, pid=os.getpid())
    target: Target | None = None
    try:
        target = resolve(job)
        if job.folder is None:
            item_id = job.request.get("item_id") or catalogue_id(target.imdb_id)
            jobs.append(job_id, "target", folder=target.folder, imdb_id=target.imdb_id,
                        item_id=item_id, film=film(target))
            job = jobs.get(job_id)
        log_request(jobs, job)

        def report(kind: str, **data: Any) -> None:
            jobs.append(job_id, kind, **data)
            if kind == "pick":
                jobs.state(job_id, J.DOWNLOADING, pid=os.getpid())

        def wait_for_user(d: run.Decision) -> None:
            raise run.Paused(d)

        async def attempt() -> run.Acquired:
            if on_task:
                on_task(asyncio.get_running_loop(), asyncio.current_task())
            return await asyncio.wait_for(run.get(
                target, s, sources=job.request.get("sources"), progress=progress, log=note,
                choose=choose or wait_for_user,
                exclude=frozenset(job.wrong), chosen=job.choice, report=report,
            ), budget_s)

        got = asyncio.run(attempt())
    except run.Paused as p:
        jobs.append(job_id, "offer", candidates=[run.describe(c) for c in p.decision.offer])
        jobs.state(job_id, J.PAUSED, p.decision.reason)
    except asyncio.CancelledError:
        jobs.state(job_id, J.CANCELLED, "cancelled while it ran")
    except TimeoutError:
        jobs.state(job_id, J.FAILED, f"gave up after {budget_s / 60:g} minutes")
    except EntertainerError as e:
        jobs.state(job_id, J.FAILED, str(e))
    except Exception as e:  # noqa: BLE001 — a job's failure must not take the worker with it
        jobs.state(job_id, J.FAILED, f"{type(e).__name__}: {e}")
    else:
        jobs.append(job_id, "filed", path=str(got.path), label=got.candidate.label,
                    source=got.candidate.source)
        jobs.state(job_id, J.FILED, f"{got.path.name} from {got.candidate.source}")
    finally:
        if target is not None:
            _drop_partial(s, target.folder)
    return jobs.get(job_id)


def search_only(
    jobs: J.Jobs, job_id: str, settings: AcquireSettings | None = None, log: Log | None = None,
) -> tuple[J.Job, Any]:
    """A dry run, recorded: the candidate list becomes an eval case, nothing is fetched."""
    s = settings or acquire_settings()
    job = jobs.get(job_id)
    target = resolve(job)
    jobs.append(job_id, "target", folder=target.folder, imdb_id=target.imdb_id,
                item_id=job.request.get("item_id") or catalogue_id(target.imdb_id),
                film=film(target))
    p = asyncio.run(run.search(target, s, sources=job.request.get("sources"), log=log or print))
    jobs.append(job_id, "plan", candidates=[run.describe(c) for c in p.candidates],
                errors=p.errors)
    jobs.state(job_id, J.SEARCHED, f"{len(p.usable)} of {len(p.candidates)} files match")
    return jobs.get(job_id), p


def _drop_partial(s: AcquireSettings, folder: str) -> None:
    if s.staging is not None:
        (s.staging / f"{folder}.part").unlink(missing_ok=True)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _age_s(stamp: str) -> float:
    then = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (datetime.now(UTC) - then).total_seconds()


class Worker:
    """The queue, in the app process. Owns no state the job files do not hold."""

    def __init__(
        self, jobs: J.Jobs | None = None,
        settings: Callable[[], AcquireSettings] = acquire_settings,
        budget_s: float = BUDGET_S, pause_s: float = PAUSE_S, idle_s: float = 30.0,
    ):
        self.jobs = jobs or J.Jobs()
        self._settings = settings
        self.budget_s, self.pause_s, self.idle_s = budget_s, pause_s, idle_s
        # An Event, not a Condition: a poke that lands before the loop waits is kept.
        self._wake = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        self._task: tuple[asyncio.AbstractEventLoop, asyncio.Task] | None = None
        self._cancelling: str | None = None
        self.progress: dict[str, tuple[int, int, float]] = {}

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="acquire-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop = True
        self._cancel_current()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def recover(self) -> None:
        """A job left running by a process that is gone failed; its partial file goes too.

        No resume: a bot deletes its copy within minutes, so resuming would fail
        anyway. A job whose process is still alive is the CLI's, and is left alone.
        """
        s = self._settings()
        for job in self.jobs.all():
            if job.state in J.RUNNING and not _alive(job.pid):
                self.jobs.state(job.id, J.FAILED, "the app stopped while this job ran; ask again")
                if job.folder:
                    _drop_partial(s, job.folder)

    # --- what the page asks for --------------------------------------------

    def submit(self, request: dict[str, Any]) -> J.Job:
        s = self._settings()
        if not s.enabled:
            raise CannotAcquire("Acquisition is switched off on this machine (ENTERTAINER_ACQUIRE).")
        run.ready(s)
        key = _key(request)
        for job in self.jobs.all():
            if not job.terminal and _key(job.request) == key:
                return job  # asking twice is one request
        job = self.jobs.create(request)
        self._poke()
        return job

    def choose(self, job_id: str, index: int) -> J.Job:
        job = self.jobs.get(job_id)
        if job.state != J.PAUSED:
            raise EntertainerError("That job is not waiting for a choice.")
        if not 0 <= index < len(job.offer):
            raise EntertainerError("Choose one of the copies offered.")
        self.jobs.append(job_id, "choice", candidate=job.offer[index])
        self.jobs.state(job_id, J.QUEUED, f"chose {job.offer[index]['label']}")
        self._poke()
        return self.jobs.get(job_id)

    def wrong(self, job_id: str) -> J.Job:
        """The filed copy is not the film: delete it and try the next, never this one again."""
        job = self.jobs.get(job_id)
        if job.state != J.FILED or not job.filed:
            raise EntertainerError("Only a filed copy can be called the wrong file.")
        path = Path(job.filed["path"])
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()  # only if empty; anything else in it is not ours
        self.jobs.append(job_id, "wrong", label=job.filed["label"], source=job.filed["source"])
        self.jobs.state(job_id, J.QUEUED, f"{job.filed['label']} was the wrong file")
        self._poke()
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> J.Job:
        job = self.jobs.get(job_id)
        if job.terminal:
            raise EntertainerError(f"That job has already {job.state}.")
        if job.state in J.RUNNING and job.pid != os.getpid() and _alive(job.pid):
            raise EntertainerError("That job is running in a terminal; stop it there with ctrl-c.")
        if job_id == self._current:
            # Before the download starts there is no task yet; _hold cancels it on arrival.
            self._cancelling = job_id
            self._cancel_current()
        else:
            self.jobs.state(job_id, J.CANCELLED, "cancelled")
        return self.jobs.get(job_id)

    def joined(self) -> list[str]:
        """Chats the agent joined and has not left. A join whose chat is unknown
        (a pending invite request) cannot be left by id and is not counted."""
        chats: list[str] = []
        for job in self.jobs.all():
            chats += [j["chat"] for j in job.joins if j["chat"] and not j["left"]]
        return list(dict.fromkeys(chats))

    def leave_joined(self, leave: Callable | None = None) -> str:
        """Undo every recorded join. Takes the Telegram session, so it waits for no
        job: while one runs, the session lock refuses and the user tries again."""
        chats = self.joined()
        if not chats:
            return "No joined channels to leave."
        if leave is None:
            from ..acquire.telegram import leave
        left, failed = asyncio.run(leave(self._settings(), chats))
        if left:
            for job in self.jobs.all():
                if any(j["chat"] in left for j in job.joins):
                    self.jobs.append(job.id, "left", chats=left)
        msg = f"Left {len(left)} channel{'s' if len(left) != 1 else ''}."
        if failed:
            msg += f" {len(failed)} could not be left: " + "; ".join(failed.values())[:300]
        return msg

    # --- the loop ----------------------------------------------------------

    def _poke(self) -> None:
        self._wake.set()

    def _cancel_current(self) -> None:
        if self._task:
            loop, task = self._task
            loop.call_soon_threadsafe(task.cancel)

    def _next(self) -> J.Job | None:
        waiting = [j for j in self.jobs.all() if j.state == J.QUEUED]
        return min(waiting, key=lambda j: j.id) if waiting else None

    def _expire_pauses(self) -> None:
        for job in self.jobs.all():
            if job.state == J.PAUSED and _age_s(job.updated) > self.pause_s:
                # A missed film costs nothing; the user can ask again.
                self.jobs.state(job.id, J.CANCELLED, "no choice within a day")

    def _loop(self) -> None:
        while not self._stop:
            try:
                self._expire_pauses()
                job = self._next()
            except Exception:  # noqa: BLE001 — an unreadable directory must not kill the thread
                job = None
            if job is None:
                self._wake.wait(self.idle_s)
                self._wake.clear()
                continue
            self._current = job.id
            try:
                run_one(self.jobs, job.id, self._settings(), budget_s=self.budget_s,
                        progress=lambda d, t, i=job.id: self._progress(i, d, t),
                        on_task=self._hold)
            except AcquireBusy:
                # `ent acquire run` has it; look again once that process moves it on.
                self._wake.wait(self.idle_s)
            finally:
                self._current, self._task, self._cancelling = None, None, None
                self.progress.pop(job.id, None)

    def _hold(self, loop: asyncio.AbstractEventLoop, task: asyncio.Task) -> None:
        self._task = (loop, task)
        if self._stop or self._cancelling == self._current:
            loop.call_soon_threadsafe(task.cancel)

    def _progress(self, job_id: str, done: int, total: int) -> None:
        self.progress[job_id] = (done, total, time.time())


def _key(request: dict[str, Any]) -> tuple:
    return (request.get("item_id"), request.get("imdb_id"), request.get("tmdb_id"))

