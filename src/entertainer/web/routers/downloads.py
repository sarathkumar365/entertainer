"""Getting a film into the library from the page, and watching it happen.

The same ``Worker`` methods the CLI calls, behind the same token as every other
write. Getting a film posts a request in the user's name and may join channels
on their account, so it is never reachable unauthenticated off loopback.

Every handler is synchronous and only reads or appends job files; the slow work
happens on the worker's thread, so a download never holds a request open.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...agent import jobs as J
from ...agent import run
from ...agent.worker import Worker
from ...config import acquire_settings
from ...errors import CannotAcquire
from ..context import AppContext, get_context

router = APIRouter()


class GetRequest(BaseModel):
    item_id: int | None = None
    tmdb_id: int | None = None
    imdb_id: str | None = None
    title: str | None = None


class Choice(BaseModel):
    index: int


def _worker(ctx: AppContext) -> Worker:
    if ctx.worker is None:
        raise HTTPException(503, "The download worker is not running in this app.")
    return ctx.worker


def _view(job: J.Job, w: Worker, full: bool = False) -> dict[str, Any]:
    out = job.detail() if full else job.summary()
    if job.id in w.progress:
        done, total, _ = w.progress[job.id]
        out["progress"] = {"done": done, "total": total}
    if full:
        out["offer"] = [_offered(c) for c in job.offer]
    return out


def _jellyfin(s) -> dict[str, Any]:
    from ...agent.library import check_jellyfin

    ok, message = check_jellyfin(s)
    return {"ok": ok, "message": message}


def _offered(c: dict) -> dict:
    return {k: c.get(k) for k in ("source", "label", "size", "score", "why")}


@router.get("/api/downloads")
def downloads(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Every job, newest first, and whether this machine can acquire at all."""
    w = _worker(ctx)
    s = acquire_settings()
    try:
        run.ready(s)
        problem = None
    except CannotAcquire as e:
        problem = str(e)
    jobs = [j for j in w.jobs.all() if not j.dry_run]
    return {
        "ready": problem is None,
        "problem": problem,
        "max_gb": s.max_gb,
        "jobs": [_view(j, w) for j in jobs],
        "joined": len(w.joined()),
        "jellyfin": _jellyfin(s),
        "measures": measures(jobs),
    }


@router.post("/api/downloads/leave-joined")
def leave_joined(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Leave every channel a bot made the account join. Registered before the
    ``{job_id}`` routes so the literal path is not read as a job id."""
    return {"message": _worker(ctx).leave_joined()}


@router.get("/api/downloads/{job_id}")
def download(job_id: str, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    w = _worker(ctx)
    return _view(w.jobs.get(job_id), w, full=True)


@router.post("/api/downloads")
def get_it(body: GetRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """"Get it": queue a job for one film, or return the one already asking for it."""
    if body.item_id is None and body.tmdb_id is None and not body.imdb_id:
        raise HTTPException(422, "Name the film by item_id, tmdb_id or imdb_id.")
    w = _worker(ctx)
    request = {k: v for k, v in body.model_dump().items() if v is not None}
    return _view(w.submit({**request, "via": "web"}), w)


@router.post("/api/downloads/{job_id}/choose")
def choose(job_id: str, body: Choice, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    w = _worker(ctx)
    return _view(w.choose(job_id, body.index), w)


@router.post("/api/downloads/{job_id}/wrong")
def wrong(job_id: str, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    w = _worker(ctx)
    return _view(w.wrong(job_id), w)


@router.post("/api/downloads/{job_id}/cancel")
def cancel(job_id: str, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    w = _worker(ctx)
    return _view(w.cancel(job_id), w)


def measures(jobs: list[J.Job]) -> dict[str, Any]:
    """The north star and its supports, from ``docs/agent/DESIGN.md`` phase 6.

    Wrong-file rate is wrong calls over files filed, counting a filed copy that
    was later called wrong as filed: it did reach the library, silently, which
    is exactly the cost the number exists to show.
    """
    wrong = sum(len(j.wrong) for j in jobs)
    filed = sum(1 for j in jobs if j.filed) + wrong
    asked = [j for j in jobs if j.folder]  # resolved to a film, so the rule had a say
    paused = sum(1 for j in asked if j.asked)
    return {
        "jobs": len(jobs),
        "filed": filed,
        "wrong": wrong,
        "wrong_rate": (wrong / filed) if filed else None,
        "pause_rate": (paused / len(asked)) if asked else None,
        "failed": sum(1 for j in jobs if j.state == J.FAILED),
        "target_wrong_rate": 0.05,
        "target_jobs": 20,
    }
