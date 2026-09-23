"""Serving the page itself.

The interface is a single-page app, so any path that is not an API call or a
static asset has to return index.html and let the router in the browser
decide. Without that, opening /taste directly — or reloading it — 404s.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

STATIC = Path(__file__).resolve().parents[1] / "static"

router = APIRouter()


@router.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@router.get("/{path:path}")
def spa_fallback(path: str) -> FileResponse:
    """Hand any unclaimed path to the browser's router.

    Registered last, so every real route wins first. API paths are excluded
    explicitly rather than by ordering alone: a mistyped endpoint should
    return 404, not a page — otherwise a broken fetch resolves with HTML and
    fails somewhere far from the cause.
    """
    if path.startswith(("api/", "static/")):
        raise HTTPException(404, "not found")
    return FileResponse(STATIC / "index.html")
