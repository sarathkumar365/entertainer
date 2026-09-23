"""Serving the page itself."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC = Path(__file__).resolve().parents[1] / "static"

router = APIRouter()


@router.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
