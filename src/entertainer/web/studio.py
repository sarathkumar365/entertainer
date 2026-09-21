"""Read-only local web application for observing ``ent setup`` runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..build_events import list_builds, read_build

STATIC = Path(__file__).parent / "static"


def create_studio_app() -> FastAPI:
    app = FastAPI(title="entertainer Build Studio", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "studio.html")

    @app.get("/api/builds")
    def builds() -> dict:
        rows = list_builds()
        return {"builds": rows, "current": rows[0] if rows else None}

    @app.get("/api/builds/{build_id}")
    def build(build_id: str) -> dict:
        if "/" in build_id or "\\" in build_id:
            raise HTTPException(404, "build not found")
        row = read_build(build_id)
        if row is None:
            raise HTTPException(404, "build not found")
        return row

    @app.get("/api/builds/{build_id}/stream")
    async def stream(build_id: str) -> StreamingResponse:
        if "/" in build_id or "\\" in build_id or read_build(build_id) is None:
            raise HTTPException(404, "build not found")

        async def events():
            previous = None
            while True:
                row = read_build(build_id)
                if row is None:
                    return
                encoded = str(row.get("updated_at")) + str(row.get("status"))
                if encoded != previous:
                    yield f"data: {json.dumps(row)}\n\n"
                    previous = encoded
                if row.get("status") in {"complete", "failed", "interrupted"}:
                    return
                await asyncio.sleep(1)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
