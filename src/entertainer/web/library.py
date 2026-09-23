"""Library read model and append-only library actions."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import store
from .presentation import present


class LibraryAction(BaseModel):
    item_id: int
    action: Literal["save", "remove", "watched"]


def active_watchlist(con) -> set[int]:
    """Titles currently saved, rather than titles that were ever saved."""
    rows = con.execute(
        """
        SELECT item_id, context FROM (
            SELECT item_id, context, row_number() OVER (
                PARTITION BY item_id ORDER BY ts DESC, event_id DESC
            ) AS rn
            FROM events WHERE kind = 'watchlist'
        ) WHERE rn = 1
        """
    ).fetchall()
    return {
        int(item_id) for item_id, context in rows
        if bool((json.loads(context or "{}") or {}).get("active"))
    }


def _rows(con, item_ids: set[int]) -> list[dict]:
    rows = store.item_rows(con, item_ids)
    return sorted(
        (present(rows[item_id]) for item_id in item_ids if item_id in rows),
        key=lambda item: ((item["title"] or "").casefold(), item["item_id"]),
    )


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api/library", tags=["library"])

    @router.get("")
    def library() -> dict:
        with store.session(read_only=True) as con:
            rated = {int(row[0]) for row in con.execute(
                "SELECT DISTINCT item_id FROM events WHERE kind = 'rate'"
            ).fetchall()}
            watched = {int(row[0]) for row in con.execute(
                "SELECT DISTINCT item_id FROM events WHERE kind = 'seen'"
            ).fetchall()} - rated
            saved = active_watchlist(con) - watched - rated
            return {"saved": _rows(con, saved), "watched": _rows(con, watched), "rated": _rows(con, rated)}

    @router.post("/actions")
    def action(body: LibraryAction) -> dict:
        with store.session() as con:
            if not con.execute("SELECT 1 FROM titles WHERE item_id = ?", [body.item_id]).fetchone():
                raise HTTPException(404, "unknown catalogue title")
            if body.action == "save":
                store.log_event(con, body.item_id, "watchlist", source="web", context={"active": True})
            elif body.action == "remove":
                store.log_event(con, body.item_id, "watchlist", source="web", context={"active": False})
            else:
                store.log_event(con, body.item_id, "seen", source="web", context={"from": "library"})
                store.log_event(con, body.item_id, "watchlist", source="web", context={"active": False})
        return {"ok": True, "action": body.action, "item_id": body.item_id}

    return router
