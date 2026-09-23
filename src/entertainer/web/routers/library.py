"""What you have saved, watched, and rated.

A library action is not a verdict, and the distinction is the point. Saving
something says you intend to watch it; it is never a training label, and it
is reversible. A verdict says what you thought; it trains the model and is
undone only by recording a different one.

Both live in the same append-only log, which is why "saved" means "the most
recent watchlist event for this title says active", not "a watchlist event
exists".
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ... import store
from ..context import AppContext, get_context
from ..present import present

router = APIRouter()


class LibraryAction(BaseModel):
    item_id: int
    action: Literal["save", "remove", "watched"]


def _shelf(con, item_ids: set[int]) -> list[dict]:
    rows = store.item_rows(con, item_ids)
    return sorted(
        (present(rows[item_id]) for item_id in item_ids if item_id in rows),
        key=lambda item: ((item["title"] or "").casefold(), item["item_id"]),
    )


@router.get("/api/library")
def library(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Three shelves, each excluding the ones further along.

    A rated title is not also listed as merely watched, and a watched one is
    not also listed as still saved — otherwise the same film appears three
    times and none of the counts mean anything.
    """
    with store.session(read_only=True) as con:
        rated = {
            int(row[0])
            for row in con.execute(
                "SELECT DISTINCT item_id FROM events WHERE kind = 'rate'"
            ).fetchall()
        }
        watched = {
            int(row[0])
            for row in con.execute(
                "SELECT DISTINCT item_id FROM events WHERE kind = 'seen'"
            ).fetchall()
        } - rated
        saved = store.active_watchlist(con) - watched - rated
        return {
            "saved": _shelf(con, saved),
            "watched": _shelf(con, watched),
            "rated": _shelf(con, rated),
        }


@router.post("/api/library/actions")
def library_action(
    body: LibraryAction,
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """Save, unsave, or mark watched. None of these is a verdict."""
    with store.session() as con:
        known = con.execute(
            "SELECT 1 FROM titles WHERE item_id = ?", [body.item_id]
        ).fetchone()
        if not known:
            raise HTTPException(404, "unknown catalogue title")

        if body.action == "save":
            store.log_event(con, body.item_id, "watchlist", source="web", context={"active": True})
        elif body.action == "remove":
            store.log_event(con, body.item_id, "watchlist", source="web", context={"active": False})
        else:
            # Watched, but with no opinion recorded. It leaves the watchlist
            # because it is no longer something you intend to get to.
            store.log_event(con, body.item_id, "seen", source="web", context={"from": "library"})
            store.log_event(con, body.item_id, "watchlist", source="web", context={"active": False})

    return {"ok": True, "action": body.action, "item_id": body.item_id}
