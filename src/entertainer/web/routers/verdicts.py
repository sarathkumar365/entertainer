"""Recording what someone thought, and reporting how much they have said."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ... import store
from ...config import language_label
from ...models.taste import VERDICTS
from ..context import AppContext, get_context
from ..present import present
from ..schemas import Verdict

router = APIRouter()


@router.post("/api/rate")
def rate(body: Verdict, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    with store.session(read_only=True) as con:
        # Without this an unknown id logs a verdict against a title that does
        # not exist: it never joins back to anything, so it is invisible in
        # every report while still counting towards the verdict total.
        known = con.execute(
            "SELECT 1 FROM titles WHERE item_id = ?", [body.item_id]
        ).fetchone()
        if not known:
            raise HTTPException(404, "unknown catalogue title")
        sealed = con.execute(
            "SELECT 1 FROM validation_cases WHERE item_id = ? AND status = 'sealed'",
            [body.item_id],
        ).fetchone()
    if sealed:
        raise HTTPException(400, "this title is a sealed validation case; reveal it through validation")
    if body.verdict == "unseen":
        with store.session() as con:
            store.log_event(con, body.item_id, "unseen", None, "web", {"answer": "unseen"})
        return {"ok": True, "verdict": "unseen"}
    if body.verdict not in VERDICTS:
        raise HTTPException(400, f"unknown verdict {body.verdict!r}")
    # A verdict given on a recommendation is the only thing that produces an
    # off-policy datapoint, and the join that finds them is temporal today —
    # a rating counts if it merely came after the impression. Stamping the
    # slate makes it exact.
    context = {}
    if body.slate_id:
        context["slate_id"] = body.slate_id
        if body.position is not None:
            context["position"] = body.position
    with store.session() as con:
        ctx.engine.record(
            con, body.item_id, body.verdict, source="web", context=context or None
        )
    return {"ok": True, "verdict": body.verdict}



@router.post("/api/undo")
def undo(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    with store.session() as con:
        # Verdicts only. A library action is reversible through the library
        # itself, and undo reaching past a rating to delete a save would
        # discard something the user did on purpose while the toast claimed
        # it had undone the rating.
        row = con.execute(
            "SELECT event_id, item_id FROM events "
            "WHERE source = 'web' AND kind IN ('rate', 'unseen') "
            "ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if not row:
            return {"ok": False}
        con.execute("DELETE FROM events WHERE event_id = ?", [row[0]])
        title = con.execute(
            "SELECT title FROM titles WHERE item_id = ?", [row[1]]
        ).fetchone()
    return {"ok": True, "item_id": int(row[1]), "title": title[0] if title else None}



@router.get("/api/progress")
def progress(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    with store.session(read_only=True) as con:
        counts = store.counts(con)
        by_lang = con.execute(
            """
            SELECT t.language, count(DISTINCT e.item_id) n
            FROM events e JOIN titles t USING (item_id)
            WHERE e.kind = 'rate' GROUP BY 1 ORDER BY n DESC, t.language ASC LIMIT 12
            """
        ).fetchall()
        recent = con.execute(
            """
            SELECT t.title, t.year, e.context FROM events e JOIN titles t USING (item_id)
            WHERE e.kind = 'rate' ORDER BY e.ts DESC LIMIT 8
            """
        ).fetchall()
    import json as _json

    return {
        "rated": counts["ratings"],
        "events": counts["events"],
        "titles": counts["titles"],
        "by_language": [
            {"code": c, "name": language_label(c), "count": int(n)} for c, n in by_lang
        ],
        "recent": [
            {
                "title": t,
                "year": y,
                "verdict": (_json.loads(ctx or "{}") or {}).get("verdict"),
            }
            for t, y, ctx in recent
        ],
    }



@router.get("/api/rated")
def rated_titles(limit: int = 200, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Latest explicit verdict per title, most recently changed first."""
    with store.session(read_only=True) as con:
        cur = con.execute(
            """
            SELECT t.*, e.value, e.context, e.ts FROM titles t JOIN (
                SELECT item_id, value, context, ts,
                       row_number() OVER (PARTITION BY item_id ORDER BY ts DESC) rn
                FROM events WHERE kind = 'rate'
            ) e USING (item_id)
            WHERE e.rn = 1 ORDER BY e.ts DESC LIMIT ?
            """,
            [max(1, min(limit, 500))],
        )
        columns = [d[0] for d in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    import json as _json

    return {
        "items": [
            {
                **present(row),
                "verdict": (_json.loads(row.get("context") or "{}") or {}).get("verdict"),
                "rated_at": str(row.get("ts") or ""),
            }
            for row in rows
        ]
    }
