"""Reading the catalogue: what languages exist, what mode we are in, what to show, and search."""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from ... import store
from ...config import has_tmdb, language_label
from ...resolve import search as resolve_search
from ..context import AppContext, _catalogue_size, get_context
from ..feed import DEFAULT_WEIGHTS, FeedRequest, fetch
from ..present import poster, present


class CheckTitlesRequest(BaseModel):
    #: Bounded because each unresolved title costs a TMDB round trip, and an
    #: unbounded paste would hold the request open for minutes.
    titles: list[str] = Field(min_length=1, max_length=20)

    @field_validator("titles")
    @classmethod
    def nonempty_titles(cls, titles: list[str]) -> list[str]:
        cleaned = [title.strip() for title in titles if title.strip()]
        if not cleaned:
            raise ValueError("provide at least one title")
        return cleaned


router = APIRouter()


@router.get("/api/languages")
def languages(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    if ctx.use_live:
        from ..live import VOTE_FLOORS

        codes = [c for c in DEFAULT_WEIGHTS if c in VOTE_FLOORS]
        return {
            "languages": [
                {
                    "code": c,
                    "name": language_label(c),
                    "titles": 0,
                    "weight": DEFAULT_WEIGHTS.get(c, 0.0),
                }
                for c in codes
            ]
        }
    with store.session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT language, count(*) n FROM titles
            WHERE poster_path IS NOT NULL AND year >= ?
            GROUP BY 1 HAVING n >= 20 ORDER BY n DESC, language ASC
            """,
            [dt.date.today().year - 6],
        ).fetchall()
    return {
        "languages": [
            {
                "code": code,
                "name": language_label(code),
                "titles": int(n),
                "weight": DEFAULT_WEIGHTS.get(code, 0.0),
            }
            for code, n in rows
            if code != "xx"
        ]
    }



@router.get("/api/mode")
def mode(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return {
        "live": ctx.use_live,
        "catalogue": _catalogue_size(),
        "tmdb": has_tmdb(),
    }



@router.get("/api/feed")
def feed(years: int = 2,
    limit: int = 60,
    langs: str = "",
    kind: str = "",
    min_quality: float = 0.0,
    page: int = 0,
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    weights = dict(DEFAULT_WEIGHTS)
    if langs.strip():
        weights = {}
        for part in langs.split(","):
            if not part.strip():
                continue
            code, _, w = part.partition(":")
            try:
                weights[code.strip()] = float(w) if w else 1.0
            except ValueError:
                weights[code.strip()] = 1.0
    if ctx.use_live:
        if not has_tmdb():
            raise HTTPException(
                400, "no catalogue and no TMDB credentials — nothing to show"
            )
        # Imported inside the handler, deliberately. tests/test_web.py
        # monkeypatches live.fetch by module attribute; binding the name at
        # import time here would make that patch a silent no-op and the test
        # would hit the real TMDB code path with a fake key.
        from ..live import LiveRequest
        from ..live import fetch as live_fetch

        since = f"{dt.date.today().year - max(years, 0)}-01-01"
        items = live_fetch(
            LiveRequest(
                languages=weights,
                since=since,
                # Over-fetch because filtering locally saved ratings below
                # can remove part of TMDB's first result page.
                limit=max(1, min(limit * 3, 200)),
                page=page + 1,
                kind=kind or "movie",
            )
        )
        with store.session(read_only=True) as con:
            rated_tmdb = {
                int(row[0])
                for row in con.execute(
                    """
                    SELECT DISTINCT t.tmdb_id FROM events e JOIN titles t USING (item_id)
                    WHERE e.kind = 'rate' AND t.tmdb_id IS NOT NULL
                    """
                ).fetchall()
            }
        items = [it for it in items if int(it.get("tmdb_id") or -1) not in rated_tmdb]
        for it in items:
            it["poster"] = poster(it.pop("poster_path", None))
            it["external"] = True
        return {"items": items[:limit], "live": True}

    req = FeedRequest(
        years=years,
        limit=max(1, min(limit, 200)),
        languages=weights,
        min_quality=min_quality,
        kind=kind or None,
        offset_seed=page,
    )
    with store.session(read_only=True) as con:
        rows = fetch(con, req, dt.date.today().year)
    return {"items": [present(r) for r in rows], "live": False}



@router.get("/api/search")
def search(q: str, limit: int = 12, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Catalogue first, then TMDB for anything the catalogue lacks."""
    if not q.strip():
        return {"catalogue": [], "tmdb": []}
    with store.session(read_only=True) as con:
        hits = resolve_search(con, q, limit=limit)
        rows = store.item_rows(con, [h.item_id for h in hits])
    # tmdb_id rides along so /api/judge can fall back to TMDB for a
    # catalogue title the item space does not hold.
    catalogue = [
        {**present(rows[h.item_id]), "tmdb_id": rows[h.item_id].get("tmdb_id")}
        for h in hits
        if h.item_id in rows
    ]

    external: list[dict] = []
    if has_tmdb():
        from ...data import tmdb

        # Built from the raw rows, not from _present output: _present
        # does not emit tmdb_id, so this set was {None} and the dedupe
        # below never fired — every catalogue title TMDB also knew was
        # listed twice.
        known_tmdb = {
            rows[h.item_id].get("tmdb_id") for h in hits if h.item_id in rows
        }
        known_tmdb.discard(None)
        try:
            for hit in tmdb.search(q)[:limit]:
                if hit.get("id") in known_tmdb:
                    continue
                date = hit.get("release_date") or hit.get("first_air_date") or ""
                external.append(
                    {
                        "tmdb_id": hit.get("id"),
                        "kind": hit["_kind"],
                        "title": hit.get("title") or hit.get("name"),
                        "original_title": hit.get("original_title")
                        or hit.get("original_name"),
                        "year": int(date[:4]) if date[:4].isdigit() else None,
                        "language": hit.get("original_language"),
                        "language_name": language_label(hit.get("original_language")),
                        "overview": (hit.get("overview") or "")[:260] or None,
                        "poster": poster(hit.get("poster_path")),
                        "rating": hit.get("vote_average"),
                        "votes": hit.get("vote_count"),
                    }
                )
        except Exception as exc:  # pragma: no cover - network dependent
            external = []
            return {"catalogue": catalogue, "tmdb": external, "tmdb_error": str(exc)}
    return {"catalogue": catalogue, "tmdb": external}


def _tmdb_matches(query: str, known_tmdb: set[int]) -> list[dict]:
    """TMDB results the catalogue does not already hold."""
    if not has_tmdb():
        return []
    # Function-local for the same reason live.fetch is: the suite patches
    # this module's tmdb by attribute.
    from ...data import tmdb

    try:
        hits = tmdb.search(query)[:5]
    except Exception:  # pragma: no cover - network dependent
        return []

    out = []
    for hit in hits:
        if hit.get("id") in known_tmdb:
            continue
        date = hit.get("release_date") or hit.get("first_air_date") or ""
        out.append(
            {
                "tmdb_id": hit.get("id"),
                "kind": hit["_kind"],
                "title": hit.get("title") or hit.get("name"),
                "original_title": hit.get("original_title") or hit.get("original_name"),
                "year": int(date[:4]) if date[:4].isdigit() else None,
                "language": hit.get("original_language"),
                "language_name": language_label(hit.get("original_language")),
                "overview": (hit.get("overview") or "")[:260] or None,
                "poster": poster(hit.get("poster_path")),
                "rating": hit.get("vote_average"),
                "votes": hit.get("vote_count"),
            }
        )
    return out


@router.post("/api/check-titles")
def check_titles(
    body: CheckTitlesRequest,
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """Resolve a pasted list of titles against the catalogue, then TMDB.

    Resolution is a separate step from prediction on purpose: a list pasted
    from somewhere else is usually full of near-misses, and scoring the wrong
    film is worse than scoring nothing.
    """
    resolved: list[tuple[str, list[dict], set[int]]] = []
    with store.session(read_only=True) as con:
        for query in body.titles:
            hits = resolve_search(con, query, limit=5)
            rows = store.item_rows(con, [hit.item_id for hit in hits])
            catalogue = [present(rows[hit.item_id]) for hit in hits if hit.item_id in rows]
            # Built from the raw rows: present() does not emit tmdb_id, and a
            # dedupe compared against its output silently never matches.
            known = {
                rows[hit.item_id]["tmdb_id"]
                for hit in hits
                if hit.item_id in rows and rows[hit.item_id].get("tmdb_id")
            }
            resolved.append((query, catalogue, known))

    # TMDB calls take seconds. Do not hold a DuckDB connection while waiting
    # on them; only catalogue resolution needs the local database.
    return {
        "results": [
            {"query": query, "catalogue": catalogue, "tmdb": _tmdb_matches(query, known)}
            for query, catalogue, known in resolved
        ]
    }
