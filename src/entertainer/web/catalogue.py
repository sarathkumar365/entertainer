"""Search and batch title-resolution HTTP boundary."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from .. import store
from ..config import has_tmdb, language_label
from ..resolve import search as resolve_search
from .presentation import poster, present


class CheckTitlesRequest(BaseModel):
    titles: list[str] = Field(min_length=1, max_length=20)

    @field_validator("titles")
    @classmethod
    def nonempty_titles(cls, titles: list[str]) -> list[str]:
        cleaned = [title.strip() for title in titles if title.strip()]
        if not cleaned:
            raise ValueError("provide at least one title")
        return cleaned


def _tmdb_matches(query: str, known_tmdb: set[int | None]) -> list[dict]:
    if not has_tmdb():
        return []
    from ..data import tmdb

    try:
        hits = tmdb.search(query)[:5]
    except Exception:  # pragma: no cover - network dependent
        return []
    out = []
    for hit in hits:
        if hit.get("id") in known_tmdb:
            continue
        date = hit.get("release_date") or hit.get("first_air_date") or ""
        out.append({
            "tmdb_id": hit.get("id"), "kind": hit.get("_kind", "movie"),
            "title": hit.get("title") or hit.get("name"),
            "original_title": hit.get("original_title") or hit.get("original_name"),
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "language": hit.get("original_language"),
            "language_name": language_label(hit.get("original_language")),
            "overview": (hit.get("overview") or "")[:260] or None,
            "poster": poster(hit.get("poster_path")),
            "rating": hit.get("vote_average"), "votes": hit.get("vote_count"),
            "external": True,
        })
    return out


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["catalogue"])

    @router.post("/check-titles")
    def check_titles(body: CheckTitlesRequest) -> dict:
        """Resolve a bounded pasted list; prediction remains an explicit next step."""
        resolved: list[tuple[str, list[dict]]] = []
        with store.session(read_only=True) as con:
            for query in body.titles:
                hits = resolve_search(con, query, limit=5)
                rows = store.item_rows(con, [hit.item_id for hit in hits])
                catalogue = [present(rows[hit.item_id]) for hit in hits if hit.item_id in rows]
                resolved.append((query, catalogue))
        # TMDB calls can take seconds. Do not occupy a DuckDB connection while
        # waiting on them; only catalogue resolution needs the local database.
        results = [
            {
                "query": query,
                "catalogue": catalogue,
                "tmdb": _tmdb_matches(query, {item.get("tmdb_id") for item in catalogue}),
            }
            for query, catalogue in resolved
        ]
        return {"results": results}

    return router
