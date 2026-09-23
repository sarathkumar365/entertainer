"""Shaping catalogue rows for the interface.

Pure functions of a row, shared by the feed, search, the rated list, slates
and additions. Kept out of the routers so that every endpoint returns the
same shape — a field added here appears everywhere at once, which is the
point.
"""

from __future__ import annotations

from ..config import language_label

POSTER_BASE = "https://image.tmdb.org/t/p/w342"


def poster(path: str | None) -> str | None:
    return f"{POSTER_BASE}{path}" if path else None


def present(row: dict) -> dict:
    """Shape a catalogue row for the interface.

    Deliberately omits tmdb_id and other internal identifiers. Callers that
    need to dedupe against TMDB should read the raw row — a search dedupe
    once compared against this output and silently never matched.
    """
    return {
        "item_id": row.get("item_id"),
        "title": row.get("title"),
        "original_title": (
            row.get("original_title")
            if row.get("original_title") and row.get("original_title") != row.get("title")
            else None
        ),
        "year": row.get("year"),
        "kind": row.get("kind"),
        "language": row.get("language"),
        "language_name": language_label(row.get("language")),
        "runtime": row.get("runtime"),
        "genres": list(row.get("genres") or [])[:3],
        "directors": list(row.get("directors") or [])[:2],
        "rating": row.get("imdb_rating"),
        "votes": row.get("imdb_votes"),
        "overview": (row.get("overview") or "")[:260] or None,
        "poster": poster(row.get("poster_path")),
    }
