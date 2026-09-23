"""Presentation-only shapes shared by the local web API."""

from __future__ import annotations

from ..config import language_label

POSTER_BASE = "https://image.tmdb.org/t/p/w342"


def poster(path: str | None) -> str | None:
    return f"{POSTER_BASE}{path}" if path else None


def present(row: dict) -> dict:
    """Return the stable, browser-safe representation of a catalogue title."""
    return {
        "item_id": row.get("item_id"),
        "tmdb_id": row.get("tmdb_id"),
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
        "rating": row.get("imdb_rating") or row.get("rating"),
        "votes": row.get("imdb_votes") or row.get("votes"),
        "overview": (row.get("overview") or "")[:260] or None,
        "poster": row.get("poster") or poster(row.get("poster_path")),
    }
