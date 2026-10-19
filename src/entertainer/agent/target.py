"""What the user asked for, as the one film the sources are matched against.

Resolved by IMDb id, never by a title search. Every title in the catalogue
already carries its IMDb id, and a title search is where the wrong film comes
from: *Vikram* is a 1986 film and a 2022 one, and "most popular" is not "the
one the user meant".
"""

from __future__ import annotations

import duckdb

from ..acquire import Target
from ..config import PRIORITY_LANGUAGES
from ..errors import EntertainerError


def from_imdb(imdb_id: str) -> Target:
    from ..data import tmdb

    d = tmdb.find_movie(imdb_id)
    if not d:
        raise EntertainerError(
            f"TMDB has no film for {imdb_id}. Only films can be fetched so far, not series."
        )
    date = d.get("release_date") or ""
    alt = d.get("original_title")
    return Target(
        title=d.get("title") or alt or imdb_id,
        year=int(date[:4]) if date[:4].isdigit() else None,
        runtime_min=d.get("runtime") or None,
        alt_titles=(alt,) if alt and alt != d.get("title") else (),
        imdb_id=imdb_id,
        language=PRIORITY_LANGUAGES.get(d.get("original_language") or ""),
    )


def from_item(con: duckdb.DuckDBPyConnection, item_id: int) -> Target:
    row = con.execute(
        "SELECT imdb_id, kind, title FROM titles WHERE item_id = ?", [int(item_id)]
    ).fetchone()
    if not row:
        raise EntertainerError(f"No title with id {item_id} in the catalogue.")
    imdb_id, kind, title = row
    if kind != "movie":
        raise EntertainerError(f"{title} is a series. Only films can be fetched so far.")
    if not imdb_id:
        raise EntertainerError(f"{title} has no IMDb id, so it cannot be matched safely.")
    return from_imdb(imdb_id)
