"""Persistence layer.

A single DuckDB file holds the catalogue and the interaction log. DuckDB was
chosen over SQLite because the catalogue work is overwhelmingly analytical
(scan 300k rows, group by language, join against a ratings matrix) and over
Parquet-only because the interaction log needs cheap appends and updates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any

import duckdb

from .config import PATHS

SCHEMA = """
CREATE TABLE IF NOT EXISTS titles (
    item_id           INTEGER PRIMARY KEY,
    imdb_id           VARCHAR UNIQUE,
    tmdb_id           INTEGER,
    movielens_id      INTEGER,
    kind              VARCHAR,          -- movie | tv
    title             VARCHAR,
    original_title    VARCHAR,
    year              INTEGER,
    end_year          INTEGER,
    runtime           INTEGER,
    genres            VARCHAR[],
    language          VARCHAR,          -- primary language, ISO 639-1
    languages         VARCHAR[],
    countries         VARCHAR[],
    imdb_rating       DOUBLE,
    imdb_votes        INTEGER,
    tmdb_rating       DOUBLE,
    tmdb_votes        INTEGER,
    popularity        DOUBLE,
    directors         VARCHAR[],
    writers           VARCHAR[],
    cast_names        VARCHAR[],
    keywords          VARCHAR[],
    overview          VARCHAR,
    tagline           VARCHAR,
    poster_path       VARCHAR,
    adult             BOOLEAN,
    quality           DOUBLE,           -- shrunk quality prior, see scoring.py
    enriched_at       TIMESTAMP,
    -- When the keyword pass last ran for this title. Distinct from
    -- `keywords` being empty: TMDB genuinely has no keywords for a large
    -- part of the catalogue, and without this marker those titles are
    -- indistinguishable from unfetched ones and get re-requested on every
    -- run, forever.
    keywords_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    event_id   BIGINT PRIMARY KEY,
    ts         TIMESTAMP,
    item_id    INTEGER,
    kind       VARCHAR,   -- rate | skip | seen | unseen | watchlist | dismiss
    value      DOUBLE,    -- rating on 0..10 for 'rate', else NULL
    source     VARCHAR,   -- elicit | rec | manual | import
    context    VARCHAR    -- JSON blob: slate position, policy version, ...
);

CREATE TABLE IF NOT EXISTS impressions (
    impression_id BIGINT PRIMARY KEY,
    ts            TIMESTAMP,
    item_id       INTEGER,
    slate_id      VARCHAR,
    position      INTEGER,
    score         DOUBLE,
    propensity    DOUBLE,   -- P(item shown | policy), for off-policy evaluation
    explored      BOOLEAN,
    policy        VARCHAR
);

CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR
);

CREATE SEQUENCE IF NOT EXISTS event_seq START 1;
CREATE SEQUENCE IF NOT EXISTS impression_seq START 1;
"""


# Columns added after the first release. DuckDB has no migration framework,
# and a catalogue takes hours to rebuild, so schema changes are applied in
# place and idempotently rather than by asking anyone to start over.
MIGRATIONS = (
    "ALTER TABLE titles ADD COLUMN IF NOT EXISTS keywords_at TIMESTAMP",
)


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    PATHS.ensure()
    con = duckdb.connect(str(PATHS.catalog_db), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
        for statement in MIGRATIONS:
            con.execute(statement)
    return con


@contextmanager
def session(read_only: bool = False):
    con = connect(read_only=read_only)
    try:
        yield con
    finally:
        con.close()


# --- meta -------------------------------------------------------------------


def set_meta(con: duckdb.DuckDBPyConnection, key: str, value: Any) -> None:
    con.execute(
        "INSERT INTO meta VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [key, json.dumps(value)],
    )


def get_meta(con: duckdb.DuckDBPyConnection, key: str, default: Any = None) -> Any:
    row = con.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
    return json.loads(row[0]) if row else default


# --- events -----------------------------------------------------------------


def log_event(
    con: duckdb.DuckDBPyConnection,
    item_id: int,
    kind: str,
    value: float | None = None,
    source: str = "manual",
    context: dict | None = None,
) -> None:
    con.execute(
        """
        INSERT INTO events
        SELECT nextval('event_seq'), now(), ?, ?, ?, ?, ?
        """,
        [item_id, kind, value, source, json.dumps(context or {})],
    )


def log_impressions(
    con: duckdb.DuckDBPyConnection,
    slate_id: str,
    rows: Sequence[tuple[int, int, float, float, bool]],
    policy: str,
) -> None:
    """rows: (item_id, position, score, propensity, explored)."""
    con.executemany(
        """
        INSERT INTO impressions
        SELECT nextval('impression_seq'), now(), ?, ?, ?, ?, ?, ?, ?
        """,
        [(r[0], slate_id, r[1], r[2], r[3], r[4], policy) for r in rows],
    )


def ratings(con: duckdb.DuckDBPyConnection) -> list[tuple[int, float]]:
    """Latest explicit rating per item, 0..10."""
    return con.execute(
        """
        SELECT item_id, value FROM (
            SELECT item_id, value, row_number() OVER (PARTITION BY item_id ORDER BY ts DESC) rn
            FROM events WHERE kind = 'rate' AND value IS NOT NULL
        ) WHERE rn = 1
        """
    ).fetchall()


def negatives(con: duckdb.DuckDBPyConnection) -> list[tuple[int, float]]:
    """(item_id, age_in_days) for titles pushed away without being watched.

    A dismissal is weaker evidence than a verdict — "not tonight" is not "I
    disliked this" — but it is evidence, and it is the only negative signal
    available for titles the user never gets round to watching. Ages are
    returned so dismissals decay with time like everything else: what you did
    not fancy two years ago says little about tonight.
    """
    return [
        (int(r[0]), float(r[1] or 0))
        for r in con.execute(
            """
            SELECT item_id, min(date_diff('day', ts, now())) AS age FROM events
            WHERE kind IN ('skip', 'dismiss')
              AND item_id NOT IN (SELECT item_id FROM events WHERE kind = 'rate')
            GROUP BY item_id
            """
        ).fetchall()
    ]


# Answering "haven't seen it" is emphatically not a reason to stop
# recommending something — it is the single best reason to keep it in play.
# So `unseen` is recorded (to avoid asking twice) but deliberately excluded
# from the set below.
CONSUMED_KINDS = ("rate", "skip", "seen", "dismiss")


def interacted(con: duckdb.DuckDBPyConnection) -> set[int]:
    """Anything already watched or judged; never recommend these again."""
    placeholders = ", ".join(f"'{k}'" for k in CONSUMED_KINDS)
    return {
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT item_id FROM events WHERE kind IN ({placeholders})"
        ).fetchall()
    }


def already_asked(con: duckdb.DuckDBPyConnection) -> set[int]:
    """Everything ever put in front of the user, including titles they had not seen.

    Used by the cold-start loop so the same question is not asked twice, and
    kept separate from `interacted` precisely because the two must not be
    conflated: one governs what to stop recommending, the other what to stop
    asking about.
    """
    return {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT item_id FROM events WHERE kind IN "
            "('rate', 'skip', 'seen', 'dismiss', 'unseen')"
        ).fetchall()
    }


def new_slate_id() -> str:
    return f"slate-{int(time.time() * 1000):x}"


def counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    def one(sql: str) -> int:
        row = con.execute(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    return {
        "titles": one("SELECT count(*) FROM titles"),
        "enriched": one("SELECT count(*) FROM titles WHERE enriched_at IS NOT NULL"),
        "ratings": one("SELECT count(DISTINCT item_id) FROM events WHERE kind = 'rate'"),
        "skips": one("SELECT count(DISTINCT item_id) FROM events WHERE kind = 'skip'"),
        "events": one("SELECT count(*) FROM events"),
        "impressions": one("SELECT count(*) FROM impressions"),
    }


def item_rows(con: duckdb.DuckDBPyConnection, item_ids: Iterable[int]) -> dict[int, dict]:
    ids = list(item_ids)
    if not ids:
        return {}
    con.execute("CREATE OR REPLACE TEMP TABLE _wanted (item_id INTEGER)")
    con.executemany("INSERT INTO _wanted VALUES (?)", [(int(i),) for i in ids])
    cur = con.execute("SELECT t.* FROM titles t JOIN _wanted w USING (item_id)")
    cols = [d[0] for d in cur.description]
    return {r[0]: dict(zip(cols, r, strict=True)) for r in cur.fetchall()}
