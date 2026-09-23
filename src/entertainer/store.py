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
from .resources import duckdb_config

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

-- A validation case is immutable at prediction time.  The ordinary events
-- table remains the source of truth for training; this table is only the
-- evidence ledger that makes later claims checkable.
CREATE TABLE IF NOT EXISTS validation_batches (
    batch_id       VARCHAR PRIMARY KEY,
    created_at     TIMESTAMP,
    item_ids       VARCHAR,           -- JSON list; all titles chosen before reveal
    full_ranking   VARCHAR,           -- JSON item ids, best first
    ridge_ranking  VARCHAR,
    manifest_id    VARCHAR
);

CREATE TABLE IF NOT EXISTS validation_cases (
    case_id          VARCHAR PRIMARY KEY,
    batch_id         VARCHAR NOT NULL,
    item_id          INTEGER NOT NULL UNIQUE,
    sealed_at        TIMESTAMP,
    full_score       DOUBLE NOT NULL,
    full_std         DOUBLE NOT NULL,
    full_like_prob   DOUBLE NOT NULL,
    ridge_score      DOUBLE NOT NULL,
    ridge_like_prob  DOUBLE NOT NULL,
    status           VARCHAR NOT NULL DEFAULT 'sealed', -- sealed | revealed | unseen
    actual_reward    DOUBLE,
    actual_verdict   VARCHAR,
    revealed_at      TIMESTAMP
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

    # A read-only connection cannot create the file, so on a machine that has
    # never run a build the first read fails outright rather than returning
    # empty results. Create and migrate it first. This matters for the rating
    # interface in live mode, where there is no catalogue by design but the
    # verdict log still has to exist.
    if read_only and not PATHS.catalog_db.exists():
        bootstrap = duckdb.connect(str(PATHS.catalog_db), config=duckdb_config())
        bootstrap.execute(SCHEMA)
        for statement in MIGRATIONS:
            bootstrap.execute(statement)
        bootstrap.close()

    con = duckdb.connect(str(PATHS.catalog_db), read_only=read_only, config=duckdb_config())
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
    ts: str | None = None,
) -> None:
    """Append one event. `ts` backdates it, for verdicts imported elsewhere.

    Chronology is load-bearing: the prequential audit replays events in `ts`
    order, and the recency half-life weights by age. Stamping an imported
    two-year-old verdict with `now()` would both scramble the replay and
    treat it as fresh evidence.
    """
    if ts is None:
        con.execute(
            """
            INSERT INTO events
            SELECT nextval('event_seq'), now(), ?, ?, ?, ?, ?
            """,
            [item_id, kind, value, source, json.dumps(context or {})],
        )
        return
    con.execute(
        """
        INSERT INTO events
        SELECT nextval('event_seq'), CAST(? AS TIMESTAMP), ?, ?, ?, ?, ?
        """,
        [ts, item_id, kind, value, source, json.dumps(context or {})],
    )


def import_event(
    con: duckdb.DuckDBPyConnection,
    item_id: int,
    kind: str,
    value: float | None,
    source: str,
    timestamp: str,
    context: dict | None = None,
) -> bool:
    """Restore one exported event once, retaining its original observation time."""
    # ``ent export`` parses and writes the JSON object in insertion order, so
    # preserving that order keeps a round-trip byte-identical to the original
    # event context without changing user-visible provenance.
    encoded = json.dumps(context or {})
    existing = con.execute(
        """
        SELECT 1 FROM events
        WHERE item_id = ? AND kind = ? AND value IS NOT DISTINCT FROM ?
          AND source = ? AND ts = CAST(? AS TIMESTAMP) AND context = ?
        LIMIT 1
        """,
        [item_id, kind, value, source, timestamp, encoded],
    ).fetchone()
    if existing:
        return False
    con.execute(
        """
        INSERT INTO events
        SELECT nextval('event_seq'), CAST(? AS TIMESTAMP), ?, ?, ?, ?, ?
        """,
        [timestamp, item_id, kind, value, source, encoded],
    )
    return True


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


def active_watchlist(con: duckdb.DuckDBPyConnection) -> set[int]:
    """Titles currently saved, rather than titles that were ever saved.

    The log is append-only, so saving and unsaving both write a ``watchlist``
    event and the most recent one for an item decides. Reading it as "has a
    watchlist event" would make removal impossible.

    Deliberately separate from a verdict: saving something says you intend to
    watch it, not that you liked it, so it is never a training label. It does
    exclude the title from future slates — there is no point recommending
    what someone has already decided to watch.
    """
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
        int(item_id)
        for item_id, context in rows
        if bool((json.loads(context or "{}") or {}).get("active"))
    }


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


#: Meta key holding the item ids of the most recent slate, so a verdict can
#: be given by position instead of by retyping a title. The writer (`ent
#: recs`) and the readers (the verdict commands) sit far apart, so the key is
#: named once here rather than spelled out at each end.
LAST_SLATE = "last_slate"


def set_last_slate(con: duckdb.DuckDBPyConnection, item_ids: list[int]) -> None:
    set_meta(con, LAST_SLATE, [int(i) for i in item_ids])


def last_slate(con: duckdb.DuckDBPyConnection) -> list[int]:
    return [int(i) for i in (get_meta(con, LAST_SLATE) or [])]


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
