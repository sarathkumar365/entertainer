"""The TMDB enrichment pass, against a mock transport.

Enrichment is the longest stage of a build and cannot be exercised live in
the suite, so the parts that decide how many requests are made, how many are
in flight, and what lands in the catalogue are pinned here instead.
"""

from __future__ import annotations

import asyncio
import re

import httpx
import pytest
from test_cli import app_env  # noqa: F401

from entertainer import store
from entertainer.commands import build
from entertainer.data import catalog, tmdb


class _NoLimit:
    def __init__(self, per_second: float):
        pass

    async def wait(self) -> None:
        return None


def _find_payload(imdb_id: str) -> dict:
    n = int(imdb_id[2:])
    return {
        "movie_results": [
            {
                "id": n,
                "overview": f" Synopsis {imdb_id}. ",
                "original_language": "ml",
                "original_title": f"Original {imdb_id}",
                "popularity": 3.5,
                "vote_average": 7.1,
                "vote_count": 42,
                "poster_path": f"/{imdb_id}.jpg",
                "genre_ids": [18, 53, 999999],
            }
        ],
        "tv_results": [],
    }


@pytest.fixture
def mock_tmdb(monkeypatch):
    """Route every TMDB request through ``handler`` and record what was asked."""
    state = {"handler": None, "calls": [], "in_flight": 0, "peak": 0}

    async def dispatch(request: httpx.Request) -> httpx.Response:
        state["calls"].append(request)
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            return await state["handler"](request)
        finally:
            state["in_flight"] -= 1

    def client(headers, concurrency):
        return httpx.AsyncClient(headers=headers, transport=httpx.MockTransport(dispatch))

    monkeypatch.setattr(tmdb, "_auth", lambda: ({}, {}))
    monkeypatch.setattr(tmdb, "_client", client)
    monkeypatch.setattr(tmdb, "RateLimiter", _NoLimit)
    return state


async def _find_handler(request: httpx.Request) -> httpx.Response:
    m = re.search(r"/find/(tt\d+)$", request.url.path)
    assert m, request.url
    await asyncio.sleep(0.001 * (int(m.group(1)[2:]) % 5))
    return httpx.Response(200, json=_find_payload(m.group(1)))


# --- worker pool ------------------------------------------------------------


def test_every_title_comes_back_exactly_once_in_full_batches(mock_tmdb):
    mock_tmdb["handler"] = _find_handler
    ids = [f"tt{i:07d}" for i in range(1, 24)]
    batches: list[list[str]] = []

    results = asyncio.run(
        tmdb.enrich_async(
            ids, concurrency=4, keywords=False,
            on_batch=lambda b: batches.append([r.imdb_id for r in b]), batch_size=5,
        )
    )

    assert sorted(r.imdb_id for r in results) == ids
    assert [len(b) for b in batches] == [5, 5, 5, 5, 3]
    assert sorted(i for b in batches for i in b) == ids
    assert len(mock_tmdb["calls"]) == len(ids)


def test_in_flight_requests_never_exceed_the_concurrency(mock_tmdb):
    mock_tmdb["handler"] = _find_handler
    asyncio.run(
        tmdb.enrich_async([f"tt{i:07d}" for i in range(1, 60)], concurrency=6, keywords=False)
    )
    # Equal, not merely at most: a pool that never fills is not a pool.
    assert mock_tmdb["peak"] == 6


def test_a_slow_request_holds_up_only_its_own_worker(mock_tmdb):
    """The lock-step batches this replaced made every batch wait on its
    slowest member. Here the other worker keeps draining the input."""
    finished: list[str] = []

    async def handler(request):
        imdb_id = request.url.path.rsplit("/", 1)[1]
        await asyncio.sleep(0.2 if imdb_id == "tt0000001" else 0.001)
        finished.append(imdb_id)
        return httpx.Response(200, json=_find_payload(imdb_id))

    mock_tmdb["handler"] = handler
    asyncio.run(
        tmdb.enrich_async([f"tt{i:07d}" for i in range(1, 21)], concurrency=2, keywords=False)
    )
    assert finished[-1] == "tt0000001"


def test_the_keyword_backfill_uses_the_same_pool(mock_tmdb):
    async def handler(request):
        tmdb_id = int(request.url.path.split("/")[-2])
        await asyncio.sleep(0.001 * (tmdb_id % 3))
        if "/tv/" in request.url.path:
            return httpx.Response(200, json={"results": [{"name": f"tv{tmdb_id}"}]})
        return httpx.Response(200, json={"keywords": [{"name": f"kw{tmdb_id}"}]})

    mock_tmdb["handler"] = handler
    targets = [(i, 1000 + i, "tv" if i % 4 == 0 else "movie") for i in range(17)]
    batches: list[list[tuple[int, list[str]]]] = []

    done = asyncio.run(
        tmdb.backfill_keywords_async(targets, concurrency=3, on_batch=batches.append, batch_size=4)
    )

    assert done == 17
    assert [len(b) for b in batches] == [4, 4, 4, 4, 1]
    got = dict(pair for b in batches for pair in b)
    assert got == {
        i: [f"{'tv' if kind == 'tv' else 'kw'}{tmdb_id}"] for i, tmdb_id, kind in targets
    }
    assert mock_tmdb["peak"] == 3


# --- one call when the tmdb_id is known -------------------------------------


def _movie_payload(tmdb_id: int, imdb_id: str) -> dict:
    return {
        "id": tmdb_id,
        "imdb_id": imdb_id,
        "overview": " A long synopsis. ",
        "tagline": " Every family has a secret. ",
        "original_language": "ml",
        "original_title": "Drishyam",
        "popularity": 9.25,
        "vote_average": 8.3,
        "vote_count": 1200,
        "poster_path": "/p.jpg",
        "genres": [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
        "keywords": {"keywords": [{"name": "murder"}, {"name": "kerala"}]},
    }


def test_a_known_tmdb_id_costs_exactly_one_request(mock_tmdb):
    async def handler(request):
        assert request.url.path.endswith("/movie/603")
        assert request.url.params["append_to_response"] == "keywords"
        return httpx.Response(200, json=_movie_payload(603, "tt10000003"))

    mock_tmdb["handler"] = handler
    [res] = asyncio.run(tmdb.enrich_async([("tt10000003", 603)], concurrency=2, keywords=False))

    assert len(mock_tmdb["calls"]) == 1
    assert res.found and res.kind == "movie" and res.tmdb_id == 603
    assert res.tagline == "Every family has a secret."
    assert res.overview == "A long synopsis."
    assert res.original_language == "ml"
    assert (res.popularity, res.tmdb_rating, res.tmdb_votes) == (9.25, 8.3, 1200)
    assert res.genres == ["Drama", "Thriller"]
    assert res.keywords == ["murder", "kerala"]
    assert res.keywords_fetched


def test_the_detail_path_maps_fields_the_way_find_does():
    """Titles enriched either way must be indistinguishable downstream."""
    find = _find_payload("tt0000007")["movie_results"][0]
    detail = {
        **find,
        "imdb_id": "tt0000007",
        "genres": [{"id": g, "name": "?"} for g in find["genre_ids"]],
    }
    via_find = tmdb._parse_find("tt0000007", {"movie_results": [find]})
    via_detail = tmdb._parse_movie("tt0000007", find["id"], detail)
    for name in ("tmdb_id", "kind", "overview", "original_language", "original_title",
                 "popularity", "tmdb_rating", "tmdb_votes", "poster_path", "genres", "found"):
        assert getattr(via_detail, name) == getattr(via_find, name), name


@pytest.mark.parametrize("dead", ["404", "mismatch"])
def test_a_stale_movielens_link_falls_back_to_find(mock_tmdb, dead):
    async def handler(request):
        if "/movie/" in request.url.path:
            if dead == "404":
                return httpx.Response(404, json={})
            return httpx.Response(200, json=_movie_payload(603, "tt9999999"))
        return await _find_handler(request)

    mock_tmdb["handler"] = handler
    [res] = asyncio.run(tmdb.enrich_async([("tt0000003", 603)], concurrency=1, keywords=False))
    assert len(mock_tmdb["calls"]) == 2
    assert res.found and res.tmdb_id == 3 and res.tagline is None
    assert not res.keywords_fetched


# --- through the command, into the catalogue --------------------------------


def _row(con, imdb_id: str) -> dict:
    cur = con.execute("SELECT * FROM titles WHERE imdb_id = ?", [imdb_id])
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone(), strict=True))


def _run_enrich(monkeypatch, floor_scale: float = 1.0):
    # Called as a plain function, so every Typer option must be passed.
    monkeypatch.setattr(build, "has_tmdb", lambda: True)
    build.data_enrich(limit=None, concurrency=4, keywords=False, floor_scale=floor_scale)


def test_titles_below_every_floor_are_skipped_and_left_unstamped(app_env, mock_tmdb, monkeypatch):  # noqa: F811
    mock_tmdb["handler"] = _find_handler
    floor = catalog.enrichment_floor(1.0)
    assert floor == min([*catalog.VOTE_FLOOR_BY_LANGUAGE.values(), catalog.VOTE_FLOOR_DEFAULT])
    with store.session() as con:
        con.execute("UPDATE titles SET enriched_at = NULL")
        con.execute(f"UPDATE titles SET imdb_votes = {floor - 1} WHERE imdb_id IN ('tt10000001', 'tt10000002')")
        con.execute("UPDATE titles SET imdb_votes = NULL WHERE imdb_id = 'tt10000003'")
        con.execute(f"UPDATE titles SET imdb_votes = {floor - 1} WHERE imdb_id = 'tt10000004'")
        rated = con.execute("SELECT item_id FROM titles WHERE imdb_id = 'tt10000004'").fetchone()[0]
        con.execute("INSERT INTO events (event_id, ts, item_id, kind, value) VALUES (1, now(), ?, 'rate', 8)", [rated])
        total = con.execute("SELECT count(*) FROM titles").fetchone()[0]

    pending = {i for i, _ in catalog.pending_enrichment()}
    assert "tt10000001" not in pending and "tt10000002" not in pending
    # The prune's own exemptions: no vote count, or the user has a verdict.
    assert {"tt10000003", "tt10000004"} <= pending
    assert len(pending) == total - 2

    _run_enrich(monkeypatch, floor_scale=1.0)
    with store.session(read_only=True) as con:
        assert _row(con, "tt10000001")["enriched_at"] is None
        assert _row(con, "tt10000002")["enriched_at"] is None
        assert _row(con, "tt10000003")["enriched_at"] is not None
        left = con.execute("SELECT count(*) FROM titles WHERE enriched_at IS NULL").fetchone()[0]
    assert left == 2

    # A later run at a lower scale still reaches them.
    assert {i for i, _ in catalog.pending_enrichment(floor_scale=0.5)} == {"tt10000001", "tt10000002"}


def test_a_known_tmdb_id_lands_with_tagline_keywords_and_keywords_at(app_env, mock_tmdb, monkeypatch):  # noqa: F811
    async def handler(request):
        if request.url.path.endswith("/movie/603"):
            return httpx.Response(200, json=_movie_payload(603, "tt10000003"))
        return await _find_handler(request)

    mock_tmdb["handler"] = handler
    with store.session() as con:
        con.execute("UPDATE titles SET enriched_at = NULL, keywords_at = NULL, tmdb_id = NULL")
        con.execute("UPDATE titles SET tmdb_id = 603, movielens_id = 1 WHERE imdb_id = 'tt10000003'")
        # A tmdb_id with no MovieLens provenance is not trusted to be a movie.
        con.execute("UPDATE titles SET tmdb_id = 777 WHERE imdb_id = 'tt10000011'")

    pending = dict(catalog.pending_enrichment())
    assert pending["tt10000003"] == 603 and pending["tt10000011"] is None

    _run_enrich(monkeypatch)
    paths = [r.url.path for r in mock_tmdb["calls"]]
    assert sum("/movie/603" in p for p in paths) == 1
    assert not any(p.endswith("/find/tt10000003") for p in paths)

    with store.session(read_only=True) as con:
        row = _row(con, "tt10000003")
        assert row["tagline"] == "Every family has a secret."
        assert list(row["keywords"]) == ["murder", "kerala"]
        assert row["keywords_at"] is not None and row["enriched_at"] is not None
        assert catalog.keyword_targets(con, 10_000) and all(
            item_id != row["item_id"] for item_id, _, _ in catalog.keyword_targets(con, 10_000)
        )
        # The /find path fetched no keywords, so it stamps nothing.
        assert _row(con, "tt10000011")["keywords_at"] is None


# --- set-based writes match the row-by-row ones they replaced --------------


def _old_apply_enrichment(con, results):
    con.executemany(
        """
        UPDATE titles SET
            tmdb_id     = coalesce(?, tmdb_id),
            overview    = coalesce(?, overview),
            tagline     = coalesce(?, tagline),
            language    = coalesce(?, language),
            popularity  = coalesce(?, popularity),
            tmdb_rating = coalesce(?, tmdb_rating),
            tmdb_votes  = coalesce(?, tmdb_votes),
            poster_path = coalesce(?, poster_path),
            keywords    = CASE WHEN len(?) > 0 THEN ? ELSE keywords END,
            genres      = list_distinct(list_concat(coalesce(genres, []), ?)),
            enriched_at = now()
        WHERE imdb_id = ?
        """,
        [
            (r.tmdb_id, r.overview, r.tagline, r.original_language, r.popularity,
             r.tmdb_rating, r.tmdb_votes, r.poster_path, r.keywords or [], r.keywords or [],
             r.genres or [], r.imdb_id)
            for r in results
        ],
    )


def _snapshot(con) -> list[tuple]:
    rows = con.execute(
        "SELECT imdb_id, tmdb_id, overview, tagline, language, popularity, tmdb_rating, "
        "tmdb_votes, poster_path, keywords, list_sort(genres), enriched_at IS NULL, "
        "keywords_at IS NULL FROM titles ORDER BY imdb_id"
    ).fetchall()
    return [tuple(tuple(v) if isinstance(v, list) else v for v in r) for r in rows]


def _twice(con, old, new) -> tuple[list, list]:
    con.execute("CREATE OR REPLACE TABLE _orig AS SELECT * FROM titles")
    old(con)
    expected = _snapshot(con)
    con.execute("DELETE FROM titles")
    con.execute("INSERT INTO titles SELECT * FROM _orig")
    new(con)
    return expected, _snapshot(con)


def test_set_based_enrichment_writes_what_the_row_by_row_update_did(app_env):  # noqa: F811
    R = tmdb.EnrichResult
    results = [
        R("tt10000001", tmdb_id=11, overview="New", tagline=None, original_language="ml",
          popularity=1.5, tmdb_rating=7.0, tmdb_votes=10, poster_path="/a.jpg",
          genres=["Drama", "Thriller"], keywords=[], found=True),
        R("tt10000002", tmdb_id=12, overview=None, original_language=None,
          genres=[], keywords=["heist", "kerala"], found=True),
        R("tt10000003"),  # not found: every field NULL, still stamped enriched
        R("tt10000012", tmdb_id=99, kind="tv", tmdb_votes=None, popularity=None,
          genres=["Mystery"], keywords=["time travel"], found=True),
        R("tt99999999", tmdb_id=1, overview="Not in the catalogue"),
    ]
    with store.session() as con:
        con.execute("UPDATE titles SET enriched_at = NULL, keywords_at = NULL")
        con.execute("UPDATE titles SET tagline = 'Old tag', poster_path = '/old.jpg', "
                    "tmdb_votes = 5 WHERE imdb_id = 'tt10000001'")
        before = _snapshot(con)
        expected, got = _twice(
            con,
            lambda c: _old_apply_enrichment(c, results),
            lambda c: catalog.apply_enrichment(c, results),
        )
    assert got == expected
    # Not vacuous: exactly the four catalogue titles were touched.
    touched = {a[0] for a, b in zip(before, got, strict=True) if a != b}
    assert touched == {"tt10000001", "tt10000002", "tt10000003", "tt10000012"}


def test_a_fetched_keyword_answer_stamps_keywords_at_even_when_empty(app_env):  # noqa: F811
    R = tmdb.EnrichResult
    with store.session() as con:
        con.execute("UPDATE titles SET keywords_at = NULL, keywords = ['kept']")
        catalog.apply_enrichment(con, [
            R("tt10000001", found=True, keywords=[], keywords_fetched=True),
            R("tt10000002", found=True, keywords=[], keywords_fetched=False),
        ])
        a, b = _row(con, "tt10000001"), _row(con, "tt10000002")
    assert a["keywords_at"] is not None and list(a["keywords"]) == ["kept"]
    assert b["keywords_at"] is None


def test_set_based_keyword_write_matches_the_row_by_row_update(app_env):  # noqa: F811
    with store.session() as con:
        ids = [r[0] for r in con.execute("SELECT item_id FROM titles ORDER BY item_id LIMIT 4").fetchall()]
        batch = [(ids[0], ["a", "b"]), (ids[1], []), (ids[2], ["c"]), (10_000_000, ["ghost"])]
        con.execute("UPDATE titles SET keywords_at = NULL")
        expected, got = _twice(
            con,
            lambda c: c.executemany(
                "UPDATE titles SET keywords = ?, keywords_at = now() WHERE item_id = ?",
                [(kws, item_id) for item_id, kws in batch],
            ),
            lambda c: catalog.apply_keywords(c, batch),
        )
    assert got == expected
