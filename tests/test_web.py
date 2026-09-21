"""End-to-end tests for the rating interface.

Driven through the HTTP API the browser actually calls, against a synthetic
catalogue, so nothing here touches the network or the real event log.
"""

from __future__ import annotations

import datetime as dt
import importlib

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

LANGS = ("ml", "ta", "en", "ko", "te", "kn", "hi")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_BEARER", raising=False)

    from entertainer import config, engine, store
    from entertainer.models import encoder, fusion

    for mod in (config, store, engine, encoder, fusion):
        importlib.reload(mod)
    from entertainer import resolve
    from entertainer.models import features
    from entertainer.web import app as webapp
    from entertainer.web import feed

    for mod in (features, resolve, feed, webapp):
        importlib.reload(mod)

    config.PATHS.ensure()
    year = dt.date.today().year

    con = store.connect()
    n = 0
    for lang in LANGS:
        for j in range(40):
            con.execute(
                """
                INSERT INTO titles (item_id, imdb_id, tmdb_id, kind, title, original_title,
                    year, language, runtime, genres, imdb_rating, imdb_votes, quality,
                    overview, poster_path, directors, cast_names, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 120, ['Drama'], ?, ?, ?, ?, ?, ['A Dir'], [], [])
                """,
                [
                    n, f"tt{n:07d}", 900000 + n,
                    "tv" if j % 9 == 0 else "movie",
                    f"{lang.upper()} Film {j}", f"{lang.upper()} Film {j}",
                    year - (j % 4),              # spread over 4 years
                    lang, 6.0 + (j % 40) / 10.0, 5_000 + j * 700,
                    0.30 + (j % 50) / 71.0,      # spread of quality
                    f"A synopsis for {lang} film {j}.",
                    f"/poster{n}.jpg",
                ],
            )
            n += 1
    con.close()

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(n, 12)).astype(np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    ids = np.arange(n, dtype=np.int32)
    fusion.save(
        fusion.FusionArtifacts(
            item_ids=ids, space=latent, components=np.eye(12, dtype=np.float32),
            block_sizes=(12, 0), cf_r2=0.5, cf_coverage=1.0,
            pca_mean=np.zeros(12, dtype=np.float32),
            # A usable imputation map, so the projection path is exercised
            # rather than short-circuited.
            ridge_coef=np.eye(12, dtype=np.float32),
            ridge_intercept=np.zeros(12, dtype=np.float32),
        )
    )
    encoder.save(ids, latent)
    return TestClient(webapp.create_app(live=False))


def test_index_serves(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "entertainer" in r.text


def test_languages_lists_what_the_catalogue_has(client):
    d = client.get("/api/languages").json()
    codes = {row["code"] for row in d["languages"]}
    assert {"ml", "ta", "en", "ko"} <= codes
    weights = {row["code"]: row["weight"] for row in d["languages"]}
    # Telugu is present but deliberately de-emphasised.
    assert 0 < weights["te"] < weights["ml"]


def test_feed_is_recent_and_balanced(client):
    items = client.get("/api/feed?years=2&limit=40").json()["items"]
    assert items
    year = dt.date.today().year
    assert all(it["year"] >= year - 2 for it in items)
    assert all(it["poster"] for it in items)
    langs = {it["language"] for it in items}
    assert len(langs) >= 4, langs
    # No single language may dominate the screen.
    top = max(sum(1 for it in items if it["language"] == c) for c in langs)
    assert top <= len(items) * 0.45, top


def test_feed_alternates_languages_rather_than_blocking(client):
    items = client.get("/api/feed?years=3&limit=24").json()["items"]
    first_five = [it["language"] for it in items[:5]]
    assert len(set(first_five)) >= 3, first_five


def test_feed_prefers_titles_well_regarded_in_their_own_industry(client):
    items = client.get("/api/feed?years=4&limit=20").json()["items"]
    all_items = client.get("/api/feed?years=4&limit=200").json()["items"]
    assert np.mean([i["rating"] for i in items]) >= np.mean(
        [i["rating"] for i in all_items]
    )


def test_language_weights_are_respected(client):
    items = client.get("/api/feed?years=4&limit=30&langs=ml:1,ta:1").json()["items"]
    assert items
    assert {it["language"] for it in items} <= {"ml", "ta"}


def test_rating_records_and_removes_from_the_feed(client):
    items = client.get("/api/feed?years=4&limit=10").json()["items"]
    target = items[0]
    assert client.post(
        "/api/rate", json={"item_id": target["item_id"], "verdict": "love"}
    ).json()["ok"]

    progress = client.get("/api/progress").json()
    assert progress["rated"] == 1
    assert progress["recent"][0]["verdict"] == "love"

    again = client.get("/api/feed?years=4&limit=200").json()["items"]
    assert target["item_id"] not in {i["item_id"] for i in again}


def test_not_seen_is_recorded_but_stays_recommendable(client):
    from entertainer import store

    items = client.get("/api/feed?years=4&limit=10").json()["items"]
    target = items[0]
    client.post("/api/rate", json={"item_id": target["item_id"], "verdict": "unseen"})

    assert client.get("/api/progress").json()["rated"] == 0
    with store.session(read_only=True) as con:
        assert target["item_id"] not in store.interacted(con)
        assert target["item_id"] in store.already_asked(con)


def test_undo_reverses_the_last_verdict(client):
    items = client.get("/api/feed?years=4&limit=10").json()["items"]
    client.post("/api/rate", json={"item_id": items[0]["item_id"], "verdict": "hate"})
    assert client.get("/api/progress").json()["rated"] == 1

    d = client.post("/api/undo").json()
    assert d["ok"] and d["item_id"] == items[0]["item_id"]
    assert client.get("/api/progress").json()["rated"] == 0


def test_undo_on_an_empty_log_is_harmless(client):
    assert client.post("/api/undo").json()["ok"] is False


def test_unknown_verdict_is_rejected(client):
    r = client.post("/api/rate", json={"item_id": 0, "verdict": "brilliant"})
    assert r.status_code == 400


def test_search_finds_catalogue_titles(client):
    d = client.get("/api/search?q=ML Film 3").json()
    assert d["catalogue"]
    assert "ML Film 3" in d["catalogue"][0]["title"]
    # No TMDB credentials in tests, so no external results.
    assert d["tmdb"] == []


def test_search_is_empty_for_a_blank_query(client):
    d = client.get("/api/search?q=  ").json()
    assert d["catalogue"] == [] and d["tmdb"] == []


def test_add_requires_credentials(client):
    """And, by extension, that the suite is not reaching TMDB at all."""
    from entertainer.config import has_tmdb

    assert not has_tmdb(), "tests must not run with real credentials loaded"
    r = client.post("/api/add", json={"tmdb_id": 123, "kind": "movie"})
    assert r.status_code == 400
    assert "TMDB" in r.text


def test_progress_breaks_down_by_language(client):
    items = client.get("/api/feed?years=4&limit=40").json()["items"]
    for it in items[:6]:
        client.post("/api/rate", json={"item_id": it["item_id"], "verdict": "like"})
    p = client.get("/api/progress").json()
    assert p["rated"] == 6
    assert sum(row["count"] for row in p["by_language"]) == 6


def test_token_gate_rejects_unauthenticated_requests(tmp_path, monkeypatch, client):
    """Off the loopback interface the page is somebody else's write access."""
    from entertainer.web import app as webapp

    guarded = TestClient(webapp.create_app(token="s3cret"))
    assert guarded.get("/api/progress").status_code == 401
    assert guarded.get("/api/feed").status_code == 401
    assert guarded.post("/api/rate", json={"item_id": 0, "verdict": "love"}).status_code == 401


def test_token_in_the_query_string_authenticates_and_sets_a_cookie(client):
    from entertainer.web import app as webapp

    guarded = TestClient(webapp.create_app(token="s3cret"))
    r = guarded.get("/api/progress?token=s3cret")
    assert r.status_code == 200
    assert "entertainer_token" in r.cookies or "entertainer_token" in r.headers.get(
        "set-cookie", ""
    )
    # The cookie carries subsequent in-page fetches.
    assert guarded.get("/api/progress").status_code == 200


def test_token_header_also_works(client):
    from entertainer.web import app as webapp

    guarded = TestClient(webapp.create_app(token="s3cret"))
    r = guarded.get("/api/progress", headers={"x-entertainer-token": "s3cret"})
    assert r.status_code == 200


def test_no_token_means_no_gate(client):
    assert client.get("/api/progress").status_code == 200


@pytest.fixture()
def bare(tmp_path, monkeypatch):
    """A machine with no catalogue at all — a fresh clone."""
    import importlib

    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path / "bare"))
    from entertainer import config, engine, store

    for mod in (config, store, engine):
        importlib.reload(mod)
    from entertainer.web import app as webapp
    from entertainer.web import live

    importlib.reload(live)
    importlib.reload(webapp)
    config.PATHS.ensure()
    store.connect().close()
    return webapp


def test_live_mode_is_chosen_automatically_when_there_is_no_catalogue(bare):
    c = TestClient(bare.create_app(live=True))
    m = c.get("/api/mode").json()
    assert m["live"] is True
    assert m["catalogue"] == 0


def test_catalogue_mode_is_chosen_when_one_exists(tmp_path, monkeypatch, client):
    """Auto-detection must not override a catalogue that exists."""
    from entertainer.web import app as webapp

    auto = TestClient(webapp.create_app())
    m = auto.get("/api/mode").json()
    assert m["live"] is False, m
    assert m["catalogue"] > webapp.EMPTY_CATALOGUE


def test_live_feed_refuses_without_credentials(bare):
    c = TestClient(bare.create_app(live=True))
    r = c.get("/api/feed")
    assert r.status_code == 400
    assert "TMDB" in r.text


def test_live_languages_do_not_need_a_catalogue(bare):
    c = TestClient(bare.create_app(live=True))
    d = c.get("/api/languages").json()
    codes = {row["code"] for row in d["languages"]}
    assert {"ml", "ta", "kn", "ko", "en"} <= codes
    weights = {row["code"]: row["weight"] for row in d["languages"]}
    assert 0 < weights["te"] < weights["ml"]


def test_live_vote_floors_are_calibrated_per_industry():
    """A single floor either floods with Hollywood or empties out Kannada."""
    from entertainer.web.live import VOTE_FLOORS

    assert VOTE_FLOORS["en"] > VOTE_FLOORS["ko"] > VOTE_FLOORS["ml"]
    assert VOTE_FLOORS["kn"] < VOTE_FLOORS["ta"]
    assert all(v > 0 for v in VOTE_FLOORS.values())
