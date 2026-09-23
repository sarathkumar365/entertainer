"""What the interface gets back when the machine is not ready yet.

A half-built machine is the normal state for hours: the catalogue exists,
the fused item space does not, and a running build owns the database. Every
one of those used to reach the browser as ``500 Internal Server Error`` with
an empty body, which a page cannot tell apart from a crash.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

#: Every route that needs the fused item space.
NEEDS_MODEL = (
    ("GET", "/api/taste"),
    ("GET", "/api/audit"),
    ("GET", "/api/similar/0"),
    ("GET", "/api/predict/0"),
    ("POST", "/api/recommendations/slate"),
)


@pytest.fixture()
def unbuilt(tmp_path, monkeypatch):
    """A catalogue, three verdicts, and no model artefacts at all.

    The verdicts matter: without them these routes stop at "not enough
    evidence" and never reach the missing item space.
    """
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_BEARER", raising=False)

    from entertainer import config, store
    from entertainer.web import app as webapp

    config.PATHS.ensure()
    con = store.connect()
    for n in range(30):
        con.execute(
            """
            INSERT INTO titles (item_id, imdb_id, tmdb_id, kind, title, year,
                language, runtime, genres, imdb_rating, imdb_votes, quality,
                overview, poster_path, directors, cast_names, keywords)
            VALUES (?, ?, ?, 'movie', ?, 2020, 'en', 120, ['Drama'], 7.0, 9000,
                    0.5, 'A synopsis.', ?, ['A Dir'], [], [])
            """,
            [n, f"tt{n:07d}", 900000 + n, f"Film {n}", f"/poster{n}.jpg"],
        )
    for n in range(4):
        store.log_event(con, n, "rate", value=8.0 - n * 2, source="manual",
                        ts=(dt.datetime.now() - dt.timedelta(days=n)).isoformat())
    con.close()
    return TestClient(webapp.create_app(live=False), raise_server_exceptions=False)


@pytest.mark.parametrize(("method", "path"), NEEDS_MODEL)
def test_model_routes_say_the_build_is_unfinished(unbuilt, method, path):
    r = unbuilt.request(method, path)
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["code"] == "model_not_ready"
    assert "build" in body["detail"]
    # Written for someone looking at a browser: no command to go and type.
    assert "ent " not in body["detail"]
    assert "scripts/" not in body["detail"]
    assert r.headers["retry-after"]


def test_the_rest_of_the_app_still_works_without_a_model(unbuilt):
    """A missing model must not take the pages that do not need one with it."""
    for path in ("/api/mode", "/api/progress", "/api/languages", "/api/feed",
                 "/api/library", "/api/rated", "/api/validation/summary"):
        assert unbuilt.get(path).status_code == 200, path


def test_rating_still_works_without_a_model(unbuilt):
    """Verdicts are what produce a model, so they cannot require one."""
    assert unbuilt.post("/api/rate", json={"item_id": 7, "verdict": "love"}).status_code == 200


def test_an_unexpected_error_answers_as_json_and_leaks_nothing():
    """Anything genuinely broken is still an answer the page can parse."""
    from fastapi import FastAPI

    from entertainer.web.failures import install

    app = FastAPI()
    install(app)

    @app.get("/api/boom")
    def boom() -> dict:
        raise ZeroDivisionError("/Users/someone/secret/path")

    @app.get("/api/fine")
    def fine() -> dict:
        return {"ok": True}

    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/boom")
    assert r.status_code == 500
    assert r.json()["code"] == "internal"
    assert "ZeroDivision" not in r.text
    assert "secret/path" not in r.text
    # Still serving: one broken route does not end the process.
    assert c.get("/api/fine").status_code == 200


def test_a_locked_database_reads_as_a_running_build(tmp_path, monkeypatch):
    """DuckDB's lock error is a normal state, not an IO failure."""
    from entertainer import config, store
    from entertainer.errors import CatalogueBusy

    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    config.PATHS.ensure()
    store.connect().close()

    env = dict(os.environ, ENTERTAINER_DATA_DIR=str(tmp_path))
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "from entertainer import store; con = store.connect();"
         " print('held', flush=True); __import__('time').sleep(30)"],
        stdout=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        deadline = time.monotonic() + 10
        while True:
            try:
                store.connect(read_only=True).close()
            except CatalogueBusy as exc:
                assert "build" in str(exc)
                break
            except Exception:  # noqa: BLE001 - the lock may not be visible yet
                pass
            assert time.monotonic() < deadline, "the holder never took the lock"
            time.sleep(0.2)
    finally:
        holder.kill()
        holder.wait()


def test_the_cli_reports_a_domain_error_as_one_line(monkeypatch, capsys):
    """The terminal gets the same courtesy as the browser: no stack trace."""
    from entertainer import cli
    from entertainer.errors import ModelNotReady

    def boom():
        raise ModelNotReady("the model is not built yet; run `ent build`")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    printed = capsys.readouterr().out
    assert "ent build" in printed
    assert "Traceback" not in printed


def test_the_console_scripts_go_through_that_wrapper():
    """`ent` pointed straight at the Typer object, so main() never ran."""
    import tomllib

    repo = pathlib.Path(__file__).resolve().parents[1]
    scripts = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["scripts"]
    assert set(scripts.values()) == {"entertainer.cli:main"}
