from __future__ import annotations

import importlib

import pytest


def test_import_event_is_idempotent_and_preserves_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer import config, store

    importlib.reload(config)
    importlib.reload(store)
    with store.session() as con:
        con.execute("INSERT INTO titles (item_id, imdb_id, title) VALUES (1, 'tt0000001', 'Film')")
        assert store.import_event(
            con, 1, "rate", 10.0, "import", "2020-01-02T03:04:05Z", {"verdict": "love"}
        )
        assert not store.import_event(
            con, 1, "rate", 10.0, "import", "2020-01-02T03:04:05Z", {"verdict": "love"}
        )
    with store.session(read_only=True) as con:
        rows = con.execute("SELECT ts FROM events").fetchall()
    assert len(rows) == 1
    assert str(rows[0][0]).startswith("2020-01-02 03:04:05")


def test_skip_enrichment_preflight_does_not_require_tmdb(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer import config, pipeline

    importlib.reload(config)
    importlib.reload(pipeline)
    monkeypatch.setattr(pipeline, "has_tmdb", lambda: False)
    assert pipeline.preflight(require_tmdb=False)["tmdb"] is False
    with pytest.raises(RuntimeError, match="TMDB credential"):
        pipeline.preflight(require_tmdb=True)
