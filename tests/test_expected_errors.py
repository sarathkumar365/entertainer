"""Conditions the build meets routinely, and what they raise.

Each of these is a state of a half-set-up machine — a dump not downloaded, a
credential not configured, an extra not installed — and each used to reach
the terminal as a traceback. They belong to the ``EntertainerError`` family
so that ``cli.main`` can print one line and stop, and the web handler can
turn them into a status code instead of a 500.
"""

from __future__ import annotations

import pytest

from entertainer.errors import (
    EntertainerError,
    MissingCredentials,
    MissingSourceData,
    NotEnoughEvidence,
)


@pytest.fixture(autouse=True)
def _scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_BEARER", raising=False)


def test_every_expected_error_is_one_the_cli_can_translate():
    """The family, not RuntimeError: `except RuntimeError` catches bugs too."""
    for kind in (MissingSourceData, MissingCredentials, NotEnoughEvidence):
        assert issubclass(kind, EntertainerError)

    from entertainer.ingest import IngestError

    # Claimed by errors.py's docstring since before it was true.
    assert issubclass(IngestError, EntertainerError)


def test_an_imdb_dump_that_was_never_fetched(tmp_path):
    from entertainer.data import imdb

    with pytest.raises(MissingSourceData) as raised:
        imdb._raw("title.basics.tsv.gz")
    assert "ent data fetch" in str(raised.value)


def test_tmdb_without_a_credential():
    from entertainer.data import tmdb

    with pytest.raises(MissingCredentials):
        tmdb._auth()


def test_the_live_feed_without_a_credential():
    from entertainer.web import live

    with pytest.raises(MissingCredentials):
        live._auth()


def test_preflight_without_a_credential():
    from entertainer import pipeline

    with pytest.raises(MissingCredentials):
        pipeline.preflight(require_tmdb=True)


def test_preflight_passes_when_tmdb_is_not_required():
    from entertainer import pipeline

    assert pipeline.preflight(require_tmdb=False)["tmdb"] is False
