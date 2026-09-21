"""The event log must survive a catalogue rebuild.

Item ids are positional within a build, so a rebuild renumbers everything.
The verdict log is the only irreplaceable thing in the project — the
catalogue can always be rebuilt from public data, and a person's taste
cannot — so these tests pin that a rebuild never silently reattaches a
verdict to a different film.
"""

from __future__ import annotations

import gzip

import pytest

from entertainer import store
from entertainer.data import catalog, imdb
from tests.test_imdb_ingest import FIXTURES


@pytest.fixture()
def catalogued(tmp_path, monkeypatch):
    root = tmp_path / "raw" / "imdb"
    root.mkdir(parents=True)
    for name, lines in FIXTURES.items():
        with gzip.open(root / name, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    class P:
        raw = tmp_path / "raw"
        root = tmp_path
        interim = tmp_path / "i"
        embeddings = tmp_path / "e"
        artifacts = tmp_path / "a"
        reports = tmp_path / "r"
        catalog_db = tmp_path / "c.duckdb"

        @classmethod
        def ensure(cls):
            for p in (cls.raw, cls.interim, cls.embeddings, cls.artifacts, cls.reports):
                p.mkdir(parents=True, exist_ok=True)
            return cls

    monkeypatch.setattr(imdb, "PATHS", P)
    monkeypatch.setattr(store, "PATHS", P)
    monkeypatch.setattr(catalog, "PATHS", P)
    catalog.build_base(min_votes=50)
    return P


def _id_of(con, imdb_id: str) -> int:
    return int(con.execute("SELECT item_id FROM titles WHERE imdb_id = ?", [imdb_id]).fetchone()[0])


def test_verdicts_follow_their_film_across_a_rebuild(catalogued):
    con = store.connect()
    target = _id_of(con, "tt01")
    store.log_event(con, target, "rate", 9.0, "manual", {"verdict": "love"})
    # Force renumbering so a positional mapping would visibly break.
    con.execute("UPDATE titles SET item_id = item_id + 500")
    con.execute("UPDATE events SET item_id = item_id + 500")
    con.close()

    catalog.build_base(min_votes=50)

    con = store.connect(read_only=True)
    row = con.execute(
        """
        SELECT t.imdb_id, e.value FROM events e JOIN titles t USING (item_id)
        WHERE e.kind = 'rate'
        """
    ).fetchone()
    con.close()
    assert row is not None, "the verdict lost its film entirely"
    assert row[0] == "tt01", f"verdict reattached to {row[0]}"
    assert row[1] == 9.0


def test_impressions_are_remapped_too(catalogued):
    con = store.connect()
    target = _id_of(con, "tt02")
    store.log_impressions(con, "slate-1", [(target, 0, 1.0, 0.1, False)], policy="test")
    con.execute("UPDATE titles SET item_id = item_id + 700")
    con.execute("UPDATE impressions SET item_id = item_id + 700")
    con.close()

    catalog.build_base(min_votes=50)

    con = store.connect(read_only=True)
    row = con.execute(
        "SELECT t.imdb_id FROM impressions i JOIN titles t USING (item_id)"
    ).fetchone()
    con.close()
    assert row is not None and row[0] == "tt02"


def test_every_verdict_still_points_at_a_real_title_after_rebuild(catalogued):
    con = store.connect()
    for imdb_id in ("tt01", "tt02", "tt03", "tt08"):
        store.log_event(con, _id_of(con, imdb_id), "rate", 7.0, "manual")
    con.execute("UPDATE titles SET item_id = item_id * 3 + 11")
    con.close()

    catalog.build_base(min_votes=50)

    con = store.connect(read_only=True)
    dangling = con.execute(
        "SELECT count(*) FROM events e LEFT JOIN titles t USING (item_id) "
        "WHERE t.item_id IS NULL"
    ).fetchone()[0]
    matched = con.execute(
        "SELECT count(*) FROM events e JOIN titles t USING (item_id)"
    ).fetchone()[0]
    con.close()
    assert dangling == 0, f"{dangling} verdicts point at nothing"
    assert matched == 4


def test_pruning_never_deletes_a_title_the_user_rated(catalogued):
    """The vote floor decides what to offer, not what to erase from your history."""
    con = store.connect()
    # tt08 is the 150-vote Kannada film; with language unknown it falls under
    # the default floor and would normally be pruned.
    obscure = _id_of(con, "tt08")
    store.log_event(con, obscure, "rate", 10.0, "manual", {"verdict": "love"})
    con.close()

    before, after = catalog.prune_by_language()
    assert before > after, "the prune did nothing, so this proves nothing"

    con = store.connect(read_only=True)
    survived = con.execute(
        "SELECT count(*) FROM titles WHERE imdb_id = 'tt08'"
    ).fetchone()[0]
    con.close()
    assert survived == 1, "a rated title was pruned out from under its verdict"


def test_pruning_still_removes_unrated_obscurities(catalogued):
    before, after = catalog.prune_by_language()
    assert after < before

    con = store.connect(read_only=True)
    remaining = {r[0] for r in con.execute("SELECT imdb_id FROM titles").fetchall()}
    con.close()
    assert "tt08" not in remaining
