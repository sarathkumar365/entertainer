"""Choosing which titles still need TMDB keywords.

`ent data keywords` has no CLI-level coverage — it makes tens of thousands of
API calls — so two real rules lived unverified inside its command body: a
schema migration disguised as an UPDATE, and a windowed query whose ordering
is the difference between a resumable pass and one that drifts.
"""

from __future__ import annotations

import pytest
from test_cli import app_env  # noqa: F401

from entertainer import store
from entertainer.data import catalog


@pytest.fixture(autouse=True)
def catalogue_with_tmdb_ids(app_env):  # noqa: F811
    """The CLI fixture builds rows without a tmdb_id, and a title with no
    tmdb_id is never a keyword target — there is nothing to ask TMDB about."""
    with store.session() as con:
        con.execute("UPDATE titles SET tmdb_id = item_id + 100000")


def test_titles_that_already_have_keywords_count_as_fetched():
    """A catalogue enriched before keywords_at existed has keywords but no
    marker. Without this every one of them is requested again."""
    with store.session() as con:
        con.execute("UPDATE titles SET keywords = ['heist'], keywords_at = NULL")
        marked = catalog.stamp_existing_keywords(con)
        assert marked > 0
        assert catalog.keyword_targets(con, 10_000) == []


def test_an_empty_keyword_list_is_not_treated_as_fetched():
    """An empty list is indistinguishable from never having asked."""
    with store.session() as con:
        con.execute("UPDATE titles SET keywords = [], keywords_at = NULL")
        assert catalog.stamp_existing_keywords(con) == 0
        assert catalog.keyword_targets(con, 10_000)


def test_stamping_is_idempotent():
    with store.session() as con:
        con.execute("UPDATE titles SET keywords = ['heist'], keywords_at = NULL")
        catalog.stamp_existing_keywords(con)
        assert catalog.stamp_existing_keywords(con) == 0


def test_targets_are_the_most_voted_titles_that_still_lack_keywords():
    with store.session() as con:
        con.execute("UPDATE titles SET keywords = NULL, keywords_at = NULL")
        targets = catalog.keyword_targets(con, 5)
        assert len(targets) == 5

        ranked = con.execute(
            "SELECT item_id FROM titles WHERE tmdb_id IS NOT NULL "
            "ORDER BY imdb_votes DESC NULLS LAST LIMIT 5"
        ).fetchall()
        assert {t[0] for t in targets} == {int(r[0]) for r in ranked}


def test_the_target_is_a_coverage_goal_not_a_batch_size():
    """Ranking first and filtering second is what makes a resumed run finish
    the same N. Filtering first would rank whatever is left, so each run
    would march on to the next N and the top of the catalogue would never be
    completed."""
    with store.session() as con:
        con.execute("UPDATE titles SET keywords = NULL, keywords_at = NULL")
        first = catalog.keyword_targets(con, 5)

        # Two of the five get done; a resumed run must ask for the other three.
        done = [t[0] for t in first[:2]]
        con.executemany(
            "UPDATE titles SET keywords = ['x'], keywords_at = now() WHERE item_id = ?",
            [(item_id,) for item_id in done],
        )
        resumed = catalog.keyword_targets(con, 5)

    assert len(resumed) == 3
    assert {t[0] for t in resumed} == {t[0] for t in first} - set(done)


def test_titles_without_a_tmdb_id_are_never_targets():
    """There is nothing to ask TMDB about."""
    with store.session() as con:
        con.execute("UPDATE titles SET keywords_at = NULL, tmdb_id = NULL")
        assert catalog.keyword_targets(con, 10_000) == []
