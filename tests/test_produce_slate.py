"""Producing and logging a recommendation slate.

The CLI and the web app each ran `recommend`, built the impression tuple by
hand and called log_impressions. Getting that tuple wrong does not fail
loudly — it writes a plausible row with the wrong propensity, and every
off-policy estimate from then on is quietly biased.
"""

from __future__ import annotations

import pytest
from test_cli import app_env, teach  # noqa: F401

from entertainer import store
from entertainer.engine import Engine
from entertainer.recommend import Filters, produce_slate


@pytest.fixture()
def ready(app_env):  # noqa: F811
    cli, runner = app_env
    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Jallikattu", "liked"), ("96", "loved")])
    engine = Engine()
    with store.session(read_only=True) as con:
        model = engine.fit(con, save=False)
        fs = engine.features(con)
        meta = engine.meta(con)
    return model, fs, meta


def impressions(con):
    return con.execute(
        "SELECT item_id, position, score, propensity, explored, policy, slate_id FROM impressions"
    ).fetchall()


def test_every_pick_is_logged_with_its_propensity(ready):
    """The propensity is the only reason an off-policy estimate is possible."""
    model, fs, meta = ready
    with store.session() as con:
        slate = produce_slate(con, model, fs, meta, k=5, policy="test", strategy="mean")
        rows = impressions(con)

    assert len(rows) == len(slate.picks) == 5
    by_item = {r[0]: r for r in rows}
    for pick in slate.picks:
        row = by_item[pick.item_id]
        assert row[1] == pick.position
        assert row[3] == pytest.approx(pick.propensity)
        assert row[4] == pick.explored
        assert row[5] == "test"
        assert row[6] == slate.slate_id


def test_one_call_is_one_slate(ready):
    model, fs, meta = ready
    with store.session() as con:
        a = produce_slate(con, model, fs, meta, k=3, policy="test", strategy="mean")
        b = produce_slate(con, model, fs, meta, k=3, policy="test", strategy="mean")
    assert a.slate_id != b.slate_id


def test_remembering_the_slate_is_opt_in(ready):
    """The browser has no positions to type, and writing last_slate there
    would repoint `ent loved 3` at a slate the terminal never saw."""
    model, fs, meta = ready
    with store.session() as con:
        produce_slate(con, model, fs, meta, k=3, policy="test", strategy="mean", remember=False)
        assert store.last_slate(con) == []

        slate = produce_slate(con, model, fs, meta, k=3, policy="test", strategy="mean")
        assert store.last_slate(con) == slate.item_ids


def test_an_empty_slate_logs_nothing(ready):
    """Filters that exclude everything must not write an empty observation."""
    model, fs, meta = ready
    with store.session() as con:
        slate = produce_slate(
            con, model, fs, meta, k=5, policy="test", strategy="mean",
            filters=Filters(languages=("zz",)),
        )
        assert slate.picks == []
        assert impressions(con) == []
        assert store.last_slate(con) == []


def test_producing_a_slate_records_no_verdicts(ready):
    """Showing something is not the user saying anything about it."""
    model, fs, meta = ready
    with store.session() as con:
        before = store.counts(con)["events"]
        produce_slate(con, model, fs, meta, k=5, policy="test", strategy="mean")
        assert store.counts(con)["events"] == before
