"""The off-policy estimate.

Extracted from `ent audit`'s render loop, where the SQL join, the propensity
filter, the item-space staleness check and the SNIPS call all lived inside a
function that also printed. The /api/audit endpoint needs the data half.

These tests exist mostly to pin the *refusals*. The estimate itself is
high-variance and the interesting question is not "what number" but "when
does it decline to give one" — every one of these guards exists because the
number would otherwise be misleading rather than merely noisy.
"""

from __future__ import annotations

import pytest
from test_cli import app_env, teach  # noqa: F401

from entertainer import store
from entertainer.engine import Engine
from entertainer.evaluation import offpolicy
from entertainer.evaluation.prequential import MIN_LOGGED


def log_slate(con, item_ids, propensity=0.1):
    store.log_impressions(
        con,
        store.new_slate_id(),
        # (item_id, position, score, propensity, explored)
        [(int(i), n, 1.0, propensity, False) for n, i in enumerate(item_ids)],
        policy="test",
    )


@pytest.fixture()
def ready(app_env):  # noqa: F811
    cli, runner = app_env
    teach(cli, runner, [("Kumbalangi Nights", "loved"), ("Jallikattu", "liked"), ("96", "loved")])
    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
    return engine, fs


def test_refuses_below_the_minimum_and_says_how_many_it_has(ready):
    engine, fs = ready
    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)
    assert result.status == "not-enough-data"
    assert result.n_usable < MIN_LOGGED
    assert result.estimate is None
    assert result.better is None


def test_impressions_without_an_outcome_do_not_count(ready):
    """A slate nobody acted on carries no information about the policy."""
    engine, fs = ready
    with store.session() as con:
        log_slate(con, [int(i) for i in fs.item_ids[:40]])
    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)
    assert result.n_usable == 0


def test_zero_propensity_impressions_are_excluded(ready):
    """A propensity of zero would divide by zero; a missing one means the
    slate predates propensity logging and cannot be reweighted."""
    engine, fs = ready
    ids = [int(i) for i in fs.item_ids[:5]]
    with store.session() as con:
        log_slate(con, ids, propensity=0.0)
        for item_id in ids:
            engine.record(con, item_id, "like", source="test")
    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)
    assert result.n_usable == 0


def test_a_rating_that_predates_its_impression_does_not_count(ready):
    """The join is temporal: only a verdict given *after* the slate was shown
    is evidence about that slate."""
    engine, fs = ready
    ids = [int(i) for i in fs.item_ids[:5]]
    with store.session() as con:
        for item_id in ids:
            engine.record(con, item_id, "like", source="test")
    with store.session() as con:
        log_slate(con, ids)
    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)
    assert result.n_usable == 0
