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


def test_a_stamped_verdict_is_matched_to_its_own_slate(ready):
    """A verdict from a recommendation records which slate produced it, so it
    can be matched exactly rather than by "came afterwards"."""
    engine, fs = ready
    ids = [int(i) for i in fs.item_ids[:3]]
    with store.session() as con:
        slate_id = store.new_slate_id()
        store.log_impressions(
            con, slate_id,
            [(i, n, 1.0, 0.2, False) for n, i in enumerate(ids)],
            policy="test",
        )
        for item_id in ids:
            engine.record(con, item_id, "like", source="web", context={"slate_id": slate_id})

    with store.session(read_only=True) as con:
        assert offpolicy.estimate(con, engine, fs).n_usable == 3


def test_a_stamped_verdict_is_not_credited_to_a_slate_it_did_not_come_from(ready):
    """Browsing to the recommendations tab logs a slate whether or not anyone
    acts on it. Without this, a verdict given elsewhere is credited to every
    earlier impression of that title."""
    engine, fs = ready
    item_id = int(fs.item_ids[0])
    with store.session() as con:
        seen_slate = store.new_slate_id()
        store.log_impressions(con, seen_slate, [(item_id, 0, 1.0, 0.2, False)], policy="test")
        ignored_slate = store.new_slate_id()
        store.log_impressions(con, ignored_slate, [(item_id, 0, 1.0, 0.9, False)], policy="test")
        engine.record(con, item_id, "love", source="web", context={"slate_id": seen_slate})

    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)
    assert result.n_usable == 1, "the unanswered slate was credited too"


def test_unstamped_verdicts_still_fall_back_to_the_temporal_join(ready):
    """Every verdict recorded before slate stamping existed, and every one
    given from the terminal, has no slate id. Those must still count."""
    engine, fs = ready
    ids = [int(i) for i in fs.item_ids[:4]]
    with store.session() as con:
        store.log_impressions(
            con, store.new_slate_id(),
            [(i, n, 1.0, 0.25, False) for n, i in enumerate(ids)],
            policy="test",
        )
    with store.session() as con:
        for item_id in ids:
            engine.record(con, item_id, "like", source="manual")

    with store.session(read_only=True) as con:
        assert offpolicy.estimate(con, engine, fs).n_usable == 4


def test_estimating_does_not_persist_a_model(ready):
    """engine.model defaults to refit=True, which saves — so reading the
    audit over HTTP rewrote the stored taste model on every page view."""
    from entertainer.config import PATHS

    engine, fs = ready
    saved = PATHS.artifacts / "taste.npz"
    if saved.exists():
        saved.unlink()
    with store.session(read_only=True) as con:
        offpolicy.estimate(con, engine, fs)
    assert not saved.exists(), "a measurement persisted a model"



def test_usable_count_agrees_with_what_the_estimate_waits_for(ready):
    """The screens say "16 of 30" and the estimate refuses below 30. Both must
    be counting the same thing.

    ``usable_count`` exists so the recommendations page can show the gap on
    every slate without paying for the model refit ``estimate`` does. Deriving
    it from a second query would let the two drift, and a page promising the
    check at thirty beside a check that refuses at thirty is the kind of
    contradiction this project has already had to fix once.
    """
    engine, fs = ready
    ids = [int(i) for i in fs.item_ids[:4]]
    with store.session() as con:
        slate_id = store.new_slate_id()
        store.log_impressions(
            con, slate_id,
            [(i, n, 1.0, 0.2, False) for n, i in enumerate(ids)],
            policy="test",
        )
        for item_id in ids:
            engine.record(con, item_id, "like", source="web", context={"slate_id": slate_id})

    with store.session(read_only=True) as con:
        assert offpolicy.usable_count(con) == 4
        assert offpolicy.estimate(con, engine, fs).n_usable == offpolicy.usable_count(con)


def test_the_count_costs_no_model_fit(ready):
    """It is read on every recommendations page view. A refit there would make
    the slate endpoint as slow as the audit page."""
    engine, fs = ready
    ids = [int(i) for i in fs.item_ids[:3]]
    with store.session() as con:
        log_slate(con, ids)
        for item_id in ids:
            engine.record(con, item_id, "like", source="test")

    class _NoFit:
        def fit(self, con, save=False):
            raise AssertionError("counting must not need a model")

    with store.session(read_only=True) as con:
        assert offpolicy.usable_count(con) == 3
        # And the estimate itself must refuse before it ever reaches a fit.
        assert offpolicy.estimate(con, _NoFit(), fs).status == "not-enough-data"


def test_the_slate_endpoint_reports_how_far_off_the_check_is(ready):
    """The recommendations page says a verdict there is worth more than the
    same verdict elsewhere, and shows the gap. That number rides on the slate
    response so the page needs no second request to draw it."""
    from fastapi.testclient import TestClient

    from entertainer.web.app import create_app

    engine, fs = ready
    client = TestClient(create_app())
    body = client.post("/api/recommendations/slate", json={"k": 3, "kind": "both"}).json()

    assert body["outcomes"]["need"] == MIN_LOGGED
    with store.session(read_only=True) as con:
        assert body["outcomes"]["have"] == offpolicy.usable_count(con)


def test_both_screens_are_told_the_same_threshold(app_env):  # noqa: F811
    """Evidence hardcoded 30 while the slate endpoint sent MIN_LOGGED.

    Two screens naming the same quantity from two sources is how they end up
    disagreeing: raising MIN_LOGGED would have left Evidence reporting a closed
    gap while the estimate still refused. Both now read it from the server, and
    this pins that they read the *same* value.

    Uses the full seed rather than the `ready` fixture because /api/audit
    refuses below MIN_VERDICTS and answers no off-policy block at all.
    """
    from fastapi.testclient import TestClient
    from test_cli_golden import SEED

    from entertainer.web.app import create_app

    cli, runner = app_env
    teach(cli, runner, SEED)

    client = TestClient(create_app())
    slate = client.post("/api/recommendations/slate", json={"k": 3, "kind": "both"}).json()
    audit = client.get("/api/audit")

    assert slate["outcomes"]["need"] == MIN_LOGGED
    assert audit.status_code == 200, audit.text
    assert audit.json()["off_policy"]["need"] == MIN_LOGGED
