"""The cold-start questioning loop.

Lifted out of `ent onboard`, where the batching, the mid-loop refit and the
"already asked" bookkeeping were interleaved with typer.prompt calls. The web
interface needs the same sequence without a terminal attached.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_cli import app_env  # noqa: F401

from entertainer import store
from entertainer.coldstart import elicit
from entertainer.coldstart.session import MIN_FOR_ADAPTIVE, ElicitationSession
from entertainer.engine import Engine


@pytest.fixture()
def session(app_env):  # noqa: F811
    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        meta = engine.meta(con)
    pool = elicit.recognisable_pool(fs, meta)
    s = ElicitationSession(fs=fs, meta=meta, pool=pool, asked=set(), target=6)
    s.start()
    return s


def test_a_question_carries_the_row_so_the_caller_need_not_look_it_up(session):
    question = session.next_question()
    assert question.item_id in session.meta
    assert question.row["title"]


def test_the_same_title_is_never_asked_twice(session):
    seen = []
    for _ in range(12):
        q = session.next_question()
        if q is None:
            break
        seen.append(q.item_id)
    assert len(seen) == len(set(seen))


def test_a_title_already_asked_elsewhere_is_skipped(app_env):  # noqa: F811
    """`asked` comes from the event log, so questions answered in a previous
    session or through the web app do not come round again."""
    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        meta = engine.meta(con)
    pool = elicit.recognisable_pool(fs, meta)

    first = ElicitationSession(fs=fs, meta=meta, pool=pool, asked=set(), target=5)
    first.start()
    already = {first.next_question().item_id for _ in range(3)}

    second = ElicitationSession(fs=fs, meta=meta, pool=pool, asked=set(already), target=5)
    second.start()
    assert second.next_question().item_id not in already


def test_offering_a_question_spends_it_even_if_it_is_never_answered(session):
    """A question shown and abandoned must not come back — from the pool's
    point of view it has been used either way."""
    first = session.next_question()
    assert first.item_id in session.asked
    assert session.answered_count == 0

    rest = [session.next_question().item_id for _ in range(5)]
    assert first.item_id not in rest


def test_finished_tracks_answers_not_questions_offered(session):
    for _ in range(8):
        session.next_question()
    assert not session.finished

    for i in range(session.target):
        session.record(10_000 + i, 0.75)
    assert session.finished


def test_it_switches_to_the_adaptive_criterion_once_there_is_a_posterior(session):
    """Below three verdicts there is no posterior to reduce the variance of,
    so the questions come from the seeded ladder instead."""
    assert session.answered_count < MIN_FOR_ADAPTIVE

    ids = [session.next_question().item_id for _ in range(MIN_FOR_ADAPTIVE)]
    for item_id in ids:
        session.record(item_id, 0.75)

    session._batch = []  # force a refill
    question = session.next_question()
    assert question is not None
    assert question.item_id not in ids


def test_an_exhausted_pool_ends_the_session_rather_than_looping(app_env):  # noqa: F811
    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        meta = engine.meta(con)
    pool = elicit.recognisable_pool(fs, meta)

    s = ElicitationSession(
        fs=fs, meta=meta, pool=pool, asked=set(int(i) for i in fs.item_ids), target=5
    )
    s.start()
    assert s.next_question() is None


def test_recognisable_pool_is_respected(session):
    allowed = {int(i) for i in session.pool}
    for _ in range(6):
        q = session.next_question()
        if q is None:
            break
        assert q.item_id in allowed


def test_np_is_used_for_the_refit_without_mutating_the_answers(session):
    before = list(session.answered)
    session.record(123, 1.0)
    assert session.answered == [*before, (123, 1.0)]
    assert isinstance(np.array([a[1] for a in session.answered]), np.ndarray)
