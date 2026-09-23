"""The holdout contract.

Both halves used to live in cli.py about 1100 lines apart — `ent data cf`
wrote the held-out users, `ent eval` checked the file existed — with neither
referring to the other and the filename spelled out three times. Neither
command has any CLI-level test coverage, so this contract was entirely
unverified.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_cli import app_env  # noqa: F401

from entertainer.errors import IntegrityRefusal
from entertainer.evaluation import integrity

pytestmark = pytest.mark.usefixtures("app_env")


def test_holdout_selection_is_seeded_so_rebuilds_stay_comparable():
    """An unseeded choice would withhold different users on every rebuild,
    and benchmark numbers across runs would stop being comparable."""
    users = np.arange(1000)
    first = integrity.choose_holdout(users, 50)
    assert np.array_equal(first, integrity.choose_holdout(users, 50))
    assert len(first) == 50
    assert len(set(first.tolist())) == 50, "users withheld more than once"


def test_holdout_is_capped_at_the_population():
    assert len(integrity.choose_holdout(np.arange(10), 500)) == 10


def test_zero_means_withhold_nobody():
    assert integrity.choose_holdout(np.arange(10), 0) is None


def test_round_trips_through_disk():
    held = integrity.choose_holdout(np.arange(200), 20)
    assert not integrity.holdout_recorded()
    integrity.save_holdout(held)
    assert integrity.holdout_recorded()
    assert np.array_equal(integrity.load_holdout(), held.astype(np.int32))


def test_load_returns_none_when_nothing_was_recorded():
    assert integrity.load_holdout() is None


def test_benchmarking_without_a_record_is_refused_not_warned():
    """A warning printed beside a plausible NDCG reads as a caveat on a real
    number. There is no real number when the prior may have read the answers."""
    with pytest.raises(IntegrityRefusal) as exc:
        integrity.require_holdout()
    assert "Refusing to report a number" in str(exc.value)


def test_recording_a_holdout_lifts_the_refusal():
    integrity.save_holdout(integrity.choose_holdout(np.arange(100), 5))
    integrity.require_holdout()
