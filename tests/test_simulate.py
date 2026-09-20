"""Tests for the offline replay harness.

The headline claim of the project — that this beats the obvious baselines —
rests entirely on this harness, so the harness itself needs testing. The
failure mode to guard against is not a crash; it is a harness that quietly
flatters the model, by leaking evaluation ratings into training, by giving
the model more answers than the baselines, or by scoring against a candidate
set the baselines could never have ranked well.

A synthetic MovieLens stands in for the real one: users drawn from two taste
groups, rating films from their own group highly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from entertainer.evaluation import simulate
from entertainer.models.features import build as build_features

N_ITEMS = 300
N_USERS = 40
GROUPS = 2


@pytest.fixture()
def fake_movielens(monkeypatch):
    rng = np.random.default_rng(0)
    group_of_item = rng.integers(0, GROUPS, size=N_ITEMS)

    users, movies, ratings = [], [], []
    for uid in range(1, N_USERS + 1):
        group = uid % GROUPS
        # Each user rates 120 films: their own group highly, the other poorly.
        picks = rng.choice(N_ITEMS, size=120, replace=False)
        for item in picks:
            liked = group_of_item[item] == group
            score = rng.choice([4.0, 4.5, 5.0]) if liked else rng.choice([0.5, 1.0, 2.0])
            users.append(uid)
            movies.append(int(item))
            ratings.append(float(score))

    df = pl.DataFrame(
        {
            "userId": pl.Series(users, dtype=pl.Int32),
            "movieId": pl.Series(movies, dtype=pl.Int32),
            "rating": pl.Series(ratings, dtype=pl.Float32),
        }
    )
    monkeypatch.setattr(simulate.cf_mod, "load_ratings", lambda: df)

    latent = np.empty((N_ITEMS, 12), dtype=np.float32)
    anchors = rng.normal(size=(GROUPS, 12))
    for i in range(N_ITEMS):
        latent[i] = anchors[group_of_item[i]] + 0.3 * rng.normal(size=12)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)

    ids = np.arange(N_ITEMS, dtype=np.int32)
    meta = {
        int(i): {
            "imdb_votes": int(500_000 - 1000 * i),
            "language": "en",
            "quality": float(0.4 + 0.5 * ((i * 17) % 100) / 100),
            "year": 2000 + (i % 25),
            "runtime": 110,
            "kind": "movie",
        }
        for i in ids
    }
    fs = build_features(ids, latent, meta)
    return fs, meta, {int(i): int(i) for i in ids}, group_of_item


def _histories(item_map):
    return simulate.load_user_histories(item_map, n_users=N_USERS, seed=0, min_history=60)


def test_histories_load_and_map_onto_catalogue_ids(fake_movielens):
    _, _, item_map, _ = fake_movielens
    histories = _histories(item_map)
    assert len(histories) == N_USERS
    assert all(len(h) >= 60 for h in histories.values())


def test_elicitation_and_evaluation_splits_are_disjoint(fake_movielens):
    """A leak here would make every number meaningless."""
    _, _, item_map, _ = fake_movielens
    histories = _histories(item_map)
    rng = np.random.default_rng(0)
    for history in histories.values():
        known, held = simulate._split(history, 0.4, rng)
        assert not (set(known) & set(held))
        assert set(known) | set(held) == set(history)


def test_every_arm_receives_identical_answers(fake_movielens, monkeypatch):
    """A win must not come from the model simply being told more."""
    fs, meta, item_map, _ = fake_movielens
    seen: list[list] = []

    original = simulate.arm_taste

    def spy(fs_, meta_, rows, answered, k):
        seen.append(list(answered))
        return original(fs_, meta_, rows, answered, k)

    monkeypatch.setitem(simulate.ARMS, "entertainer", spy)
    for name in ("popularity", "weighted-kNN"):
        base = simulate.ARMS[name]

        def make(fn):
            def wrapped(fs_, meta_, rows, answered, k):
                seen.append(list(answered))
                return fn(fs_, meta_, rows, answered, k)

            return wrapped

        monkeypatch.setitem(simulate.ARMS, name, make(base))

    cfg = simulate.SimConfig(n_users=6, budget=12, seed=1)
    histories = dict(list(_histories(item_map).items())[:6])
    simulate.run(fs, meta, histories, cfg, arms=("popularity", "weighted-kNN", "entertainer"))

    assert seen
    # Answers are recorded per user, once per arm: every group of three must agree.
    for i in range(0, len(seen), 3):
        group = seen[i : i + 3]
        assert all(g == group[0] for g in group)


def test_the_model_beats_popularity_on_planted_taste(fake_movielens):
    fs, meta, item_map, _ = fake_movielens
    cfg = simulate.SimConfig(n_users=N_USERS, budget=20, seed=2)
    histories = _histories(item_map)
    results = simulate.run(
        fs, meta, histories, cfg, arms=("popularity", "quality-prior", "entertainer")
    )
    ours = results["entertainer"].summary()["ndcg@10"]
    for baseline in ("popularity", "quality-prior"):
        assert ours > results[baseline].summary()["ndcg@10"], baseline


def test_paired_bootstrap_detects_a_real_difference_and_ignores_a_fake_one():
    strong = simulate.ArmResult("a", [{"ndcg@10": 0.6 + 0.01 * i} for i in range(60)])
    weak = simulate.ArmResult("b", [{"ndcg@10": 0.3 + 0.01 * i} for i in range(60)])
    diff, p = simulate.paired_bootstrap(strong, weak)
    assert diff > 0 and p < 0.01

    rng = np.random.default_rng(0)
    a = simulate.ArmResult("a", [{"ndcg@10": float(v)} for v in rng.normal(0.5, 0.1, 60)])
    b = simulate.ArmResult("b", [{"ndcg@10": float(v)} for v in rng.normal(0.5, 0.1, 60)])
    _, p_null = simulate.paired_bootstrap(a, b)
    assert p_null > 0.05


def test_candidate_set_excludes_only_what_the_engine_was_told(fake_movielens):
    """Held-out positives must remain findable in the full catalogue haystack."""
    fs, meta, item_map, _ = fake_movielens
    cfg = simulate.SimConfig(n_users=4, budget=10, seed=3)
    histories = dict(list(_histories(item_map).items())[:4])
    results = simulate.run(fs, meta, histories, cfg, arms=("entertainer",))
    rows = results["entertainer"].per_user
    assert rows
    # Some hits must actually be achievable, otherwise the task is impossible
    # and the comparison is vacuous.
    assert max(r["recall@10"] for r in rows) > 0.0


def test_elicitation_is_not_recomputed_per_user(fake_movielens, monkeypatch):
    """The opening ladder depends only on the catalogue, so it is built once."""
    calls = {"n": 0}
    original = simulate.elicit.seed_questions

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(simulate.elicit, "seed_questions", counted)

    fs, meta, item_map, _ = fake_movielens
    cfg = simulate.SimConfig(n_users=8, budget=10, seed=4)
    histories = dict(list(_histories(item_map).items())[:8])
    simulate.run(fs, meta, histories, cfg, arms=("entertainer",))
    assert calls["n"] == 1, calls


def test_the_population_prior_arm_is_separable(fake_movielens):
    """Both engine arms must run, so the prior's contribution can be measured."""
    fs, meta, item_map, _ = fake_movielens
    cfg = simulate.SimConfig(n_users=10, budget=12, seed=5)
    histories = dict(list(_histories(item_map).items())[:10])
    results = simulate.run(
        fs, meta, histories, cfg, arms=("entertainer", "entertainer-flat-prior")
    )
    assert results["entertainer"].per_user
    assert results["entertainer-flat-prior"].per_user
    # With prior=None the two arms are the same computation and must agree.
    a = results["entertainer"].summary()["ndcg@10"]
    b = results["entertainer-flat-prior"].summary()["ndcg@10"]
    assert a == pytest.approx(b)
