"""End-to-end plumbing test on a synthetic catalogue.

Real data takes an hour to download and another to process, which is far too
slow a loop to catch a transposed matrix in. This builds a small synthetic
world with a *known* ground-truth taste function and checks that the whole
chain — fusion, feature assembly, elicitation, posterior fitting, ranking,
axis discovery — recovers it.

The synthetic world has planted structure: items sit in four themed clusters,
and the simulated viewer likes exactly two of them. If the pipeline works, the
engine should find those two without ever being told a cluster exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from entertainer.coldstart import elicit
from entertainer.models import fusion
from entertainer.models.discover import describe_axes, nearest_liked
from entertainer.models.features import build as build_features
from entertainer.models.taste import fit
from entertainer.recommend import Filters, recommend

N_ITEMS = 1200
N_CLUSTERS = 4
CONTENT_DIM = 48
CF_DIM = 24
LIKED_CLUSTERS = (1, 3)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)

    cluster = rng.integers(0, N_CLUSTERS, size=N_ITEMS)
    centres = rng.normal(size=(N_CLUSTERS, CONTENT_DIM))
    content = centres[cluster] + 0.45 * rng.normal(size=(N_ITEMS, CONTENT_DIM))
    content /= np.linalg.norm(content, axis=1, keepdims=True)
    content = content.astype(np.float32)

    # Collaborative factors exist for only 60% of items, mirroring the real
    # gap between MovieLens coverage and the full catalogue.
    cf_centres = rng.normal(size=(N_CLUSTERS, CF_DIM))
    covered = np.sort(rng.choice(N_ITEMS, size=int(N_ITEMS * 0.6), replace=False))
    cf_factors = (cf_centres[cluster[covered]] + 0.4 * rng.normal(size=(len(covered), CF_DIM)))
    cf_factors = cf_factors.astype(np.float32)

    item_ids = np.arange(N_ITEMS, dtype=np.int32)
    movielens_of_item = {int(i): int(i) for i in covered}

    art = fusion.build(
        item_ids, content, covered.astype(np.int32), cf_factors, movielens_of_item, dim=32, seed=0
    )

    meta = {}
    langs = ["en", "ml", "ko", "ta"]
    for i in range(N_ITEMS):
        c = int(cluster[i])
        meta[i] = {
            "item_id": i,
            "title": f"Title {i}",
            "year": 1980 + (i % 45),
            "kind": "tv" if i % 9 == 0 else "movie",
            "language": langs[i % len(langs)],
            "runtime": 80 + (i % 90),
            "genres": [f"genre-{c}"],
            "keywords": [f"theme-{c}", f"motif-{c}-{i % 3}"],
            "directors": [f"Director {c}-{i % 7}"],
            "cast_names": [],
            "imdb_rating": 5.5 + 0.001 * (i % 2000),
            "imdb_votes": 200_000 - 150 * i,
            "quality": 0.4 + 0.5 * ((i * 37) % 100) / 100,
        }

    fs = build_features(art.item_ids, art.space, meta)

    # Ground truth: the viewer likes clusters 1 and 3, dislikes the rest.
    true_reward = np.where(np.isin(cluster, LIKED_CLUSTERS), 0.85, 0.2)
    true_reward = np.clip(true_reward + rng.normal(0, 0.05, N_ITEMS), 0, 1)

    return dict(fs=fs, meta=meta, cluster=cluster, reward=true_reward, art=art)


def test_fusion_imputes_missing_collaborative_factors(world):
    art = world["art"]
    assert art.space.shape == (N_ITEMS, 32)
    assert 0.5 < art.cf_coverage < 0.7
    # Content genuinely predicts the CF factors in this world, so the ridge
    # map should be clearly better than predicting the mean.
    assert art.cf_r2 > 0.3
    norms = np.linalg.norm(art.space, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_latent_space_separates_planted_clusters(world):
    fs, cluster = world["fs"], world["cluster"]
    within, between = [], []
    rng = np.random.default_rng(0)
    for _ in range(500):
        a, b = rng.integers(0, N_ITEMS, size=2)
        sim = float(fs.latent[a] @ fs.latent[b])
        (within if cluster[a] == cluster[b] else between).append(sim)
    assert np.mean(within) > np.mean(between) + 0.2


def test_engine_recovers_taste_from_verdicts_alone(world):
    fs, reward, cluster = world["fs"], world["reward"], world["cluster"]
    rng = np.random.default_rng(3)
    train = rng.choice(N_ITEMS, size=40, replace=False)

    model = fit(fs.vectors_for(train), reward[train])
    held = np.setdiff1d(np.arange(N_ITEMS), train)
    pred = model.predict(fs.matrix[held], with_std=False)

    liked = np.isin(cluster[held], LIKED_CLUSTERS)
    # The model was never told clusters exist; it saw forty numbers.
    assert pred[liked].mean() > pred[~liked].mean() + 0.2
    assert np.corrcoef(pred, reward[held])[0, 1] > 0.7


def test_recommendations_land_in_the_liked_clusters(world):
    fs, meta, reward, cluster = world["fs"], world["meta"], world["reward"], world["cluster"]
    rng = np.random.default_rng(5)
    train = rng.choice(N_ITEMS, size=50, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    picks = recommend(
        model, fs, meta, k=10,
        filters=Filters(exclude=frozenset(int(i) for i in train)),
        rng=np.random.default_rng(11), strategy="mean",
    )
    assert len(picks) == 10
    hit_rate = np.mean([cluster[p.item_id] in LIKED_CLUSTERS for p in picks])
    assert hit_rate >= 0.8


def test_filters_are_hard_constraints(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(5)
    train = rng.choice(N_ITEMS, size=50, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    picks = recommend(
        model, fs, meta, k=10,
        filters=Filters(languages=("ml",), kind="movie", min_year=2000),
        rng=np.random.default_rng(1), strategy="mean",
    )
    assert picks
    for p in picks:
        assert meta[p.item_id]["language"] == "ml"
        assert meta[p.item_id]["kind"] == "movie"
        assert meta[p.item_id]["year"] >= 2000


def test_thompson_sampling_explores_more_than_greedy(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(5)
    train = rng.choice(N_ITEMS, size=20, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    def slate(strategy, seed):
        return {
            p.item_id
            for p in recommend(
                model, fs, meta, k=10, rng=np.random.default_rng(seed), strategy=strategy
            )
        }

    greedy = [slate("mean", s) for s in (1, 2, 3)]
    sampled = [slate("thompson", s) for s in (1, 2, 3)]
    assert greedy[0] == greedy[1] == greedy[2], "greedy must be deterministic"
    assert len(set.union(*sampled)) > len(set.union(*greedy))


def test_active_elicitation_beats_random_questioning(world):
    """The whole justification for the question-selection machinery."""
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    pool = np.arange(N_ITEMS)
    held = np.arange(N_ITEMS)

    active_scores, random_scores = [], []
    for trial in range(5):
        rng = np.random.default_rng(100 + trial)

        seeds = elicit.seed_questions(fs, meta, k=8, pool=pool)
        answered = list(seeds)
        model = fit(fs.vectors_for(answered), reward[answered], allow_rff=False)
        for _ in range(3):
            batch = elicit.next_questions(model, fs, meta, set(answered), k=6, pool=pool)
            answered.extend(batch)
            model = fit(fs.vectors_for(answered), reward[answered], allow_rff=False)
        pred = model.predict(fs.matrix[held], with_std=False)
        active_scores.append(np.corrcoef(pred, reward[held])[0, 1])

        rnd = rng.choice(N_ITEMS, size=len(answered), replace=False)
        rmodel = fit(fs.vectors_for(rnd), reward[rnd], allow_rff=False)
        rpred = rmodel.predict(fs.matrix[held], with_std=False)
        random_scores.append(np.corrcoef(rpred, reward[held])[0, 1])

    assert np.mean(active_scores) > np.mean(random_scores)


def test_discovered_axes_name_the_planted_structure(world):
    fs, meta, reward, cluster = world["fs"], world["meta"], world["reward"], world["cluster"]
    rng = np.random.default_rng(9)
    train = rng.choice(N_ITEMS, size=60, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    axes = describe_axes(model, fs.item_ids, fs.latent, meta, n_axes=3, pole_size=150)
    assert axes
    # The vocabulary planted in the liked clusters should surface on the
    # preferred pole of at least one discovered axis.
    wanted = {f"theme-{c}" for c in LIKED_CLUSTERS}
    surfaced = {term for ax in axes for term in ax.liked_pole}
    assert wanted & surfaced, f"expected one of {wanted}, got {surfaced}"
    assert all(ax.liked_examples for ax in axes)


def test_nearest_liked_explanations_are_relevant(world):
    fs, cluster = world["fs"], world["cluster"]
    liked = [i for i in range(N_ITEMS) if cluster[i] == LIKED_CLUSTERS[0]][:20]
    labels = [f"Title {i}" for i in liked]
    target = [i for i in range(N_ITEMS) if cluster[i] == LIKED_CLUSTERS[0] and i not in liked][0]

    reasons = nearest_liked(fs.latent[target], fs.latent[fs.rows_for(liked)], labels, top=3)
    assert reasons
    assert all(sim > 0 for _, sim in reasons)
