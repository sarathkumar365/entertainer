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

    return dict(
        fs=fs, meta=meta, cluster=cluster, reward=true_reward, art=art,
        content=content, covered=covered,
    )


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


def test_active_elicitation_beats_random_at_a_small_budget(world):
    """The justification for the question-selection machinery.

    Measured at ten questions, which is where it matters. Given enough
    answers, any reasonable selection converges to the same place and the
    comparison stops being meaningful — this synthetic world saturates past
    about twenty. The real comparison lives in the MovieLens replay.
    """
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    pool = np.arange(N_ITEMS)
    budget = 10

    active, random_ = [], []
    for trial in range(8):
        rng = np.random.default_rng(300 + trial)

        answered = list(elicit.seed_questions(fs, meta, k=6, pool=pool))
        model = fit(fs.vectors_for(answered), reward[answered], allow_rff=False)
        while len(answered) < budget:
            batch = elicit.next_questions(
                model, fs, meta, set(answered), k=budget - len(answered), pool=pool, rng=rng
            )
            if not batch:
                break
            answered.extend(batch)
            model = fit(fs.vectors_for(answered), reward[answered], allow_rff=False)
        active.append(
            np.corrcoef(model.predict(fs.matrix, with_std=False), reward)[0, 1]
        )

        chosen = rng.choice(N_ITEMS, size=len(answered), replace=False)
        rmodel = fit(fs.vectors_for(chosen), reward[chosen], allow_rff=False)
        random_.append(
            np.corrcoef(rmodel.predict(fs.matrix, with_std=False), reward)[0, 1]
        )

    assert np.mean(active) > np.mean(random_)


def test_v_optimal_score_matches_a_brute_force_posterior_update(world):
    """The closed form must equal what an explicit rank-one update produces.

    This is the check that matters for the V-optimal criterion: the algebra
    collapses a per-candidate posterior update into one quadratic form, and a
    sign or transpose error there would be invisible in any end-to-end metric.
    """
    from entertainer.coldstart.elicit import variance_reduction

    fs, reward = world["fs"], world["reward"]
    rng = np.random.default_rng(2)
    train = rng.choice(N_ITEMS, size=30, replace=False)
    model = fit(fs.vectors_for(train), reward[train], allow_rff=False)

    candidates = rng.choice(N_ITEMS, size=15, replace=False)
    reference = fs.matrix[rng.choice(N_ITEMS, size=200, replace=False)]

    closed_form = variance_reduction(model, fs.matrix[candidates], reference)

    phi_ref = model.feature_map(reference)
    cov = model.cov
    brute = []
    for c in candidates:
        x = model.feature_map(fs.matrix[c : c + 1])[0]
        denom = 1.0 + model.beta * float(x @ cov @ x)
        cov_after = cov - model.beta * np.outer(cov @ x, x @ cov) / denom
        before = np.einsum("ij,jk,ik->i", phi_ref, cov, phi_ref).sum()
        after = np.einsum("ij,jk,ik->i", phi_ref, cov_after, phi_ref).sum()
        brute.append((before - after) / len(phi_ref))

    assert np.allclose(closed_form, brute, rtol=1e-6, atol=1e-12)


def test_both_elicitation_criteria_produce_distinct_usable_batches(world):
    fs, meta = world["fs"], world["meta"]
    reward = world["reward"]
    pool = np.arange(N_ITEMS)
    rng = np.random.default_rng(4)
    train = rng.choice(N_ITEMS, size=25, replace=False)
    model = fit(fs.vectors_for(train), reward[train], allow_rff=False)

    d_picks = elicit.next_questions(
        model, fs, meta, set(train.tolist()), k=12, pool=pool, criterion="d-optimal"
    )
    v_picks = elicit.next_questions(
        model, fs, meta, set(train.tolist()), k=12, pool=pool, criterion="v-optimal", rng=rng
    )
    for picks in (d_picks, v_picks):
        assert len(picks) == len(set(picks)) == 12
        assert not (set(picks) & set(train.tolist()))
    assert set(d_picks) != set(v_picks), "the two criteria should not coincide exactly"


def test_discovered_axes_name_the_planted_structure(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
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


def test_explore_knob_widens_the_slate(world):
    """The risk dial must actually change behaviour in the stated direction."""
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(5)
    train = rng.choice(N_ITEMS, size=30, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    def spread(explore):
        slates = [
            {
                p.item_id
                for p in recommend(
                    model, fs, meta, k=10, rng=np.random.default_rng(s),
                    strategy="thompson", explore=explore,
                )
            }
            for s in range(6)
        ]
        return len(set.union(*slates))

    assert spread(0.0) < spread(1.0) < spread(3.0)


def test_novelty_penalty_shifts_towards_less_seen_titles(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(5)
    train = rng.choice(N_ITEMS, size=60, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    def mean_votes(novelty):
        picks = recommend(
            model, fs, meta, k=10, rng=np.random.default_rng(3),
            strategy="mean", novelty=novelty,
        )
        return np.mean([meta[p.item_id]["imdb_votes"] for p in picks])

    assert mean_votes(1.5) < mean_votes(0.0)


def test_feature_space_survives_entirely_missing_metadata():
    """A feature nobody has must contribute zero, never NaN."""
    from entertainer.models.features import build as build_features

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(50, 12)).astype(np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    ids = np.arange(50, dtype=np.int32)
    # No year, no runtime, no quality anywhere.
    meta = {int(i): {"language": "en", "imdb_votes": 1000} for i in ids}

    fs = build_features(ids, latent, meta)
    assert np.isfinite(fs.matrix).all()
    assert np.allclose(fs.side[:, 0], 0.0)


def test_shortlisted_variance_matches_the_exact_computation(world):
    """The fast path must agree with the slow one it replaced."""
    from entertainer.recommend import _predictive_std

    fs, reward = world["fs"], world["reward"]
    rng = np.random.default_rng(6)
    train = rng.choice(N_ITEMS, size=200, replace=False)   # enough to trigger RFF
    model = fit(fs.vectors_for(train), reward[train])

    rows = rng.choice(N_ITEMS, size=40, replace=False)
    phi = model.feature_map(fs.matrix[rows])
    fast = _predictive_std(model, phi)
    _, exact = model.predict(fs.matrix[rows])
    assert np.allclose(fast, exact, rtol=1e-5, atol=1e-8)


def test_ucb_respects_the_explore_setting(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(6)
    train = rng.choice(N_ITEMS, size=25, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    def slate(kappa):
        return [
            p.item_id
            for p in recommend(
                model, fs, meta, k=10, strategy="ucb", explore=kappa,
                rng=np.random.default_rng(0),
            )
        ]

    greedy = slate(0.0)
    optimistic = slate(3.0)
    assert greedy != optimistic
    # Zero kappa must reproduce the pure posterior-mean ranking.
    assert greedy == [p.item_id for p in recommend(
        model, fs, meta, k=10, strategy="mean", rng=np.random.default_rng(0)
    )]


def test_every_recommendation_carries_a_finite_uncertainty(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(8)
    train = rng.choice(N_ITEMS, size=40, replace=False)
    model = fit(fs.vectors_for(train), reward[train])
    picks = recommend(model, fs, meta, k=12, rng=np.random.default_rng(2))
    assert picks
    assert all(np.isfinite(p.std) and p.std > 0 for p in picks)
    assert all(0.0 < p.propensity <= 1.0 for p in picks)


def test_out_of_sample_projection_matches_the_in_sample_coordinates(world):
    """A title projected after the fact must land where it would have during the build.

    This is what lets the catalogue grow — a film too obscure to have made the
    vote floor can be fetched, embedded and placed into the same space without
    rebuilding anything. If the projection drifted, those titles would be
    scored in a subtly different geometry from every other one.

    Checked on the titles that had no genuine collaborative factors, since
    those are the ones whose in-sample coordinates came purely through the
    imputation path the projection has to reproduce.
    """
    art, content, covered = world["art"], world["content"], world["covered"]
    uncovered = np.setdiff1d(np.arange(N_ITEMS), covered)
    assert uncovered.size > 100

    rng = np.random.default_rng(11)
    sample = rng.choice(uncovered, size=40, replace=False)

    projected = art.project(content[sample])
    assert projected.shape == (40, art.space.shape[1])
    cosines = np.einsum("ij,ij->i", projected, art.space[sample])
    assert cosines.min() > 0.999, cosines.min()


def test_projection_of_a_covered_title_is_close_but_not_identical(world):
    """Titles MovieLens covered used their real factors, not the imputed ones.

    The projection can only ever reproduce the imputed path, so it should land
    near a covered title but not on it. Asserting this keeps anyone from
    later assuming projection is exact for every title.
    """
    art, content, covered = world["art"], world["content"], world["covered"]
    rng = np.random.default_rng(12)
    sample = rng.choice(covered, size=40, replace=False)
    cosines = np.einsum("ij,ij->i", art.project(content[sample]), art.space[sample])
    assert 0.5 < float(np.mean(cosines)) < 0.999


def test_projection_refuses_a_space_that_cannot_support_it():
    from entertainer.models.fusion import FusionArtifacts

    art = FusionArtifacts(
        item_ids=np.arange(3, dtype=np.int32),
        space=np.eye(3, dtype=np.float32),
        components=np.eye(3, dtype=np.float32),
        block_sizes=(2, 1),
        cf_r2=0.5,
        cf_coverage=1.0,
    )
    with pytest.raises(RuntimeError, match="rebuild"):
        art.project(np.zeros((1, 2), dtype=np.float32))


def test_mood_vector_nudges_ranking_without_overriding_taste(world):
    """A mood is a second opinion, not a constraint."""
    fs, meta, reward, cluster = world["fs"], world["meta"], world["reward"], world["cluster"]
    rng = np.random.default_rng(13)
    train = rng.choice(N_ITEMS, size=60, replace=False)
    model = fit(fs.vectors_for(train), reward[train])

    # A "mood" pointing squarely at one of the clusters the user dislikes.
    disliked = [i for i in range(N_ITEMS) if cluster[i] not in LIKED_CLUSTERS]
    mood = fs.latent[fs.rows_for(disliked[:80])].mean(axis=0)
    mood /= np.linalg.norm(mood)

    def hit_rate(weight):
        picks = recommend(
            model, fs, meta, k=10, strategy="mean",
            rng=np.random.default_rng(1), mood=mood, mood_weight=weight,
        )
        return np.mean([cluster[p.item_id] in LIKED_CLUSTERS for p in picks])

    # With no mood the slate is the user's taste; a strong mood pulls it away.
    assert hit_rate(0.0) > hit_rate(3.0)
    # But a gentle mood must not flip the slate wholesale.
    assert hit_rate(0.3) >= hit_rate(3.0)


def test_mood_leaves_hard_filters_alone(world):
    fs, meta, reward = world["fs"], world["meta"], world["reward"]
    rng = np.random.default_rng(14)
    train = rng.choice(N_ITEMS, size=40, replace=False)
    model = fit(fs.vectors_for(train), reward[train])
    mood = fs.latent[7]

    picks = recommend(
        model, fs, meta, k=8, strategy="mean", rng=np.random.default_rng(1),
        filters=Filters(languages=("ml",)), mood=mood, mood_weight=5.0,
    )
    assert picks
    assert all(meta[p.item_id]["language"] == "ml" for p in picks)


def test_filtering_is_vectorised_and_agrees_with_the_obvious_loop(world):
    """The fast mask must match a naive implementation exactly."""
    fs, meta = world["fs"], world["meta"]
    cases = [
        Filters(),
        Filters(languages=("ml",)),
        Filters(languages=("ml", "ko")),
        Filters(kind="movie"),
        Filters(kind="tv"),
        Filters(min_year=2000),
        Filters(max_year=1995),
        Filters(min_runtime=120),
        Filters(max_runtime=100),
        Filters(min_votes=50_000),
        Filters(languages=("ta",), kind="movie", min_year=1990, max_runtime=150),
        Filters(exclude=frozenset({0, 5, 9, 500})),
        Filters(languages=("nonexistent",)),
    ]
    for f in cases:
        fast = f.mask(fs)
        naive = np.ones(len(fs.item_ids), dtype=bool)
        for i, iid in enumerate(fs.item_ids.tolist()):
            r = meta.get(int(iid))
            ok = r is not None
            if ok and int(iid) in f.exclude:
                ok = False
            if ok and f.languages and r.get("language") not in f.languages:
                ok = False
            if ok and f.kind and r.get("kind") != f.kind:
                ok = False
            if ok and f.min_year and (r.get("year") or 0) < f.min_year:
                ok = False
            if ok and f.max_year and not (0 < (r.get("year") or 0) <= f.max_year):
                ok = False
            if ok and f.max_runtime and not (0 < (r.get("runtime") or 0) <= f.max_runtime):
                ok = False
            if ok and f.min_runtime and (r.get("runtime") or 0) < f.min_runtime:
                ok = False
            if ok and f.min_votes and (r.get("imdb_votes") or 0) < f.min_votes:
                ok = False
            naive[i] = ok
        assert np.array_equal(fast, naive), f


def test_an_unknown_runtime_fails_a_runtime_constraint(world):
    """A hard constraint must never be relaxed for missing data."""
    from entertainer.models.features import build as build_features

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(4, 8)).astype(np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    ids = np.arange(4, dtype=np.int32)
    meta = {
        0: {"language": "en", "kind": "movie", "runtime": 90, "imdb_votes": 10},
        1: {"language": "en", "kind": "movie", "imdb_votes": 10},        # no runtime
        2: {"language": "en", "kind": "movie", "runtime": 200, "imdb_votes": 10},
        3: {"language": "en", "kind": "movie", "runtime": 95, "imdb_votes": 10},
    }
    fs = build_features(ids, latent, meta)
    keep = Filters(max_runtime=100).mask(fs)
    assert list(keep) == [True, False, False, True]


def test_truncating_collaborative_factors_improves_imputation(world):
    """The measurement that set CF_RANK, reproduced on synthetic data.

    iALS spreads variance across all its dimensions, but only the leading
    directions are recoverable from text. Keeping the rest halves the
    imputation accuracy for every title MovieLens never covered — which is
    most of a multilingual catalogue.
    """
    content, covered = world["content"], world["covered"]
    cluster = world["cluster"]
    rng = np.random.default_rng(21)

    # Collaborative factors: a few content-aligned directions, then noise that
    # no synopsis could predict — the shape the real factors turned out to have.
    signal = rng.normal(size=(N_CLUSTERS, 8))
    factors = np.hstack(
        [
            signal[cluster[covered]] + 0.3 * rng.normal(size=(len(covered), 8)),
            rng.normal(size=(len(covered), 56)),
        ]
    ).astype(np.float32)

    item_ids = np.arange(N_ITEMS, dtype=np.int32)
    mapping = {int(i): int(i) for i in covered}

    low = fusion.build(
        item_ids, content, covered.astype(np.int32), factors, mapping,
        dim=32, seed=0, cf_rank=8,
    )
    full = fusion.build(
        item_ids, content, covered.astype(np.int32), factors, mapping,
        dim=32, seed=0, cf_rank=64,
    )
    assert low.cf_r2 > full.cf_r2, (low.cf_r2, full.cf_r2)
    assert low.block_sizes[1] == 8
    assert full.block_sizes[1] == 64


def test_truncation_keeps_the_projection_consistent(world):
    """Out-of-sample projection must still land on in-sample coordinates."""
    content, covered = world["content"], world["covered"]
    item_ids = np.arange(N_ITEMS, dtype=np.int32)
    rng = np.random.default_rng(22)
    factors = rng.normal(size=(len(covered), 40)).astype(np.float32)

    art = fusion.build(
        item_ids, content, covered.astype(np.int32), factors,
        {int(i): int(i) for i in covered}, dim=24, seed=0, cf_rank=12,
    )
    uncovered = np.setdiff1d(np.arange(N_ITEMS), covered)[:30]
    cosines = np.einsum(
        "ij,ij->i", art.project(content[uncovered]), art.space[uncovered]
    )
    assert cosines.min() > 0.999, cosines.min()


def test_a_compressed_reward_band_still_produces_a_usable_model(world):
    """The failure that sank the first real benchmark.

    People rate what they expected to like, so real verdicts arrive as a
    narrow band near the top of the scale. Fitting on those alone made every
    weight collapse to zero and the model predict a constant — NDCG driven
    entirely by tie-breaking. Sampled negatives supply the missing contrast.
    """
    from entertainer.engine import NEGATIVE_REWARD, NEGATIVE_WEIGHT

    fs, reward, cluster = world["fs"], world["reward"], world["cluster"]
    rng = np.random.default_rng(31)

    # Only titles the viewer liked, rated in a tight band — the real shape.
    liked = np.array([i for i in range(N_ITEMS) if cluster[i] in LIKED_CLUSTERS])
    train = rng.choice(liked, size=30, replace=False)
    compressed = 0.78 + 0.09 * rng.standard_normal(30)

    bare = fit(fs.vectors_for(train), compressed, allow_rff=False)
    bare_pred = bare.predict(fs.matrix, with_std=False)

    negatives = rng.choice(N_ITEMS, size=600, replace=False)
    X = np.vstack([fs.vectors_for(train), fs.matrix[negatives]])
    y = np.concatenate([compressed, np.full(len(negatives), NEGATIVE_REWARD)])
    w = np.concatenate([np.ones(30), np.full(len(negatives), NEGATIVE_WEIGHT)])
    with_neg = fit(X, y, sample_weight=w, allow_rff=False)
    neg_pred = with_neg.predict(fs.matrix, with_std=False)

    # Without negatives the posterior collapses: nothing to rank by.
    assert bare_pred.std() < 0.02, bare_pred.std()
    assert neg_pred.std() > bare_pred.std() * 5

    # And the model can now actually separate the liked clusters.
    is_liked = np.isin(cluster, LIKED_CLUSTERS)
    assert neg_pred[is_liked].mean() > neg_pred[~is_liked].mean() + 0.05


def test_sampled_negatives_exclude_titles_the_user_rated(world):
    from entertainer.engine import _add_sampled_negatives

    fs, reward = world["fs"], world["reward"]
    rng = np.random.default_rng(3)
    ids = rng.choice(N_ITEMS, size=40, replace=False)
    X = fs.vectors_for(ids)
    rewards = reward[ids]
    weights = np.ones(40)

    X2, y2, w2 = _add_sampled_negatives(
        fs, X, rewards, weights, known=set(int(i) for i in ids), count=500, seed=1
    )
    assert X2.shape[0] == len(y2) == len(w2) > 40
    # Original rows untouched, appended rows all weak negatives.
    assert np.allclose(y2[:40], rewards)
    assert np.all(y2[40:] < 0.2)
    assert np.all(w2[40:] < 1.0)
