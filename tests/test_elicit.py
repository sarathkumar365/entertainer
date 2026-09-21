import numpy as np

from entertainer.coldstart.elicit import greedy_dpp, information_gain, next_questions
from entertainer.models.features import build as build_features
from entertainer.models.taste import fit


def _space(n=300, d=24, seed=0, meta=None):
    """Build a feature space the same way the pipeline does.

    Going through `build_features` rather than constructing FeatureSpace by
    hand matters: the vectorised filtering and pool selection read the
    metadata columns it derives, and a hand-built space has none.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, d))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    ids = np.arange(n, dtype=np.int32)
    if meta is None:
        meta = {
            int(i): {"language": "en", "imdb_votes": 100_000, "quality": 0.7, "kind": "movie"}
            for i in ids.tolist()
        }
    return build_features(ids, latent.astype(np.float32), meta)


def test_dpp_avoids_near_duplicates():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(200, 16))
    V[:150] = V[0] + 0.01 * rng.normal(size=(150, 16))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    picked = greedy_dpp(V, np.ones(200), k=8)
    assert len(picked) == 8
    from_cluster = sum(1 for p in picked if p < 150)
    assert from_cluster <= 2


def test_dpp_respects_quality_weighting():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(50, 8))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    q = np.full(50, 0.1)
    q[7] = 10.0
    assert greedy_dpp(V, q, k=1) == [7]


def test_information_gain_is_higher_where_the_model_is_ignorant():
    fs = _space()
    rng = np.random.default_rng(0)
    seen = np.arange(40)
    rewards = rng.uniform(size=40)
    model = fit(fs.matrix[seen], rewards, allow_rff=False)
    gains = information_gain(model, fs.matrix)
    assert gains.min() >= 0
    # Items already observed should be among the better-understood ones.
    assert gains[seen].mean() < gains.mean()


def test_next_questions_are_distinct_and_unasked():
    fs = _space()
    meta = {
        int(i): {"imdb_votes": 100_000, "language": "en", "quality": 0.7, "kind": "movie"}
        for i in fs.item_ids.tolist()
    }
    rng = np.random.default_rng(0)
    model = fit(fs.matrix[:20], rng.uniform(size=20), allow_rff=False)
    asked = {0, 1, 2, 3}
    picks = next_questions(model, fs, meta, asked, k=10, pool=np.arange(300))
    assert len(picks) == len(set(picks)) == 10
    assert not (set(picks) & asked)


def test_seed_questions_do_not_collapse_onto_one_language():
    """Guards the failure mode where the opening set is all English."""
    from collections import Counter

    from entertainer.coldstart.elicit import seed_questions

    fs = _space(n=600, d=24, seed=2)
    rng = np.random.default_rng(3)
    meta = {}
    for i in fs.item_ids.tolist():
        # A catalogue shaped like the real one: English dominates in both
        # count and quality.
        english = i < 480
        meta[int(i)] = {
            "imdb_votes": int(rng.integers(50_000, 900_000)) if english else int(rng.integers(5_000, 60_000)),
            "language": "en" if english else ["ml", "ta", "ko"][i % 3],
            "quality": float(rng.uniform(0.7, 0.95)) if english else float(rng.uniform(0.4, 0.7)),
        }
    picks = seed_questions(fs, meta, k=20, pool=np.arange(600), max_language_share=0.4)
    assert len(picks) == 20
    counts = Counter(meta[p]["language"] for p in picks)
    assert counts["en"] <= 8, counts
    assert len(counts) >= 3, counts


def test_small_language_catalogues_keep_a_usable_question_pool():
    """Ten per cent of 200 Kannada films is twenty — too thin to question from."""
    from entertainer.coldstart.elicit import MIN_POOL_PER_LANGUAGE, recognisable_pool

    rng = np.random.default_rng(6)
    meta = {
        int(i): {
            # 1200 English titles, 200 Kannada.
            "language": "en" if i < 1200 else "kn",
            "imdb_votes": int(rng.integers(1_000, 900_000)),
            "quality": 0.6,
            "kind": "movie",
        }
        for i in range(1400)
    }
    fs = _space(n=1400, d=16, seed=5, meta=meta)
    pool = recognisable_pool(fs, meta, quantile=0.90)
    by_lang: dict[str, int] = {}
    for row in pool.tolist():
        by_lang[meta[int(fs.item_ids[row])]["language"]] = (
            by_lang.get(meta[int(fs.item_ids[row])]["language"], 0) + 1
        )
    assert by_lang.get("kn", 0) >= MIN_POOL_PER_LANGUAGE, by_lang
    # English is large enough that the quantile governs, not the floor.
    assert by_lang.get("en", 0) >= 100, by_lang


def test_a_language_with_a_handful_of_titles_is_not_a_stratum():
    from entertainer.coldstart.elicit import recognisable_pool

    meta = {
        int(i): {
            "language": "en" if i < 295 else "zz",
            "imdb_votes": 50_000,
            "quality": 0.6,
            "kind": "movie",
        }
        for i in range(300)
    }
    fs = _space(n=300, d=16, seed=7, meta=meta)
    pool = recognisable_pool(fs, meta)
    langs = {meta[int(fs.item_ids[r])]["language"] for r in pool.tolist()}
    assert langs == {"en"}, langs


def test_pool_still_prefers_the_better_known_titles():
    from entertainer.coldstart.elicit import recognisable_pool

    meta = {
        int(i): {
            "language": "en",
            "imdb_votes": int(i) * 100,
            "quality": 0.6,
            "kind": "movie",
        }
        for i in range(1, 401)
    }
    fs = _space(n=401, d=16, seed=8, meta=meta)
    pool = recognisable_pool(fs, meta, quantile=0.90)
    votes = [meta[int(fs.item_ids[r])]["imdb_votes"] for r in pool.tolist()]
    assert min(votes) > max(
        meta[int(i)]["imdb_votes"] for i in range(1, 51)
    ), "the least-voted titles must not be in the question pool"
