import numpy as np

from entertainer.coldstart.elicit import greedy_dpp, information_gain, next_questions
from entertainer.models.features import FeatureSpace
from entertainer.models.taste import fit


def _space(n=300, d=24, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, d))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    side = np.zeros((n, 5), dtype=np.float32)
    ids = np.arange(n, dtype=np.int32)
    return FeatureSpace(ids, latent.astype(np.float32), side, {int(i): i for i in range(n)})


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
        int(i): {"imdb_votes": 100_000, "language": "en", "quality": 0.7}
        for i in fs.item_ids.tolist()
    }
    rng = np.random.default_rng(0)
    model = fit(fs.matrix[:20], rng.uniform(size=20), allow_rff=False)
    asked = {0, 1, 2, 3}
    picks = next_questions(model, fs, meta, asked, k=10, pool=np.arange(300))
    assert len(picks) == len(set(picks)) == 10
    assert not (set(picks) & asked)
