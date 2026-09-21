import numpy as np
import pytest

from entertainer.models.taste import (
    FeatureMap,
    fit,
    taste_direction,
    verdict_to_reward,
)


def _synthetic(n=200, d=32, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = 1.0 / (1.0 + np.exp(-4.0 * (X @ w)))
    return X, np.clip(y + rng.normal(0, 0.05, n), 0, 1), w


def test_verdict_mapping_is_monotone():
    order = ["hate", "dislike", "meh", "like", "love"]
    values = [verdict_to_reward(v) for v in order]
    assert values == sorted(values)
    assert verdict_to_reward(8) == pytest.approx(0.8)
    assert verdict_to_reward(0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        verdict_to_reward("brilliant")


def test_feature_map_shapes():
    fm = FeatureMap(dim=8, n_rff=16, gamma=0.5, seed=1)
    out = fm(np.zeros((5, 8), dtype=np.float32))
    assert out.shape == (5, fm.out_dim) == (5, 8 + 16 + 1)
    assert np.allclose(out[:, 0], 1.0), "intercept column must be present"


def test_posterior_improves_with_data():
    X, y, w = _synthetic(n=400)
    Xt, yt, _ = _synthetic(n=500, seed=99)
    # Same latent direction for train and test.
    Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
    yt = 1.0 / (1.0 + np.exp(-4.0 * (Xt @ w)))

    corrs = []
    for n in (10, 50, 300):
        model = fit(X[:n], y[:n])
        pred = model.predict(Xt, with_std=False)
        corrs.append(np.corrcoef(pred, yt)[0, 1])
    assert corrs[0] < corrs[1] < corrs[2]
    assert corrs[-1] > 0.85


def test_uncertainty_shrinks_with_data():
    X, y, _ = _synthetic(n=400)
    _, sd_small = fit(X[:10], y[:10]).predict(X[300:])
    _, sd_large = fit(X[:300], y[:300]).predict(X[300:])
    assert sd_large.mean() < sd_small.mean()


def test_thompson_draws_are_coherent_and_varied():
    X, y, _ = _synthetic(n=60)
    model = fit(X, y)
    a = model.thompson_scores(X, np.random.default_rng(1))
    b = model.thompson_scores(X, np.random.default_rng(2))
    assert not np.allclose(a, b), "draws must differ, otherwise there is no exploration"
    # A single shared weight draw keeps the ranking broadly coherent.
    assert np.corrcoef(a, b)[0, 1] > 0.4


def test_capacity_selection_is_conservative_when_data_is_tiny():
    X, y, _ = _synthetic(n=30)
    assert fit(X, y).feature_map.n_rff == 0


def test_taste_direction_has_latent_dimension():
    X, y, _ = _synthetic(n=80, d=16)
    model = fit(X, y)
    assert taste_direction(model).shape == (16,)


def test_roundtrip_persistence(tmp_path):
    from entertainer.models.taste import TasteModel

    X, y, _ = _synthetic(n=150)
    model = fit(X, y)
    path = model.to_npz(tmp_path / "taste.npz")
    restored = TasteModel.from_npz(path)
    assert np.allclose(model.predict(X, with_std=False), restored.predict(X, with_std=False))
    assert restored.n_obs == model.n_obs


def test_sampled_negatives_do_not_license_extra_capacity():
    """Padding rows must not buy model flexibility.

    A thousand sampled negatives push n past the threshold where the evidence
    starts considering a random-feature lift, but they carry no information
    about where this person's taste curves. Capacity is decided by the count
    of real verdicts.
    """
    X, y, _ = _synthetic(n=400)
    real, padded = 20, 380

    naive = fit(X, y, allow_rff=True)
    gated = fit(X, y, allow_rff=True, capacity_obs=real)
    assert gated.feature_map.n_rff == 0, gated.feature_map.n_rff
    assert naive.feature_map.n_rff >= 0
    del padded


def test_capacity_gate_still_allows_a_lift_on_real_evidence():
    X, y, _ = _synthetic(n=400)
    lifted = fit(X, y, allow_rff=True, capacity_obs=400)
    # With 400 genuine labels the evidence is free to choose either; the point
    # is only that the gate is not forcing the linear model.
    assert lifted.log_evidence == pytest.approx(
        fit(X, y, allow_rff=True).log_evidence
    )


def test_pinned_penalty_behaves_like_ridge_with_that_alpha():
    """With the penalty fixed, this is ridge — so it should match one."""
    from sklearn.linear_model import Ridge

    X, y, _ = _synthetic(n=120, d=16)
    model = fit(X, y, allow_rff=False, penalty=5.0)
    ridge = Ridge(alpha=5.0).fit(X, y)

    ours = model.predict(X, with_std=False)
    theirs = ridge.predict(X)
    # Same estimator, same penalty: rankings must agree closely.
    assert np.corrcoef(ours, theirs)[0, 1] > 0.999


def test_pinned_penalty_resists_padding_that_fools_empirical_bayes():
    """The failure the pin exists to prevent.

    A thousand identical pseudo-negatives are trivially easy to fit, so the
    inferred noise precision comes out high and the effective penalty comes
    out far too weak. Pinning it makes the fit insensitive to how much
    constant padding is present.
    """
    rng = np.random.default_rng(0)
    X, y, _ = _synthetic(n=40, d=16)

    def effective(n_pad, penalty):
        pad = rng.normal(size=(n_pad, 16))
        pad /= np.linalg.norm(pad, axis=1, keepdims=True)
        Xa = np.vstack([X, pad])
        ya = np.concatenate([y, np.full(n_pad, 0.15)])
        m = fit(Xa, ya, allow_rff=False, penalty=penalty, capacity_obs=len(y))
        return m.alpha / m.beta

    inferred = [effective(n, None) for n in (100, 1000)]
    pinned = [effective(n, 10.0) for n in (100, 1000)]

    # Inferred regularisation drifts with the amount of padding...
    assert max(inferred) / min(inferred) > 1.5, inferred
    # ...pinned does not.
    assert pinned[0] == pytest.approx(pinned[1], rel=1e-6), pinned
    assert pinned[0] == pytest.approx(10.0, rel=1e-6)


def test_penalty_none_restores_pure_empirical_bayes():
    X, y, _ = _synthetic(n=80)
    free = fit(X, y, allow_rff=False, penalty=None)
    pinned = fit(X, y, allow_rff=False, penalty=10.0)
    assert not np.isclose(free.alpha / free.beta, 10.0)
    assert pinned.alpha / pinned.beta == pytest.approx(10.0, rel=1e-6)
