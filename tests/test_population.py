"""Tests for the population prior.

Two things need proving. That the reparameterisation is exact — an identity
prior must reproduce the isotropic fit to numerical precision, or every
comparison between the two is confounded. And that the prior actually helps
where it is claimed to help: at small n, when there is almost no evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from entertainer.models.population import PopulationPrior
from entertainer.models.taste import fit


def _world(d=24, n_items=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_items, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, rng


def _population(d, rng, n_users=400, manifold=4):
    """Taste vectors confined to a low-dimensional subspace, as real ones are."""
    basis = rng.normal(size=(manifold, d))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coeffs = rng.normal(size=(n_users, manifold))
    W = coeffs @ basis + 0.05 * rng.normal(size=(n_users, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W, basis


def _prior_from(W, shrinkage=0.15):
    mean = W.mean(axis=0)
    centred = W - mean
    cov = (centred.T @ centred) / (len(W) - 1)
    target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    cov = (1 - shrinkage) * cov + shrinkage * target
    return PopulationPrior(mean=mean, cov=cov, n_users=len(W), dim=cov.shape[0])


def test_identity_prior_reproduces_the_isotropic_fit():
    """The reparameterisation must be exact, or every comparison is confounded."""
    X, rng = _world()
    d = X.shape[1]
    w = rng.normal(size=d)
    y = np.clip(0.5 + 0.3 * (X @ w / np.linalg.norm(w)), 0, 1)

    identity = PopulationPrior(
        mean=np.zeros(d), cov=np.eye(d), n_users=0, dim=d
    )
    # Both sides with pure empirical Bayes: the pinned penalty applies only
    # to the isotropic branch, so it has to be switched off here for the
    # reparameterisation to be comparable at all.
    plain = fit(X[:60], y[:60], allow_rff=False, penalty=None)
    reparam = fit(X[:60], y[:60], allow_rff=False, prior=identity, alpha_anchor=0.0)

    assert np.allclose(plain.mean, reparam.mean, atol=1e-8)
    assert np.allclose(plain.cov, reparam.cov, atol=1e-10)
    assert plain.log_evidence == pytest.approx(reparam.log_evidence)


def test_the_hyperprior_is_what_rescues_the_small_sample_case():
    """Isolates the anchor from the prior, so the cause is not guessed at."""
    X, rng = _world(seed=3)
    d = X.shape[1]
    W, basis = _population(d, rng)
    prior = _prior_from(W)

    free, anchored = [], []
    for trial in range(40):
        trial_rng = np.random.default_rng(700 + trial)
        target = trial_rng.normal(size=basis.shape[0]) @ basis
        target /= np.linalg.norm(target)
        truth = np.clip(0.5 + 0.3 * (X @ target), 0, 1)
        train = trial_rng.choice(len(X), size=5, replace=False)
        held = np.setdiff1d(np.arange(len(X)), train)

        for store, anchor in ((free, 0.0), (anchored, None)):
            model = fit(
                X[train], truth[train], allow_rff=False, prior=prior, alpha_anchor=anchor
            )
            store.append(
                np.corrcoef(model.predict(X[held], with_std=False), truth[held])[0, 1]
            )

    assert np.mean(anchored) > np.mean(free) + 0.2, (np.mean(anchored), np.mean(free))


def test_population_prior_helps_most_when_evidence_is_scarce():
    """The claim the prior exists to make."""
    X, rng = _world(seed=1)
    d = X.shape[1]
    W, basis = _population(d, rng)
    prior = _prior_from(W)

    gains = {}
    for budget in (4, 8, 16, 64):
        with_prior, without = [], []
        for trial in range(25):
            trial_rng = np.random.default_rng(500 + trial)
            # A new user drawn from the same population the prior describes.
            target = trial_rng.normal(size=basis.shape[0]) @ basis
            target /= np.linalg.norm(target)
            truth = np.clip(0.5 + 0.3 * (X @ target), 0, 1)

            train = trial_rng.choice(len(X), size=budget, replace=False)
            held = np.setdiff1d(np.arange(len(X)), train)

            a = fit(X[train], truth[train], allow_rff=False, prior=prior)
            # Compared against pure empirical Bayes, which is what the prior
            # replaces. The pinned penalty is a separate, later change and
            # measuring both at once would confound them.
            b = fit(X[train], truth[train], allow_rff=False, penalty=None)
            with_prior.append(
                np.corrcoef(a.predict(X[held], with_std=False), truth[held])[0, 1]
            )
            without.append(
                np.corrcoef(b.predict(X[held], with_std=False), truth[held])[0, 1]
            )
        gains[budget] = float(np.mean(with_prior) - np.mean(without))

    # The prior earns its keep precisely where evidence is thinnest.
    assert gains[4] > 0.2, gains
    assert gains[8] > 0.2, gains
    assert gains[16] > 0.1, gains
    # And the advantage must shrink as evidence accumulates — a prior that
    # still dominated at n=64 would be overriding the user rather than
    # informing them.
    assert gains[64] < 0.05, gains
    assert gains[4] > gains[64], gains


def test_a_user_who_contradicts_the_population_is_not_overridden():
    """Empirical Bayes must be able to walk away from the prior."""
    X, rng = _world(seed=2)
    d = X.shape[1]
    W, basis = _population(d, rng)
    prior = _prior_from(W)

    # A taste direction orthogonal to everything the population contains.
    u, _, _ = np.linalg.svd(basis.T, full_matrices=True)
    contrarian = u[:, -1]
    truth = np.clip(0.5 + 0.3 * (X @ contrarian), 0, 1)

    train = rng.choice(len(X), size=250, replace=False)
    held = np.setdiff1d(np.arange(len(X)), train)
    model = fit(X[train], truth[train], allow_rff=False, prior=prior)
    corr = np.corrcoef(model.predict(X[held], with_std=False), truth[held])[0, 1]
    assert corr > 0.9, corr


def test_prior_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    W, _ = _population(12, rng, n_users=100, manifold=3)
    prior = _prior_from(W)
    path = tmp_path / "prior.npz"
    prior.save(path)
    restored = PopulationPrior.load(path)
    assert np.allclose(prior.mean, restored.mean)
    assert np.allclose(prior.cov, restored.cov)
    assert restored.n_users == prior.n_users


def test_cholesky_is_usable_even_with_a_degenerate_covariance():
    d = 8
    cov = np.zeros((d, d))
    cov[0, 0] = 1.0        # rank one
    prior = PopulationPrior(mean=np.zeros(d), cov=cov, n_users=10, dim=d)
    chol = prior.cholesky()
    assert np.all(np.isfinite(chol))
    assert chol.shape == (d, d)
