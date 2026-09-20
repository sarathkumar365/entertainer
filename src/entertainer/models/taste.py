"""The preference model: what the engine learns about one person.

Constraints that shaped this design, in order of importance.

1. **The supervision is names and a verdict, nothing else.** No genre, no
   reason, no rating scale the user has to calibrate. So the model has to
   find the structure itself, in the latent space built by ``fusion.py``.

2. **There will never be much data.** A person might label two hundred films
   in a year, not two million. Anything with the capacity to overfit two
   hundred points will overfit two hundred points. That rules out the
   fashionable answers — a fine-tuned transformer, a deep two-tower net —
   not because they are bad ideas but because they are the wrong ideas at
   n=200.

3. **The model must know what it does not know.** A recommender that cannot
   distinguish "I am confident you will love this" from "I have no idea, but
   my point estimate is high" will spend its life re-recommending the middle
   of the distribution the user already occupies. Uncertainty is what buys
   exploration, and exploration is what lets taste be *discovered* rather
   than merely confirmed.

The answer to all three at once is Bayesian linear regression with
empirical-Bayes hyperparameters, optionally lifted through random Fourier
features once there is enough data to support curvature. It is closed form,
it fits in microseconds, its posterior covariance is exact rather than
approximated by dropout, and its posterior mean *is* the interpretable taste
vector that ``discover.py`` reads back. Thompson sampling over that exact
posterior is the exploration policy.

This is a deliberate choice of the right tool over the impressive one. The
scaling laws that make deep recommenders win start several orders of
magnitude above a single human's viewing history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import PATHS

# Verdict vocabulary. Deliberately coarse: people are reliable about
# "loved it / liked it / fine / no", and unreliable about 7 versus 8.
VERDICTS: dict[str, float] = {
    "love": 1.0,
    "like": 0.75,
    "ok": 0.4,
    "meh": 0.4,
    "dislike": 0.1,
    "hate": 0.0,
}

# Below this many labels, extra capacity is pure variance.
RFF_MIN_OBS = 120


def verdict_to_reward(verdict: str | float) -> float:
    if isinstance(verdict, int | float):
        v = float(verdict)
        return max(0.0, min(1.0, v / 10.0 if v > 1.0 else v))
    key = str(verdict).strip().lower()
    if key not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; use one of {sorted(VERDICTS)}")
    return VERDICTS[key]


@dataclass
class FeatureMap:
    """Identity features, optionally augmented with random Fourier features.

    RFFs approximate an RBF kernel, which lets the model express "I like this
    *region* of taste space" rather than only "I like this direction". That
    matters for people with genuinely multi-modal taste — arthouse dramas and
    stupid action comedies, nothing in between — which a purely linear model
    would average into a preference for neither.
    """

    dim: int
    n_rff: int = 0
    gamma: float = 1.0
    seed: int = 0
    W: np.ndarray | None = field(default=None, repr=False)
    b: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_rff and self.W is None:
            rng = np.random.default_rng(self.seed)
            self.W = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(self.dim, self.n_rff))
            self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_rff)

    @property
    def out_dim(self) -> int:
        return self.dim + self.n_rff + 1  # +1 intercept

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        parts = [np.ones((X.shape[0], 1), dtype=np.float32), X]
        if self.n_rff and self.W is not None and self.b is not None:
            proj = np.sqrt(2.0 / self.n_rff) * np.cos(X @ self.W + self.b)
            parts.append(proj.astype(np.float32))
        return np.hstack(parts)

    def to_dict(self) -> dict:
        return {
            "dim": self.dim, "n_rff": self.n_rff,
            "gamma": self.gamma, "seed": self.seed,
        }


@dataclass
class TasteModel:
    """Posterior over one person's preference function."""

    feature_map: FeatureMap
    mean: np.ndarray            # (F,) posterior mean weights
    cov: np.ndarray             # (F, F) posterior covariance
    alpha: float                # prior precision
    beta: float                 # noise precision
    n_obs: int
    y_mean: float
    log_evidence: float
    _chol: np.ndarray | None = field(default=None, repr=False)

    # --- inference ---------------------------------------------------------

    def predict(self, X: np.ndarray, with_std: bool = True):
        phi = self.feature_map(X)
        mu = phi @ self.mean + self.y_mean
        if not with_std:
            return mu
        # Predictive variance = epistemic (parameter) + aleatoric (noise).
        var = np.einsum("ij,jk,ik->i", phi, self.cov, phi) + 1.0 / self.beta
        return mu, np.sqrt(np.maximum(var, 1e-12))

    def sample_weights(
        self, rng: np.random.Generator, n: int = 1, temperature: float = 1.0
    ) -> np.ndarray:
        """Draw weight vectors from the posterior.

        ``temperature`` scales the posterior standard deviation. 1.0 is exact
        Thompson sampling and is the theoretically right default. Below 1.0
        the policy becomes more conservative, above 1.0 more adventurous —
        which is not a better algorithm, it is a knob for a person who knows
        whether they want a safe evening or a surprising one.
        """
        if self._chol is None:
            jitter = 1e-8 * np.eye(self.cov.shape[0])
            self._chol = np.linalg.cholesky(self.cov + jitter)
        z = rng.standard_normal(size=(self.mean.shape[0], n))
        return (self.mean[:, None] + temperature * (self._chol @ z)).T

    def thompson_scores(
        self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0
    ) -> np.ndarray:
        """One posterior draw, scored across every candidate.

        A single shared draw (not one per item) is what makes this Thompson
        sampling rather than noisy greedy: it samples a coherent hypothesis
        about the user and then acts optimally under it.
        """
        w = self.sample_weights(rng, 1, temperature=temperature)[0]
        return self.feature_map(X) @ w + self.y_mean

    def ucb_scores(self, X: np.ndarray, kappa: float = 1.0) -> np.ndarray:
        mu, sd = self.predict(X)
        return mu + kappa * sd

    # --- persistence -------------------------------------------------------

    def to_npz(self, path: Path | None = None) -> Path:
        PATHS.ensure()
        path = path or (PATHS.artifacts / "taste.npz")
        fm = self.feature_map
        np.savez_compressed(
            path,
            mean=self.mean, cov=self.cov,
            alpha=np.array([self.alpha]), beta=np.array([self.beta]),
            n_obs=np.array([self.n_obs]), y_mean=np.array([self.y_mean]),
            log_evidence=np.array([self.log_evidence]),
            spec=np.array([json.dumps(fm.to_dict())]),
            W=fm.W if fm.W is not None else np.zeros(0, dtype=np.float32),
            b=fm.b if fm.b is not None else np.zeros(0, dtype=np.float32),
        )
        return path

    @classmethod
    def from_npz(cls, path: Path | None = None) -> TasteModel:
        path = path or (PATHS.artifacts / "taste.npz")
        z = np.load(path, allow_pickle=False)
        spec = json.loads(str(z["spec"][0]))
        W = z["W"] if z["W"].size else None
        b = z["b"] if z["b"].size else None
        fm = FeatureMap(**spec, W=W, b=b)
        return cls(
            feature_map=fm, mean=z["mean"], cov=z["cov"],
            alpha=float(z["alpha"][0]), beta=float(z["beta"][0]),
            n_obs=int(z["n_obs"][0]), y_mean=float(z["y_mean"][0]),
            log_evidence=float(z["log_evidence"][0]),
        )

    @staticmethod
    def exists(path: Path | None = None) -> bool:
        return (path or (PATHS.artifacts / "taste.npz")).exists()


# --- fitting ----------------------------------------------------------------


def _evidence_fit(
    phi: np.ndarray, y: np.ndarray, iters: int = 200, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Empirical-Bayes (MacKay) fit of a Bayesian linear model.

    Both the prior precision and the noise precision are learned from the data
    by maximising the marginal likelihood, so there is nothing to
    cross-validate — which matters when the entire dataset is forty films and
    a held-out split would be mostly noise.

    The Gram matrix is diagonalised once up front. In the eigenbasis the
    posterior is diagonal, so each hyperparameter iteration costs O(f) rather
    than the O(f^3) of re-inverting the precision matrix every time. That
    turns the hyperparameter search from the dominant cost into a rounding
    error, which is what makes it affordable to refit from scratch on every
    command and to run a full leave-future-out audit over an entire history.
    """
    n, f = phi.shape
    gram = phi.T @ phi
    rhs = phi.T @ y

    # gram = V diag(eigs) V^T, symmetric positive semi-definite.
    eigs, V = np.linalg.eigh(gram)
    eigs = np.maximum(eigs, 0.0)
    Vt_rhs = V.T @ rhs
    y_sq = float(y @ y)

    alpha, beta = 1.0, 4.0
    for _ in range(iters):
        denom = alpha + beta * eigs
        m_eig = beta * Vt_rhs / denom          # posterior mean in the eigenbasis
        mtm = float(m_eig @ m_eig)
        gamma = float(np.sum(beta * eigs / denom))
        # ||y - phi m||^2 evaluated without forming phi m.
        resid = max(y_sq - 2.0 * float(m_eig @ Vt_rhs) + float((m_eig ** 2) @ eigs), 1e-12)

        new_alpha = float(np.clip(gamma / max(mtm, 1e-9), 1e-4, 1e6))
        new_beta = float(np.clip(max(n - gamma, 1e-6) / resid, 1e-4, 1e6))
        converged = abs(new_alpha - alpha) < tol and abs(new_beta - beta) < tol
        alpha, beta = new_alpha, new_beta
        if converged:
            break

    denom = alpha + beta * eigs
    m_eig = beta * Vt_rhs / denom
    mean = V @ m_eig
    cov = (V / denom) @ V.T
    resid = max(y_sq - 2.0 * float(m_eig @ Vt_rhs) + float((m_eig ** 2) @ eigs), 1e-12)
    logdet = float(np.sum(np.log(denom)))
    log_evidence = 0.5 * (
        f * np.log(alpha) + n * np.log(beta)
        - beta * resid - alpha * float(m_eig @ m_eig)
        - logdet - n * np.log(2 * np.pi)
    )
    return mean, cov, alpha, beta, float(log_evidence)


def fit(
    X: np.ndarray,
    rewards: np.ndarray,
    sample_weight: np.ndarray | None = None,
    allow_rff: bool = True,
    rff_candidates: tuple[int, ...] = (0, 128, 256),
    gamma_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    seed: int = 0,
) -> TasteModel:
    """Fit the taste posterior, selecting model capacity by marginal likelihood.

    Capacity selection is itself Bayesian: candidate feature maps are scored
    by their evidence, which automatically penalises the extra parameters of
    the RFF lift. With forty labels the evidence reliably picks the plain
    linear model; somewhere past a couple of hundred it starts preferring
    curvature, which is exactly the behaviour wanted.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    y = np.asarray(rewards, dtype=np.float64).ravel()
    n, d = X.shape
    if n == 0:
        raise ValueError("no labelled examples")

    if sample_weight is None:
        sqrt_w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        w = w / w.mean()
        sqrt_w = np.sqrt(w)

    # Centre on the weighted mean, and apply the weights to the *design
    # matrix* rather than to X, so that the intercept column the feature map
    # adds is weighted along with everything else. Scaling X alone leaves the
    # intercept unweighted, which quietly biases the fit whenever weak
    # negatives (skips) are mixed with stated verdicts.
    y_mean = float(np.average(y, weights=sqrt_w**2))
    yc = y - y_mean

    candidates: list[tuple[int, float]] = [(0, 1.0)]
    if allow_rff and n >= RFF_MIN_OBS:
        candidates += [(r, g) for r in rff_candidates if r for g in gamma_candidates]

    best: TasteModel | None = None
    for n_rff, gamma in candidates:
        fm = FeatureMap(dim=d, n_rff=n_rff, gamma=gamma, seed=seed)
        # More features than observations is fine for a Bayesian model, but
        # the evidence computation gets numerically fragile past ~8x.
        if fm.out_dim > max(8 * n, 64) and n_rff:
            continue
        phi = fm(X) * sqrt_w[:, None]
        mean, cov, alpha, beta, ev = _evidence_fit(phi, yc * sqrt_w)
        model = TasteModel(
            feature_map=fm, mean=mean, cov=cov, alpha=alpha, beta=beta,
            n_obs=n, y_mean=y_mean, log_evidence=ev,
        )
        if best is None or ev > best.log_evidence:
            best = model

    assert best is not None
    return best


def taste_direction(model: TasteModel) -> np.ndarray:
    """The learned preference direction in the *original* latent space.

    Only the linear part is returned: the RFF block encodes local curvature
    that has no single direction. This vector is what ``discover.py`` decodes
    into human-readable statements about what the person actually likes.
    """
    d = model.feature_map.dim
    return np.asarray(model.mean[1 : 1 + d], dtype=np.float32)
