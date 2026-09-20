"""A prior over what human taste actually looks like.

The preference model starts from a prior over the weight vector. Left
isotropic, that prior asserts something obviously false: that on day one,
before any evidence, every direction in taste space is equally plausible.
It is not. Real preference vectors live on a thin, structured manifold —
people who like unhurried character studies tend to dislike the same things
as each other, and nobody's taste is a random direction in 192 dimensions.

MovieLens contains two hundred thousand examples of what a real preference
vector looks like. Fitting one weight vector per user in the same fused space
the engine uses, then taking the mean and covariance of that population,
yields a prior that already knows the shape of human taste before its own
user has answered a single question. This is empirical Bayes at the
population level — the same "learn the prior from related tasks" idea that
underlies meta-learning, in the one setting where it is fully closed form.

The payoff is concentrated exactly where it is needed: at small n. With forty
answers the likelihood dominates and the prior barely matters. With five, the
prior is most of the posterior, and the difference between "any direction is
equally likely" and "directions look like this" is the difference between a
useful first slate and a random one.

Leakage matters here more than anywhere. Users reserved for offline
evaluation are excluded from the population fit, or the replay would be
scoring strangers against a prior built partly from their own opinions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from rich.console import Console

from ..config import PATHS

console = Console()

# A user needs enough ratings on both sides for their weight vector to mean
# anything; below this it is mostly noise and would only blur the population.
MIN_RATINGS = 25
MIN_EACH_SIDE = 5
LIKED_AT = 4.0
DISLIKED_AT = 2.5


@dataclass
class PopulationPrior:
    mean: np.ndarray        # (d,) average taste direction across the population
    cov: np.ndarray         # (d, d) covariance of taste directions
    n_users: int
    dim: int

    def cholesky(self) -> np.ndarray:
        """Lower-triangular factor, for whitening the design matrix."""
        d = self.cov.shape[0]
        jitter = 1e-8 * np.trace(self.cov) / max(d, 1)
        return np.linalg.cholesky(self.cov + jitter * np.eye(d))

    def save(self, path=None) -> None:
        PATHS.ensure()
        path = path or (PATHS.artifacts / "population_prior.npz")
        np.savez_compressed(
            path, mean=self.mean, cov=self.cov,
            n_users=np.array([self.n_users]), dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path=None) -> PopulationPrior:
        path = path or (PATHS.artifacts / "population_prior.npz")
        z = np.load(path)
        return cls(
            mean=z["mean"], cov=z["cov"],
            n_users=int(z["n_users"][0]), dim=int(z["dim"][0]),
        )

    @staticmethod
    def exists(path=None) -> bool:
        return (path or (PATHS.artifacts / "population_prior.npz")).exists()


def _user_weight(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """One user's taste direction: ridge regression of their ratings onto features.

    Ridge rather than a difference-of-centroids because the catalogue is not
    balanced — a user who has rated forty English films and three Korean ones
    would otherwise have their Korean opinions swamped by the axis that
    separates the two, rather than by what they actually thought.
    """
    d = X.shape[1]
    gram = X.T @ X + ridge * np.eye(d)
    return np.linalg.solve(gram, X.T @ (y - y.mean()))


def fit(
    fs,
    item_of_movielens: dict[int, int],
    holdout_users: np.ndarray | None = None,
    max_users: int = 20_000,
    seed: int = 0,
    shrinkage: float = 0.15,
) -> PopulationPrior:
    """Estimate the population prior from MovieLens users.

    ``shrinkage`` pulls the covariance towards a scaled identity. With a few
    hundred dimensions and tens of thousands of users the sample covariance is
    estimable, but its smallest eigenvalues are still badly conditioned, and a
    prior that is near-singular in some direction forbids that direction
    outright — which is a much stronger claim than the data supports.
    """
    from . import cf as cf_mod

    df = cf_mod.load_ratings()
    if holdout_users is not None and len(holdout_users):
        held = pl.Series(np.asarray(holdout_users, dtype=np.int32)).implode()
        df = df.filter(~pl.col("userId").is_in(held))

    known = pl.Series("movieId", list(item_of_movielens.keys()), dtype=pl.Int32)
    df = df.filter(pl.col("movieId").is_in(known.implode()))

    counts = df.group_by("userId").len().filter(pl.col("len") >= MIN_RATINGS)
    candidates = counts["userId"].to_numpy()
    if candidates.size == 0:
        raise RuntimeError("no MovieLens users with enough ratings to fit a population prior")

    rng = np.random.default_rng(seed)
    if candidates.size > max_users:
        candidates = rng.choice(candidates, size=max_users, replace=False)
    df = df.filter(pl.col("userId").is_in(pl.Series(candidates.astype(np.int32)).implode()))

    console.print(f"[dim]fitting taste vectors for {len(candidates):,} MovieLens users[/dim]")

    vectors: list[np.ndarray] = []
    for (uid,), sub in df.group_by(["userId"]):
        del uid
        items = sub["movieId"].to_numpy()
        ratings = sub["rating"].to_numpy().astype(np.float64)
        rows, keep = [], []
        for pos, movie in enumerate(items.tolist()):
            item = item_of_movielens.get(int(movie))
            if item is not None and item in fs.index:
                rows.append(fs.index[item])
                keep.append(pos)
        if len(rows) < MIN_RATINGS:
            continue
        y = ratings[keep]
        if (y >= LIKED_AT).sum() < MIN_EACH_SIDE or (y <= DISLIKED_AT).sum() < MIN_EACH_SIDE:
            continue
        X = fs.matrix[np.array(rows)]
        w = _user_weight(X, (y - 0.5) / 4.5)
        norm = np.linalg.norm(w)
        if norm > 1e-9:
            # Direction, not magnitude: how strongly someone rates is a
            # property of how they use a scale, not of what they like.
            vectors.append(w / norm)

    if len(vectors) < 50:
        raise RuntimeError(f"only {len(vectors)} usable taste vectors; need at least 50")

    W = np.vstack(vectors)
    mean = W.mean(axis=0)
    centred = W - mean
    cov = (centred.T @ centred) / max(len(W) - 1, 1)

    target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    cov = (1.0 - shrinkage) * cov + shrinkage * target

    console.print(
        f"[dim]population prior from {len(W):,} users, "
        f"effective rank {np.trace(cov) ** 2 / np.trace(cov @ cov):.1f} of {cov.shape[0]}[/dim]"
    )
    return PopulationPrior(mean=mean, cov=cov, n_users=len(W), dim=cov.shape[0])
