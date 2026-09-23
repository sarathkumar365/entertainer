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


# Upper bound on the padded (users x ratings x d) block built per solve, in
# float64 elements (~256 MB). Users are length-sorted before chunking, so
# padding waste stays small even though MovieLens counts span 25..30k.
_CHUNK_ELEMENTS = 32_000_000


def _taste_vectors(df: pl.DataFrame, fs, item_of_movielens: dict[int, int]) -> np.ndarray:
    """Unit taste direction per qualifying user, in ascending userId order.

    Batched form of ``_user_weight``: users are grouped into chunks, their
    zero-padded design matrices stacked, and one ``np.linalg.solve`` call
    handles the whole chunk. Padding rows are zero in both X and the centred
    target, so they add nothing to either side of the normal equations.
    """
    pairs = [(int(m), fs.index[i]) for m, i in item_of_movielens.items() if i in fs.index]
    lookup = pl.DataFrame(
        {"movieId": [m for m, _ in pairs], "_row": [r for _, r in pairs]},
        schema={"movieId": pl.Int32, "_row": pl.Int64},
    )
    df = df.select("userId", "movieId", "rating").cast({"movieId": pl.Int32})
    df = df.join(lookup, on="movieId", how="inner")

    stats = (
        df.group_by("userId")
        .agg(
            n=pl.len(),
            liked=(pl.col("rating") >= LIKED_AT).sum(),
            disliked=(pl.col("rating") <= DISLIKED_AT).sum(),
        )
        .filter(
            (pl.col("n") >= MIN_RATINGS)
            & (pl.col("liked") >= MIN_EACH_SIDE)
            & (pl.col("disliked") >= MIN_EACH_SIDE)
        )
        .sort("n", "userId")
    )
    d = fs.matrix.shape[1]
    if stats.height == 0:
        return np.empty((0, d))

    df = df.join(stats.select("userId", "n"), on="userId", how="inner").sort(
        "n", "userId", maintain_order=True
    )
    rows = df["_row"].to_numpy()
    y_all = (df["rating"].to_numpy().astype(np.float64) - 0.5) / 4.5
    uid_all = stats["userId"].to_numpy()
    counts = stats["n"].to_numpy().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    user_of_row = np.repeat(np.arange(len(counts)), counts)
    pos_in_user = np.arange(len(rows)) - starts[user_of_row]
    # Centre per user, exactly as `_user_weight` does with y - y.mean().
    means = np.add.reduceat(y_all, starts[:-1]) / counts
    yc_all = y_all - means[user_of_row]

    matrix = fs.matrix
    ridge = np.eye(d)
    out = np.empty((len(counts), d))
    u = 0
    while u < len(counts):
        # Lengths ascend, so the last user in the chunk sets the padding.
        v = u + 1
        while v < len(counts) and (v + 1 - u) * counts[v] * d <= _CHUNK_ELEMENTS:
            v += 1
        B, L = v - u, int(counts[v - 1])
        lo, hi = starts[u], starts[v]
        local_user = user_of_row[lo:hi] - u
        local_pos = pos_in_user[lo:hi]
        X = np.zeros((B, L, d))
        X[local_user, local_pos] = matrix[rows[lo:hi]]
        yc = np.zeros((B, L))
        yc[local_user, local_pos] = yc_all[lo:hi]
        Xt = X.transpose(0, 2, 1)
        gram = Xt @ X + ridge
        rhs = (Xt @ yc[:, :, None])[:, :, 0]
        out[u:v] = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
        u = v

    norms = np.linalg.norm(out, axis=1)
    # Direction, not magnitude: how strongly someone rates is a property of
    # how they use a scale, not of what they like.
    keep = norms > 1e-9
    W = out[keep] / norms[keep, None]
    order = np.argsort(uid_all[keep], kind="stable")
    return W[order]


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
    # group_by order is arbitrary; sort so the seeded sample is reproducible.
    candidates = np.sort(candidates)
    if candidates.size > max_users:
        candidates = rng.choice(candidates, size=max_users, replace=False)
    df = df.filter(pl.col("userId").is_in(pl.Series(candidates.astype(np.int32)).implode()))

    console.print(f"[dim]fitting taste vectors for {len(candidates):,} MovieLens users[/dim]")

    W = _taste_vectors(df, fs, item_of_movielens)
    if len(W) < 50:
        raise RuntimeError(f"only {len(W)} usable taste vectors; need at least 50")

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
