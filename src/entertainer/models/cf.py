"""Collaborative-filtering tower.

Text embeddings know what a film *is about*. They do not know that people who
love `Prisoners` tend to love `Memories of Murder`, or that a particular kind
of viewer bounces off `Birdman` despite every surface feature matching their
stated taste. That knowledge only exists in co-consumption data, and
MovieLens-32M is the largest openly licensed source of it.

Implicit ALS is used rather than a neural sequence model for a specific
reason: the signal available here is a static preference matrix with no
reliable timestamps of *consumption* (only of rating entry), so the extra
capacity of a transformer buys nothing while costing a great deal of compute
and reproducibility. iALS remains a genuinely strong baseline — repeated
benchmark work through the 2020s found it competitive with far heavier
neural rankers once properly tuned.

The output is used as a *feature space*, not as a ranker. See ``fusion.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from rich.console import Console

from ..config import CF_FACTORS, PATHS

console = Console()

# Ratings at or above this are treated as positive implicit feedback. 3.5 on
# MovieLens' 0.5..5 scale is the conventional cut: it keeps "liked" distinct
# from "watched and shrugged".
POSITIVE_THRESHOLD = 3.5
MIN_USER_RATINGS = 10
MIN_ITEM_RATINGS = 5


def _ratings_path():
    return PATHS.raw / "ml-32m" / "ratings.csv"


def load_ratings() -> pl.DataFrame:
    path = _ratings_path()
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `entertainer fetch` first")
    return pl.read_csv(
        path,
        schema_overrides={"userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32},
    ).select("userId", "movieId", "rating")


def build_matrix(
    df: pl.DataFrame,
    positive_threshold: float = POSITIVE_THRESHOLD,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return (user x item CSR of confidences, user ids, movielens item ids)."""
    df = df.filter(pl.col("rating") >= positive_threshold)

    ucount = df.group_by("userId").len().filter(pl.col("len") >= MIN_USER_RATINGS)
    icount = df.group_by("movieId").len().filter(pl.col("len") >= MIN_ITEM_RATINGS)
    df = df.join(ucount.select("userId"), on="userId").join(icount.select("movieId"), on="movieId")

    users = np.sort(df["userId"].unique().to_numpy())
    items = np.sort(df["movieId"].unique().to_numpy())
    uidx = {u: i for i, u in enumerate(users.tolist())}
    iidx = {m: i for i, m in enumerate(items.tolist())}

    rows = np.fromiter((uidx[u] for u in df["userId"].to_list()), dtype=np.int32, count=df.height)
    cols = np.fromiter((iidx[m] for m in df["movieId"].to_list()), dtype=np.int32, count=df.height)

    # Confidence grows with how far above the "liked" line the rating sits.
    vals = 1.0 + 2.0 * (df["rating"].to_numpy() - positive_threshold)
    mat = sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items


def fit(
    factors: int = CF_FACTORS,
    regularization: float = 0.05,
    iterations: int = 20,
    alpha: float = 12.0,
    seed: int = 0,
    holdout_users: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit iALS. Returns (movielens_item_ids, item_factor_matrix).

    ``holdout_users`` removes those MovieLens users from the training matrix
    entirely. The offline simulation in ``evaluation/`` replays held-out users
    as cold-start strangers, and that measurement is worthless if their own
    ratings helped shape the item factors those strangers are scored against.
    """
    from implicit.als import AlternatingLeastSquares

    df = load_ratings()
    console.print(f"[dim]MovieLens ratings: {df.height:,}[/dim]")
    if holdout_users is not None and len(holdout_users):
        before = df.height
        df = df.filter(~pl.col("userId").is_in(pl.Series(holdout_users.astype(np.int32))))
        console.print(f"[dim]held out {len(holdout_users):,} users "
                      f"({before - df.height:,} ratings) from CF training[/dim]")
    mat, users, items = build_matrix(df)
    console.print(f"[dim]implicit matrix: {mat.shape[0]:,} users x {mat.shape[1]:,} items, "
                  f"{mat.nnz:,} nonzeros[/dim]")

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        calculate_training_loss=True,
        random_state=seed,
        use_gpu=False,
    )
    model.fit(mat)

    item_factors = np.asarray(model.item_factors, dtype=np.float32)
    return items.astype(np.int32), item_factors


def save(item_ids: np.ndarray, factors: np.ndarray) -> None:
    PATHS.ensure()
    np.save(PATHS.embeddings / "cf_movielens_ids.npy", item_ids)
    np.save(PATHS.embeddings / "cf_factors.npy", factors)


def load() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.load(PATHS.embeddings / "cf_movielens_ids.npy"),
        np.load(PATHS.embeddings / "cf_factors.npy"),
    )


def exists() -> bool:
    return (PATHS.embeddings / "cf_factors.npy").exists()
