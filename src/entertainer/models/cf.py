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

import os

import numpy as np
import polars as pl
import scipy.sparse as sp
from rich.console import Console

from ..config import CF_FACTORS, PATHS
from ..errors import MissingSourceData

console = Console()

# Two ways to read a MovieLens rating.
#
# The conventional implicit-feedback recipe keeps only ratings at or above
# 3.5 and treats them as ones. That reads the matrix as "who liked what".
#
# The alternative reads it as "who watched what", keeping every rating and
# grading the confidence by how much they liked it. It is the better choice
# here, for a reason specific to how these factors get used. They are not a
# ranker — they are a feature space, and the ridge map in ``fusion.py`` has
# to learn to predict them from text for the ~80% of the catalogue MovieLens
# never covered. Coverage of the *map's training set* therefore matters more
# than purity of the signal: liked-only yields factors for 30,810 titles,
# co-watch for 55,420. Eighty per cent more supervision for the map, and the
# preference information is not lost, only demoted from a hard filter to a
# confidence weight.
WATCH_CONFIDENCE = 0.3     # someone watched it and disliked it: still a co-watch
LIKE_CONFIDENCE = 1.7      # additional confidence at the top of the scale
POSITIVE_THRESHOLD = 3.5   # only used by signal="liked"
MIN_USER_RATINGS = 10
MIN_ITEM_RATINGS = 3


def _ratings_path():
    return PATHS.raw / "ml-32m" / "ratings.csv"


_RATINGS_SCHEMA = {"userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32}


def load_ratings() -> pl.DataFrame:
    """MovieLens ratings as (userId Int32, movieId Int32, rating Float32).

    Parsing the 836 MB CSV costs tens of seconds and `data cf`, `data prior`
    and the offline simulation each need it, so the first read leaves a
    sibling parquet behind. It is rebuilt whenever the CSV changes.
    """
    path = _ratings_path()
    cache = path.with_suffix(".parquet")
    stamp = cache.with_name(cache.name + ".source")
    # Size and mtime together, compared for equality rather than "newer":
    # re-extracting the zip restores the archive's old mtime, which a
    # newer-than check would take as proof the stale cache is still good.
    source = f"{path.stat().st_size}:{path.stat().st_mtime_ns}" if path.exists() else None
    if cache.exists() and stamp.exists() and (
        source is None or stamp.read_text().strip() == source
    ):
        return pl.read_parquet(cache).select("userId", "movieId", "rating").cast(_RATINGS_SCHEMA)
    if not path.exists():
        raise MissingSourceData(f"missing {path}; run `ent data fetch` first")
    df = pl.read_csv(path, schema_overrides=_RATINGS_SCHEMA).select("userId", "movieId", "rating")
    # Write-then-rename, stamp last: a crash mid-write must never leave a
    # truncated cache that a matching stamp would vouch for.
    tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
    try:
        stamp.unlink(missing_ok=True)
        df.write_parquet(tmp)
        os.replace(tmp, cache)
        stamp.write_text(source)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        console.print(f"[yellow]could not cache {cache.name}: {exc}[/yellow]")
    return df


def build_matrix(
    df: pl.DataFrame,
    signal: str = "watched",
    positive_threshold: float = POSITIVE_THRESHOLD,
    min_user_ratings: int = MIN_USER_RATINGS,
    min_item_ratings: int = MIN_ITEM_RATINGS,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return (user x item CSR of confidences, user ids, movielens item ids)."""
    if signal == "liked":
        df = df.filter(pl.col("rating") >= positive_threshold)

    ucount = df.group_by("userId").len().filter(pl.col("len") >= min_user_ratings)
    icount = df.group_by("movieId").len().filter(pl.col("len") >= min_item_ratings)
    df = df.join(ucount.select("userId"), on="userId").join(icount.select("movieId"), on="movieId")

    users = np.sort(df["userId"].unique().to_numpy())
    items = np.sort(df["movieId"].unique().to_numpy())
    # `users`/`items` are sorted and contain every id in the frame, so a
    # binary search is the exact positional index — no 32M-entry dict walk.
    rows = np.searchsorted(users, df["userId"].to_numpy()).astype(np.int32)
    cols = np.searchsorted(items, df["movieId"].to_numpy()).astype(np.int32)

    ratings = df["rating"].to_numpy().astype(np.float32)
    if signal == "liked":
        vals = 1.0 + 2.0 * (ratings - positive_threshold)
    else:
        # 0.5 stars -> WATCH_CONFIDENCE, 5 stars -> WATCH + LIKE.
        scaled = np.clip((ratings - 0.5) / 4.5, 0.0, 1.0)
        vals = WATCH_CONFIDENCE + LIKE_CONFIDENCE * scaled

    mat = sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items


def _use_gpu() -> bool:
    """CUDA ALS when implicit was built with it; ENTERTAINER_CF_GPU=0/1 overrides."""
    flag = os.environ.get("ENTERTAINER_CF_GPU", "").strip()
    if flag in ("0", "1"):
        return flag == "1"
    try:
        import implicit.gpu
    except ImportError:
        return False
    return bool(implicit.gpu.HAS_CUDA)


def fit(
    factors: int = CF_FACTORS,
    regularization: float = 0.05,
    iterations: int = 20,
    alpha: float = 12.0,
    seed: int = 0,
    holdout_users: np.ndarray | None = None,
    signal: str = "watched",
    ratings: pl.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit iALS. Returns (movielens_item_ids, item_factor_matrix).

    ``holdout_users`` removes those MovieLens users from the training matrix
    entirely. The offline simulation in ``evaluation/`` replays held-out users
    as cold-start strangers, and that measurement is worthless if their own
    ratings helped shape the item factors those strangers are scored against.
    """
    from implicit.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits

    df = load_ratings() if ratings is None else ratings
    console.print(f"[dim]MovieLens ratings: {df.height:,}[/dim]")
    if holdout_users is not None and len(holdout_users):
        before = df.height
        df = df.filter(~pl.col("userId").is_in(pl.Series(holdout_users.astype(np.int32)).implode()))
        console.print(f"[dim]held out {len(holdout_users):,} users "
                      f"({before - df.height:,} ratings) from CF training[/dim]")
    mat, users, items = build_matrix(df, signal=signal)
    console.print(f"[dim]implicit matrix: {mat.shape[0]:,} users x {mat.shape[1]:,} items, "
                  f"{mat.nnz:,} nonzeros[/dim]")

    use_gpu = _use_gpu()
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        calculate_training_loss=True,
        random_state=seed,
        use_gpu=use_gpu,
    )
    console.print(f"[dim]ALS on {'GPU' if use_gpu else 'CPU'}[/dim]")
    # implicit parallelises ALS itself; a BLAS threadpool inside each of its
    # workers oversubscribes every core. The env var is read once at BLAS
    # load (long before this point), so the limit has to be applied live.
    with threadpool_limits(1, "blas"):
        model.fit(mat)

    raw = model.item_factors
    # GPU models hold factors as implicit.gpu.Matrix, not ndarray.
    if hasattr(raw, "to_numpy"):
        raw = raw.to_numpy()
    item_factors = np.asarray(raw, dtype=np.float32)
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
