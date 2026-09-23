"""Vectorised build steps must reproduce the loops they replaced.

Each test keeps a verbatim copy of the old implementation as the reference,
so a later edit to the fast path cannot quietly drift from what the build
used to produce.
"""

from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from entertainer.data.catalog import quality_prior, quality_prior_expr
from entertainer.models import cf, fusion, population


def _ratings(seed=0, n=4000, users=120, items=90):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "userId": rng.integers(1, users, n).astype(np.int32) * 7,
            "movieId": rng.integers(1, items, n).astype(np.int32) * 13,
            "rating": (rng.integers(1, 11, n) / 2).astype(np.float32),
        }
    ).unique(["userId", "movieId"], keep="first", maintain_order=True)


# --- ratings cache -----------------------------------------------------------


def test_load_ratings_caches_parquet_and_refreshes_when_csv_is_newer(tmp_path, monkeypatch):
    csv = tmp_path / "ratings.csv"
    csv.write_text("userId,movieId,rating,timestamp\n1,10,4.5,0\n2,10,1.0,0\n")
    monkeypatch.setattr(cf, "_ratings_path", lambda: csv)

    first = cf.load_ratings()
    cache = tmp_path / "ratings.parquet"
    assert cache.exists()
    assert first.schema == pl.Schema(
        {"userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32}
    )
    assert not list(tmp_path.glob(".*.tmp"))

    again = cf.load_ratings()
    assert again.equals(first)
    assert again.schema == first.schema

    csv.write_text("userId,movieId,rating,timestamp\n3,11,2.0,0\n")
    later = time.time() + 10
    os.utime(csv, (later, later))
    refreshed = cf.load_ratings()
    assert refreshed["userId"].to_list() == [3]


def test_load_ratings_refreshes_when_csv_is_restored_with_an_older_mtime(tmp_path, monkeypatch):
    csv = tmp_path / "ratings.csv"
    csv.write_text("userId,movieId,rating,timestamp\n1,10,4.5,0\n")
    monkeypatch.setattr(cf, "_ratings_path", lambda: csv)
    cf.load_ratings()

    # Re-extracting a zip restores the archive's timestamp, older than the cache.
    csv.write_text("userId,movieId,rating,timestamp\n3,11,2.0,0\n4,11,3.0,0\n")
    os.utime(csv, (1_000_000, 1_000_000))
    assert cf.load_ratings()["userId"].to_list() == [3, 4]


def test_load_ratings_missing_everything_raises(tmp_path, monkeypatch):
    """A dump that was never fetched is a state of the build, not an IO fault."""
    from entertainer.errors import MissingSourceData

    monkeypatch.setattr(cf, "_ratings_path", lambda: tmp_path / "ratings.csv")
    with pytest.raises(MissingSourceData):
        cf.load_ratings()


# --- CF matrix -----------------------------------------------------------------


def _old_build_matrix(df, signal="watched", positive_threshold=cf.POSITIVE_THRESHOLD,
                      min_user_ratings=cf.MIN_USER_RATINGS,
                      min_item_ratings=cf.MIN_ITEM_RATINGS):
    if signal == "liked":
        df = df.filter(pl.col("rating") >= positive_threshold)
    ucount = df.group_by("userId").len().filter(pl.col("len") >= min_user_ratings)
    icount = df.group_by("movieId").len().filter(pl.col("len") >= min_item_ratings)
    df = df.join(ucount.select("userId"), on="userId").join(icount.select("movieId"), on="movieId")
    users = np.sort(df["userId"].unique().to_numpy())
    items = np.sort(df["movieId"].unique().to_numpy())
    uidx = {u: i for i, u in enumerate(users.tolist())}
    iidx = {m: i for i, m in enumerate(items.tolist())}
    rows = np.fromiter((uidx[u] for u in df["userId"].to_list()), dtype=np.int32, count=df.height)
    cols = np.fromiter((iidx[m] for m in df["movieId"].to_list()), dtype=np.int32, count=df.height)
    ratings = df["rating"].to_numpy().astype(np.float32)
    if signal == "liked":
        vals = 1.0 + 2.0 * (ratings - positive_threshold)
    else:
        scaled = np.clip((ratings - 0.5) / 4.5, 0.0, 1.0)
        vals = cf.WATCH_CONFIDENCE + cf.LIKE_CONFIDENCE * scaled
    mat = sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items


@pytest.mark.parametrize("signal", ["watched", "liked"])
def test_build_matrix_matches_dict_version(signal):
    df = _ratings()
    new_mat, new_u, new_i = cf.build_matrix(df, signal=signal)
    old_mat, old_u, old_i = _old_build_matrix(df, signal=signal)
    assert np.array_equal(new_u, old_u)
    assert np.array_equal(new_i, old_i)
    assert new_mat.shape == old_mat.shape
    assert new_mat.dtype == old_mat.dtype
    assert (new_mat != old_mat).nnz == 0
    assert np.array_equal(new_mat.indptr, old_mat.indptr)
    assert np.array_equal(new_mat.indices, old_mat.indices)


@pytest.mark.parametrize(("flag", "expected"), [("0", False), ("1", True)])
def test_cf_gpu_env_override(monkeypatch, flag, expected):
    monkeypatch.setenv("ENTERTAINER_CF_GPU", flag)
    assert cf._use_gpu() is expected


# --- population prior ---------------------------------------------------------


def _old_taste_vectors(df, fs, item_of_movielens):
    vectors = {}
    for (uid,), sub in df.group_by(["userId"]):
        items = sub["movieId"].to_numpy()
        ratings = sub["rating"].to_numpy().astype(np.float64)
        rows, keep = [], []
        for pos, movie in enumerate(items.tolist()):
            item = item_of_movielens.get(int(movie))
            if item is not None and item in fs.index:
                rows.append(fs.index[item])
                keep.append(pos)
        if len(rows) < population.MIN_RATINGS:
            continue
        y = ratings[keep]
        if ((y >= population.LIKED_AT).sum() < population.MIN_EACH_SIDE
                or (y <= population.DISLIKED_AT).sum() < population.MIN_EACH_SIDE):
            continue
        X = fs.matrix[np.array(rows)]
        w = population._user_weight(X, (y - 0.5) / 4.5)
        norm = np.linalg.norm(w)
        if norm > 1e-9:
            vectors[uid] = w / norm
    return np.vstack([vectors[u] for u in sorted(vectors)])


def _population_world(dtype, seed=1):
    rng = np.random.default_rng(seed)
    n_items, d = 300, 12
    matrix = rng.normal(size=(n_items, d)).astype(dtype)
    # Catalogue has 300 rows; MovieLens knows 260 titles, 240 of them in fs.
    fs = SimpleNamespace(matrix=matrix, index={i + 1000: i for i in range(240)})
    item_of_ml = {m: 1000 + m for m in range(260)}
    lengths = rng.integers(10, 140, size=90)
    frames = []
    for u, n in enumerate(lengths):
        movies = rng.choice(260, size=n, replace=False).astype(np.int32)
        frames.append(pl.DataFrame({
            "userId": np.full(n, u * 3 + 1, dtype=np.int32),
            "movieId": movies,
            "rating": (rng.integers(1, 11, n) / 2).astype(np.float32),
        }))
    df = pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=seed)
    return df, fs, item_of_ml


def test_batched_taste_vectors_match_per_user_loop(monkeypatch):
    df, fs, item_of_ml = _population_world(np.float64)
    # Force several chunks so chunk boundaries are exercised.
    monkeypatch.setattr(population, "_CHUNK_ELEMENTS", 20_000)
    new = population._taste_vectors(df, fs, item_of_ml)
    old = _old_taste_vectors(df, fs, item_of_ml)
    assert new.shape == old.shape
    assert len(new) > 20
    assert np.allclose(new, old, atol=1e-8, rtol=0)


def test_batched_taste_vectors_close_on_float32_features():
    """Production features are float32; the batched path solves in float64."""
    df, fs, item_of_ml = _population_world(np.float32)
    new = population._taste_vectors(df, fs, item_of_ml)
    old = _old_taste_vectors(df, fs, item_of_ml)
    assert np.allclose(new, old, atol=1e-4, rtol=0)


# --- fusion ridge map ---------------------------------------------------------


def _old_fit_cf_map(content_overlap, cf_overlap, seed=0):
    best_alpha, best_r2 = fusion._RIDGE_ALPHAS[0], -np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for alpha in fusion._RIDGE_ALPHAS:
        scores = []
        for tr, te in kf.split(content_overlap):
            m = Ridge(alpha=alpha).fit(content_overlap[tr], cf_overlap[tr])
            pred = m.predict(content_overlap[te])
            ss_res = float(((cf_overlap[te] - pred) ** 2).sum())
            ss_tot = float(((cf_overlap[te] - cf_overlap[tr].mean(0)) ** 2).sum())
            scores.append(1.0 - ss_res / max(ss_tot, 1e-9))
        mean_r2 = float(np.mean(scores))
        if mean_r2 > best_r2:
            best_alpha, best_r2 = alpha, mean_r2
    return Ridge(alpha=best_alpha).fit(content_overlap, cf_overlap), best_r2


@pytest.mark.parametrize("noise", [0.3, 3.0])
def test_cf_map_matches_per_alpha_ridge_folds(noise):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 40))
    B = rng.normal(size=(40, 6)) * 0.3
    Y = X @ B + noise * rng.normal(size=(400, 6)) + 2.0
    new_model, new_r2 = fusion._fit_cf_map(X, Y, seed=3)
    old_model, old_r2 = _old_fit_cf_map(X, Y, seed=3)
    assert new_model.alpha == old_model.alpha
    assert new_r2 == pytest.approx(old_r2, abs=1e-9)
    assert np.allclose(new_model.coef_, old_model.coef_)


def test_fusion_alignment_handles_missing_links():
    rng = np.random.default_rng(0)
    ids = np.arange(30, dtype=np.int32)
    content = rng.normal(size=(30, 8)).astype(np.float32)
    cf_ids = np.array([500 + i for i in range(0, 30, 2)], dtype=np.int32)
    cf_factors = rng.normal(size=(len(cf_ids), 5)).astype(np.float32)
    links = {int(i): 500 + int(i) for i in ids if i % 3 != 0}
    art = fusion.build(ids, content, cf_ids, cf_factors, links, dim=6, cf_rank=3)
    expected = np.mean([(i % 3 != 0) and (i % 2 == 0) for i in range(30)])
    assert art.cf_coverage == pytest.approx(expected)


# --- manifests ------------------------------------------------------------------


def test_manifest_hashes_the_fused_file_fusion_writes(tmp_path, monkeypatch):
    from entertainer import manifests

    monkeypatch.setattr(fusion, "fused_path", lambda: tmp_path / "fused.npz")
    (tmp_path / "fused.npz").write_bytes(b"space")
    monkeypatch.setattr(manifests, "PATHS", SimpleNamespace(
        ensure=lambda: None, reports=tmp_path / "reports",
        catalog_db=tmp_path / "none.duckdb", artifacts=tmp_path,
    ))
    record = manifests.write("test", {})
    assert record["artifacts"]["fused"] is not None


# --- catalogue quality ----------------------------------------------------------


def test_quality_expression_matches_scalar_prior():
    rng = np.random.default_rng(7)
    n = 3000
    ratings = rng.uniform(1.0, 10.0, n)
    votes = rng.integers(-5, 3_000_000, n)
    votes[:200] = 0
    votes[200:300] = rng.integers(1, 5, 100)
    rating_col = [None if i % 17 == 0 else float(r) for i, r in enumerate(ratings)]
    vote_col = [None if i % 23 == 0 else int(v) for i, v in enumerate(votes)]
    df = pl.DataFrame(
        {"imdb_rating": rating_col, "imdb_votes": vote_col},
        schema={"imdb_rating": pl.Float64, "imdb_votes": pl.Int64},
    )
    got = df.select(q=quality_prior_expr("imdb_rating", "imdb_votes"))["q"].to_list()
    want = [quality_prior(r, v) for r, v in zip(rating_col, vote_col, strict=True)]
    assert all(math.isclose(g, w, rel_tol=0, abs_tol=1e-12) for g, w in zip(got, want, strict=True))
    assert df.select(q=quality_prior_expr("imdb_rating", "imdb_votes"))["q"].dtype == pl.Float64
