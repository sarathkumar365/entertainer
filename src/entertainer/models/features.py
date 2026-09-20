"""Assemble the feature vector the preference model actually sees.

The fused latent space carries *what a title is like*. A handful of side
features carry *what kind of object it is* — how long, how old, how widely
liked, film or series. These are kept separate from the latent block and
appended rather than folded in, for two reasons.

They are interpretable on their own, so a person can be told "you lean towards
longer, older films" without any axis-naming machinery. And critically, the
consensus-quality feature is *learned* rather than imposed. Most recommenders
hard-blend a popularity or quality prior into the final score with a tuned
coefficient, which silently decides on the user's behalf how much they should
care what everyone else thinks. Here it is simply another input, and the
posterior works out for itself whether this particular person tracks the
critical consensus, ignores it, or runs against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SIDE_FEATURE_NAMES = (
    "consensus quality",
    "how widely seen",
    "release recency",
    "runtime",
    "is a series",
)


@dataclass
class FeatureSpace:
    item_ids: np.ndarray     # (N,)
    latent: np.ndarray       # (N, D) fused PCA space, unit rows
    side: np.ndarray         # (N, E) standardised side features
    index: dict[int, int]    # item_id -> row

    @property
    def n_latent(self) -> int:
        return self.latent.shape[1]

    @property
    def matrix(self) -> np.ndarray:
        return np.hstack([self.latent, self.side]).astype(np.float32)

    def rows_for(self, item_ids) -> np.ndarray:
        return np.array([self.index[int(i)] for i in item_ids], dtype=np.int64)

    def vectors_for(self, item_ids) -> np.ndarray:
        return self.matrix[self.rows_for(item_ids)]


def _standardise(col: np.ndarray) -> np.ndarray:
    mu, sd = float(np.nanmean(col)), float(np.nanstd(col))
    out = (np.nan_to_num(col, nan=mu) - mu) / (sd if sd > 1e-9 else 1.0)
    # Side features enter on the same scale as a single latent axis, so that
    # the prior does not implicitly favour them.
    return np.clip(out, -4.0, 4.0) * 0.15


def build(
    item_ids: np.ndarray,
    latent: np.ndarray,
    meta: dict[int, dict],
    reference_year: int = 2026,
) -> FeatureSpace:
    n = len(item_ids)
    quality = np.full(n, np.nan, dtype=np.float64)
    votes = np.full(n, np.nan, dtype=np.float64)
    year = np.full(n, np.nan, dtype=np.float64)
    runtime = np.full(n, np.nan, dtype=np.float64)
    is_tv = np.zeros(n, dtype=np.float64)

    for i, iid in enumerate(item_ids.tolist()):
        r = meta.get(int(iid))
        if not r:
            continue
        if r.get("quality") is not None:
            quality[i] = r["quality"]
        if r.get("imdb_votes"):
            votes[i] = math.log1p(r["imdb_votes"])
        if r.get("year"):
            year[i] = reference_year - int(r["year"])
        if r.get("runtime"):
            runtime[i] = min(int(r["runtime"]), 300)
        is_tv[i] = 1.0 if r.get("kind") == "tv" else 0.0

    side = np.column_stack(
        [
            _standardise(quality),
            _standardise(votes),
            _standardise(-year),  # positive = more recent
            _standardise(runtime),
            (is_tv - is_tv.mean()) * 0.15,
        ]
    ).astype(np.float32)

    return FeatureSpace(
        item_ids=item_ids.astype(np.int32),
        latent=latent.astype(np.float32),
        side=side,
        index={int(v): i for i, v in enumerate(item_ids.tolist())},
    )
