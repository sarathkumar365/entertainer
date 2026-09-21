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
class Columns:
    """Catalogue metadata as parallel arrays, for vectorised filtering.

    Filtering and pool selection both used to walk a 270,000-entry dict in
    Python on every command, which is about a second of latency on something
    that should feel instant. The same facts as columns cost microseconds.

    Languages are held as integer codes against a vocabulary rather than as
    strings, so a language filter is an integer comparison rather than 270,000
    string comparisons.
    """

    language_code: np.ndarray    # (N,) int32 index into `languages`
    languages: list[str]
    is_tv: np.ndarray            # (N,) bool
    year: np.ndarray             # (N,) int32, 0 where unknown
    runtime: np.ndarray          # (N,) int32, 0 where unknown
    votes: np.ndarray            # (N,) int64, 0 where unknown
    quality: np.ndarray          # (N,) float32
    known: np.ndarray            # (N,) bool — present in the catalogue at all

    def codes_for(self, wanted) -> np.ndarray:
        """Language codes for a set of language strings; -1 entries dropped."""
        lookup = {name: i for i, name in enumerate(self.languages)}
        return np.array([lookup[w] for w in wanted if w in lookup], dtype=np.int32)


@dataclass
class FeatureSpace:
    item_ids: np.ndarray     # (N,)
    latent: np.ndarray       # (N, D) fused PCA space, unit rows
    side: np.ndarray         # (N, E) standardised side features
    index: dict[int, int]    # item_id -> row
    columns: Columns | None = None
    _matrix: np.ndarray | None = None

    @property
    def n_latent(self) -> int:
        return self.latent.shape[1]

    @property
    def matrix(self) -> np.ndarray:
        """Latent and side features concatenated, built once and reused.

        At catalogue scale this array is a couple of hundred megabytes, and
        scoring, elicitation and evaluation all touch it repeatedly within a
        single command. Rebuilding it per access was the difference between a
        recommendation taking half a second and taking ten.
        """
        if self._matrix is None:
            object.__setattr__(
                self, "_matrix", np.hstack([self.latent, self.side]).astype(np.float32)
            )
        return self._matrix

    def rows_for(self, item_ids) -> np.ndarray:
        return np.array([self.index[int(i)] for i in item_ids], dtype=np.int64)

    def vectors_for(self, item_ids) -> np.ndarray:
        return self.matrix[self.rows_for(item_ids)]


def _standardise(col: np.ndarray) -> np.ndarray:
    """Centre and scale, tolerating a column that is entirely absent.

    A feature nobody in the catalogue has — runtime on a catalogue of series,
    say, or quality before the first build — must contribute nothing rather
    than NaN. A single NaN here propagates through the fused matrix into every
    prediction, and does so silently.
    """
    observed = col[~np.isnan(col)]
    if observed.size == 0:
        return np.zeros_like(col, dtype=np.float64)
    mu, sd = float(observed.mean()), float(observed.std())
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

    languages: list[str] = []
    lang_lookup: dict[str, int] = {}
    language_code = np.full(n, -1, dtype=np.int32)
    known = np.zeros(n, dtype=bool)
    year_col = np.zeros(n, dtype=np.int32)
    runtime_col = np.zeros(n, dtype=np.int32)
    votes_col = np.zeros(n, dtype=np.int64)

    for i, iid in enumerate(item_ids.tolist()):
        r = meta.get(int(iid))
        if not r:
            continue
        known[i] = True
        lang = r.get("language") or "xx"
        code = lang_lookup.get(lang)
        if code is None:
            code = len(languages)
            lang_lookup[lang] = code
            languages.append(lang)
        language_code[i] = code
        if r.get("year"):
            year_col[i] = int(r["year"])
        if r.get("runtime"):
            runtime_col[i] = int(r["runtime"])
        if r.get("imdb_votes"):
            votes_col[i] = int(r["imdb_votes"])

    columns = Columns(
        language_code=language_code,
        languages=languages,
        is_tv=is_tv.astype(bool),
        year=year_col,
        runtime=runtime_col,
        votes=votes_col,
        quality=np.nan_to_num(quality, nan=0.5).astype(np.float32),
        known=known,
    )

    return FeatureSpace(
        item_ids=item_ids.astype(np.int32),
        latent=latent.astype(np.float32),
        side=side,
        index={int(v): i for i, v in enumerate(item_ids.tolist())},
        columns=columns,
    )
