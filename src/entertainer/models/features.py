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
    #: (mean, sd) per side column as ``build`` measured it on the catalogue,
    #: so a title outside the space can be placed on the same scale.
    side_stats: tuple[tuple[float, float], ...] | None = None
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

    def side_for(self, row: dict, reference_year: int = 2026) -> np.ndarray:
        """Side features for a title that is not in the space.

        Scaled by the catalogue's own statistics, so the vector is what
        ``build`` would have given it. A value the row lacks sits at the
        mean, which is to say it contributes nothing.
        """
        if self.side_stats is None:
            raise ValueError("this feature space was built without side statistics")
        raw = _raw_side(row, reference_year)
        out = [
            _scale(np.array([value]), mu, sd)[0]
            for value, (mu, sd) in zip(raw[:-1], self.side_stats[:-1], strict=True)
        ]
        tv_mean = self.side_stats[-1][0]
        out.append((raw[-1] - tv_mean) * 0.15)
        return np.array(out, dtype=np.float32)


def _raw_side(r: dict, reference_year: int) -> tuple[float, ...]:
    """One title's side features before scaling, in SIDE_FEATURE_NAMES order.

    Recency is negated here, so a larger value is always more recent.
    """
    nan = float("nan")
    return (
        float(r["quality"]) if r.get("quality") is not None else nan,
        math.log1p(r["imdb_votes"]) if r.get("imdb_votes") else nan,
        -(reference_year - int(r["year"])) if r.get("year") else nan,
        min(int(r["runtime"]), 300) if r.get("runtime") else nan,
        1.0 if r.get("kind") == "tv" else 0.0,
    )


def _stats(col: np.ndarray) -> tuple[float, float]:
    observed = col[~np.isnan(col)]
    if observed.size == 0:
        return (float("nan"), 1.0)
    return (float(observed.mean()), float(observed.std()))


def _scale(col: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Centre and scale, tolerating a column that is entirely absent.

    A feature nobody in the catalogue has — runtime on a catalogue of series,
    say, or quality before the first build — must contribute nothing rather
    than NaN. A single NaN here propagates through the fused matrix into every
    prediction, and does so silently.
    """
    if math.isnan(mu):
        return np.zeros_like(col, dtype=np.float64)
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
    raw = np.full((n, len(SIDE_FEATURE_NAMES)), np.nan, dtype=np.float64)
    raw[:, -1] = 0.0
    for i, iid in enumerate(item_ids.tolist()):
        r = meta.get(int(iid))
        if r:
            raw[i] = _raw_side(r, reference_year)
    quality = raw[:, 0]
    is_tv = raw[:, -1]

    stats = [_stats(raw[:, j]) for j in range(raw.shape[1] - 1)]
    side = np.column_stack(
        [_scale(raw[:, j], *stats[j]) for j in range(len(stats))]
        + [(is_tv - is_tv.mean()) * 0.15]
    ).astype(np.float32)
    side_stats = (*stats, (float(is_tv.mean()), 1.0))

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
        side_stats=side_stats,
    )
