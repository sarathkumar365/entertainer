"""Orchestration: hold the pieces together and keep them consistent.

The engine is deliberately stateless between invocations. Every command
reloads the catalogue, replays the event log, refits the posterior and throws
it all away again. Refitting costs milliseconds because the posterior is
closed form, and the alternative — an incrementally-updated model file that
can silently drift out of sync with the log — is the kind of bug that takes
a month to notice and invalidates everything measured in between.

The event log is the single source of truth. The model is a pure function of
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import store
from .config import PATHS
from .models import fusion
from .models.features import FeatureSpace
from .models.features import build as build_features
from .models.taste import TasteModel, verdict_to_reward
from .models.taste import fit as fit_taste

# Roughly three years. Long enough that a full history still counts, short
# enough that a decisive shift in taste is reflected within a season or two.
HALF_LIFE_DAYS = 1100.0

# A skip says "not tonight", not "I disliked this", so it enters as a mild
# negative at half the weight of a stated verdict.
SKIP_REWARD = 0.25
SKIP_WEIGHT = 0.5

# Columns needed for filtering, scoring and axis naming. The synopsis is
# deliberately excluded: it is only needed for the handful of titles actually
# displayed, and loading 300k of them costs hundreds of megabytes for nothing.
_LEAN_COLUMNS = (
    "item_id, title, original_title, year, kind, language, runtime, genres, "
    "keywords, directors, cast_names, imdb_rating, imdb_votes, quality"
)


@dataclass
class Engine:
    _meta: dict[int, dict] | None = field(default=None, repr=False)
    _fs: FeatureSpace | None = field(default=None, repr=False)

    # --- catalogue ---------------------------------------------------------

    def meta(self, con=None) -> dict[int, dict]:
        if self._meta is not None:
            return self._meta
        own = con is None
        con = con or store.connect(read_only=True)
        try:
            cur = con.execute(f"SELECT {_LEAN_COLUMNS} FROM titles")
            cols = [d[0] for d in cur.description]
            self._meta = {r[0]: dict(zip(cols, r, strict=True)) for r in cur.fetchall()}
        finally:
            if own:
                con.close()
        return self._meta

    def features(self, con=None) -> FeatureSpace:
        if self._fs is not None:
            return self._fs
        if not fusion.exists():
            raise RuntimeError("no fused item space; run `entertainer build` first")
        art = fusion.load()
        self._fs = build_features(art.item_ids, art.space, self.meta(con))
        return self._fs

    # --- labels ------------------------------------------------------------

    def labels(self, con) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (item_ids, rewards, ages_in_days, is_weak) from the event log.

        Latest verdict per title wins, so changing your mind about something
        simply overwrites the old opinion rather than averaging with it.
        """
        fs = self.features(con)
        rows = con.execute(
            """
            SELECT item_id, value, date_diff('day', ts, now()) AS age FROM (
                SELECT item_id, value, ts,
                       row_number() OVER (PARTITION BY item_id ORDER BY ts DESC) rn
                FROM events WHERE kind = 'rate' AND value IS NOT NULL
            ) WHERE rn = 1
            """
        ).fetchall()
        ids, rewards, ages, weak = [], [], [], []
        for item_id, value, age in rows:
            if int(item_id) in fs.index:
                ids.append(int(item_id))
                rewards.append(float(value) / 10.0)
                ages.append(float(age or 0))
                weak.append(False)
        # Explicit skips are weak negatives: the user declined to engage, which
        # is informative but far less so than saying they disliked it. They are
        # flagged rather than identified later by their reward value, since a
        # stated verdict could coincide with that value.
        for item_id in store.negatives(con):
            if int(item_id) in fs.index:
                ids.append(int(item_id))
                rewards.append(SKIP_REWARD)
                ages.append(0.0)
                weak.append(True)
        return (
            np.array(ids, dtype=np.int64),
            np.array(rewards, dtype=np.float64),
            np.array(ages, dtype=np.float64),
            np.array(weak, dtype=bool),
        )

    def fit(self, con, save: bool = True, half_life_days: float = HALF_LIFE_DAYS) -> TasteModel | None:
        ids, rewards, ages, weak = self.labels(con)
        if ids.size < 3:
            return None
        fs = self.features(con)
        X = fs.vectors_for(ids)

        # Skips carry half the weight of a stated verdict.
        weights = np.where(weak, SKIP_WEIGHT, 1.0)
        # Taste drifts. Old verdicts still count, but a film you loved four
        # years ago is weaker evidence about what you want tonight than one
        # you loved last month. The half-life is deliberately long: this is a
        # gentle tilt towards the present, not a forgetting mechanism.
        if half_life_days > 0:
            weights = weights * np.exp(-np.log(2.0) * ages / half_life_days)
        weights = np.maximum(weights, 1e-3)

        model = fit_taste(X, rewards, sample_weight=weights)
        if save:
            model.to_npz()
        return model

    def model(self, con, refit: bool = True) -> TasteModel | None:
        if refit:
            return self.fit(con)
        return TasteModel.from_npz() if TasteModel.exists() else None

    # --- recording ---------------------------------------------------------

    def record(self, con, item_id: int, verdict: str, source: str = "manual") -> float:
        reward = verdict_to_reward(verdict)
        store.log_event(con, item_id, "rate", reward * 10.0, source, {"verdict": verdict})
        return reward

    # --- reporting ---------------------------------------------------------

    @cached_property
    def artifacts_present(self) -> dict[str, bool]:
        from .models import cf, encoder

        return {
            "catalog": PATHS.catalog_db.exists(),
            "content_embeddings": encoder.exists(),
            "cf_factors": cf.exists(),
            "fused_space": fusion.exists(),
            "taste_model": TasteModel.exists(),
        }


def liked_titles(con, engine: Engine, min_reward: float = 0.7, limit: int = 60):
    """The user's own positives, for use in explanations."""
    ids, rewards, _, _ = engine.labels(con)
    if ids.size == 0:
        return [], []
    order = np.argsort(-rewards)
    picked = [int(ids[i]) for i in order if rewards[i] >= min_reward][:limit]
    meta = engine.meta(con)
    labels = []
    for i in picked:
        r = meta.get(i, {})
        title = r.get("title", str(i))
        labels.append(f"{title} ({r['year']})" if r.get("year") else title)
    return picked, labels
