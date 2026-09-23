"""What today's model would have scored on yesterday's slates.

Every slate records the probability each title had of being shown. That is
what makes this measurable at all: the verdicts on record were collected
under an older, worse model, so comparing them directly to anything is apples
to oranges. Importance weighting corrects for it.

Separated from the prequential numbers and reported with a hedge, because it
genuinely is the weaker measurement — the estimator is high-variance on a few
hundred samples even self-normalised, and it assumes the candidate set has
not shifted underneath it.

This module returns data. The CLI decides how to phrase it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prequential import MIN_LOGGED, snips


@dataclass
class OffPolicy:
    """The outcome of an off-policy check.

    ``status`` says why there is or is not a number:

    * ``"ok"`` — logged_value and estimate are both set
    * ``"not-enough-data"`` — fewer than MIN_LOGGED usable impressions
    * ``"no-model"`` — too few verdicts to fit anything to compare against
    * ``"item-space-changed"`` — a logged title has left the item space, so
      its vector no longer means what it meant when the slate was shown
    * ``"estimate-unavailable"`` — the estimator itself declined
    """

    status: str
    n_usable: int
    logged_value: float | None = None
    estimate: float | None = None

    @property
    def better(self) -> bool | None:
        if self.estimate is None or self.logged_value is None:
            return None
        return self.estimate > self.logged_value


def estimate(con, engine, fs) -> OffPolicy:
    """Join logged impressions to the verdicts that followed them.

    Only impressions with a positive propensity are usable: a zero would
    divide by zero, and a missing one means the slate predates propensity
    logging.
    """
    # Two joins, and the exact one wins where it applies.
    #
    # A verdict given on a recommendation now records which slate it came
    # from, so it can be matched to that slate and no other. Verdicts
    # predating that, or given anywhere else, fall back to the temporal join:
    # any rating of the item after the impression. That fallback is loose —
    # it credits a slate the user may never have looked at — so it is only
    # used for impressions no stamped verdict claims.
    exact = con.execute(
        """
        SELECT i.item_id, i.propensity, e.value / 10.0 AS reward
        FROM impressions i
        JOIN events e
          ON e.item_id = i.item_id
         AND json_extract_string(e.context, '$.slate_id') = i.slate_id
        WHERE e.kind = 'rate' AND e.value IS NOT NULL
        """
    ).fetchall()
    claimed = {slate for (slate,) in con.execute(
        """
        SELECT DISTINCT i.slate_id
        FROM impressions i
        JOIN events e
          ON e.item_id = i.item_id
         AND json_extract_string(e.context, '$.slate_id') = i.slate_id
        WHERE e.kind = 'rate'
        """
    ).fetchall()}
    loose = con.execute(
        """
        SELECT i.item_id, i.propensity, e.value / 10.0 AS reward, i.slate_id
        FROM impressions i
        JOIN events e ON e.item_id = i.item_id
        WHERE e.kind = 'rate' AND e.value IS NOT NULL AND e.ts >= i.ts
          AND json_extract_string(e.context, '$.slate_id') IS NULL
        """
    ).fetchall()
    rows = list(exact) + [
        (item_id, propensity, reward)
        for item_id, propensity, reward, slate_id in loose
        if slate_id not in claimed
    ]

    usable = [(float(reward), float(p), 0.0) for _, p, reward in rows if p and p > 0]
    item_ids = [int(item_id) for item_id, p, _ in rows if p and p > 0]

    if len(usable) < MIN_LOGGED:
        return OffPolicy(status="not-enough-data", n_usable=len(usable))

    # save=False: this is a measurement, not a fit worth keeping. engine.model
    # defaults to refit=True, which persists taste.npz — so reading the audit
    # over HTTP rewrote the stored model on every page view.
    model = engine.fit(con, save=False)
    if model is None:
        return OffPolicy(status="no-model", n_usable=len(usable))

    if any(item_id not in fs.index for item_id in item_ids):
        return OffPolicy(status="item-space-changed", n_usable=len(usable))

    scores = model.predict(fs.vectors_for(item_ids), with_std=False)
    value = snips(usable, list(scores))
    logged = float(np.mean([reward for reward, _, _ in usable]))
    if value is None:
        return OffPolicy(status="estimate-unavailable", n_usable=len(usable), logged_value=logged)

    return OffPolicy(status="ok", n_usable=len(usable), logged_value=logged, estimate=value)
