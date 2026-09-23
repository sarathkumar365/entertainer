"""Producing a recommendation slate, logged as an observation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ... import store
from ...models.taste import to_display_scale
from ..context import AppContext, get_context
from ..present import present


def _band(std: float) -> float:
    """The +/- half-width, on the same 0-10 scale and always a real number."""
    value = float(std)
    if value != value:  # NaN
        return 10.0
    return min(max(value, 0.0) * 10.0, 10.0)


router = APIRouter()


@router.post("/api/recommendations/slate")
def recommendation_slate(k: int = 10, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Create and log an observational production slate with propensities."""
    from ...engine import liked_titles
    from ...recommend import Filters, attach_reasons, produce_slate

    with store.session() as con:
        # Recommendations are derived from the event log on every call;
        # do not persist a transient model merely to create a slate.
        model = ctx.engine.fit(con, save=False)
        if model is None:
            raise HTTPException(409, "at least three explicit verdicts are needed before recommendations")
        fs = ctx.engine.features(con)
        meta = ctx.engine.meta(con)
        # remember=False: a slate position exists to be typed at a prompt,
        # and there is no prompt in a browser. Writing last_slate here
        # would repoint `ent loved 3` at a slate the terminal never saw.
        slate = produce_slate(
            con, model, fs, meta, k=max(1, min(k, 20)),
            policy="bayesian-thompson-v1", remember=False,
            strategy="thompson",
            filters=Filters(exclude=frozenset(store.interacted(con))),
        )
        recs = slate.picks
        # "because you liked X" — the same explanation `ent recs` prints.
        liked_ids, liked_labels = liked_titles(con, ctx.engine)
        attach_reasons(recs, fs, liked_ids, liked_labels)
        # Engine.meta is a lean column set — no poster, no overview — because
        # it is loaded for the whole catalogue and the synopses alone would be
        # tens of megabytes. Fetching full rows for the handful actually
        # recommended is what makes them renderable.
        rows = store.item_rows(con, slate.item_ids)

    return {
        "slate_id": slate.slate_id,
        "observational": True,
        "items": [
            {
                **present(rows.get(r.item_id) or meta[r.item_id]),
                # Clamped for the same reason `ent recs` clamps: an
                # unbounded posterior predicts 10.5 for something squarely
                # inside what you love, and a bar drawn past its own axis
                # reads as a bug.
                "score": to_display_scale(r.mean),
                # A pick outside the shortlist carries std=NaN, and Python's
                # json emits a bare NaN token that JSON.parse rejects — one
                # such item would blank the whole slate rather than one card.
                "std": _band(r.std),
                "propensity": r.propensity,
                "explored": r.explored,
                "reasons": r.reasons,
            }
            for r in recs
        ],
    }
