"""Producing a recommendation slate, logged as an observation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ... import store
from ..context import AppContext, get_context
from ..present import present

router = APIRouter()


@router.post("/api/recommendations/slate")
def recommendation_slate(k: int = 10, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Create and log an observational production slate with propensities."""
    from ...recommend import Filters, produce_slate

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
    return {
        "slate_id": slate.slate_id,
        "observational": True,
        "items": [{**present(meta[r.item_id]), "score": r.mean * 10.0, "std": r.std * 10.0,
                   "propensity": r.propensity, "explored": r.explored} for r in recs],
    }
