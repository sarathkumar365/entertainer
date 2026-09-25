"""Producing a recommendation slate, logged as an observation."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ... import store
from ...models.taste import to_display_scale
from ..context import AppContext, get_context
from ..present import present


class SlateRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=20)
    kind: Literal["movie", "tv", "both"] = "both"


def _band(std: float) -> float:
    """The +/- half-width, on the same 0-10 scale and always a real number."""
    value = float(std)
    if value != value:  # NaN
        return 10.0
    return min(max(value, 0.0) * 10.0, 10.0)


router = APIRouter()


@router.post("/api/recommendations/slate")
def recommendation_slate(
    body: SlateRequest | None = None,
    k: int = Query(default=10, ge=1, le=20),
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """Create and log an observational production slate with propensities."""
    request = body or SlateRequest(k=k)
    from ...engine import liked_titles
    from ...evaluation import offpolicy
    from ...evaluation.prequential import MIN_LOGGED
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
        # Saved titles are excluded alongside interacted ones: there is no
        # point recommending something the user has already decided to watch.
        excluded = store.interacted(con) | store.active_watchlist(con)
        slate = produce_slate(
            con, model, fs, meta, k=request.k,
            # The kind is part of the policy label, not just a filter. A
            # films-only slate is a different policy from an unrestricted
            # one, and the off-policy analysis groups by this string.
            policy=f"bayesian-thompson-v1:{request.kind}",
            remember=False,
            strategy="thompson",
            filters=Filters(
                kind=None if request.kind == "both" else request.kind,
                exclude=frozenset(excluded),
            ),
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
        # How close the off-policy check is to being able to run. The page says
        # so, because a verdict given here is the only kind that moves this
        # number and nothing on any screen used to mention that.
        outcomes = offpolicy.usable_count(con)

    return {
        "slate_id": slate.slate_id,
        "observational": True,
        "kind": request.kind,
        "outcomes": {"have": outcomes, "need": MIN_LOGGED},
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
