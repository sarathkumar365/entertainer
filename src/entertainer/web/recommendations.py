"""HTTP boundary for intentionally generated production recommendation slates."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import store
from ..engine import Engine
from ..recommend import Filters, recommend
from .library import active_watchlist
from .presentation import present


class SlateRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=20)
    kind: Literal["movie", "tv", "both"] = "both"


def create_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

    @router.post("/slate")
    def recommendation_slate(
        body: SlateRequest | None = None,
        k: int = Query(default=10, ge=1, le=20),
    ) -> dict:
        """Generate one intentional observational slate and log its policy inputs."""
        request = body or SlateRequest(k=k)
        with store.session() as con:
            model = engine.fit(con, save=False)
            if model is None:
                raise HTTPException(409, "at least three explicit verdicts are needed before recommendations")
            fs = engine.features(con)
            meta = engine.meta(con)
            excluded = store.interacted(con) | active_watchlist(con)
            slate_id = store.new_slate_id()
            recs = recommend(
                model, fs, meta, k=request.k, strategy="thompson",
                filters=Filters(
                    kind=None if request.kind == "both" else request.kind,
                    exclude=frozenset(excluded),
                ),
            )
            store.log_impressions(
                con, slate_id,
                [(r.item_id, r.position, r.score, r.propensity, r.explored) for r in recs],
                policy=f"bayesian-thompson-v1:{request.kind}",
            )
        return {
            "slate_id": slate_id,
            "observational": True,
            "kind": request.kind,
            "items": [
                {**present(meta[r.item_id]), "score": r.mean * 10.0, "std": r.std * 10.0,
                 "propensity": r.propensity, "explored": r.explored}
                for r in recs
            ],
        }

    return router
