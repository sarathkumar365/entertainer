"""The sealed blind test, and single-title predictions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ... import store
from ...config import has_tmdb
from ..context import AppContext, get_context
from ..present import present
from ..schemas import JudgeRequest, ValidationRevealRequest, ValidationSealRequest

router = APIRouter()


@router.post("/api/validation/seal")
def seal_validation(body: ValidationSealRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Seal a user-selected watched pool before any verdict is revealed."""
    from ...evaluation import personal

    try:
        return personal.seal(ctx.engine, body.item_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc



@router.post("/api/validation/{case_id}/reveal")
def reveal_validation(case_id: str, body: ValidationRevealRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Reveal exactly one sealed validation verdict and add it to training."""
    from ...evaluation import personal

    try:
        return personal.reveal(ctx.engine, case_id, body.verdict)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc



@router.get("/api/validation/cases")
def validation_cases(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Sealed cases still awaiting a verdict.

    The page used to hold the item_id-to-case_id mapping in memory only,
    so a reload stranded the sealed pool: those titles cannot be rated
    through /api/rate by design, and without their case ids they cannot
    be revealed either.
    """
    from ...evaluation import personal

    return {"cases": personal.open_cases()}



@router.get("/api/validation/summary")
def validation_summary(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Interval-aware personal evidence; no bare point-estimate claims."""
    from ...evaluation import personal

    return personal.summary()



@router.get("/api/predict/{item_id}")
def predict(item_id: int, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Score a catalogue title without recording or changing a verdict."""
    from ...evaluation import personal

    try:
        return personal.predict(ctx.engine, item_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/api/judge")
def judge(body: JudgeRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Would you like this? For any title, including one the catalogue lacks.

    A title outside the catalogue is judged from its TMDB details alone:
    encoded, projected into the item space and scored, with nothing written.
    Adding it would be a side effect of asking a question.
    """
    from ...evaluation import personal

    item_id = body.item_id
    if item_id is None and body.tmdb_id is not None:
        with store.session(read_only=True) as con:
            hit = con.execute(
                "SELECT item_id FROM titles WHERE tmdb_id = ? AND kind = ? LIMIT 1",
                [body.tmdb_id, body.kind],
            ).fetchone()
        item_id = int(hit[0]) if hit else None

    if item_id is not None and item_id in ctx.engine.features().index:
        try:
            prediction = personal.predict(ctx.engine, item_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        with store.session(read_only=True) as con:
            row = store.item_rows(con, [item_id]).get(item_id)
        # The space and the catalogue can disagree after a prune or rebuild.
        if row is None:
            raise HTTPException(404, "that title is no longer in the catalogue")
        return {"item": present(row), "prediction": prediction, "source": "catalogue"}

    if body.tmdb_id is None:
        raise HTTPException(404, "that title is not in the catalogue")
    if not has_tmdb():
        raise HTTPException(400, "no TMDB credentials configured")

    import numpy as np

    from ...data import tmdb
    from ...data.catalog import quality_prior
    from ...models import encoder, fusion
    from ...models.itemcard import build_card

    payload = tmdb.detail(body.tmdb_id, body.kind)
    if not payload:
        raise HTTPException(404, "TMDB returned nothing for that id")
    row = tmdb.detail_to_row(payload, body.kind)
    # The same quality the title would get if it were added, so asking now and
    # asking after adding it give the same answer.
    row["quality"] = quality_prior(row.get("tmdb_rating"), row.get("tmdb_votes"), prior_weight=400.0)

    fs = ctx.engine.features()
    content = encoder.encode_texts([build_card(row)], show_progress=False)
    latent = fusion.load().project(content)[0]
    x = np.concatenate([latent, fs.side_for(row)])
    with store.session(read_only=True) as con:
        prediction = personal.predict_vector(ctx.engine, con, x)
    item = present({**row, "item_id": None})
    return {"item": item, "prediction": prediction, "source": "text"}
