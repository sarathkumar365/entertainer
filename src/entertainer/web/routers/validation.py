"""The sealed blind test, and single-title predictions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext, get_context
from ..schemas import ValidationRevealRequest, ValidationSealRequest

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
