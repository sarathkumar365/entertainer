"""What the engine has worked out, and how well it is doing.

Everything here was reachable only from the terminal: `ent taste`, `ent
audit` and `ent similar` computed it and printed it, and the browser had no
way to ask.

These are the slow endpoints. `audit` refits the model once per verdict in
the history and grows roughly linearly with it — two seconds at 169 verdicts
— so it is a request the interface must show progress for rather than block
silently on.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ... import store
from ...errors import NotEnoughEvidence
from ...evaluation import offpolicy
from ...evaluation.prequential import MIN_LOGGED, MIN_VERDICTS
from ...evaluation.prequential import readings as prequential_readings
from ...evaluation.prequential import run as prequential
from ..context import AppContext, get_context
from ..present import present

router = APIRouter()


@router.get("/api/taste")
def taste(axes: int = 6, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """The learned latent axes, and the named side features.

    Each axis carries both poles and example titles from each, which is what
    makes it legible — an axis is only meaningful relative to what it points
    away from.
    """
    from ...models.discover import describe_axes, surface_preferences

    with store.session(read_only=True) as con:
        model = ctx.engine.fit(con, save=False)
        if model is None:
            # A domain error rather than a bare HTTPException, so the answer
            # carries the "not_enough_evidence" code the page branches on.
            raise NotEnoughEvidence(
                "at least three verdicts are needed before there is a taste to describe"
            )
        fs = ctx.engine.features(con)
        meta = ctx.engine.meta(con)

    described = describe_axes(
        model, fs.item_ids, fs.latent, meta, n_axes=max(1, min(axes, 24))
    )
    return {
        "n_verdicts": model.n_real,
        "capacity": {
            "rff": model.feature_map.n_rff,
            "log_evidence": model.log_evidence,
        },
        "axes": [
            {
                "index": a.index,
                "weight": a.weight,
                "strength": a.strength,
                "towards": {"terms": a.liked_pole, "examples": a.liked_examples},
                "away": {"terms": a.other_pole, "examples": a.other_examples},
            }
            for a in described
        ],
        "side_features": [
            {"name": name, "weight": weight}
            for name, weight in surface_preferences(model, fs)
        ],
    }


@router.get("/api/audit")
def audit(interval: float = 0.90, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """The learning curve, its readings, and the off-policy check.

    The raw per-step arrays are returned alongside the summary rows so the
    interface can plot the curve rather than re-deriving it.
    """
    with store.session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT item_id, value FROM events
            WHERE kind = 'rate' AND value IS NOT NULL ORDER BY ts
            """
        ).fetchall()
        fs = ctx.engine.features(con)

    seen: set[int] = set()
    items: list[int] = []
    rewards: list[float] = []
    for item_id, value in rows:
        iid = int(item_id)
        if iid in seen or iid not in fs.index:
            continue
        seen.add(iid)
        items.append(iid)
        rewards.append(float(value) / 10.0)

    if len(items) < MIN_VERDICTS:
        raise HTTPException(
            409,
            f"only {len(items)} verdicts — this needs at least {MIN_VERDICTS} "
            "to say anything honest",
        )

    result = prequential(fs, items, rewards, interval=interval)
    with store.session(read_only=True) as con:
        policy = offpolicy.estimate(con, ctx.engine, fs)

    return {
        "n_verdicts": len(items),
        "interval": interval,
        "readings": [
            {"measure": r.measure, "value": r.value, "reading": r.reading, "tone": r.tone}
            for r in prequential_readings(result, interval=interval, n_verdicts=len(items))
        ],
        "curve": {
            "steps": result.steps,
            "absolute_error": result.absolute_error,
            "baseline_error": result.baseline_error,
            # The control's error minus the model's, per step. Positive means
            # the model beat predicting the running average on that verdict.
            # Plotted rather than re-derived, because the readings above are
            # computed from it and the two must not drift apart.
            "skill": result.skill(),
            "informative": result.informative(),
            "predicted": result.predicted,
            "actual": result.actual,
            "inside_interval": result.inside_interval,
        },
        "off_policy": {
            "status": policy.status,
            "n_usable": policy.n_usable,
            # The threshold travels with the number it gates. Hardcoding 30 in
            # the page let Evidence and the recommendations screen disagree
            # about the same quantity the moment MIN_LOGGED moved.
            "need": MIN_LOGGED,
            "logged_value": policy.logged_value,
            "estimate": policy.estimate,
            "better": policy.better,
        },
    }


@router.get("/api/similar/{item_id}")
def similar(
    item_id: int,
    k: int = 10,
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """Titles nearest this one in the latent space.

    Pure geometry — this ignores the taste model, so it answers "what is like
    this" rather than "what would you enjoy".
    """
    from ...recommend import neighbours

    with store.session(read_only=True) as con:
        fs = ctx.engine.features(con)
        meta = ctx.engine.meta(con)
        try:
            found = neighbours(fs, item_id, meta, k=max(1, min(k, 50)))
        except KeyError as exc:
            raise HTTPException(404, "that title is not in the item space") from exc
        rows = store.item_rows(con, [n.item_id for n in found])

    return {
        "items": [
            {**present(rows[n.item_id]), "similarity": n.similarity}
            for n in found
            if n.item_id in rows
        ]
    }
