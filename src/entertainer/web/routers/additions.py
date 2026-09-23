"""Pulling a title TMDB knows into the catalogue."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ... import store
from ...config import has_tmdb
from ..context import AppContext, get_context
from ..present import present
from ..schemas import AddRequest

router = APIRouter()


@router.post("/api/add")
def add(body: AddRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """Pull a title TMDB knows into the catalogue and place it in the item space."""
    import numpy as np

    from ...data import catalog, tmdb
    from ...models import encoder, fusion
    from ...models.itemcard import build_card

    if not has_tmdb():
        raise HTTPException(400, "no TMDB credentials configured")
    payload = tmdb.detail(body.tmdb_id, body.kind)
    if not payload:
        raise HTTPException(404, "TMDB returned nothing for that id")
    row = tmdb.detail_to_row(payload, body.kind)

    art = None
    if not ctx.use_live:
        art = fusion.load()
        if art.pca_mean is None:
            raise HTTPException(
                500, "item space predates projection; run `ent data fuse`"
            )

    with store.session() as con:
        item_id = catalog.insert_title(con, row)
        already = con.execute(
            "SELECT count(*) FROM titles WHERE item_id = ?", [item_id]
        ).fetchone()[0]
    del already

    # Placing the title in the item space needs the encoder, which is a
    # 1.2GB download. On a machine that only collects verdicts that is a
    # steep price for something the main machine will redo anyway, so it
    # is skipped when the space is not present. The verdict is still
    # recorded and still merges back, because it keys on the IMDb id.
    if not ctx.use_live and item_id not in ctx.engine.features().index:
        content = encoder.encode_texts([build_card(row)], show_progress=False)
        try:
            latent = art.project(content)
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        ids, mat = encoder.load()
        encoder.save(np.append(ids, np.int32(item_id)), np.vstack([mat, content]))
        art.item_ids = np.append(art.item_ids, np.int32(item_id))
        art.space = np.vstack([art.space, latent])
        fusion.save(art)
        ctx.engine._fs = None
        ctx.engine._meta = None

    if body.verdict:
        with store.session() as con:
            if body.verdict == "unseen":
                store.log_event(con, item_id, "unseen", None, "web")
            else:
                ctx.engine.record(con, item_id, body.verdict, source="web")

    with store.session(read_only=True) as con:
        rows = store.item_rows(con, [item_id])
    return {"ok": True, "item": present(rows[item_id])}
