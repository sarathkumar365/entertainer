"""Pull a title TMDB knows into the catalogue and place it in the item space.

Both the web app's "add a missing film" button and the Netflix import need
exactly this, and it is subtle enough — the encoder matrix and the fused
space have to stay index-aligned, and the engine's caches have to be busted
afterwards — that having two copies of it would be a bug waiting to happen.
"""

from __future__ import annotations

import numpy as np

from . import store
from .data import catalog, tmdb
from .errors import EntertainerError


class IngestError(EntertainerError):
    """A title could not be pulled in. The message is safe to show a user."""


def add_title(engine, tmdb_id: int, kind: str, *, place: bool = True) -> tuple[int, dict]:
    """Insert a TMDB title, returning its item id and the flattened row.

    `place` is what the encoder costs: putting a title in the item space
    needs a 1.2GB model, which is a steep price on a machine that only
    collects verdicts. Skipping it still records a usable verdict, because
    verdicts key on the IMDb id and merge back later.
    """
    payload = tmdb.detail(tmdb_id, kind)
    if not payload:
        raise IngestError("TMDB returned nothing for that id")
    row = tmdb.detail_to_row(payload, kind)

    art = None
    if place:
        from .models import fusion

        art = fusion.load()
        if art.pca_mean is None:
            raise IngestError("item space predates projection; run `ent data fuse`")

    with store.session() as con:
        item_id = catalog.insert_title(con, row)

    if place and item_id not in engine.features().index:
        from .models import encoder, fusion
        from .models.itemcard import build_card

        content = encoder.encode_texts([build_card(row)], show_progress=False)
        latent = art.project(content)
        ids, mat = encoder.load()
        encoder.save(np.append(ids, np.int32(item_id)), np.vstack([mat, content]))
        art.item_ids = np.append(art.item_ids, np.int32(item_id))
        art.space = np.vstack([art.space, latent])
        fusion.save(art)
        engine._fs = None
        engine._meta = None

    return item_id, row
