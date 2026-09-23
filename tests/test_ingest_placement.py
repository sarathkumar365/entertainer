"""Placing a new title into the item space.

The encoder matrix and the fused space are plain arrays kept aligned by row
order — row N of each belongs to item_ids[N]. There is no key and no join, so
appending to one and not the other, or appending twice, silently misaligns
every lookup past that point. Nothing raises; recommendations just start
returning the wrong film's vector.

`ent add` carried its own copy of this sequence, without the guard against a
title already in the space.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_cli import app_env  # noqa: F401

from entertainer.engine import Engine
from entertainer.models import encoder, fusion

DETAIL = {
    "id": 424242,
    "title": "A Film Not In The Catalogue",
    "release_date": "2021-05-01",
    "original_language": "ml",
    "overview": "Something.",
    "runtime": 100,
    "genres": [{"name": "Drama"}],
    "vote_average": 7.5,
    "vote_count": 900,
    "external_ids": {"imdb_id": "tt42424242"},
}


@pytest.fixture()
def placeable(app_env, monkeypatch):  # noqa: F811
    """A space that can actually accept a new title.

    The CLI fixture's fusion artifact has no imputation map, so projection
    refuses. Rebuild it with one, at the dimensionality the fixture encoder
    already uses, so the real projection path runs rather than being skipped.
    """
    from entertainer.data import tmdb

    monkeypatch.setattr(tmdb, "detail", lambda tmdb_id, kind: dict(DETAIL, id=tmdb_id))

    ids, matrix = encoder.load()
    dim = matrix.shape[1]
    monkeypatch.setattr(
        encoder,
        "encode_texts",
        lambda texts, **kw: np.ones((len(texts), dim), dtype=np.float32),
    )
    # project() hstacks the content block, the imputed collaborative block and
    # one confidence scalar before rotating, so the PCA basis is over
    # 2 * dim + 1 columns, not dim.
    joint = 2 * dim + 1
    fusion.save(
        fusion.FusionArtifacts(
            item_ids=ids, space=matrix.astype(np.float32),
            components=np.eye(joint, dtype=np.float32)[:dim],
            block_sizes=(dim, dim), cf_r2=0.5, cf_coverage=1.0,
            pca_mean=np.zeros(joint, dtype=np.float32),
            ridge_coef=np.eye(dim, dtype=np.float32),
            ridge_intercept=np.zeros(dim, dtype=np.float32),
        )
    )


def sizes():
    ids, matrix = encoder.load()
    art = fusion.load()
    return len(ids), matrix.shape[0], len(art.item_ids), art.space.shape[0]


def test_placing_a_title_extends_both_arrays_by_exactly_one(placeable):
    from entertainer.ingest import add_title

    before = sizes()
    item_id, row = add_title(Engine(), 424242, "movie")

    after = sizes()
    assert after == tuple(n + 1 for n in before)
    assert len(set(after)) == 1, "the two arrays fell out of alignment"
    assert row["title"] == DETAIL["title"]

    ids, _ = encoder.load()
    assert int(ids[-1]) == item_id
    assert int(fusion.load().item_ids[-1]) == item_id


def test_adding_the_same_title_twice_does_not_append_a_duplicate_row(placeable):
    """`ent add` had no such guard, so a second add appended a duplicate row
    and misaligned every lookup past it — silently."""
    from entertainer.ingest import add_title

    first_id, _ = add_title(Engine(), 424242, "movie")
    after_first = sizes()

    second_id, _ = add_title(Engine(), 424242, "movie")
    assert second_id == first_id, "the same TMDB title became two catalogue rows"
    assert sizes() == after_first
    assert len(set(sizes())) == 1
