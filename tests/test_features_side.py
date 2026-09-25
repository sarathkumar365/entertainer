"""Side features for a title the space does not hold."""

from __future__ import annotations

import numpy as np

from entertainer.models.features import build


def _space():
    rng = np.random.default_rng(0)
    ids = np.arange(1, 61)
    meta = {
        int(i): {
            "quality": float(rng.random()) if i % 3 else None,
            "imdb_votes": int(rng.integers(1, 100_000)) if i % 4 else None,
            "year": int(rng.integers(1950, 2026)) if i % 5 else None,
            "runtime": int(rng.integers(20, 400)) if i % 2 else None,
            "kind": "tv" if i % 7 == 0 else "movie",
        }
        for i in ids
    }
    return build(ids, rng.normal(size=(len(ids), 4)), meta), meta


def test_side_for_reproduces_what_build_gave_a_catalogue_title():
    fs, meta = _space()
    for row, item_id in enumerate(fs.item_ids.tolist()):
        assert np.allclose(fs.side_for(meta[item_id]), fs.side[row], atol=1e-6)


def test_missing_facts_contribute_nothing():
    fs, _ = _space()
    side = fs.side_for({"kind": "movie"})
    assert np.allclose(side[:4], 0.0)
