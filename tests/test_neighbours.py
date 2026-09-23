"""Nearest-neighbour search in the latent space.

Lifted out of the `ent similar` command body, where it was written inline and
reachable only by invoking a Typer callback. The web layer needs the same
search for /api/similar, so it is now a function with its own tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from entertainer.models.features import FeatureSpace
from entertainer.recommend import neighbours


def space(vectors: dict[int, list[float]]) -> FeatureSpace:
    ids = np.array(sorted(vectors), dtype=np.int32)
    latent = np.array([vectors[int(i)] for i in ids], dtype=np.float32)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    return FeatureSpace(
        item_ids=ids,
        latent=latent,
        side=np.zeros((len(ids), 0), dtype=np.float32),
        index={int(i): r for r in range(len(ids)) for i in [ids[r]]},
    )


META = {1: {"language": "ml"}, 2: {"language": "ml"}, 3: {"language": "ta"}, 4: {"language": "en"}}


@pytest.fixture()
def fs():
    return space({
        1: [1.0, 0.0],
        2: [0.99, 0.14],   # nearest to 1
        3: [0.7, 0.71],
        4: [-1.0, 0.0],    # opposite
    })


def test_orders_by_similarity_and_excludes_the_query_itself(fs):
    got = neighbours(fs, 1, META, k=3)
    assert [n.item_id for n in got] == [2, 3, 4]
    assert all(got[i].similarity >= got[i + 1].similarity for i in range(len(got) - 1))


def test_respects_k(fs):
    assert len(neighbours(fs, 1, META, k=1)) == 1


def test_language_filter(fs):
    got = neighbours(fs, 1, META, k=5, languages=("ta",))
    assert [n.item_id for n in got] == [3]


def test_skips_items_absent_from_meta(fs):
    """A row with no metadata cannot be rendered, so it must not be returned
    rather than being returned and blowing up in the caller."""
    got = neighbours(fs, 1, {1: META[1], 3: META[3]}, k=5)
    assert [n.item_id for n in got] == [3]


def test_unknown_item_raises(fs):
    with pytest.raises(KeyError):
        neighbours(fs, 999, META, k=3)


def test_tied_similarities_come_back_in_a_stable_order():
    """argpartition would be faster but reorders ties, and exact ties are
    common in a 73k catalogue — the displayed order would wobble."""
    tied = space({1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.0, 1.0], 4: [0.0, 1.0]})
    meta = {i: {"language": "en"} for i in (1, 2, 3, 4)}
    first = [n.item_id for n in neighbours(tied, 1, meta, k=3)]
    for _ in range(4):
        assert [n.item_id for n in neighbours(tied, 1, meta, k=3)] == first
