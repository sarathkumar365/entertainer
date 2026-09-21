"""Resumability of the encoding pass.

Encoding the catalogue is forty minutes of GPU work with nothing to show for
itself until the end, and it has already been killed once mid-pipeline by a
session teardown. These tests pin that an interruption costs only the shard
in flight.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def sharded(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer import config
    from entertainer.models import encoder

    importlib.reload(config)
    importlib.reload(encoder)
    config.PATHS.ensure()
    return encoder


def _fake_encoder(monkeypatch, encoder, dim=8, counter=None):
    """Stand in for the real model: deterministic vectors, counts the work."""

    def fake_encode_texts(texts, batch_size=64, dim=dim, model=None, show_progress=True):
        if counter is not None:
            counter["encoded"] += len(texts)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, hash(text) % dim] = 1.0
        return out

    monkeypatch.setattr(encoder, "encode_texts", fake_encode_texts)
    monkeypatch.setattr(encoder, "load_model", lambda name=None: object())


def test_encoding_is_resumable_and_does_not_redo_finished_shards(sharded, monkeypatch):
    encoder = sharded
    counter = {"encoded": 0}
    _fake_encoder(monkeypatch, encoder, counter=counter)

    ids = np.arange(250, dtype=np.int32)
    texts = [f"card {i}" for i in ids]

    ids_a, mat_a = encoder.encode_resumable(ids, texts, dim=8, shard_size=100)
    assert counter["encoded"] == 250
    assert mat_a.shape == (250, 8)

    counter["encoded"] = 0
    ids_b, mat_b = encoder.encode_resumable(ids, texts, dim=8, shard_size=100)
    assert counter["encoded"] == 0, "a completed run must re-encode nothing"
    assert np.array_equal(ids_a, ids_b)
    assert np.array_equal(mat_a, mat_b)


def test_only_the_missing_shard_is_recomputed(sharded, monkeypatch):
    encoder = sharded
    counter = {"encoded": 0}
    _fake_encoder(monkeypatch, encoder, counter=counter)

    ids = np.arange(250, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=8, shard_size=100)

    shards = sorted(encoder._shard_dir("content").glob("*.npz"))
    assert len(shards) == 3
    shards[1].unlink()

    counter["encoded"] = 0
    _, mat = encoder.encode_resumable(ids, texts, dim=8, shard_size=100)
    assert counter["encoded"] == 100, counter
    assert mat.shape == (250, 8)


def test_a_changed_catalogue_invalidates_only_affected_shards(sharded, monkeypatch):
    """Shards key on the ids they cover, so new ids cannot inherit old vectors."""
    encoder = sharded
    counter = {"encoded": 0}
    _fake_encoder(monkeypatch, encoder, counter=counter)

    ids = np.arange(200, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=8, shard_size=100)

    # Second shard now covers different titles entirely.
    changed = np.concatenate([np.arange(100), np.arange(900, 1000)]).astype(np.int32)
    counter["encoded"] = 0
    out_ids, _ = encoder.encode_resumable(
        changed, [f"card {i}" for i in changed], dim=8, shard_size=100
    )
    assert counter["encoded"] == 100, counter
    assert np.array_equal(out_ids, changed)


def test_a_truncated_shard_is_discarded_rather_than_trusted(sharded, monkeypatch):
    encoder = sharded
    counter = {"encoded": 0}
    _fake_encoder(monkeypatch, encoder, counter=counter)

    ids = np.arange(100, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=8, shard_size=100)

    shard = next(encoder._shard_dir("content").glob("*.npz"))
    shard.write_bytes(b"not a real npz")

    counter["encoded"] = 0
    _, mat = encoder.encode_resumable(ids, texts, dim=8, shard_size=100)
    assert counter["encoded"] == 100
    assert mat.shape == (100, 8)


def test_clear_shards_forces_a_full_recompute(sharded, monkeypatch):
    encoder = sharded
    counter = {"encoded": 0}
    _fake_encoder(monkeypatch, encoder, counter=counter)

    ids = np.arange(100, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=8, shard_size=100)
    assert encoder.clear_shards() == 1

    counter["encoded"] = 0
    encoder.encode_resumable(ids, texts, dim=8, shard_size=100)
    assert counter["encoded"] == 100
