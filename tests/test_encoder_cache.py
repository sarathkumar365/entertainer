"""The content-addressed encoding cache, device choice and model reuse.

Encoding the catalogue is the slowest stage of a build, and most rebuilds
change a handful of cards. These pin that a vector is computed once per
(model, width, card text), that an interruption costs only the chunk in
flight, and that the output is always in catalogue order whatever order the
work was done in.

No torch, no model download: a fake model stands in, and a fake ``torch``
module stands in where device selection is under test.
"""

from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

WIDTH = 16
DIM = 8


class FakeModel:
    """Deterministic vectors derived from the text; counts what it is asked to encode."""

    def __init__(self, fail_after: int | None = None):
        self.encoded = 0
        self.calls = 0
        self.fail_after = fail_after

    def encode(self, texts, batch_size=64, **kw):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        self.calls += 1
        self.encoded += len(texts)
        out = np.empty((len(texts), WIDTH), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "little")
            out[i] = np.random.default_rng(seed).normal(size=WIDTH)
        return out


@pytest.fixture()
def enc(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer import config
    from entertainer.models import encoder

    config.PATHS.ensure()
    model = FakeModel()
    monkeypatch.setattr(encoder, "load_model", lambda name=None: model)
    monkeypatch.setattr(encoder, "resolve_batch_size", lambda requested: requested or 4)
    return encoder, model


def _expected(texts):
    from entertainer.models.encoder import _truncate

    return _truncate(FakeModel().encode(texts), DIM)


def test_unchanged_cards_are_never_encoded_twice(enc):
    encoder, model = enc
    ids = np.arange(50, dtype=np.int32)
    texts = [f"card {i} " + "x" * (i % 7) for i in ids]

    _, first = encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=16)
    assert model.encoded == 50

    model.encoded = 0
    _, second = encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=16)
    assert model.encoded == 0
    assert np.array_equal(first, second)


def test_one_changed_card_encodes_exactly_one(enc):
    encoder, model = enc
    ids = np.arange(40, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=16)

    texts[17] = "card 17, now with a longer synopsis"
    model.encoded = 0
    _, mat = encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=16)
    assert model.encoded == 1
    assert np.allclose(mat, _expected(texts), atol=1e-6)


def test_output_is_in_item_order_not_encoding_order(enc):
    """Work is done longest card first; the result must not be."""
    encoder, _ = enc
    ids = np.array([5, 3, 9, 1, 7], dtype=np.int32)
    texts = ["a", "a much longer card", "mid card", "the longest card of them all", "bb"]

    out_ids, mat = encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=2)
    assert np.array_equal(out_ids, ids)
    assert np.allclose(mat, _expected(texts), atol=1e-6)

    # And again purely from cache.
    out_ids, mat = encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=2)
    assert np.allclose(mat, _expected(texts), atol=1e-6)


def test_misses_are_encoded_longest_first(enc):
    encoder, model = enc
    seen: list[list[str]] = []
    original = model.encode

    def spy(texts, **kw):
        seen.append(list(texts))
        return original(texts, **kw)

    model.encode = spy
    texts = ["aa", "a", "aaaa", "aaa"]
    encoder.encode_resumable(np.arange(4), texts, dim=DIM, chunk_size=2)
    assert seen == [["aaaa", "aaa"], ["aa", "a"]]


def test_a_digest_ending_in_nul_still_hits(enc):
    """numpy's fixed-width bytes drop trailing NULs; the cache must not."""
    encoder, model = enc
    name = encoder.ENCODER_MODEL
    text = next(
        t
        for t in (f"card {i}" for i in range(10_000))
        if encoder.card_keys([t], name, DIM)[0].endswith(b"\x00")
    )
    encoder.encode_resumable(np.arange(1), [text], dim=DIM)
    model.encoded = 0
    encoder.encode_resumable(np.arange(1), [text], dim=DIM)
    assert model.encoded == 0


def test_duplicate_cards_are_encoded_once(enc):
    encoder, model = enc
    texts = ["same card"] * 5 + ["other"]
    _, mat = encoder.encode_resumable(np.arange(6), texts, dim=DIM)
    assert model.encoded == 2
    assert np.allclose(mat[0], mat[4])


def test_fresh_ignores_the_cache(enc):
    encoder, model = enc
    ids = np.arange(20, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=DIM)

    model.encoded = 0
    encoder.encode_resumable(ids, texts, dim=DIM, fresh=True)
    assert model.encoded == 20


def test_an_interruption_keeps_every_finished_chunk(enc, monkeypatch):
    encoder, _ = enc
    ids = np.arange(30, dtype=np.int32)
    texts = [f"card {i:02d}" for i in ids]

    dying = FakeModel(fail_after=2)
    monkeypatch.setattr(encoder, "load_model", lambda name=None: dying)
    with pytest.raises(KeyboardInterrupt):
        encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=10)
    assert dying.encoded == 20

    survivor = FakeModel()
    monkeypatch.setattr(encoder, "load_model", lambda name=None: survivor)
    _, mat = encoder.encode_resumable(ids, texts, dim=DIM, chunk_size=10)
    assert survivor.encoded == 10
    assert np.allclose(mat, _expected(texts), atol=1e-6)


def test_a_torn_chunk_is_discarded_rather_than_trusted(enc):
    encoder, model = enc
    ids = np.arange(10, dtype=np.int32)
    texts = [f"card {i}" for i in ids]
    encoder.encode_resumable(ids, texts, dim=DIM)

    chunk = next(encoder._cache_dir("content").glob("*.npz"))
    chunk.write_bytes(b"not a real npz")

    model.encoded = 0
    _, mat = encoder.encode_resumable(ids, texts, dim=DIM)
    assert model.encoded == 10
    assert mat.shape == (10, DIM)


def test_a_different_width_or_model_does_not_reuse_vectors(enc):
    encoder, model = enc
    texts = ["one", "two"]
    encoder.encode_resumable(np.arange(2), texts, dim=DIM)

    model.encoded = 0
    encoder.encode_resumable(np.arange(2), texts, dim=4)
    assert model.encoded == 2

    model.encoded = 0
    encoder.encode_resumable(np.arange(2), texts, dim=DIM, name="some/other-model")
    assert model.encoded == 2


def test_stale_entries_are_compacted_away(enc):
    encoder, model = enc
    for round_ in range(4):
        texts = [f"round {round_} card {i}" for i in range(10)]
        encoder.encode_resumable(np.arange(10), texts, dim=DIM)
    chunks = list(encoder._cache_dir("content").glob("*.npz"))
    rows = sum(len(np.load(p)["hashes"]) for p in chunks)
    assert rows <= 20

    model.encoded = 0
    encoder.encode_resumable(np.arange(10), texts, dim=DIM)
    assert model.encoded == 0


def test_a_partial_pass_does_not_compact_the_full_catalogue_away(enc):
    encoder, model = enc
    texts = [f"card {i}" for i in range(30)]
    encoder.encode_resumable(np.arange(30), texts, dim=DIM)

    encoder.encode_resumable(np.arange(3), texts[:3], dim=DIM, compact=False)

    model.encoded = 0
    encoder.encode_resumable(np.arange(30), texts, dim=DIM)
    assert model.encoded == 0


# --- ent data embed ---------------------------------------------------------


def _catalogue(rows):
    from entertainer import store

    con = store.connect()
    for item_id, title, overview in rows:
        con.execute(
            "INSERT INTO titles (item_id, imdb_id, kind, title, original_title, year, "
            "language, overview, adult) VALUES (?, ?, 'movie', ?, ?, 2000, 'en', ?, false)",
            [item_id, f"tt{item_id:08d}", title, title, overview],
        )
    con.close()


def test_data_embed_rebuild_encodes_only_the_changed_title(enc):
    from entertainer import store
    from entertainer.commands.build import data_embed

    encoder, model = enc
    # Inserted out of order: content.npy must follow item_id regardless.
    _catalogue([(30, "Gamma", "third"), (10, "Alpha", "first"), (20, "Beta", "second")])

    data_embed(batch_size=None, limit=None, fresh=False)
    ids, mat = encoder.load()
    assert ids.tolist() == [10, 20, 30]
    assert model.encoded == 3

    con = store.connect()
    con.execute("UPDATE titles SET overview = 'rewritten' WHERE item_id = 20")
    con.close()

    model.encoded = 0
    data_embed(batch_size=None, limit=None, fresh=False)
    ids2, mat2 = encoder.load()
    assert model.encoded == 1
    assert ids2.tolist() == [10, 20, 30]
    assert np.array_equal(mat2[[0, 2]], mat[[0, 2]])
    assert not np.allclose(mat2[1], mat[1])

    model.encoded = 0
    data_embed(batch_size=None, limit=None, fresh=True)
    assert model.encoded == 3


# --- device and model loading ----------------------------------------------


def _fake_torch(monkeypatch, cuda=False, mps=False, bf16=True, free_gib=0.0):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        is_bf16_supported=lambda: bf16,
        mem_get_info=lambda: (int(free_gib * 2**30), int(24 * 2**30)),
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("ENTERTAINER_DEVICE", raising=False)
    return torch


def test_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    from entertainer.models import encoder

    _fake_torch(monkeypatch, cuda=True, mps=True)
    assert encoder._device() == "cuda"
    _fake_torch(monkeypatch, cuda=False, mps=True)
    assert encoder._device() == "mps"
    _fake_torch(monkeypatch)
    assert encoder._device() == "cpu"


def test_device_env_override_wins(monkeypatch):
    from entertainer.models import encoder

    _fake_torch(monkeypatch, cuda=True)
    monkeypatch.setenv("ENTERTAINER_DEVICE", "CPU")
    assert encoder._device() == "cpu"


@pytest.mark.parametrize(
    ("free", "expected"), [(20.0, 512), (14.0, 512), (8.0, 256), (3.0, 64)]
)
def test_batch_size_follows_free_vram(monkeypatch, free, expected):
    from entertainer.models import encoder

    _fake_torch(monkeypatch, cuda=True, free_gib=free)
    assert encoder.auto_batch_size() == expected
    assert encoder.resolve_batch_size(None) == expected
    assert encoder.resolve_batch_size(48) == 48, "an explicit size always wins"


def test_batch_size_off_cuda(monkeypatch):
    from entertainer.models import encoder

    _fake_torch(monkeypatch, mps=True)
    assert encoder.auto_batch_size() == 32
    _fake_torch(monkeypatch)
    assert encoder.auto_batch_size() == 64


def test_cuda_precision_and_attention(monkeypatch):
    from entertainer.models import encoder

    _fake_torch(monkeypatch, cuda=True, bf16=True)
    first, second = encoder._model_kwarg_attempts("cuda")
    assert first == {"torch_dtype": "bfloat16", "attn_implementation": "sdpa"}
    assert second == {"torch_dtype": "bfloat16"}

    _fake_torch(monkeypatch, cuda=True, bf16=False)
    assert encoder._model_kwarg_attempts("cuda")[0]["torch_dtype"] == "float16"
    assert encoder._model_kwarg_attempts("mps") == [{}]


def test_sdpa_refusal_retries_without_it(monkeypatch):
    from entertainer.models import encoder

    _fake_torch(monkeypatch, cuda=True)
    tried = []

    def construct(target, dev, model_kwargs):
        tried.append(model_kwargs)
        if "attn_implementation" in model_kwargs:
            raise ValueError("sdpa not supported")
        return object()

    monkeypatch.setattr(encoder, "_construct", construct)
    encoder._build("m", "cuda")
    assert [("attn_implementation" in k) for k in tried] == [True, False]


def test_the_model_is_loaded_once_per_process(monkeypatch):
    from entertainer.models import encoder

    _fake_torch(monkeypatch)
    monkeypatch.setattr(encoder, "_MODELS", {})
    monkeypatch.setattr(encoder, "_MODEL_NAMES", {})
    built = []

    class Loaded:
        max_seq_length = 0

    def build(target, dev):
        built.append((target, dev))
        return Loaded()

    monkeypatch.setattr(encoder, "_build", build)
    a = encoder.load_model()
    b = encoder.load_model()
    assert a is b
    assert len(built) == 1
    assert a.max_seq_length == 384


def test_a_fallback_model_is_not_cached_under_the_primary_name(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTERTAINER_DATA_DIR", str(tmp_path))
    from entertainer.config import ENCODER_FALLBACK, ENCODER_MODEL
    from entertainer.models import encoder

    _fake_torch(monkeypatch)
    monkeypatch.setattr(encoder, "_MODELS", {})
    monkeypatch.setattr(encoder, "_MODEL_NAMES", {})
    fake = FakeModel()

    def build(target, dev):
        if target == ENCODER_MODEL:
            raise OSError("no network")
        return fake

    monkeypatch.setattr(encoder, "_build", build)
    monkeypatch.setattr(encoder, "resolve_batch_size", lambda requested: 4)
    encoder.encode_resumable(np.arange(2), ["a", "b"], dim=DIM)
    assert encoder.model_name(fake) == ENCODER_FALLBACK

    hashes = {
        bytes(row)
        for p in encoder._cache_dir("content").glob("*.npz")
        for row in np.load(p)["hashes"]
    }
    assert hashes == set(encoder.card_keys(["a", "b"], ENCODER_FALLBACK, DIM))
