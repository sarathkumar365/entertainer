"""Item text tower.

Qwen3-Embedding-0.6B is the default: as of 2026 the Qwen3 embedding family
leads the MTEB multilingual board among open weights, covers 100+ languages
(which is the whole ballgame for a Tamil/Malayalam/Korean catalogue), and at
0.6B in half precision it leaves plenty of room on an 8GB card for a large batch.

Its outputs are Matryoshka-trained, so the 1024-dim vector can be truncated to
256 and renormalised with very little loss. Storing 256 dims instead of 1024
makes the downstream dense scan four times cheaper and keeps the whole item
matrix in RAM.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from ..config import ENCODER_DIM_TARGET, ENCODER_FALLBACK, ENCODER_MODEL, PATHS

console = Console()

_EMB_FILE = "content.npy"
_IDS_FILE = "content_ids.npy"

DEFAULT_BATCH = 64


def _device() -> str:
    """cuda > mps > cpu, unless ENTERTAINER_DEVICE names one explicitly."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the encode extra: uv pip install -e '.[encode]'") from exc
    override = os.environ.get("ENTERTAINER_DEVICE", "").strip().lower()
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def auto_batch_size(dev: str | None = None) -> int:
    """Size the batch to the card rather than to the smallest one it might run on.

    64 leaves a 16-24GB GPU mostly idle. Free memory rather than total is
    measured because the web app or another process may already hold some.
    """
    dev = dev or _device()
    if dev.startswith("cuda"):
        import torch

        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:
            return DEFAULT_BATCH
        gib = free / 2**30
        if gib >= 14:
            return 512
        if gib >= 7:
            return 256
        return DEFAULT_BATCH
    if dev == "mps":
        # Unified memory is shared with everything else on the machine.
        return 32
    return DEFAULT_BATCH


def resolve_batch_size(requested: int | None) -> int:
    """An explicit batch size always wins; None means size it to the device."""
    return int(requested) if requested else auto_batch_size()


def _model_kwarg_attempts(dev: str) -> list[dict]:
    """Most capable configuration first, each later one giving something up.

    mps stays in fp32: Qwen3 in half precision on mps is not reliable enough
    to bet a long run on.
    """
    if not dev.startswith("cuda"):
        return [{}]
    import torch

    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    base = {"torch_dtype": dtype}
    return [{**base, "attn_implementation": "sdpa"}, base]


def _construct(target: str, dev: str, model_kwargs: dict):
    """Instantiate the encoder, tolerating the 5.x -> 6.x keyword rename.

    Qwen3 embedding models want left padding (they pool the final token), and
    sentence-transformers renamed the argument that sets it. Trying the new
    name first and falling back keeps this working on either version rather
    than pinning the whole project to one.
    """
    from sentence_transformers import SentenceTransformer

    pad = {"padding_side": "left"}
    for kw in ({"processor_kwargs": pad}, {"tokenizer_kwargs": pad}, {}):
        try:
            return SentenceTransformer(target, device=dev, model_kwargs=model_kwargs, **kw)
        except TypeError:
            continue
    return SentenceTransformer(target, device=dev, model_kwargs=model_kwargs)


def _build(target: str, dev: str):
    attempts = _model_kwarg_attempts(dev)
    for i, model_kwargs in enumerate(attempts):
        try:
            return _construct(target, dev, model_kwargs)
        except Exception:
            # sdpa is refused by older transformers and some model classes;
            # that is a reason to run slower, not to fall back to another model.
            if i == len(attempts) - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


# One model per (name, device) for the life of the process: the web app's
# add-title path would otherwise pay a multi-second load on every request.
_MODELS: dict[tuple[str, str], object] = {}
_MODEL_NAMES: dict[int, str] = {}
_MODELS_LOCK = threading.Lock()


def load_model(name: str | None = None):
    dev = _device()
    target = name or ENCODER_MODEL
    key = (target, dev)
    with _MODELS_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached
        effective = target
        try:
            model = _build(target, dev)
        except Exception as exc:
            console.print(
                f"[yellow]{target} unavailable ({exc}); falling back to {ENCODER_FALLBACK}[/yellow]"
            )
            effective = ENCODER_FALLBACK
            model = _build(ENCODER_FALLBACK, dev)
        model.max_seq_length = 384
        _MODELS[key] = model
        _MODEL_NAMES[id(model)] = effective
        return model


def model_name(model, default: str | None = None) -> str:
    """The checkpoint a loaded model really is, which differs after a fallback."""
    return _MODEL_NAMES.get(id(model), default or ENCODER_MODEL)


def _truncate(mat: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation followed by renormalisation.

    Copies rather than slicing in place: ``mat[:, :dim]`` is a view, and
    normalising through it would silently rewrite the caller's array.
    """
    out = np.array(mat[:, :dim] if mat.shape[1] > dim else mat, dtype=np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(norms, 1e-9)
    return out


def encode_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH,
    dim: int = ENCODER_DIM_TARGET,
    model=None,
    show_progress: bool = True,
) -> np.ndarray:
    model = model or load_model()
    mat = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)
    return _truncate(mat, dim)


def encode_query(text: str, model=None, dim: int = ENCODER_DIM_TARGET) -> np.ndarray:
    """Encode a free-text taste description with the retrieval instruction."""
    from .itemcard import QUERY_INSTRUCTION

    model = model or load_model()
    prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
    vec = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    return _truncate(vec.astype(np.float32), dim)[0]


# --- content-addressed, resumable encoding ---------------------------------

CHUNK_SIZE = 8_192
_KEY_BYTES = 20  # sha1


def _cache_dir(stem: str) -> Path:
    return Path(PATHS.embeddings) / f"{stem}_cache"


def card_keys(texts: list[str], name: str, dim: int) -> list[bytes]:
    """One key per card: a vector depends on the model, the width and the text, nothing else."""
    prefix = f"{name}\x00{dim}\x00".encode()
    return [hashlib.sha1(prefix + t.encode("utf-8")).digest() for t in texts]


def _load_cache(directory: Path) -> tuple[dict[bytes, np.ndarray], int]:
    """Every cached vector, later chunks overriding earlier ones.

    Also returns how many rows the chunks hold in total, so the caller can
    tell when stale entries have piled up.
    """
    found: dict[bytes, np.ndarray] = {}
    rows = 0
    for path in sorted(directory.glob("*.npz")):
        try:
            with np.load(path) as z:
                hashes, vectors = z["hashes"], z["vectors"]
        except Exception:
            # A chunk torn by the interruption this cache exists to survive.
            path.unlink(missing_ok=True)
            continue
        # Width is part of every key, so no dimension check is needed here.
        if vectors.ndim != 2 or hashes.shape != (len(vectors), _KEY_BYTES):
            continue
        rows += len(vectors)
        raw = hashes.tobytes()
        for i, v in enumerate(vectors):
            found[raw[i * _KEY_BYTES : (i + 1) * _KEY_BYTES]] = v
    return found, rows


def _write_chunk(directory: Path, hashes: list[bytes], vectors: np.ndarray) -> Path:
    # Named by write time so a sorted listing replays chunks in order, and
    # written under a temporary name first: a half-written chunk must never
    # look like a finished one.
    path = directory / f"{time.time_ns():020d}_{os.getpid()}.npz"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            # Raw bytes, not an "S20" array: numpy strips trailing NULs from
            # fixed-width byte strings, which would corrupt one digest in 256.
            hashes=np.frombuffer(b"".join(hashes), dtype=np.uint8).reshape(-1, _KEY_BYTES),
            vectors=np.asarray(vectors, dtype=np.float32),
        )
    tmp.replace(path)
    return path


def _compact(directory: Path, keep: dict[bytes, np.ndarray], dim: int) -> None:
    """Rewrite the cache as one chunk holding only what the catalogue uses."""
    old = sorted(directory.glob("*.npz"))
    hashes = list(keep)
    vectors = (
        np.vstack([keep[h] for h in hashes]) if hashes else np.zeros((0, dim), dtype=np.float32)
    )
    written = _write_chunk(directory, hashes, vectors)
    for path in old:
        if path != written:
            path.unlink(missing_ok=True)


def encode_resumable(
    ids: np.ndarray,
    texts: list[str],
    batch_size: int | None = None,
    dim: int = ENCODER_DIM_TARGET,
    stem: str = "content",
    chunk_size: int = CHUNK_SIZE,
    fresh: bool = False,
    name: str | None = None,
    compact: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a whole catalogue, reusing every card that has been seen before.

    Encoding 270k item cards is the one stage with nothing to show for itself
    until the very end, and most rebuilds change only a handful of cards. Each
    vector is therefore cached under a hash of the model, width and card text:
    a rebuild encodes only new or edited cards, and an interruption loses at
    most the chunk in flight, because every chunk is persisted as it finishes.

    Misses are encoded longest first across the whole catalogue, which keeps
    batches evenly padded and surfaces an out-of-memory on the first chunk
    rather than at the end. The result is always in the order of ``ids``.
    ``fresh`` ignores what is cached, though new vectors are still written.
    ``compact`` drops cached vectors that ``ids`` does not use; only a pass over
    the whole catalogue knows which those are.
    """
    directory = _cache_dir(stem)
    directory.mkdir(parents=True, exist_ok=True)
    name = name or ENCODER_MODEL

    cache, cached_rows = ({}, 0) if fresh else _load_cache(directory)
    keys = card_keys(texts, name, dim)

    def misses() -> list[int]:
        first: dict[bytes, int] = {}
        for i, k in enumerate(keys):
            if k not in cache and k not in first:
                first[k] = i
        return list(first.values())

    todo = misses()
    if todo:
        model = load_model(name)
        effective = model_name(model, name)
        if effective != name:
            # A fallback model's vectors must not be filed under the primary's name.
            name = effective
            keys = card_keys(texts, name, dim)
            todo = misses()

    if todo:
        bs = resolve_batch_size(batch_size)
        todo.sort(key=lambda i: len(texts[i]), reverse=True)
        console.print(
            f"[dim]encoding {len(todo):,} new or changed cards (batch {bs}); "
            f"{len(texts) - len(todo):,} from cache[/dim]"
        )
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            console.print(f"[dim]encoding {start:,}–{start + len(chunk):,} of {len(todo):,}[/dim]")
            mat = encode_texts(
                [texts[i] for i in chunk], batch_size=bs, dim=dim, model=model, show_progress=True
            ).astype(np.float32)
            chunk_keys = [keys[i] for i in chunk]
            _write_chunk(directory, chunk_keys, mat)
            cached_rows += len(chunk)
            for k, v in zip(chunk_keys, mat, strict=True):
                cache[k] = v
    elif texts:
        console.print(f"[green]all {len(texts):,} cards already encoded[/green]")

    all_mat = (
        np.vstack([cache[k] for k in keys]).astype(np.float32)
        if keys
        else np.zeros((0, dim), dtype=np.float32)
    )

    live = {k: cache[k] for k in keys}
    if compact and not fresh and cached_rows > 2 * max(len(live), 1):
        _compact(directory, live, dim)
    return np.asarray(ids).astype(np.int32), all_mat


def save(ids: np.ndarray, mat: np.ndarray, stem: str = "content") -> None:
    PATHS.ensure()
    np.save(PATHS.embeddings / f"{stem}_ids.npy", ids.astype(np.int32))
    np.save(PATHS.embeddings / f"{stem}.npy", mat.astype(np.float32))


def load(stem: str = "content") -> tuple[np.ndarray, np.ndarray]:
    ids = np.load(PATHS.embeddings / f"{stem}_ids.npy")
    mat = np.load(PATHS.embeddings / f"{stem}.npy")
    return ids, mat


def exists(stem: str = "content") -> bool:
    return (PATHS.embeddings / f"{stem}.npy").exists()
