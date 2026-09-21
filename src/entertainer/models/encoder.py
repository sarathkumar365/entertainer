"""Item text tower.

Qwen3-Embedding-0.6B is the default: as of 2026 the Qwen3 embedding family
leads the MTEB multilingual board among open weights, covers 100+ languages
(which is the whole ballgame for a Tamil/Malayalam/Korean catalogue), and at
0.6B in fp16 it leaves plenty of room on an 8GB card for a large batch.

Its outputs are Matryoshka-trained, so the 1024-dim vector can be truncated to
256 and renormalised with very little loss. Storing 256 dims instead of 1024
makes the downstream dense scan four times cheaper and keeps the whole item
matrix in RAM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from rich.console import Console

from ..config import ENCODER_DIM_TARGET, ENCODER_FALLBACK, ENCODER_MODEL, PATHS

console = Console()

_EMB_FILE = "content.npy"
_IDS_FILE = "content_ids.npy"


def _device() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the encode extra: uv pip install -e '.[encode]'") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build(target: str, dev: str):
    """Instantiate the encoder, tolerating the 5.x -> 6.x keyword rename.

    Qwen3 embedding models want left padding (they pool the final token), and
    sentence-transformers renamed the argument that sets it. Trying the new
    name first and falling back keeps this working on either version rather
    than pinning the whole project to one.
    """
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": "float16"} if dev == "cuda" else {}
    pad = {"padding_side": "left"}
    for kw in ({"processor_kwargs": pad}, {"tokenizer_kwargs": pad}, {}):
        try:
            return SentenceTransformer(target, device=dev, model_kwargs=model_kwargs, **kw)
        except TypeError:
            continue
    return SentenceTransformer(target, device=dev)


def load_model(name: str | None = None):
    dev = _device()
    target = name or ENCODER_MODEL
    try:
        model = _build(target, dev)
    except Exception as exc:
        console.print(
            f"[yellow]{target} unavailable ({exc}); falling back to {ENCODER_FALLBACK}[/yellow]"
        )
        model = _build(ENCODER_FALLBACK, dev)
    model.max_seq_length = 384
    return model


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
    batch_size: int = 64,
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


# --- sharded, resumable encoding -------------------------------------------

SHARD_SIZE = 20_000


def _shard_dir(stem: str) -> Path:
    return Path(PATHS.embeddings) / f"{stem}_shards"


def encode_resumable(
    ids: np.ndarray,
    texts: list[str],
    batch_size: int = 64,
    dim: int = ENCODER_DIM_TARGET,
    stem: str = "content",
    shard_size: int = SHARD_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a whole catalogue, surviving interruption.

    Encoding 270k item cards is roughly forty minutes of GPU time, and it is
    the one stage with nothing to show for itself until the very end. A
    dropped SSH session, an OOM, or a laptop lid closing at minute 38 would
    otherwise cost the entire run.

    Work is therefore written in shards as it completes, and a restart skips
    what is already on disk. Shards are keyed by the ids they cover, not by
    position, so a catalogue that changed between runs invalidates only the
    shards it actually affected rather than silently pairing new ids with old
    vectors.
    """
    directory = _shard_dir(stem)
    directory.mkdir(parents=True, exist_ok=True)

    model = None
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    total = len(ids)
    reused = 0

    for start in range(0, total, shard_size):
        chunk_ids = ids[start : start + shard_size]
        digest = hashlib.sha1(chunk_ids.tobytes()).hexdigest()[:12]
        path = directory / f"{start:08d}_{digest}.npz"

        if path.exists():
            try:
                z = np.load(path)
                if z["ids"].shape[0] == chunk_ids.shape[0] and z["mat"].shape[1] == dim:
                    parts.append((z["ids"], z["mat"]))
                    reused += len(chunk_ids)
                    continue
            except Exception:
                path.unlink(missing_ok=True)

        if model is None:
            model = load_model()
        console.print(
            f"[dim]encoding {start:,}–{min(start + shard_size, total):,} of {total:,}[/dim]"
        )
        mat = encode_texts(
            texts[start : start + shard_size],
            batch_size=batch_size,
            dim=dim,
            model=model,
            show_progress=True,
        )
        # Write to a temporary name first: a shard truncated by the same
        # interruption this exists to survive would be worse than no shard.
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=chunk_ids.astype(np.int32), mat=mat.astype(np.float32))
        tmp.replace(path)
        parts.append((chunk_ids.astype(np.int32), mat.astype(np.float32)))

    if reused:
        console.print(f"[green]reused {reused:,} already-encoded cards[/green]")

    all_ids = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.int32)
    all_mat = (
        np.vstack([p[1] for p in parts]) if parts else np.zeros((0, dim), dtype=np.float32)
    )
    (directory / "manifest.json").write_text(
        json.dumps({"total": int(total), "dim": int(dim), "shard_size": int(shard_size)}),
        encoding="utf-8",
    )
    return all_ids, all_mat


def clear_shards(stem: str = "content") -> int:
    """Delete cached shards. Returns how many were removed."""
    directory = _shard_dir(stem)
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob("*.npz"):
        path.unlink()
        removed += 1
    return removed


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
