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


def load_model(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    dev = _device()
    target = name or ENCODER_MODEL
    try:
        model = SentenceTransformer(
            target,
            device=dev,
            model_kwargs={"torch_dtype": "float16"} if dev == "cuda" else {},
            tokenizer_kwargs={"padding_side": "left"},
        )
    except Exception as exc:
        console.print(f"[yellow]{target} unavailable ({exc}); falling back to {ENCODER_FALLBACK}[/yellow]")
        model = SentenceTransformer(ENCODER_FALLBACK, device=dev)
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
