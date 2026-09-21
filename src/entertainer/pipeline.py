"""Build preflight checks kept separate from the expensive pipeline stages."""

from __future__ import annotations

import shutil

from .config import PATHS, has_tmdb

MIN_FREE_BYTES = 10 * 1024**3


def preflight() -> dict[str, int | bool | str]:
    """Fail early when a local full build cannot complete safely."""
    PATHS.ensure()
    free = shutil.disk_usage(PATHS.root).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"{free / 1024**3:.1f} GiB free at {PATHS.root}; the local build needs "
            "at least 10 GiB. Raw IMDb downloads can be deleted after a successful build."
        )
    if not has_tmdb():
        raise RuntimeError(
            "no usable TMDB credential: set an ASCII TMDB_API_KEY or TMDB_BEARER in .env"
        )
    return {"data_dir": str(PATHS.root), "free_bytes": free, "tmdb": True}
