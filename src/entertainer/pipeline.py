"""Where the build has got to, and what to do next.

The preflight checks live here too, kept apart from the expensive stages so
that a full build can fail in the first second rather than the third hour.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from .config import PATHS, has_tmdb

MIN_FREE_BYTES = 10 * 1024**3

#: Below this share of the catalogue embedded into the item space, the gap is
#: worth warning about: anything outside the space cannot be recommended at
#: all, so a silent 3% shortfall is 2,000 titles that simply never appear.
STALE_COVERAGE = 0.995

#: Verdicts needed before recommendations mean anything, and before the audit
#: can say anything honest. Mirrors taste.fit and prequential respectively.
MIN_FOR_RECS = 3
MIN_FOR_AUDIT = 8


def preflight(require_tmdb: bool = True) -> dict[str, int | bool | str]:
    """Fail early when a local full build cannot complete safely."""
    PATHS.ensure()
    free = shutil.disk_usage(PATHS.root).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"{free / 1024**3:.1f} GiB free at {PATHS.root}; the local build needs "
            "at least 10 GiB. Raw IMDb downloads can be deleted after a successful build."
        )
    if require_tmdb and not has_tmdb():
        raise RuntimeError(
            "no usable TMDB credential: set an ASCII TMDB_API_KEY or TMDB_BEARER in .env"
        )
    return {"data_dir": str(PATHS.root), "free_bytes": free, "tmdb": has_tmdb()}


@dataclass
class Step:
    """What to run next, and why.

    ``note`` is the aside the CLI prints in dim text. It is separate from the
    command so that a caller which is not a terminal can use one without the
    other — the previous version returned a single string with rich markup
    embedded in it, which made this function unusable anywhere else.
    """

    command: str
    note: str = ""


def catalogue_exists() -> bool:
    return PATHS.catalog_db.exists()


def next_step(present: dict[str, bool], n_ratings: int) -> Step:
    """The first unmet dependency, in build order.

    A pipeline with eight stages and a three-hour critical path needs to be
    able to say where it got to.
    """
    if not present["catalog"]:
        return Step("ent setup")
    if not present["content_embeddings"]:
        return Step("ent data embed")
    if not present["cf_factors"]:
        return Step("ent data cf")
    if not present["fused_space"]:
        return Step("ent data fuse")
    if n_ratings < MIN_FOR_RECS:
        return Step("ent onboard", "or: ent bulk seed.example.txt")
    if n_ratings < MIN_FOR_AUDIT:
        return Step("ent recs", "a few more verdicts and `ent audit` will work too")
    return Step("ent recs")


def coverage_is_stale(coverage: dict[str, int]) -> float | None:
    """The embedded share, when it is low enough to be worth reporting.

    Returns None when coverage is fine or cannot be computed, so the caller
    does not repeat the threshold comparison.
    """
    if not coverage.get("catalog") :
        return None
    share = coverage["fused"] / coverage["catalog"]
    return share if share < STALE_COVERAGE else None
