"""Search every source, then try the best matches one at a time until one verifies.

Two steps, not one, because a person may need to step in between them. ``plan``
searches and ranks and touches nothing on disk; ``execute`` downloads the
candidates it is given, in order, and verifies each before handing it back.
Whatever sits between the two — the decision rule, a pause for the user, a
"wrong file, try the next" — belongs to the agent, not to this package.

A file whose duration disagrees with the film's runtime is deleted and the next
candidate is tried: the name said it was the film, the file says otherwise, and
the file wins.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import NotAcquired
from .match import VIDEO_EXTS, Candidate, Target, score
from .sources import Progress, Source
from .verify import verify

GB = 1 << 30

Log = Callable[[str], None]


@dataclass
class Plan:
    target: Target
    candidates: list[Candidate]  # every file seen, scored, best first; refused ones score 0
    errors: dict[str, str] = field(default_factory=dict)  # source name -> why it failed

    @property
    def usable(self) -> list[Candidate]:
        return [c for c in self.candidates if c.score > 0]


@dataclass
class Fetched:
    """A verified download, still in staging."""

    path: Path
    candidate: Candidate
    duration_min: float | None


async def plan(target: Target, sources: dict[str, Source], log: Log = print) -> Plan:
    found: list[Candidate] = []
    errors: dict[str, str] = {}
    for name, src in sources.items():
        try:
            hits = await src.search(target)
        except Exception as e:  # noqa: BLE001 — one source down must not sink the others
            errors[name] = f"{type(e).__name__}: {e}"
            log(f"[{name}] search failed: {errors[name]}")
            continue
        for c in hits:
            c.source = name
            found.append(score(target, c))
    found.sort(key=lambda c: -c.score)
    p = Plan(target, found, errors)
    log(f"{len(found)} files seen, {len(p.usable)} match {target.folder}")
    return p


async def execute(
    target: Target,
    candidates: list[Candidate],
    sources: dict[str, Source],
    staging: Path,
    *,
    reserve_gb: float,
    max_tries: int = 3,
    progress: Progress | None = None,
    log: Log = print,
) -> Fetched:
    if not candidates:
        raise NotAcquired(f"No candidate to try for {target.folder}.")
    staging.mkdir(parents=True, exist_ok=True)
    tried, no_room = 0, []
    for cand in candidates:
        if tried >= max_tries:
            break
        if not _fits(staging, cand.size, reserve_gb):
            # A smaller copy further down may still fit; skipping costs nothing.
            no_room.append(cand)
            log(f"skipping [{cand.source}] {cand.label}: {_gb(cand.size)} does not fit")
            continue
        tried += 1
        part = staging / f"{target.folder}.part"
        log(f"trying [{cand.source}] {cand.label} ({_gb(cand.size)}, score {cand.score:.1f})")
        try:
            await sources[cand.source].fetch(cand, part, progress)
        except Exception as e:  # noqa: BLE001 — any source failure means try the next
            log(f"  download failed: {e}")
            part.unlink(missing_ok=True)
            continue
        if cand.ext and cand.ext not in VIDEO_EXTS:
            log(f"  rejected after download: .{cand.ext} is not a video")
            part.unlink(missing_ok=True)
            continue
        ok, dur, why = verify(part, target.runtime_min)
        if not ok:
            log(f"  rejected after download: {why}")
            part.unlink(missing_ok=True)
            continue
        return Fetched(part, cand, dur)
    if no_room and not tried:
        free = shutil.disk_usage(staging).free
        raise NotAcquired(
            f"Not enough disk for {target.folder}: the smallest copy is "
            f"{_gb(min(c.size or 0 for c in no_room))}, {_gb(free)} free, {reserve_gb:g} GB kept "
            "in reserve. Free space or lower ENTERTAINER_ACQUIRE_RESERVE_GB."
        )
    raise NotAcquired(f"Every candidate for {target.folder} failed to download or verify.")


def _fits(where: Path, size: int | None, reserve_gb: float) -> bool:
    return not size or shutil.disk_usage(where).free - size >= reserve_gb * GB


def _gb(n: int | None) -> str:
    return f"{n / GB:.2f} GB" if n else "? GB"
