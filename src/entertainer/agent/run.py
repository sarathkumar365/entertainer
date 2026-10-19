"""One request, start to finish: check, search, download, verify, file.

The CLI calls these directly today; the worker thread will call the same
functions in phase 3, so the web app and the CLI stay two front ends over one
code path.

Until the decision rule lands (phase 2) the pick is simply "best score first,
within the size cap". Over-cap copies are refused here rather than offered;
phase 2 turns that refusal into a pause.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..acquire import Candidate, Plan, Target, execute, open_sources, plan
from ..acquire.core import GB
from ..acquire.sources import Progress
from ..config import AcquireSettings, acquire_settings
from ..errors import CannotAcquire, NotAcquired
from . import library

Log = Callable[[str], None]


@dataclass
class Acquired:
    path: Path
    candidate: Candidate
    duration_min: float | None


def ready(settings: AcquireSettings | None = None, sources: list[str] | None = None) -> None:
    """Refuse, naming every missing piece, unless this machine can acquire."""
    s = settings or acquire_settings()
    names = sources or list(s.sources)
    problems = []
    if not s.enabled:
        problems.append("acquisition is switched off (ENTERTAINER_ACQUIRE=off)")
    if s.library_dir is None:
        problems.append("ENTERTAINER_LIBRARY_DIR is not set")
    elif not s.library_dir.is_dir():
        problems.append(f"the library {s.library_dir} does not exist or is not mounted")
    if any(n.startswith("telegram") for n in names):
        if not (s.tg_api_id and s.tg_api_hash):
            problems.append("TELEGRAM_API_ID and TELEGRAM_API_HASH are not set")
        elif not s.tg_session.exists():
            problems.append("Telegram is not logged in (run: ent acquire login)")
    if problems:
        raise CannotAcquire("This machine cannot acquire: " + "; ".join(problems) + ".")


async def search(
    target: Target, settings: AcquireSettings | None = None,
    sources: list[str] | None = None, log: Log = print,
) -> Plan:
    """The planning mode: search and rank, no download. Joins nothing either,
    since joins happen only when a bot is asked for a file."""
    s = settings or acquire_settings()
    async with open_sources(s, sources) as live:
        return await plan(target, live, log)


async def get(
    target: Target, settings: AcquireSettings | None = None, *,
    sources: list[str] | None = None, progress: Progress | None = None, log: Log = print,
) -> Acquired:
    s = settings or acquire_settings()
    ready(s, sources)
    assert s.library_dir is not None and s.staging is not None
    if (have := library.existing(target, s.library_dir)) is not None:
        raise NotAcquired(f"{target.folder} is already in the library: {have}")
    async with open_sources(s, sources) as live:
        p = await plan(target, live, log)
        picks = _within_cap(p, s.max_gb)
        fetched = await execute(
            target, picks, live, s.staging, reserve_gb=s.reserve_gb, progress=progress, log=log
        )
    dest = library.file_into(fetched, target, s.library_dir)
    library.refresh_jellyfin(s, log)
    return Acquired(dest, fetched.candidate, fetched.duration_min)


def _within_cap(p: Plan, max_gb: float) -> list[Candidate]:
    usable = p.usable
    if not usable:
        refused = "; ".join(sorted({c.why for c in p.candidates})[:5]) or "nothing found"
        failed = "".join(f" {n} failed ({e})." for n, e in p.errors.items())
        raise NotAcquired(f"No source had {p.target.folder}. Refused: {refused}.{failed}")
    fits = [c for c in usable if not (c.size and c.size > max_gb * GB)]
    if not fits:
        smallest = min(c.size or 0 for c in usable)
        raise NotAcquired(
            f"Every copy of {p.target.folder} is over the {max_gb:g} GB cap; the smallest is "
            f"{smallest / GB:.2f} GB. Raise ENTERTAINER_ACQUIRE_MAX_GB to take it."
        )
    return fits
