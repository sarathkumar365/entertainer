"""One request, start to finish: check, search, download, verify, file.

The CLI calls these directly today; the worker thread will call the same
functions in phase 3, so the web app and the CLI stay two front ends over one
code path.

When the decision rule pauses, ``get`` asks ``choose``: the CLI prompts, and
the worker (phase 3) will park the job until the web page answers. With no one
to ask, a pause is a refusal that says why.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..acquire import Candidate, Plan, Target, execute, open_sources, plan
from ..acquire.core import GB
from ..acquire.sources import Progress
from ..config import AcquireSettings, acquire_settings
from ..errors import CannotAcquire, NotAcquired
from . import library
from .decide import Decision, decide

Log = Callable[[str], None]
# Shown a paused decision; returns the copy to take, or None to give up.
Choose = Callable[[Decision], Candidate | None]
# Told what happened, as a kind and its fields, so a job record can keep it.
Report = Callable[..., None]


class Paused(Exception):  # noqa: N818 — a state the job enters, not a fault
    """Raised by a ``choose`` that cannot answer now: the job waits for the user."""

    def __init__(self, decision: Decision):
        super().__init__(decision.reason)
        self.decision = decision


def describe(c: Candidate) -> dict[str, Any]:
    """A candidate as plain data, without the source's live reference."""
    # Not asdict: it deep-copies the reference, which for Telegram is a live message.
    return {f.name: getattr(c, f.name) for f in fields(c) if f.name != "ref"}


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
    choose: Choose | None = None, exclude: frozenset[str] = frozenset(),
    chosen: dict | None = None, report: Report | None = None,
) -> Acquired:
    """``exclude`` holds labels already filed and called wrong; ``chosen`` is a
    copy the user picked at an earlier pause, found again by source and label
    because a bot's buttons do not outlive the search that showed them;
    ``report`` hears the plan, the pick and every join, for the job record."""
    s = settings or acquire_settings()
    tell = report or (lambda kind, **_: None)
    ready(s, sources)
    assert s.library_dir is not None and s.staging is not None
    if (have := library.existing(target, s.library_dir)) is not None:
        raise NotAcquired(f"{target.folder} is already in the library: {have}")
    on_join = (lambda link, chat: tell("join", link=link, chat=chat)) if report else None
    async with open_sources(s, sources, on_join=on_join) as live:
        p = await plan(target, live, log)
        tell("plan", candidates=[describe(c) for c in p.candidates], errors=p.errors)
        if exclude:
            for c in p.candidates:
                if c.label in exclude:
                    c.score, c.why = 0.0, "called the wrong file earlier"
            p.candidates.sort(key=lambda c: -c.score)
        picks = _again(p, chosen) if chosen else _picks(decide(p, s.max_gb), target, choose, log)
        tell("pick", candidates=[describe(c) for c in picks])
        # A copy the user accepted over the cap is theirs to take; the disk check still holds.
        cap = None if any(c.size and c.size > s.max_gb * GB for c in picks) \
            else int(s.max_gb * GB)
        fetched = await execute(
            target, picks, live, s.staging, reserve_gb=s.reserve_gb, max_bytes=cap,
            progress=progress, log=log,
        )
    dest = library.file_into(fetched, target, s.library_dir)
    library.refresh_jellyfin(s, log)
    return Acquired(dest, fetched.candidate, fetched.duration_min)


def _again(p: Plan, chosen: dict) -> list[Candidate]:
    for c in p.candidates:
        if (c.source, c.label) == (chosen.get("source"), chosen.get("label")):
            return [c]
    raise NotAcquired(
        f"The copy chosen for {p.target.folder} ({chosen.get('label')}) is no longer offered. "
        "Ask for the film again to see what is there now."
    )


def _picks(d: Decision, target: Target, choose: Choose | None, log: Log) -> list[Candidate]:
    if not d.paused:
        log(f"picked: {d.reason}")
        return d.picks
    if choose is None:
        raise NotAcquired(f"{target.folder} needs a choice: {d.reason}.")
    chosen = choose(d)
    if chosen is None:
        raise NotAcquired(f"{target.folder}: no copy chosen.")
    # The user picked this file; if it fails to verify, guessing another is not ours to do.
    return [chosen]
