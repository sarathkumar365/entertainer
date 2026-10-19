"""The one shape every source has, and the only place sources are built.

Where a title comes from is a configuration choice (``ENTERTAINER_ACQUIRE_SOURCES``),
not something inherited from whichever library was convenient. Every source is
opened here, so the Telegram ones share a single client and a single session
lock however many of them a job uses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Protocol

from ..config import AcquireSettings
from ..errors import EntertainerError
from .match import Candidate, Target

Progress = Callable[[int, int], None]

KNOWN = ("telegram-bot", "telegram", "archive")


class Source(Protocol):
    name: str

    async def __aenter__(self) -> Source: ...

    async def __aexit__(self, *exc) -> None: ...

    async def search(self, target: Target) -> list[Candidate]: ...

    async def fetch(self, cand: Candidate, dest: Path, progress: Progress | None = None) -> Path:
        """Download to ``dest``. May set ``cand.ext`` once the real file is known."""
        ...


@asynccontextmanager
async def open_sources(
    settings: AcquireSettings, names: tuple[str, ...] | list[str] | None = None
) -> AsyncIterator[dict[str, Source]]:
    names = list(names or settings.sources)
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise EntertainerError(
            f"Unknown source {', '.join(unknown)}. Choose from {', '.join(KNOWN)}."
        )
    async with AsyncExitStack() as stack:
        tg = None
        if any(n.startswith("telegram") for n in names):
            from .telegram import connect

            tg = await stack.enter_async_context(connect(settings))
        live: dict[str, Source] = {}
        for n in names:
            live[n] = await stack.enter_async_context(_make(n, settings, tg))
        yield live


def _make(name: str, settings: AcquireSettings, tg) -> Source:
    if name == "telegram":
        from .telegram import TelegramSource

        return TelegramSource(tg, settings.tg_channels)
    if name == "telegram-bot":
        from .telegram import BotGroupSource

        return BotGroupSource(tg, settings.tg_groups)
    from .archive import ArchiveSource

    return ArchiveSource()
