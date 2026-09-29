"""Spike entry point: ``python -m entertainer.acquire get "Title" --year 1994``.

Resolves the title through TMDB first, so every source is matched against the
same canonical title, year and runtime rather than against what was typed.
Not a registered ``ent`` verb yet — that waits on the spike being accepted.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from ..config import PRIORITY_LANGUAGES as _LANGUAGES
from ..errors import EntertainerError
from .core import acquire, library_dir, rank
from .match import Target

SOURCES = ("telegram-bot", "telegram", "archive")


def resolve(title: str, year: int | None) -> Target:
    from ..data import tmdb

    hits = tmdb.search(title, year, "movie")
    if not hits:
        raise EntertainerError(f"TMDB has no film called {title!r}. Check the spelling or year.")
    d = tmdb.detail(hits[0]["id"], "movie") or {}
    date = d.get("release_date") or ""
    alt = d.get("original_title")
    lang = _LANGUAGES.get(d.get("original_language") or "")
    return Target(
        title=d.get("title") or title,
        year=int(date[:4]) if date[:4].isdigit() else year,
        runtime_min=d.get("runtime") or None,
        alt_titles=(alt,) if alt and alt != d.get("title") else (),
        imdb_id=d.get("imdb_id"),
        language=lang,
    )


def build_sources(names: list[str]):
    out = []
    for n in names:
        if n == "telegram":
            from .telegram import TelegramSource

            out.append(TelegramSource())
        elif n == "telegram-bot":
            from .telegram import BotGroupSource

            out.append(BotGroupSource())
        elif n == "archive":
            from .archive import ArchiveSource

            out.append(ArchiveSource())
    return out


def _bar(done: int, total: int) -> None:
    pct = f"{100 * done / total:5.1f}%" if total else ""
    print(f"\r  {done / 2**20:8.1f} MB {pct}", end="", file=sys.stderr, flush=True)


async def _login() -> None:
    from .telegram import client

    c = client()

    def phone() -> str:
        # Telegram wants international format; a bare local number fails with
        # PhoneNumberInvalidError after a round trip, so refuse it here.
        while not (p := input("Phone, with country code (e.g. +1..., +91...): ").strip()) \
                .startswith("+"):
            print("Start with + and the country code.")
        return p.replace(" ", "")

    await c.start(phone=phone)  # code (and 2FA password, if set) prompted on this terminal
    me = await c.get_me()
    print(f"Logged in as {me.username or me.first_name}. Session saved.")
    await c.disconnect()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m entertainer.acquire")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login", help="log the Telegram user client in, once")
    g = sub.add_parser("get", help="find and download a film")
    g.add_argument("title")
    g.add_argument("--year", type=int)
    g.add_argument("--sources", default=",".join(SOURCES))
    g.add_argument("--dry-run", action="store_true", help="rank candidates, download nothing")
    g.add_argument("--reserve-gb", type=float, default=5.0)
    g.add_argument("--max-gb", type=float, help="skip files larger than this")
    a = p.parse_args(argv)

    try:
        if a.cmd == "login":
            asyncio.run(_login())
            return 0
        target = resolve(a.title, a.year)
        print(f"target: {target}")
        sources = build_sources([s.strip() for s in a.sources.split(",") if s.strip()])
        if a.dry_run:
            for c in asyncio.run(rank(target, sources))[:15]:
                mark = "ok " if c.score > 0 else "no "
                print(f"{mark}{c.score:5.1f} [{c.source}] {c.label}  — {c.why}")
            return 0
        got = asyncio.run(acquire(target, sources, reserve_gb=a.reserve_gb, max_gb=a.max_gb, progress=_bar))
        print(f"\nacquired: {got.path}  ({got.duration_min:.0f} min, from {got.candidate.source})")
        print(f"library:  {library_dir()}")
        return 0
    except EntertainerError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
