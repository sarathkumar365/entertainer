"""Getting a film into the library from the terminal.

The same functions the web app will call (``agent``); this module only turns a
typed title into an IMDb id and prints what happened. On the server it is also
the way to log the Telegram session in, over ssh.
"""

from __future__ import annotations

import asyncio
import sys

import typer

from .. import agent, store
from ..acquire.core import GB
from ..config import acquire_settings
from ..errors import EntertainerError
from ..render import console
from ..resolve import split_year
from ._apps import acquire_app
from ._shared import pick, require_catalog


@acquire_app.command("login")
def acquire_login() -> None:
    """Log the Telegram account in, once. Asks for the phone number and the code."""
    asyncio.run(_login())


async def _login() -> None:
    from ..acquire.telegram import client, session_lock

    settings = acquire_settings()
    with session_lock(settings):
        c = client(settings)

        def phone() -> str:
            # Telegram wants international format; a bare local number fails
            # with PhoneNumberInvalidError after a round trip, so refuse it here.
            while not (p := input("Phone, with country code (e.g. +1..., +91...): ").strip()) \
                    .startswith("+"):
                print("Start with + and the country code.")
            return p.replace(" ", "")

        await c.start(phone=phone)  # the code, and a 2FA password if set, are asked here
        me = await c.get_me()
        await c.disconnect()
        settings.tg_session.chmod(0o600)
    console.print(f"Logged in as {me.username or me.first_name}. Session saved.")


@acquire_app.command("get")
def acquire_get(
    title: str = typer.Argument(None, help="Title, optionally with a year: 'Vikram 2022'."),
    imdb: str = typer.Option(None, help="An IMDb id, for a film the catalogue does not have."),
    sources: str = typer.Option(None, help="Comma-separated sources (default: from .env)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Search and rank; download nothing."),
) -> None:
    """Find a film, download the best copy, verify it and file it for Jellyfin."""
    target = agent.from_imdb(imdb) if imdb else _target_for(title)
    console.print(f"[bold]{target.folder}[/bold]  {target.imdb_id}  "
                  f"{target.runtime_min or '?'} min  {target.language or ''}")
    names = [s.strip() for s in sources.split(",") if s.strip()] if sources else None

    if dry_run:
        p = asyncio.run(agent.search(target, sources=names, log=_log))
        for c in p.candidates[:15]:
            mark = "[green]ok[/green]" if c.score > 0 else "[dim]no[/dim]"
            size = f"{c.size / GB:5.2f} GB" if c.size else "    ? GB"
            console.print(f"{mark} {c.score:5.1f} {size} [{c.source}] {c.label}  [dim]{c.why}[/dim]")
        return

    got = asyncio.run(agent.get(target, sources=names, progress=_bar, log=_log))
    print(file=sys.stderr)
    dur = f"{got.duration_min:.0f} min, " if got.duration_min else ""
    console.print(f"[green]filed[/green] {got.path}  ({dur}from {got.candidate.source})")


def _target_for(title: str | None):
    if not title:
        raise typer.BadParameter("Give a title, or --imdb.")
    require_catalog()
    with store.session() as con:
        match = pick(con, title)
        if match is None:
            raise typer.Exit(code=1)
        # The resolver drops a typed year when nothing matches it, and then
        # confidently returns the film of the same name from another decade.
        # Rating that is recoverable; downloading it is the wrong-target failure.
        _, year = split_year(title)
        if year and match.year and abs(match.year - year) > 1:
            raise EntertainerError(
                f"The catalogue has no {title!r}; the nearest is {match.label()}. "
                "Use --imdb with the film's IMDb id."
            )
        return agent.from_item(con, match.item_id)


def _log(msg: str) -> None:
    print(file=sys.stderr, end="\r")
    console.print(f"[dim]{msg}[/dim]")


def _bar(done: int, total: int) -> None:
    pct = f"{100 * done / total:5.1f}%" if total else ""
    print(f"\r  {done / 2**20:8.1f} MB {pct}", end="", file=sys.stderr, flush=True)
