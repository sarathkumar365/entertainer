"""Helpers used by more than one command group.

These sit in the command layer rather than in a library module because each
one talks to a person: `pick` prompts, `record` prints, and both raise
``typer.Exit``. The resolution and recording underneath them live in
``resolve`` and ``engine``, where they can be called without a terminal.
"""

from __future__ import annotations

import typer

from .. import pipeline, store
from ..engine import Engine
from ..render import console
from ..render import fail as _fail
from ..resolve import Match, resolve_one
from ..resolve import from_slate as resolve_from_slate

#: Verdict counts worth a nudge. The engine refits on every command, so there
#: is no threshold at which it "starts working" — these are the points where
#: saying so is useful rather than noise.
MILESTONES = (3, 10, 25, 50, 100)


def require_catalog() -> None:
    if not pipeline.catalogue_exists():
        _fail("no catalogue yet — run `ent setup` first")


def pick(con, query: str, kind: str | None = None) -> Match | None:
    """Resolve a typed title, asking the user only when genuinely ambiguous."""
    from_slate = resolve_from_slate(con, query)
    if from_slate is not None:
        return from_slate

    match, alternatives = resolve_one(con, query, kind=kind)
    if match:
        return match
    if not alternatives:
        console.print(f"[yellow]nothing matching {query!r} in the catalogue[/yellow]")
        return None

    console.print(f"[bold]which one?[/bold] ({query!r})")
    for i, alt in enumerate(alternatives, start=1):
        console.print(f"  [cyan]{i}[/cyan]  {alt.label()}")
    console.print("  [cyan]0[/cyan]  none of these")
    try:
        choice = typer.prompt("number", type=int, default=1)
    except (typer.Abort, EOFError):
        return None
    if choice <= 0 or choice > len(alternatives):
        return None
    return alternatives[choice - 1]


def record(query: str, verdict: str, kind: str | None = None) -> None:
    require_catalog()
    engine = Engine()
    with store.session() as con:
        match = pick(con, query, kind=kind)
        if not match:
            raise typer.Exit(code=1)
        engine.record(con, match.item_id, verdict)
        console.print(f"[green]{verdict}[/green] — {match.label()}")
        n = len(store.ratings(con))
        if n in MILESTONES:
            console.print(f"[dim]{n} verdicts recorded; model refits on every command[/dim]")
