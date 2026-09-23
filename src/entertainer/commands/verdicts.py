"""Saying what you thought.

The engine learns from verdicts and nothing else, so these are the commands
that actually make it better. Everything else reads what they write."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from .. import profile_io, store
from ..config import language_label
from ..engine import Engine
from ..render import console
from ..render import fail as _fail
from ..resolve import resolve_one
from ._apps import app
from ._shared import pick, record, require_catalog


@app.command()
def loved(title: str = typer.Argument(..., help="Title, optionally with a year.")) -> None:
    """Record that you loved something."""
    record(title, "love")


@app.command()
def liked(title: str) -> None:
    """Record that you liked something."""
    record(title, "like")


@app.command()
def meh(title: str) -> None:
    """Record that something left you cold."""
    record(title, "meh")


@app.command()
def disliked(title: str) -> None:
    """Record that you disliked something."""
    record(title, "dislike")


@app.command()
def hated(title: str) -> None:
    """Record that you hated something."""
    record(title, "hate")


@app.command()
def seen(title: str) -> None:
    """Mark something as already watched, with no opinion attached."""
    require_catalog()
    with store.session() as con:
        match = pick(con, title)
        if not match:
            raise typer.Exit(code=1)
        store.log_event(con, match.item_id, "seen", None, "manual")
        console.print(f"[dim]noted as seen[/dim] — {match.label()}")


@app.command()
def dismiss(title: str) -> None:
    """Say you are not interested, without claiming to have watched it.

    Distinct from a verdict: it removes the title from circulation and counts
    as a mild negative at half weight, because declining to watch something is
    real evidence about taste but much weaker than having watched it and
    disliked it.
    """
    require_catalog()
    with store.session() as con:
        match = pick(con, title)
        if not match:
            raise typer.Exit(code=1)
        store.log_event(con, match.item_id, "dismiss", None, "manual")
        console.print(f"[dim]dismissed[/dim] — {match.label()}")


@app.command()
def bulk(
    path: Path = typer.Argument(..., help="Text file: one title per line, optional `| verdict`."),
    default_verdict: str = typer.Option("like", help="Verdict for lines with none given."),
) -> None:
    """Import many titles at once from a plain text file.

    Lines look like `Kumbalangi Nights` or `Morbius | hate`. Unresolvable or
    ambiguous titles are reported at the end rather than guessed at.
    """
    require_catalog()
    engine = Engine()
    unresolved: list[str] = []
    resolved = 0
    entries = profile_io.parse_bulk_lines(
        path.read_text(encoding="utf-8"), default_verdict=default_verdict
    )
    with store.session() as con:
        for entry in entries:
            match, alts = resolve_one(con, entry.title)
            if not match:
                unresolved.append(
                    f"{entry.title}" + (f"  (closest: {alts[0].label()})" if alts else "")
                )
                continue
            engine.record(con, match.item_id, entry.verdict, source="import")
            resolved += 1
    console.print(f"[green]recorded {resolved} verdicts[/green]")
    if unresolved:
        console.print(f"[yellow]{len(unresolved)} unresolved:[/yellow]")
        for line in unresolved:
            console.print(f"  {line}")


@app.command()
def forget(title: str) -> None:
    """Remove every verdict you have given a title, as if it were never rated."""
    require_catalog()
    with store.session() as con:
        match = pick(con, title)
        if not match:
            raise typer.Exit(code=1)
        n = con.execute(
            "SELECT count(*) FROM events WHERE item_id = ?", [match.item_id]
        ).fetchone()[0]
        if not n:
            console.print(f"[yellow]nothing recorded for {match.label()}[/yellow]")
            return
        con.execute("DELETE FROM events WHERE item_id = ?", [match.item_id])
    console.print(f"[green]forgot {n} event(s)[/green] — {match.label()}")


@app.command()
def history(limit: int = typer.Option(30)) -> None:
    """List the verdicts you have given, most recent first."""
    require_catalog()
    with store.session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT e.ts, t.title, t.year, t.language, e.value, e.context
            FROM events e JOIN titles t USING (item_id)
            WHERE e.kind = 'rate' ORDER BY e.ts DESC LIMIT ?
            """,
            [limit],
        ).fetchall()
    table = Table("when", "title", "year", "lang", "verdict")
    for ts, title, year, lang, value, context in rows:
        verdict = (json.loads(context) or {}).get("verdict", f"{value:.0f}")
        table.add_row(str(ts)[:16], title, str(year or ""), lang or "", verdict)
    console.print(table)


@app.command()
def onboard(
    questions: int = typer.Option(40, "--n", help="How many titles to ask about."),
    languages: str = typer.Option("", help="Comma-separated language codes to focus on."),
) -> None:
    """Cold start: answer a short, adaptively chosen set of questions."""

    from ..coldstart import elicit
    from ..coldstart.session import ElicitationSession

    require_catalog()
    engine = Engine()
    langs = tuple(x.strip() for x in languages.split(",") if x.strip())

    with store.session() as con:
        fs = engine.features(con)
        meta = engine.meta(con)
        pool = elicit.recognisable_pool(fs, meta, langs)
        if pool.size == 0:
            _fail("no recognisable titles for those languages")

        asked = store.already_asked(con)
        console.print(
            Panel.fit(
                "[bold]l[/bold]oved   l[bold]i[/bold]ked   [bold]m[/bold]eh   "
                "[bold]d[/bold]isliked   [bold]h[/bold]ated\n"
                "[bold]n[/bold] = haven't seen it   [bold]q[/bold] = stop",
                title="cold start",
                border_style="cyan",
            )
        )
        keymap = {
            "l": "love", "i": "like", "m": "meh", "d": "dislike", "h": "hate",
        }

        session = ElicitationSession(
            fs=fs, meta=meta, pool=pool, asked=asked, languages=langs,
            prior=engine.prior(con), target=questions,
        )
        session.start()

        while not session.finished:
            question = session.next_question()
            if question is None:
                break
            row = question.row
            label = f"[bold]{row['title']}[/bold]"
            if row.get("original_title") and row["original_title"] != row["title"]:
                label += f" [dim]({row['original_title']})[/dim]"
            tail = ", ".join(
                str(x) for x in (row.get("year"), language_label(row.get("language")),
                                 "series" if row.get("kind") == "tv" else None) if x
            )
            console.print(
                f"\n[cyan]{session.answered_count + 1}/{questions}[/cyan]  "
                f"{label}  [dim]{tail}[/dim]"
            )
            try:
                key = typer.prompt("", default="n", show_default=False).strip().lower()[:1]
            except (typer.Abort, EOFError):
                break
            if key == "q":
                break
            if key not in keymap:
                # Not a verdict: record that the question was asked so it is
                # not repeated, without removing the title from circulation.
                store.log_event(
                    con, question.item_id, "unseen", None, "elicit", {"answer": "unseen"}
                )
                continue
            reward = engine.record(con, question.item_id, keymap[key], source="elicit")
            session.record(question.item_id, reward)

        answered_count = session.answered_count
        console.print(f"\n[green]{answered_count} verdicts recorded[/green]")
        if answered_count >= 3:
            engine.fit(con)
            console.print("try [cyan]ent recs[/cyan] or [cyan]ent taste[/cyan]")


# --- recommendations --------------------------------------------------------
