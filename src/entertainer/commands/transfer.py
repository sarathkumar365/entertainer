"""Moving verdicts and catalogues between machines.

A catalogue takes hours to build and is identical for everyone; a verdict
history is small, personal, and the only thing that cannot be rebuilt."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from .. import profile_io, store
from ..engine import Engine
from ..errors import EntertainerError
from ..render import console
from ..render import fail as _fail
from ._apps import app, bundle_app
from ._shared import require_catalog


@bundle_app.command("export")
def bundle_export(
    path: Path = typer.Option(Path("entertainer-bundle.zip")),
    space: bool = typer.Option(True, help="Include the item space (needed for `ent recs`)."),
    encodings: bool = typer.Option(
        False, help="Include raw content embeddings (only `ent add` needs them)."
    ),
) -> None:
    """Pack the built catalogue so another machine can use it without rebuilding.

    Four hours of TMDB round trips become a file. Verdicts are deliberately
    not included — use `ent export` for those.
    """
    from .. import bundle

    require_catalog()
    info = bundle.export(path, include_space=space, include_encodings=encodings)
    console.print(
        Panel.fit(
            f"[bold]{info.path}[/bold]\n"
            f"{info.bytes / 1e6:.1f} MB · {info.titles:,} titles\n"
            f"[dim]{', '.join(info.contents)}[/dim]\n\n"
            "on the other machine:\n"
            "  [cyan]git clone <repo> && uv pip install -e '.[web]'[/cyan]\n"
            f"  [cyan]ent bundle import {info.path.name}[/cyan]\n"
            "  [cyan]ent rate[/cyan]",
            title="bundle written",
            border_style="cyan",
        )
    )
    console.print(
        "[yellow]This is IMDb and TMDB derived data. Keep it private — "
        "their terms do not permit redistributing it publicly.[/yellow]"
    )


@bundle_app.command("info")
def bundle_info(path: Path = typer.Argument(...)) -> None:
    """Show what a bundle contains without unpacking it."""
    from .. import bundle

    data = bundle.inspect(path)
    table = Table("field", "value")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


@bundle_app.command("import")
def bundle_import(
    path: Path = typer.Argument(...),
    overwrite: bool = typer.Option(
        False, help="Replace an existing catalogue even if this machine has verdicts."
    ),
) -> None:
    """Unpack a bundle written by `ent bundle export`."""
    from .. import bundle

    try:
        manifest = bundle.restore(path, overwrite=overwrite)
    except RuntimeError as exc:
        _fail(str(exc))
    for warning in manifest.get("warnings", []):
        console.print(f"[yellow]{warning}[/yellow]")
    console.print(
        f"[green]{manifest['restored_titles']:,} titles restored[/green]"
        + (
            f", [dim]{manifest['preserved_events']} existing events kept[/dim]"
            if manifest.get("preserved_events")
            else ""
        )
    )
    console.print(f"[dim]contents: {', '.join(manifest.get('contents', []))}[/dim]")
    console.print("\n[bold]next:[/bold] [cyan]ent rate[/cyan]")


@app.command("export")
def export_profile(path: Path = typer.Option(Path("profile.jsonl"))) -> None:
    """Export your verdicts so the catalogue can be rebuilt without losing them.

    The default filename is gitignored, because this repository is public and
    the export is a complete record of what you watch.
    """
    require_catalog()
    with store.session(read_only=True) as con:
        n = profile_io.export_events(con, path)
    console.print(f"[green]{n:,} events -> {path}[/green]")
    console.print("[dim]this file is your viewing history; keep it out of public repos[/dim]")


@app.command("import")
def import_profile(path: Path = typer.Argument(...)) -> None:
    """Re-import an exported profile, matching on IMDb id."""
    require_catalog()
    with store.session() as con:
        try:
            counts = profile_io.import_events(con, path)
        except EntertainerError as exc:
            _fail(str(exc))
    message = f"[green]imported {counts.imported:,}[/green]"
    if counts.duplicates:
        message += f", [dim]{counts.duplicates} already present[/dim]"
    if counts.missing:
        message += f", [yellow]{counts.missing} not in catalogue[/yellow]"
    console.print(message)


@app.command("netflix")
def netflix_import(
    paths: list[Path] = typer.Argument(..., help="Netflix jsonGraph pages, in any order."),
    apply: bool = typer.Option(False, "--apply", help="Write the confident matches."),
    review: Path = typer.Option(Path("netflix-review.json"), help="Where to put everything else."),
    decisions: Path = typer.Option(None, help="Rulings for the ambiguous titles, keyed by Netflix id."),
    place: bool = typer.Option(True, help="Embed new titles into the item space (needs the encoder)."),
    replace: bool = typer.Option(False, help="Clear previously imported Netflix verdicts first."),
) -> None:
    """Import a Netflix thumbs history, resolving titles against TMDB.

    Netflix's internal `movieID` maps to nothing public, so titles have to be
    matched by text. That is ambiguous, and attaching a verdict to the wrong
    film is worse than not importing it, so only unambiguous matches are
    written. The rest land in a review file with their candidates.

    Runs as a dry run unless `--apply` is given.
    """
    from ..data import netflix
    from ..ingest import IngestError, add_title

    require_catalog()

    ratings: list[netflix.Rating] = []
    for path in paths:
        ratings.extend(netflix.parse(path))
    if not ratings:
        _fail("no rating items found in those files")

    unique = netflix.dedupe(ratings)
    if len(unique) != len(ratings):
        console.print(f"[dim]{len(ratings) - len(unique)} duplicate rows dropped[/dim]")

    with console.status(f"resolving {len(unique)} titles against TMDB..."):
        resolutions = netflix.flag_collisions(netflix.resolve(unique))
    if decisions:
        resolutions = netflix.apply_decisions(
            resolutions, json.loads(decisions.read_text(encoding="utf-8"))
        )

    buckets = netflix.bucket(resolutions)

    table = Table(title="Netflix import")
    table.add_column("confidence")
    table.add_column("n", justify="right")
    table.add_column("what happens")
    for name, note in (
        ("high", "written" if apply else "would be written"),
        ("ambiguous", "sent to review — several films share the title"),
        ("low", "sent to review — no exact title match"),
        ("unresolvable", "skipped — Netflix exported no title"),
        ("skipped", "skipped — ruled out by hand"),
    ):
        if buckets.get(name):
            table.add_row(name, str(len(buckets[name])), note)
    console.print(table)

    needs_eyes = [r for r in resolutions if r.confidence not in ("high", "skipped")]
    if needs_eyes:
        review.write_text(
            json.dumps(netflix.review_payload(needs_eyes), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[yellow]{len(needs_eyes)} need a human -> {review}[/yellow]")

    confident = buckets.get("high", [])
    if not apply:
        for res in confident[:10]:
            m = res.match or {}
            console.print(
                f"  [dim]{res.rating.verdict:<7}[/dim] {m.get('title')} "
                f"({m.get('year')}) [dim]{m.get('language')}[/dim]"
            )
        if len(confident) > 10:
            console.print(f"  [dim]... and {len(confident) - 10} more[/dim]")
        console.print("\n[bold]dry run.[/bold] re-run with [cyan]--apply[/cyan] to write these.")
        return

    if replace:
        with store.session() as con:
            cleared = netflix.clear_previous(con)
        console.print(f"[dim]cleared {cleared} previously imported Netflix verdicts[/dim]")

    engine = Engine()
    written, failed = 0, 0
    for res in confident:
        m = res.match or {}
        try:
            item_id, _ = add_title(engine, int(m["tmdb_id"]), m["kind"], place=place)
        except (IngestError, KeyError, TypeError) as exc:
            console.print(f"[yellow]skipped {res.rating.title}: {exc}[/yellow]")
            failed += 1
            continue
        with store.session() as con:
            engine.record(
                con, item_id, res.rating.verdict, source="netflix",
                context={"netflix_id": res.rating.netflix_id},
                ts=str(res.rating.rated_on) if res.rating.rated_on else None,
            )
        written += 1

    console.print(f"[green]wrote {written} verdicts[/green]" + (f", [yellow]{failed} failed[/yellow]" if failed else ""))
    console.print("\n[bold]next:[/bold] [cyan]ent taste[/cyan] then [cyan]ent audit[/cyan]")
