"""Publishing a built catalogue on one machine and pulling it on another."""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from .. import releases
from ..config import PATHS
from ..errors import EntertainerError
from ..render import console
from ..render import fail as _fail
from ._apps import app, release_app
from ._shared import require_catalog

_REPO_HELP = f"Private repository holding the builds (default: ${releases.REPO_ENV} or {releases.DEFAULT_REPO})."


@release_app.command("publish")
def release_publish(
    repo: str = typer.Option(None, help=_REPO_HELP),
    tag: str = typer.Option(None, help="Release tag (default: build-YYYYMMDD-HHMM, UTC)."),
    notes: str = typer.Option(None, help="Release notes (default: a summary of the bundle)."),
) -> None:
    """Export the full catalogue, encodings included, as a release on the builds repo."""
    require_catalog()
    target = releases.releases_repo(repo)
    try:
        with console.status(f"exporting and uploading to {target}..."):
            done = releases.publish(target, tag=tag, notes=notes)
    except EntertainerError as exc:
        _fail(str(exc))
    m = done.info.manifest
    console.print(
        Panel.fit(
            f"[bold]{done.repo}[/bold] · {done.tag}\n"
            f"{done.info.bytes / 1e6:.1f} MB · {m['titles']:,} titles · build {m['build_id']}\n"
            f"[dim]{', '.join(m['contents'])}[/dim]\n\n"
            "on another machine:  [cyan]./scripts/entertainer pull[/cyan]",
            title="published",
            border_style="cyan",
        )
    )


@release_app.command("list")
def release_list(repo: str = typer.Option(None, help=_REPO_HELP)) -> None:
    """Show the published catalogues, newest first."""
    target = releases.releases_repo(repo)
    try:
        found = releases.list_releases(target)
    except EntertainerError as exc:
        _fail(str(exc))
    if not found:
        console.print(f"[dim]no releases in {target}[/dim]")
        return
    table = Table("tag", "published", "size", title=target)
    for rel in found:
        size = sum(a.get("size", 0) for a in rel.get("assets", []))
        table.add_row(rel.get("tag_name", "?"), (rel.get("published_at") or "")[:16], f"{size / 1e6:.1f} MB")
    console.print(table)


@app.command("pull")
def pull(
    repo: str = typer.Option(None, help=_REPO_HELP),
    tag: str = typer.Option("latest", help="Release to pull."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Replace the local catalogue without asking."),
    force: bool = typer.Option(False, help="Import even if this build is already here."),
) -> None:
    """Replace the local catalogue with a published build, keeping your verdicts.

    Your events are moved onto the new catalogue by IMDb id; if any of them
    names a title the new build lacks, nothing is changed.
    """
    target = releases.releases_repo(repo)
    PATHS.ensure()
    try:
        releases.require_app_stopped()
        with tempfile.TemporaryDirectory(dir=PATHS.root, prefix=".pull-") as work:
            plan = releases.plan_pull(target, tag, Path(work))
            m = plan.manifest
            table = Table("", "this machine", plan.tag, show_header=True)
            table.add_row("build", plan.local_build or "[dim]unknown[/dim]", str(m.get("build_id")))
            table.add_row("created", "", str(m.get("created_at", "?")))
            table.add_row("titles", f"{plan.local_titles:,}", f"{m.get('titles', 0):,}")
            table.add_row("your events", f"{plan.local_events:,}", "[dim]kept, remapped by IMDb id[/dim]")
            console.print(table)
            for warning in plan.warnings:
                console.print(f"[yellow]{warning}[/yellow]")
            if plan.up_to_date and not force:
                console.print("[green]already up to date[/green]")
                return
            if (plan.local_titles or plan.local_events) and not yes:
                typer.confirm("Replace the local catalogue?", abort=True)
            with console.status("downloading and importing..."):
                result = releases.apply_pull(plan, Path(work))
    except EntertainerError as exc:
        _fail(str(exc))
    console.print(
        f"[green]{result['restored_titles']:,} titles from {plan.tag}[/green]"
        + (f", [dim]{result['preserved_events']} events kept[/dim]" if result.get("preserved_events") else "")
    )


@release_app.command("status")
def release_status(repo: str = typer.Option(None, help=_REPO_HELP)) -> None:
    """Show whether this machine is set up to build, publish and pull."""
    from .. import setup_status

    report = setup_status.collect(repo)
    marks = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]",
             "missing": "[red]✗[/red]", "unknown": "[dim]?[/dim]"}
    console.print(f"[dim]{report['repo']} · this machine is a {report['role']} "
                  f"(set ENTERTAINER_ROLE to change)[/dim]")
    table = Table(show_header=False, box=None, pad_edge=False)
    for check in report["checks"]:
        table.add_row(marks[check["status"]], check["label"], check["detail"])
    console.print(table)
    fixes = [c for c in report["checks"] if c["fix"]]
    if fixes:
        console.print("\n[bold]to fix[/bold]")
        for check in fixes:
            console.print(f"[dim]{check['label']}:[/dim]")
            console.print(check["fix"], markup=False, soft_wrap=True)
