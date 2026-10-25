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
from ..acquire import Candidate
from ..acquire.core import GB
from ..agent import worker
from ..agent.jobs import FILED, QUEUED, Jobs
from ..config import acquire_settings
from ..errors import AcquireBusy, EntertainerError
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
    """Find a film, download the best copy, verify it and file it for Jellyfin.

    Every run is a job, as on the web page, so `ent acquire jobs` and the
    Downloads page show it, and a dry run's candidate list becomes an eval case.
    """
    names = [s.strip() for s in sources.split(",") if s.strip()] if sources else None
    request = {"imdb_id": imdb} if imdb else _request_for(title)
    request.update(via="cli", sources=names, dry_run=dry_run)
    jobs = Jobs()

    if dry_run:
        job, p = worker.search_only(jobs, jobs.create(request, run_here=True).id, log=_log)
        console.print(f"[bold]{job.folder}[/bold]  {job.imdb_id}  [dim]{job.id}[/dim]")
        for i, c in enumerate(p.candidates[:15], 1):
            mark = "[green]ok[/green]" if c.score > 0 else "[dim]no[/dim]"
            console.print(f"{i:>2}. {mark} {c.score:5.1f} {_size(c)} [{c.source}] {c.label}  "
                          f"[dim]{c.why}[/dim]")
        if p.usable:
            d = agent.decide(p, acquire_settings().max_gb)
            console.print(f"[bold]{'would pause' if d.paused else 'would take'}[/bold]: "
                          f"{d.reason}")
        console.print(f"[dim]If one of these is the right file: ent acquire confirm {job.id} N[/dim]")
        return

    agent.ready(sources=names)
    job = worker.run_one(jobs, jobs.create(request, run_here=True).id, choose=_choose,
                         progress=_bar, log=_log)
    print(file=sys.stderr)
    if job.state != FILED:
        raise EntertainerError(f"{job.folder or 'The film'}: {job.reason} (job {job.id})")
    console.print(f"[green]filed[/green] {job.filed['path']}  (from {job.filed['source']})")


@acquire_app.command("jobs")
def acquire_jobs(limit: int = typer.Option(15, help="How many, newest first.")) -> None:
    """Recent download jobs, from the web page and the terminal alike."""
    shown = [j for j in Jobs().all() if not j.dry_run][:limit]
    if not shown:
        console.print("No download jobs yet.")
    for j in shown:
        console.print(f"{j.id}  [bold]{j.state:<11}[/bold] {j.folder or j.request}  "
                      f"[dim]{j.reason}[/dim]")
        for i, c in enumerate(j.offer, 1):
            console.print(f"    {i}. {_size_of(c.get('size'))} [{c['source']}] {c['label']}")


@acquire_app.command("choose")
def acquire_choose(job_id: str, number: int = typer.Argument(..., help="As listed, from 1.")) -> None:
    """Answer a paused job with one of the copies it offered."""
    job = worker.Worker().choose(job_id, number - 1)
    console.print(f"{job.id} queued with {job.choice['label']}. Run `ent acquire run`, or let "
                  "the app pick it up.")


@acquire_app.command("wrong")
def acquire_wrong(job_id: str) -> None:
    """The filed copy is not the film: delete it and try the next copy."""
    job = worker.Worker().wrong(job_id)
    console.print(f"Deleted. {job.id} is queued again without {job.wrong[-1]}.")


@acquire_app.command("cancel")
def acquire_cancel(job_id: str) -> None:
    """Cancel a queued or paused job."""
    console.print(f"{worker.Worker().cancel(job_id).id} cancelled.")


@acquire_app.command("check")
def acquire_check() -> None:
    """Whether this machine can acquire, and whether Jellyfin will hear about new films."""
    from ..agent.library import check_jellyfin

    settings = acquire_settings()
    try:
        agent.ready(settings)
        console.print("[green]ready[/green] to acquire into " + str(settings.library_dir))
    except EntertainerError as e:
        console.print(f"[red]not ready[/red]: {e}")
    ok, message = check_jellyfin(settings)
    console.print(f"[{'green' if ok else 'yellow'}]{message}[/]")


@acquire_app.command("confirm")
def acquire_confirm(
    job_id: str, number: int = typer.Argument(..., help="As the dry run listed it, from 1."),
) -> None:
    """Name the right file in a dry run's list, making it a labelled replay case."""
    jobs = Jobs()
    job = jobs.get(job_id)
    if not 1 <= number <= len(job.candidates):
        raise EntertainerError(f"{job_id} listed {len(job.candidates)} files; choose one of them.")
    label = job.candidates[number - 1]["label"]
    jobs.append(job_id, "confirm", label=label)
    console.print(f"{job.folder}: the right file is {label}")


@acquire_app.command("replay")
def acquire_replay(
    save: bool = typer.Option(False, "--save", help="Make this run the baseline to beat."),
) -> None:
    """Re-run matching and the decision rule over every recorded job, offline.

    Exits non-zero when a change picks more wrong files than the saved baseline:
    that is the merge gate for matcher changes.
    """
    from ..agent import replay as R

    r = R.replay()
    for c in r.cases:
        if c.outcome != "right":
            console.print(f"{c.outcome:<10} {c.folder}  [dim]{c.picked or c.why}[/dim]")
    t = r.totals
    console.print(f"[bold]{len(r.cases)} cases[/bold]: {t['right']} right, {t['wrong']} wrong, "
                  f"{t['paused']} paused, {t['unlabelled']} unlabelled, {t['nothing']} nothing")
    base = R.load_baseline()
    if save:
        console.print(f"baseline saved to {R.save_baseline(r)}")
    elif base and t["wrong"] > base["totals"]["wrong"]:
        raise EntertainerError(
            f"{t['wrong']} wrong picks against {base['totals']['wrong']} in the baseline. "
            "The change makes matching worse; it does not merge."
        )
    elif base:
        console.print(f"baseline: {base['totals']['wrong']} wrong, "
                      f"{base['totals']['paused']} paused")


@acquire_app.command("leave")
def acquire_leave() -> None:
    """Leave every channel a bot made the account join, as the join log records them."""
    console.print(worker.Worker().leave_joined())


@acquire_app.command("run")
def acquire_run() -> None:
    """Run every queued job now, in this terminal, prompting at any pause."""
    jobs = Jobs()
    busy: set[str] = set()
    while queued := sorted((j for j in jobs.all() if j.state == QUEUED and j.id not in busy),
                           key=lambda j: j.id):
        try:
            job = worker.run_one(jobs, queued[0].id, choose=_choose, progress=_bar, log=_log)
        except AcquireBusy:
            busy.add(queued[0].id)  # the app is running it
            continue
        print(file=sys.stderr)
        console.print(f"{job.id}  [bold]{job.state}[/bold]  {job.folder}  [dim]{job.reason}[/dim]")


def _request_for(title: str | None) -> dict:
    if not title:
        raise typer.BadParameter("Give a title, or --imdb.")
    require_catalog()
    with store.session(read_only=True) as con:
        match = pick(con, title)
        if match is None:
            raise typer.Exit(code=1)
        _refuse_other_year(title, match)
        return {"item_id": match.item_id, "title": match.label()}


def _refuse_other_year(title: str, match) -> None:
    # The resolver drops a typed year when nothing matches it, and then
    # confidently returns the film of the same name from another decade.
    # Rating that is recoverable; downloading it is the wrong-target failure.
    _, year = split_year(title)
    if year and match.year and abs(match.year - year) > 1:
        raise EntertainerError(
            f"The catalogue has no {title!r}; the nearest is {match.label()}. "
            "Use --imdb with the film's IMDb id."
        )


def _choose(d: agent.Decision) -> Candidate | None:
    console.print(f"[yellow]paused[/yellow]: {d.reason}.")
    for i, c in enumerate(d.offer, 1):
        console.print(f"  {i}. {c.score:5.1f} {_size(c)} [{c.source}] {c.label}")
    if not sys.stdin.isatty():
        return None
    pick = typer.prompt(f"Take which (1-{len(d.offer)}, 0 to give up)", default="0")
    return d.offer[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(d.offer) else None


def _size(c: Candidate) -> str:
    return _size_of(c.size)


def _size_of(n: int | None) -> str:
    return f"{n / GB:5.2f} GB" if n else "    ? GB"


def _log(msg: str) -> None:
    print(file=sys.stderr, end="\r")
    console.print(f"[dim]{msg}[/dim]")


def _bar(done: int, total: int) -> None:
    pct = f"{100 * done / total:5.1f}%" if total else ""
    print(f"\r  {done / 2**20:8.1f} MB {pct}", end="", file=sys.stderr, flush=True)
