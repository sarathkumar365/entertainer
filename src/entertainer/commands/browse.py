"""Asking the engine things.

Recommendations, explanations, neighbours and the learned taste itself.
None of these change what the engine knows."""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from .. import store
from ..config import has_tmdb, language_label
from ..engine import Engine, liked_titles
from ..render import as_ten as _as_ten
from ..render import console
from ..render import fail as _fail
from ..render import pm as _pm
from ..resolve import resolve_one, search
from ._apps import app
from ._shared import pick, require_catalog


@app.command()
def add(
    title: str,
    year: int | None = typer.Option(None, help="Disambiguate by release year."),
    kind: str | None = typer.Option(None, help="movie | tv"),
) -> None:
    """Fetch a title TMDB knows but the catalogue does not, and place it in the space.

    The catalogue has a vote floor, so genuinely obscure films are missing
    from it. That should never stop someone teaching the engine about a film
    they loved. The title is fetched, encoded, projected into the existing
    fused space by the stored imputation map, and appended — no rebuild.
    """
    from ..data import tmdb
    from ..ingest import IngestError, add_title
    from ..models import fusion
    from ..models.itemcard import build_card

    require_catalog()
    if not has_tmdb():
        _fail("no TMDB credentials — put TMDB_BEARER or TMDB_API_KEY in .env")
    if not fusion.exists():
        _fail("no fused item space — run `ent setup` first")
    engine = Engine()

    with store.session(read_only=True) as con:
        existing, _ = resolve_one(con, title, kind=kind)
    if existing:
        console.print(f"[yellow]already in the catalogue:[/yellow] {existing.label()}")
        return

    hits = tmdb.search(title, year=year, kind=kind)
    if not hits:
        _fail(f"TMDB has nothing matching {title!r}")

    console.print(f"[bold]TMDB matches for[/bold] {title!r}")
    for i, h in enumerate(hits[:8], start=1):
        name = h.get("title") or h.get("name") or "?"
        date = (h.get("release_date") or h.get("first_air_date") or "")[:4]
        console.print(
            f"  [cyan]{i}[/cyan]  {name} [dim]({date or '?'}, {h['_kind']}, "
            f"{h.get('original_language')})[/dim]"
        )
    console.print("  [cyan]0[/cyan]  none of these")
    try:
        choice = typer.prompt("number", type=int, default=1)
    except (typer.Abort, EOFError):
        return
    if choice <= 0 or choice > len(hits[:8]):
        return
    chosen = hits[choice - 1]

    # Placement — inserting the row, encoding it, and appending to both the
    # encoder matrix and the fused space in the same order — is owned by
    # ingest.add_title, which `ent netflix` and the web app already use. This
    # command used to carry its own copy of that sequence, and the copy had
    # no guard against a title already in the space: adding an existing title
    # appended a second, misaligned row.
    try:
        item_id, row = add_title(engine, int(chosen["id"]), chosen["_kind"], place=True)
    except IngestError as exc:
        _fail(str(exc))

    console.print(Panel.fit(build_card(row), title="item card", border_style="dim"))
    console.print(f"[green]added[/green] {row['title']} ({row.get('year')}) as item {item_id}")
    console.print("[dim]rate it with[/dim] [cyan]ent loved \"" + str(row["title"]) + "\"[/cyan]")


@app.command()
def find(query: str, limit: int = typer.Option(8)) -> None:
    """Search the catalogue by title."""
    require_catalog()
    with store.session(read_only=True) as con:
        hits = search(con, query, limit=limit)
    if not hits:
        console.print("[yellow]no matches[/yellow]")
        return
    table = Table("title", "year", "lang", "kind", "IMDb", "votes")
    for h in hits:
        table.add_row(
            h.title, str(h.year or ""), h.language or "", h.kind,
            f"{h.imdb_rating:.1f}" if h.imdb_rating else "",
            f"{h.imdb_votes:,}" if h.imdb_votes else "",
        )
    console.print(table)


# --- onboarding -------------------------------------------------------------


@app.command()
def recs(
    k: int = typer.Option(10, "-k", help="How many titles."),
    language: str = typer.Option("", "--lang", help="Comma-separated language codes."),
    movies: bool = typer.Option(False, "--movies", help="Films only."),
    series: bool = typer.Option(False, "--series", help="Series only."),
    since: int | None = typer.Option(None, help="Only titles released on or after this year."),
    until: int | None = typer.Option(None, help="Only titles released on or before this year."),
    max_runtime: int | None = typer.Option(None, help="Maximum runtime in minutes."),
    strategy: str = typer.Option(
        "thompson", help="thompson (explores) | mean (safest) | ucb (optimistic)."
    ),
    explore: float = typer.Option(
        1.0, help="Appetite for risk. 1.0 is exact Thompson sampling; 0 is greedy; 2 is reckless."
    ),
    novelty: float = typer.Option(
        0.0, help="Push away from the canon. 0 = no penalty, 1 = strongly prefer the obscure."
    ),
    mood: str = typer.Option(
        "", help="Free text: 'slow-burn, quiet, no action'. Nudges the ranking, never overrides it."
    ),
    mood_weight: float = typer.Option(0.6, help="How hard the mood text pulls."),
    why: bool = typer.Option(True, help="Show which of your own titles each pick resembles."),
) -> None:
    """Recommend what to watch next."""
    from ..recommend import Filters, attach_reasons, recommend

    require_catalog()
    engine = Engine()
    with store.session() as con:
        model = engine.model(con)
        if model is None:
            _fail("not enough verdicts yet — run `ent onboard`, or record at least three titles")

        fs = engine.features(con)
        meta = engine.meta(con)
        filters = Filters(
            languages=tuple(x.strip() for x in language.split(",") if x.strip()),
            kind="movie" if movies else ("tv" if series else None),
            min_year=since,
            max_year=until,
            max_runtime=max_runtime,
            exclude=frozenset(store.interacted(con)),
        )
        mood_vec = None
        if mood.strip():
            from ..models import encoder, fusion

            art = fusion.load()
            if art.pca_mean is None:
                _fail("this item space predates mood queries — run `ent data fuse` again")
            console.print("[dim]loading the text encoder…[/dim]")
            mood_vec = art.project(encoder.encode_query(mood)[None, :])[0]

        picks = recommend(
            model, fs, meta, k=k, filters=filters, strategy=strategy,
            explore=explore, novelty=novelty, mood=mood_vec, mood_weight=mood_weight,
        )
        if not picks:
            _fail("no candidates survived those filters")

        if why:
            ids, labels = liked_titles(con, engine)
            attach_reasons(picks, fs, ids, labels)

        slate = store.new_slate_id()
        store.log_impressions(
            con,
            slate,
            [(p.item_id, p.position, p.score, p.propensity, p.explored) for p in picks],
            # n_real, not n_obs: the policy label should say how many verdicts
            # were behind the slate, not how many rows the fit happened to see.
            policy=f"{strategy}-n{model.n_real}",
        )
        # Remembered so a verdict can be given by position rather than by
        # retyping a title.
        store.set_last_slate(con, [p.item_id for p in picks])

        full = store.item_rows(con, [p.item_id for p in picks])

    console.print()
    for n, p in enumerate(picks, start=1):
        row = full[p.item_id]
        head = f"[dim]{n:>2}[/dim] [bold]{row['title']}[/bold]"
        if row.get("original_title") and row["original_title"] != row["title"]:
            head += f" [dim]({row['original_title']})[/dim]"
        facts = [str(row["year"])] if row.get("year") else []
        if row.get("language"):
            facts.append(language_label(row["language"]))
        if row.get("kind") == "tv":
            facts.append("series")
        if row.get("runtime"):
            facts.append(f"{row['runtime']}m")
        if row.get("imdb_rating"):
            facts.append(f"IMDb {row['imdb_rating']:.1f}")
        marker = "[magenta]◇[/magenta]" if p.explored else "[green]◆[/green]"
        console.print(f"{marker}{head}  [dim]{' · '.join(facts)}[/dim]")

        genres = ", ".join((row.get("genres") or [])[:4])
        if genres:
            console.print(f"    [dim]{genres}[/dim]")
        overview = (row.get("overview") or "").strip()
        if overview:
            console.print(f"    {overview[:190]}{'…' if len(overview) > 190 else ''}")
        if p.reasons:
            because = "; ".join(f"{name}" for name, _ in p.reasons)
            console.print(f"    [dim]close to your: {because}[/dim]")
        console.print(
            f"    [dim]predicted {_as_ten(p.mean):.1f}/10 ± {_pm(p.std):.1f}[/dim]\n"
        )

    console.print(
        "[dim]◆ confident pick   ◇ exploratory pick[/dim]\n"
        "[dim]tell it what happened: [/dim][cyan]ent loved 3[/cyan]"
        "[dim] (by position) or [/dim][cyan]ent loved \"<title>\"[/cyan]"
    )


@app.command()
def why(title: str) -> None:
    """Explain how a specific title scores against what the engine knows about you."""
    from ..models.discover import nearest_liked

    require_catalog()
    engine = Engine()
    with store.session() as con:
        match = pick(con, title)
        if not match:
            raise typer.Exit(code=1)
        model = engine.model(con)
        if model is None:
            _fail("not enough verdicts yet")
        fs = engine.features(con)
        if match.item_id not in fs.index:
            _fail("that title has no embedding — rebuild the item space")

        vec = fs.matrix[fs.index[match.item_id]]
        mean, std = model.predict(vec[None, :])
        ids, labels = liked_titles(con, engine)
        neighbours = []
        if ids:
            neighbours = nearest_liked(
                fs.latent[fs.index[match.item_id]], fs.latent[fs.rows_for(ids)], labels, top=5
            )

    console.print(
        Panel.fit(
            f"[bold]{match.label()}[/bold]\n\n"
            f"predicted  [bold]{_as_ten(mean[0]):.1f}/10[/bold]  ± {_pm(std[0]):.1f}\n"
            f"the ± is the model's own uncertainty; a wide band means it is guessing",
            border_style="cyan",
        )
    )
    if neighbours:
        console.print("[bold]closest to titles you rated well:[/bold]")
        for name, sim in neighbours:
            console.print(f"  [dim]{sim:+.2f}[/dim]  {name}")


@app.command()
def similar(
    title: str,
    k: int = typer.Option(10, "-k"),
    language: str = typer.Option("", "--lang"),
    same_language: bool = typer.Option(False, help="Restrict to the title's own language."),
) -> None:
    """Find titles closest to a given one in the learned latent space.

    Pure geometry — this ignores your taste model entirely, so it answers
    "what is like this" rather than "what would you enjoy". Useful when you
    know exactly what mood you are in.
    """
    require_catalog()
    engine = Engine()
    with store.session(read_only=True) as con:
        match = pick(con, title)
        if not match:
            raise typer.Exit(code=1)
        fs = engine.features(con)
        meta = engine.meta(con)
        if match.item_id not in fs.index:
            _fail("that title has no embedding — rebuild the item space")

        langs = tuple(x.strip() for x in language.split(",") if x.strip())
        if same_language and match.language:
            langs = (match.language,)

        from ..recommend import neighbours

        picked = neighbours(fs, match.item_id, meta, k=k, languages=langs)
        rows = store.item_rows(con, [n.item_id for n in picked])

    console.print(f"[dim]closest to[/dim] [bold]{match.label()}[/bold]\n")
    table = Table("similarity", "title", "year", "lang", "IMDb")
    for n in picked:
        r = rows[n.item_id]
        table.add_row(
            f"{n.similarity:.3f}", r["title"], str(r.get("year") or ""),
            r.get("language") or "", f"{r['imdb_rating']:.1f}" if r.get("imdb_rating") else "",
        )
    console.print(table)


@app.command()
def taste(axes: int = typer.Option(6, help="How many latent axes to describe.")) -> None:
    """Show what the engine has worked out about your taste."""
    from ..models.discover import describe_axes, surface_preferences

    require_catalog()
    engine = Engine()
    with store.session() as con:
        model = engine.model(con)
        if model is None:
            _fail("not enough verdicts yet — run `ent onboard`")
        fs = engine.features(con)
        meta = engine.meta(con)
        found = describe_axes(model, fs.item_ids, fs.latent, meta, n_axes=axes)

    console.print(
        Panel.fit(
            f"learned from [bold]{model.n_real}[/bold] verdicts · "
            f"capacity: {'linear' if model.feature_map.n_rff == 0 else f'linear + {model.feature_map.n_rff} RFF'} "
            f"(chosen by marginal likelihood, log Z = {model.log_evidence:.1f})",
            border_style="dim",
        )
    )

    if not found:
        console.print("[yellow]not enough signal to describe the latent axes yet[/yellow]")
    for ax in found:
        console.print(f"\n[bold cyan]axis {ax.index}[/bold cyan]  [dim]influence {ax.strength:.3f}[/dim]")
        console.print(f"  [green]towards[/green]  {', '.join(ax.liked_pole[:6]) or '—'}")
        console.print(f"  [dim]e.g. {', '.join(ax.liked_examples)}[/dim]")
        console.print(f"  [red]away from[/red]  {', '.join(ax.other_pole[:5]) or '—'}")
        console.print(f"  [dim]e.g. {', '.join(ax.other_examples)}[/dim]")

    surface = surface_preferences(model, fs)
    if surface:
        console.print("\n[bold]surface preferences[/bold] [dim](learned, not assumed)[/dim]")
        for name, w in surface:
            lean = "prefers more" if w > 0 else "prefers less"
            bar = "█" * min(20, int(abs(w) * 60))
            console.print(f"  {name:>18}  [dim]{lean:>12}[/dim]  {bar} [dim]{w:+.3f}[/dim]")


# --- introspection ----------------------------------------------------------
