"""Command line interface.

The whole interaction model is: type a title, say what you thought. Everything
else the engine does is downstream of that. Commands are therefore named after
what a person would say out loud — ``ent loved parasite`` — rather than after
the machinery underneath.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer
from rich.panel import Panel
from rich.table import Table

from . import profile_io, store
from .build_events import Reporter
from .config import PATHS, has_tmdb, language_label
from .engine import Engine, liked_titles
from .errors import EntertainerError
from .manifests import write as write_manifest
from .pipeline import preflight as pipeline_preflight
from .render import as_ten as _as_ten
from .render import console, tables
from .render import fail as _fail
from .render import pm as _pm
from .render import toned as _toned
from .resolve import Match, resolve_one, search

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A personal, self-improving recommendation engine for film and television.",
)
data_app = typer.Typer(no_args_is_help=True, help="Build and maintain the catalogue.")
app.add_typer(data_app, name="data")

# --- helpers ----------------------------------------------------------------


def _require_catalog() -> None:
    if not PATHS.catalog_db.exists():
        _fail("no catalogue yet — run `ent setup` first")


def _from_last_slate(con, query: str) -> Match | None:
    """Let a bare number refer to a position in the last recommended slate.

    The daily loop is: ask for recommendations, watch one, say what you
    thought. Retyping a title you just read off the screen — often a
    transliterated one — is the friction most likely to stop someone giving
    feedback at all, and feedback is the only thing this system runs on.
    """
    if not query.strip().isdigit():
        return None
    position = int(query.strip())
    slate = store.get_meta(con, "last_slate", [])
    if not slate or not (1 <= position <= len(slate)):
        return None
    item_id = int(slate[position - 1])
    row = con.execute(
        "SELECT item_id, title, original_title, year, kind, language, imdb_votes, imdb_rating "
        "FROM titles WHERE item_id = ?",
        [item_id],
    ).fetchone()
    if not row:
        return None
    return Match(
        item_id=int(row[0]), title=row[1], original_title=row[2], year=row[3],
        kind=row[4], language=row[5], imdb_votes=row[6], imdb_rating=row[7], score=1.0,
    )


def _pick(con, query: str, kind: str | None = None) -> Match | None:
    """Resolve a typed title, asking the user only when genuinely ambiguous."""
    from_slate = _from_last_slate(con, query)
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


def _record(query: str, verdict: str, kind: str | None = None) -> None:
    _require_catalog()
    engine = Engine()
    with store.session() as con:
        match = _pick(con, query, kind=kind)
        if not match:
            raise typer.Exit(code=1)
        engine.record(con, match.item_id, verdict)
        console.print(f"[green]{verdict}[/green] — {match.label()}")
        n = len(store.ratings(con))
        if n in (3, 10, 25, 50, 100):
            console.print(f"[dim]{n} verdicts recorded; model refits on every command[/dim]")


# --- setup and data ---------------------------------------------------------


@app.command()
def setup(
    skip_enrich: bool = typer.Option(False, help="Skip the TMDB enrichment pass."),
    enrich_limit: int = typer.Option(0, help="Enrich only the N most-voted titles (0 = all)."),
    min_votes: int = typer.Option(50, help="Flat IMDb vote floor at build time."),
    floor_scale: float = typer.Option(1.0, help="<1 widens the catalogue, >1 narrows it."),
) -> None:
    """Run everything: download, build, enrich, prune, embed, factorise, fuse, prior."""
    from .data import catalog, download

    info = pipeline_preflight(require_tmdb=not skip_enrich)
    reporter = Reporter(settings={
        "skip_enrich": skip_enrich,
        "enrich_limit": enrich_limit,
        "min_votes": min_votes,
        "floor_scale": floor_scale,
        "free_bytes": info["free_bytes"],
    })
    console.print(f"[dim]preflight passed: {info['free_bytes'] / 1024**3:.1f} GiB free[/dim]")
    console.rule("[bold]1/8 downloading source data")
    with reporter.stage("sources"):
        download.fetch_imdb()
        download.fetch_movielens()

    console.rule("[bold]2/8 building catalogue")
    with reporter.stage("catalogue"):
        n = catalog.build_base(min_votes=min_votes)
    console.print(f"[green]{n:,} titles[/green]")

    if not skip_enrich and has_tmdb():
        console.rule("[bold]3/8 enriching from TMDB")
        with reporter.stage("tmdb"):
            data_enrich(limit=enrich_limit or None, concurrency=40, keywords=False)
            data_keywords(top=150_000, concurrency=40)
    else:
        console.rule("[bold]3/8 TMDB enrichment skipped")
        reporter.skip("tmdb", "disabled by --skip-enrich or no TMDB credentials")

    console.rule("[bold]4/8 pruning by language")
    with reporter.stage("prune"):
        data_prune(scale=floor_scale, dry_run=False)
    console.rule("[bold]5/8 encoding item text")
    with reporter.stage("embeddings"):
        data_embed(batch_size=64, limit=None, fresh=False)
    console.rule("[bold]6/8 factorising MovieLens")
    with reporter.stage("cf"):
        data_cf(factors=192, iterations=20, holdout=2_000, signal="watched")
    console.rule("[bold]7/8 fusing item space")
    with reporter.stage("fusion"):
        data_fuse(dim=192)
    console.rule("[bold]8/8 learning the population prior")
    with reporter.stage("prior"):
        data_prior(max_users=20_000, shrinkage=0.15)

    with store.session(read_only=True) as con:
        counts = store.counts(con)
    manifest = write_manifest("build", {"counts": counts, "min_votes": min_votes, "floor_scale": floor_scale})
    reporter.complete(manifest["id"])
    console.print(f"[dim]build manifest: {manifest['path']}[/dim]")

    console.print(Panel.fit("[bold green]ready[/bold green]\nnext: [cyan]ent onboard[/cyan]"))


@app.command("preflight")
def preflight_command() -> None:
    """Check local storage and TMDB configuration before a full build."""
    try:
        info = pipeline_preflight()
    except RuntimeError as exc:
        _fail(str(exc))
    console.print(
        f"[green]ready[/green] — {info['free_bytes'] / 1024**3:.1f} GiB free at {info['data_dir']}"
    )


@data_app.command("fetch")
def data_fetch() -> None:
    """Download the IMDb and MovieLens bulk datasets."""
    from .data import download

    download.fetch_imdb()
    download.fetch_movielens()
    console.print("[green]downloaded[/green]")


@data_app.command("prune")
def data_prune(
    scale: float = typer.Option(1.0, help="<1 keeps more obscure titles, >1 fewer."),
    dry_run: bool = typer.Option(False, help="Report what would go, change nothing."),
) -> None:
    """Apply per-language vote floors, after TMDB has supplied the languages."""
    from .data import catalog

    if not dry_run:
        catalog.recalibrate_quality()
    before, after = catalog.prune_by_language(scale=scale, dry_run=dry_run)
    verb = "would remove" if dry_run else "removed"
    console.print(f"[green]{before:,} -> {after:,}[/green] ({verb} {before - after:,})")
    if not dry_run:
        console.print(tables.language_histogram(catalog.language_histogram(25)))


@data_app.command("build")
def data_build(min_votes: int = typer.Option(50, help="Flat IMDb vote floor at build time.")) -> None:
    """Build the catalogue table from the bulk datasets."""
    from .data import catalog

    n = catalog.build_base(min_votes=min_votes)
    console.print(f"[green]{n:,} titles[/green]")
    console.print(tables.language_histogram(catalog.language_histogram(20)))


@data_app.command("enrich")
def data_enrich(
    limit: int | None = typer.Option(None, help="Only the N most-voted unenriched titles."),
    concurrency: int = typer.Option(40),
    keywords: bool = typer.Option(True, help="Fetch the keyword vocabulary (doubles requests)."),
) -> None:
    """Fill in synopses, keywords and languages from TMDB."""
    from .data import catalog, tmdb

    if not has_tmdb():
        _fail("no TMDB credentials — put TMDB_BEARER or TMDB_API_KEY in .env")
    pending = catalog.pending_enrichment(limit)
    if not pending:
        console.print("[green]nothing left to enrich[/green]")
        return
    console.print(f"[dim]{len(pending):,} titles to enrich[/dim]")

    con = store.connect()
    written = {"n": 0}

    def flush(batch):
        catalog.apply_enrichment(con, batch)
        written["n"] += len(batch)

    try:
        tmdb.enrich(pending, concurrency=concurrency, keywords=keywords, on_batch=flush)
    finally:
        con.close()
    console.print(f"[green]enriched {written['n']:,} titles[/green]")


@data_app.command("keywords")
def data_keywords(
    top: int = typer.Option(
        150_000, help="Ensure the N most-voted titles have keywords. Idempotent."
    ),
    concurrency: int = typer.Option(40),
) -> None:
    """Backfill TMDB keywords for the titles most likely to be encountered."""
    from .data import tmdb

    if not has_tmdb():
        _fail("no TMDB credentials")
    con = store.connect()
    # Anything with keywords already stored counts as fetched, so a catalogue
    # enriched before this marker existed is not re-requested wholesale.
    con.execute(
        "UPDATE titles SET keywords_at = now() "
        "WHERE keywords_at IS NULL AND keywords IS NOT NULL AND len(keywords) > 0"
    )
    # `top` is a coverage target, not a batch size: "the N most-voted titles
    # should have keywords". Ranking first and filtering second makes a
    # resumed run finish the same N rather than moving on to the next N.
    targets = con.execute(
        """
        SELECT item_id, tmdb_id, kind FROM (
            SELECT item_id, tmdb_id, kind, keywords_at,
                   row_number() OVER (ORDER BY imdb_votes DESC NULLS LAST) AS rank
            FROM titles WHERE tmdb_id IS NOT NULL
        ) WHERE rank <= ? AND keywords_at IS NULL
        """,
        [top],
    ).fetchall()
    if not targets:
        console.print("[green]nothing to backfill[/green]")
        con.close()
        return
    console.print(f"[dim]{len(targets):,} titles need keywords[/dim]")

    written = {"n": 0}

    def flush(batch):
        # keywords_at is stamped even when the list comes back empty: "TMDB
        # has none for this title" is a result, not a failure to fetch.
        con.executemany(
            "UPDATE titles SET keywords = ?, keywords_at = now() WHERE item_id = ?",
            [(kws, item_id) for item_id, kws in batch],
        )
        written["n"] += len(batch)

    try:
        tmdb.backfill_keywords([(int(a), int(b), c) for a, b, c in targets],
                               concurrency=concurrency, on_batch=flush)
    finally:
        con.close()
    console.print(f"[green]keywords for {written['n']:,} titles[/green]")


@data_app.command("embed")
def data_embed(
    batch_size: int = typer.Option(64),
    limit: int | None = typer.Option(None),
    fresh: bool = typer.Option(False, help="Discard cached shards and re-encode everything."),
) -> None:
    """Encode every item card with the multilingual text encoder.

    Resumable: completed shards are written as they finish, so an interrupted
    run picks up where it stopped rather than repeating forty minutes of GPU
    work.
    """
    from .models import encoder
    from .models.itemcard import build_card

    _require_catalog()
    con = store.connect(read_only=True)
    cur = con.execute(
        "SELECT item_id, title, original_title, year, kind, language, runtime, genres, "
        "keywords, directors, cast_names, overview, tagline FROM titles "
        "ORDER BY item_id" + (f" LIMIT {int(limit)}" if limit else "")
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    con.close()
    if not rows:
        _fail("catalogue is empty")

    if fresh:
        console.print(f"[dim]cleared {encoder.clear_shards()} cached shards[/dim]")

    ids = np.array([r["item_id"] for r in rows], dtype=np.int32)
    cards = [build_card(r) for r in rows]
    console.print(f"[dim]{len(cards):,} item cards[/dim]")
    console.print(Panel.fit(cards[0], title="example item card", border_style="dim"))

    ids, mat = encoder.encode_resumable(ids, cards, batch_size=batch_size)
    encoder.save(ids, mat)
    console.print(f"[green]embeddings: {mat.shape}[/green]")


@data_app.command("cf")
def data_cf(
    factors: int = typer.Option(192),
    iterations: int = typer.Option(20),
    holdout: int = typer.Option(
        2000, help="MovieLens users withheld from training, reserved for offline evaluation."
    ),
    signal: str = typer.Option(
        "watched", help="watched (all ratings, graded confidence) | liked (>=3.5 only)."
    ),
) -> None:
    """Factorise the MovieLens co-consumption matrix."""
    import numpy as np

    from .models import cf

    ratings = cf.load_ratings()
    users = np.sort(ratings["userId"].unique().to_numpy())
    rng = np.random.default_rng(0)
    held = rng.choice(users, size=min(holdout, len(users)), replace=False) if holdout else None

    ids, item_factors = cf.fit(
        factors=factors, iterations=iterations, holdout_users=held, signal=signal
    )
    cf.save(ids, item_factors)
    if held is not None:
        np.save(PATHS.artifacts / "cf_holdout_users.npy", held.astype(np.int32))
    console.print(f"[green]CF factors: {item_factors.shape}[/green]")


@data_app.command("prior")
def data_prior(
    max_users: int = typer.Option(20_000, help="MovieLens users to fit taste vectors for."),
    shrinkage: float = typer.Option(0.15, help="Pull the covariance towards a scaled identity."),
) -> None:
    """Learn what human taste vectors look like, to use as the cold-start prior."""
    import numpy as np

    from .models import fusion
    from .models.population import fit as fit_prior

    if not fusion.exists():
        _fail("no fused item space — run `ent data fuse` first")

    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        ml = dict(
            con.execute(
                "SELECT movielens_id, item_id FROM titles WHERE movielens_id IS NOT NULL"
            ).fetchall()
        )
    item_of_ml = {int(k): int(v) for k, v in ml.items() if int(v) in fs.index}
    if not item_of_ml:
        _fail("no MovieLens identities in the catalogue — rebuild after downloading ml-32m")

    held_path = PATHS.artifacts / "cf_holdout_users.npy"
    held = np.load(held_path) if held_path.exists() else None

    prior = fit_prior(fs, item_of_ml, holdout_users=held, max_users=max_users,
                      shrinkage=shrinkage)
    prior.save()
    console.print(
        f"[green]population prior from {prior.n_users:,} users[/green] "
        f"({prior.dim} dimensions)"
    )


@data_app.command("fuse")
def data_fuse(dim: int = typer.Option(192)) -> None:
    """Fuse the content and collaborative towers into one latent space."""
    from .models import cf, encoder, fusion

    if not encoder.exists():
        _fail("no content embeddings — run `ent data embed`")
    if not cf.exists():
        _fail("no CF factors — run `ent data cf`")

    ids, content = encoder.load()
    cf_ids, cf_factors = cf.load()

    con = store.connect(read_only=True)
    ml = dict(
        con.execute(
            "SELECT item_id, movielens_id FROM titles WHERE movielens_id IS NOT NULL"
        ).fetchall()
    )
    con.close()

    art = fusion.build(ids, content, cf_ids, cf_factors, {int(k): int(v) for k, v in ml.items()}, dim=dim)
    fusion.save(art)
    console.print(
        f"[green]fused space: {art.space.shape}[/green] "
        f"(CF coverage {art.cf_coverage:.1%}, imputation R²={art.cf_r2:.3f})"
    )


# --- taste input ------------------------------------------------------------


@app.command()
def loved(title: str = typer.Argument(..., help="Title, optionally with a year.")) -> None:
    """Record that you loved something."""
    _record(title, "love")


@app.command()
def liked(title: str) -> None:
    """Record that you liked something."""
    _record(title, "like")


@app.command()
def meh(title: str) -> None:
    """Record that something left you cold."""
    _record(title, "meh")


@app.command()
def disliked(title: str) -> None:
    """Record that you disliked something."""
    _record(title, "dislike")


@app.command()
def hated(title: str) -> None:
    """Record that you hated something."""
    _record(title, "hate")


@app.command()
def seen(title: str) -> None:
    """Mark something as already watched, with no opinion attached."""
    _require_catalog()
    with store.session() as con:
        match = _pick(con, title)
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
    _require_catalog()
    with store.session() as con:
        match = _pick(con, title)
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
    _require_catalog()
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
    from .data import catalog, tmdb
    from .models import encoder, fusion
    from .models.itemcard import build_card

    _require_catalog()
    if not has_tmdb():
        _fail("no TMDB credentials — put TMDB_BEARER or TMDB_API_KEY in .env")
    if not fusion.exists():
        _fail("no fused item space — run `ent setup` first")

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

    payload = tmdb.detail(int(chosen["id"]), chosen["_kind"])
    if not payload:
        _fail("could not fetch details from TMDB")
    row = tmdb.detail_to_row(payload, chosen["_kind"])

    art = fusion.load()
    if art.pca_mean is None:
        _fail("this fused space predates out-of-sample projection — run `ent data fuse` again")

    card = build_card(row)
    console.print(Panel.fit(card, title="item card", border_style="dim"))
    content = encoder.encode_texts([card], show_progress=False)
    latent = art.project(content)

    with store.session() as con:
        item_id = catalog.insert_title(con, row)

    ids, mat = encoder.load()
    encoder.save(np.append(ids, np.int32(item_id)), np.vstack([mat, content]))
    art.item_ids = np.append(art.item_ids, np.int32(item_id))
    art.space = np.vstack([art.space, latent])
    fusion.save(art)

    console.print(f"[green]added[/green] {row['title']} ({row.get('year')}) as item {item_id}")
    console.print("[dim]rate it with[/dim] [cyan]ent loved \"" + str(row["title"]) + "\"[/cyan]")


@app.command()
def find(query: str, limit: int = typer.Option(8)) -> None:
    """Search the catalogue by title."""
    _require_catalog()
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
def onboard(
    questions: int = typer.Option(40, "--n", help="How many titles to ask about."),
    languages: str = typer.Option("", help="Comma-separated language codes to focus on."),
) -> None:
    """Cold start: answer a short, adaptively chosen set of questions."""
    import numpy as np

    from .coldstart import elicit
    from .models.taste import fit as fit_taste

    _require_catalog()
    engine = Engine()
    langs = tuple(x.strip() for x in languages.split(",") if x.strip())

    with store.session() as con:
        fs = engine.features(con)
        meta = engine.meta(con)
        pool = elicit.recognisable_pool(fs, meta, langs)
        if pool.size == 0:
            _fail("no recognisable titles for those languages")

        asked = store.already_asked(con)
        answered: list[tuple[int, float]] = []
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

        batch = elicit.seed_questions(fs, meta, k=max(questions, 24), languages=langs, pool=pool)
        batch = [b for b in batch if b not in asked]
        cursor = 0
        answered_count = 0

        while answered_count < questions:
            if cursor >= len(batch):
                if len(answered) >= 3:
                    ids = np.array([a[0] for a in answered])
                    rewards = np.array([a[1] for a in answered])
                    model = fit_taste(
                        fs.vectors_for(ids), rewards, allow_rff=False, prior=engine.prior(con)
                    )
                    batch = elicit.next_questions(
                        model, fs, meta, asked, k=16, languages=langs, pool=pool
                    )
                else:
                    batch = [
                        b
                        for b in elicit.seed_questions(
                            fs, meta, k=questions * 3, languages=langs, pool=pool
                        )
                        if b not in asked
                    ]
                cursor = 0
                if not batch:
                    break

            item = batch[cursor]
            cursor += 1
            if item in asked:
                continue
            asked.add(item)
            row = meta[item]
            label = f"[bold]{row['title']}[/bold]"
            if row.get("original_title") and row["original_title"] != row["title"]:
                label += f" [dim]({row['original_title']})[/dim]"
            tail = ", ".join(
                str(x) for x in (row.get("year"), language_label(row.get("language")),
                                 "series" if row.get("kind") == "tv" else None) if x
            )
            console.print(f"\n[cyan]{answered_count + 1}/{questions}[/cyan]  {label}  [dim]{tail}[/dim]")
            try:
                key = typer.prompt("", default="n", show_default=False).strip().lower()[:1]
            except (typer.Abort, EOFError):
                break
            if key == "q":
                break
            if key not in keymap:
                # Not a verdict: record that the question was asked so it is
                # not repeated, without removing the title from circulation.
                store.log_event(con, item, "unseen", None, "elicit", {"answer": "unseen"})
                continue
            reward = engine.record(con, item, keymap[key], source="elicit")
            answered.append((item, reward))
            answered_count += 1

        console.print(f"\n[green]{answered_count} verdicts recorded[/green]")
        if answered_count >= 3:
            engine.fit(con)
            console.print("try [cyan]ent recs[/cyan] or [cyan]ent taste[/cyan]")


# --- recommendations --------------------------------------------------------


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
    from .recommend import Filters, attach_reasons, recommend

    _require_catalog()
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
            from .models import encoder, fusion

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
            policy=f"{strategy}-n{model.n_obs}",
        )
        # Remembered so a verdict can be given by position rather than by
        # retyping a title.
        store.set_meta(con, "last_slate", [p.item_id for p in picks])

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
    from .models.discover import nearest_liked

    _require_catalog()
    engine = Engine()
    with store.session() as con:
        match = _pick(con, title)
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
    _require_catalog()
    engine = Engine()
    with store.session(read_only=True) as con:
        match = _pick(con, title)
        if not match:
            raise typer.Exit(code=1)
        fs = engine.features(con)
        meta = engine.meta(con)
        if match.item_id not in fs.index:
            _fail("that title has no embedding — rebuild the item space")

        langs = tuple(x.strip() for x in language.split(",") if x.strip())
        if same_language and match.language:
            langs = (match.language,)

        from .recommend import neighbours

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
def forget(title: str) -> None:
    """Remove every verdict you have given a title, as if it were never rated."""
    _require_catalog()
    with store.session() as con:
        match = _pick(con, title)
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
def taste(axes: int = typer.Option(6, help="How many latent axes to describe.")) -> None:
    """Show what the engine has worked out about your taste."""
    from .models.discover import describe_axes
    from .models.features import SIDE_FEATURE_NAMES
    from .models.taste import taste_direction

    _require_catalog()
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
            f"learned from [bold]{model.n_obs}[/bold] verdicts · "
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

    direction = taste_direction(model)
    side = direction[fs.n_latent :]
    if side.size == len(SIDE_FEATURE_NAMES):
        console.print("\n[bold]surface preferences[/bold] [dim](learned, not assumed)[/dim]")
        for name, w in zip(SIDE_FEATURE_NAMES, side, strict=True):
            lean = "prefers more" if w > 0 else "prefers less"
            bar = "█" * min(20, int(abs(w) * 60))
            console.print(f"  {name:>18}  [dim]{lean:>12}[/dim]  {bar} [dim]{w:+.3f}[/dim]")


# --- introspection ----------------------------------------------------------


def _next_step(present: dict[str, bool], n_ratings: int) -> str:
    """What to run next, given what exists.

    A pipeline with eight stages and a three-hour critical path needs to be
    able to say where it got to. Ordered by dependency, first gap wins.
    """
    if not present["catalog"]:
        return "ent setup"
    if not present["content_embeddings"]:
        return "ent data embed"
    if not present["cf_factors"]:
        return "ent data cf"
    if not present["fused_space"]:
        return "ent data fuse"
    if n_ratings < 3:
        return "ent onboard    [dim](or: ent bulk seed.example.txt)[/dim]"
    if n_ratings < 8:
        return "ent recs    [dim](a few more verdicts and `ent audit` will work too)[/dim]"
    return "ent recs"


@app.command()
def rate(
    port: int = typer.Option(8756, help="Port to serve on."),
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by default."),
    lan: bool = typer.Option(
        False, help="Bind to all interfaces so another device on the network can reach it."
    ),
    token: str = typer.Option("", help="Shared secret. Generated automatically when --lan."),
    live: bool = typer.Option(
        None,
        "--live/--catalogue",
        help="Serve titles from TMDB instead of a local catalogue. "
        "Chosen automatically when there is no catalogue.",
    ),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Open the rating interface — a grid of posters you click through.

    The engine learns from verdicts and nothing else, so how fast you can give
    verdicts is the rate-limiting step in making it good. Recognising a poster
    is far quicker than recalling and typing a transliterated title, and the
    friction compounds over the hundred-odd ratings the model needs.

    Everything written here goes into the same event log the CLI uses, so
    `ent recs`, `ent taste` and `ent audit` see it immediately.
    """
    try:
        import uvicorn
    except ImportError:
        _fail("install the web extra: uv pip install -e '.[web]'")

    from .web.app import EMPTY_CATALOGUE, _catalogue_size, create_app

    size = _catalogue_size()
    use_live = live if live is not None else size < EMPTY_CATALOGUE
    if use_live and not has_tmdb():
        _fail(
            "no catalogue on this machine and no TMDB credentials.\n"
            "Either put TMDB_BEARER in .env, or import a bundle with "
            "`ent bundle import <file>`."
        )
    if not use_live and size < EMPTY_CATALOGUE:
        _fail("no catalogue — run `ent setup`, `ent bundle import <file>`, or use --live")

    if lan:
        host = "0.0.0.0"  # noqa: S104 - deliberate, and gated behind a token
    off_loopback = host not in ("127.0.0.1", "localhost", "::1")

    # A token is mandatory off the loopback interface. The page writes to the
    # verdict log, so an unauthenticated copy on a shared network is somebody
    # else's write access to your taste profile.
    if off_loopback and not token:
        import secrets

        token = secrets.token_urlsafe(12)

    display_host = host
    if host == "0.0.0.0":  # noqa: S104
        import socket

        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            display_host = probe.getsockname()[0]
            probe.close()
        except OSError:
            display_host = "localhost"

    url = f"http://{display_host}:{port}"
    if token:
        url += f"?token={token}"
    console.print(
        Panel.fit(
            f"[bold]{url}[/bold]\n\n"
            "[bold]♥[/bold] loved   [bold]+[/bold] liked   [bold]~[/bold] fine   "
            "[bold]−[/bold] disliked   [bold]?[/bold] not seen\n"
            + (
                "[yellow]live mode — titles come from TMDB, no local catalogue[/yellow]\n"
                if use_live
                else f"[dim]{size:,} titles in the local catalogue[/dim]\n"
            )
            + "search finds anything TMDB knows, even outside the catalogue\n\n"
            + (
                "[yellow]reachable on your local network — the link above "
                "contains its access token[/yellow]\n"
                if off_loopback
                else ""
            )
            + "[dim]ctrl-c to stop · nothing leaves this machine[/dim]",
            title="rate what you have seen",
            border_style="cyan",
        )
    )
    if open_browser and not off_loopback:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        create_app(token=token or None, live=use_live),
        host=host, port=port, log_level="warning",
    )


@app.command()
def studio(
    port: int = typer.Option(8757, help="Port to serve on."),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Open the read-only Build Studio, then run ``ent setup`` separately."""
    try:
        import uvicorn
    except ImportError:
        _fail("install the web extra: uv pip install -e '.[web]'")
    from .web.studio import create_studio_app

    url = f"http://127.0.0.1:{port}"
    console.print(Panel.fit(
        f"[bold]{url}[/bold]\n\nOpen this page, then run [cyan]ent setup[/cyan] in another terminal.\n"
        "[dim]Studio observes saved local build progress; it cannot alter the build.[/dim]",
        title="Build Studio", border_style="cyan",
    ))
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_studio_app(), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def stats() -> None:
    """Show what exists, how much the engine knows, and what to run next."""
    engine = Engine()
    present = engine.artifacts_present
    table = Table("component", "state")
    for name, ok in present.items():
        table.add_row(name.replace("_", " "), "[green]ready[/green]" if ok else "[red]missing[/red]")
    table.add_row(
        "population prior",
        "[green]ready[/green]"
        if (PATHS.artifacts / "population_prior.npz").exists()
        else "[yellow]absent[/yellow]  [dim]optional: ent data prior[/dim]",
    )
    table.add_row("tmdb credentials", "[green]set[/green]" if has_tmdb() else "[yellow]absent[/yellow]")
    console.print(table)

    if not PATHS.catalog_db.exists():
        console.print("\n[bold]next:[/bold] [cyan]ent setup[/cyan]")
        return
    with store.session(read_only=True) as con:
        c = store.counts(con)
        langs = con.execute(
            # Tiebreak on language, so equal counts do not reorder between runs.
            "SELECT language, count(*) n FROM titles GROUP BY 1 "
            "ORDER BY n DESC, language ASC LIMIT 12"
        ).fetchall()
        cover = engine.coverage(con)

    if cover["catalog"] and present["fused_space"]:
        share = cover["fused"] / cover["catalog"]
        if share < 0.995:
            console.print(
                f"\n[yellow]item space covers {cover['fused']:,} of "
                f"{cover['catalog']:,} titles ({share:.1%}).[/yellow] "
                "The rest cannot be recommended. Run [cyan]ent data embed[/cyan] "
                "then [cyan]ent data fuse[/cyan]."
            )

    t2 = Table("metric", "value")
    for key, value in c.items():
        t2.add_row(key, f"{value:,}")
    console.print(t2)
    console.print(tables.language_histogram(langs))
    console.print(f"\n[bold]next:[/bold] [cyan]{_next_step(present, c['ratings'])}[/cyan]")


@app.command()
def history(limit: int = typer.Option(30)) -> None:
    """List the verdicts you have given, most recent first."""
    _require_catalog()
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


bundle_app = typer.Typer(no_args_is_help=True, help="Move a built catalogue between machines.")
app.add_typer(bundle_app, name="bundle")


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
    from . import bundle

    _require_catalog()
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
    from . import bundle

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
    from . import bundle

    try:
        manifest = bundle.restore(path, overwrite=overwrite)
    except RuntimeError as exc:
        _fail(str(exc))
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
    _require_catalog()
    with store.session(read_only=True) as con:
        n = profile_io.export_events(con, path)
    console.print(f"[green]{n:,} events -> {path}[/green]")
    console.print("[dim]this file is your viewing history; keep it out of public repos[/dim]")


@app.command("import")
def import_profile(path: Path = typer.Argument(...)) -> None:
    """Re-import an exported profile, matching on IMDb id."""
    _require_catalog()
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


@app.command()
def audit(
    interval: float = typer.Option(0.90, help="Nominal coverage of the predictive interval."),
) -> None:
    """Measure whether the engine is actually learning *you*, on your own history.

    Walks your verdicts in order, refits on everything before each one and
    predicts it blind. Every number here comes from a model that had not seen
    the answer.
    """
    from .evaluation.prequential import MIN_VERDICTS
    from .evaluation.prequential import readings as prequential_readings
    from .evaluation.prequential import run as prequential

    _require_catalog()
    engine = Engine()
    with store.session(read_only=True) as con:
        rows = con.execute(
            """
            SELECT item_id, value, ts FROM events
            WHERE kind = 'rate' AND value IS NOT NULL ORDER BY ts
            """
        ).fetchall()
        fs = engine.features(con)

    seen: set[int] = set()
    items, rewards = [], []
    for item_id, value, _ in rows:
        iid = int(item_id)
        if iid in seen or iid not in fs.index:
            continue
        seen.add(iid)
        items.append(iid)
        rewards.append(float(value) / 10.0)

    if len(items) < MIN_VERDICTS:
        _fail(
            f"only {len(items)} verdicts — this needs at least {MIN_VERDICTS} "
            "to say anything honest"
        )

    res = prequential(fs, items, rewards, interval=interval)

    table = Table("measure", "value", "reading")
    for row in prequential_readings(res, interval=interval, n_verdicts=len(items)):
        table.add_row(row.measure, row.value, _toned(row.reading, row.tone))
    console.print(table)
    console.print(
        "[dim]every prediction above was made by a model that had not seen that verdict[/dim]"
    )

    _off_policy_report(engine, fs)


def _off_policy_report(engine: Engine, fs) -> None:
    """Render the off-policy check.

    The measurement lives in evaluation.offpolicy; this decides how to phrase
    it. Hedged deliberately — it is the weaker of the two numbers `ent audit`
    prints, and reads as authoritative if presented plainly.
    """
    from .evaluation import offpolicy
    from .evaluation.prequential import MIN_LOGGED

    with store.session(read_only=True) as con:
        result = offpolicy.estimate(con, engine, fs)

    if result.status in ("not-enough-data", "no-model"):
        console.print(
            f"\n[dim]off-policy check: {result.n_usable} logged recommendations with an "
            f"outcome; needs {MIN_LOGGED} before the estimate means anything[/dim]"
        )
        return
    if result.status == "item-space-changed":
        console.print("\n[dim]off-policy check skipped: the item space has changed[/dim]")
        return
    if result.status != "ok":
        return

    verdict = "[green]better[/green]" if result.better else "[yellow]no better[/yellow]"
    console.print(
        f"\n[bold]off-policy estimate[/bold] [dim](weaker evidence; indicative only)[/dim]\n"
        f"  slates actually shown scored  {result.logged_value * 10:.2f}/10\n"
        f"  today's model would have      {result.estimate * 10:.2f}/10   {verdict}\n"
        f"  [dim]over {result.n_usable} logged recommendations you later rated[/dim]"
    )


@app.command("eval")
def evaluate(
    users: int = typer.Option(300, help="How many held-out MovieLens users to replay."),
    budget: int = typer.Option(30, help="Answered questions each simulated user gives."),
    elicitation: str = typer.Option(
        "v-optimal",
        help="Question-selection criterion: v-optimal | d-optimal | random (control).",
    ),
    out: Path | None = typer.Option(None, help="Write the full report as JSON."),
) -> None:
    """Benchmark the engine against baselines on held-out MovieLens users."""
    from .evaluation.simulate import ARMS, SimConfig, load_user_histories, paired_bootstrap, run

    _require_catalog()
    engine = Engine()
    with store.session(read_only=True) as con:
        fs = engine.features(con)
        meta = engine.meta(con)
        ml_map = dict(
            con.execute(
                "SELECT movielens_id, item_id FROM titles WHERE movielens_id IS NOT NULL"
            ).fetchall()
        )

    item_of_ml = {int(k): int(v) for k, v in ml_map.items() if int(v) in fs.index}
    cfg = SimConfig(n_users=users, budget=budget)
    histories = load_user_histories(item_of_ml, users, cfg.seed)
    if not histories:
        _fail("no usable MovieLens histories — is the catalogue too narrow?")
    console.print(f"[dim]{len(histories)} simulated users, budget {budget} answers[/dim]")

    prior = engine.prior()
    if prior is None:
        console.print("[yellow]no population prior fitted — run `ent data prior`[/yellow]")
    else:
        held_path = PATHS.artifacts / "cf_holdout_users.npy"
        if not held_path.exists():
            console.print(
                "[red]no record of which users were held out; the prior may contain "
                "the very users being replayed. Refusing to report a number.[/red]"
            )
            raise typer.Exit(code=1)

    results = run(fs, meta, histories, cfg, elicitation=elicitation, prior=prior)
    evaluation_manifest = write_manifest(
        "offline-evaluation",
        {
            "config": cfg.__dict__,
            "n_users": len(histories),
            "elicitation": elicitation,
            "arms": list(results),
        },
    )

    table = Table("arm", "NDCG@10", "P@10", "MAP@10", "MRR@10", "novelty", "diversity", "serend.")
    for name, res in results.items():
        s = res.summary()
        if not s:
            continue
        style = "bold green" if name == "entertainer" else ""
        table.add_row(
            f"[{style}]{name}[/{style}]" if style else name,
            f"{s['ndcg@10']:.4f} ±{s['ndcg@10_se']:.4f}",
            f"{s['precision@10']:.4f}",
            f"{s['map@10']:.4f}",
            f"{s['mrr@10']:.4f}",
            f"{s['novelty']:.2f}",
            f"{s['diversity']:.3f}",
            f"{s['serendipity']:.3f}",
        )
    console.print(table)

    if "entertainer" in results:
        console.print("\n[bold]paired bootstrap vs each baseline (NDCG@10)[/bold]")
        for name in ARMS:
            if name == "entertainer" or name not in results:
                continue
            diff, p = paired_bootstrap(results["entertainer"], results[name])
            verdict = "[green]significant[/green]" if p < 0.05 else "[yellow]not significant[/yellow]"
            console.print(f"  vs {name:<18} Δ={diff:+.4f}  p={p:.4f}  {verdict}")

    if out:
        payload = {
            "config": cfg.__dict__,
            "n_users": len(histories),
            "elicitation": elicitation,
            "arms": {name: res.summary() for name, res in results.items()},
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]report -> {out}[/green]")
    console.print(f"[dim]evaluation manifest: {evaluation_manifest['path']}[/dim]")


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
    from .data import netflix
    from .ingest import IngestError, add_title

    _require_catalog()

    ratings: list[netflix.Rating] = []
    for path in paths:
        ratings.extend(netflix.parse(path))
    if not ratings:
        _fail("no rating items found in those files")

    # Netflix pages overlap when they are grabbed by hand. Later pages are
    # older, so the first occurrence of a title is the most recent verdict.
    seen: set[tuple[str, int | None]] = set()
    unique: list[netflix.Rating] = []
    for rating in ratings:
        key = (netflix.normalise(rating.title), rating.netflix_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rating)
    if len(unique) != len(ratings):
        console.print(f"[dim]{len(ratings) - len(unique)} duplicate rows dropped[/dim]")

    with console.status(f"resolving {len(unique)} titles against TMDB..."):
        resolutions = netflix.flag_collisions(netflix.resolve(unique))
    if decisions:
        resolutions = netflix.apply_decisions(
            resolutions, json.loads(decisions.read_text(encoding="utf-8"))
        )

    buckets: dict[str, list[netflix.Resolution]] = {}
    for res in resolutions:
        buckets.setdefault(res.confidence, []).append(res)

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
            json.dumps(
                [
                    {
                        "title": r.rating.title,
                        "thumbs": r.rating.thumbs,
                        "verdict": r.rating.verdict,
                        "netflix_id": r.rating.netflix_id,
                        "rated_on": str(r.rating.rated_on) if r.rating.rated_on else None,
                        "confidence": r.confidence,
                        "reason": r.reason,
                        "best_guess": r.match,
                        "alternatives": r.alternatives,
                    }
                    for r in needs_eyes
                ],
                indent=2,
                ensure_ascii=False,
            ),
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
        # Re-running after settling the ambiguous titles would otherwise stack
        # a second verdict on every title the first pass already wrote. The
        # latest verdict wins, so nothing breaks, but the duplicates distort
        # the prequential replay, which walks the log in order.
        with store.session() as con:
            cleared = con.execute(
                "SELECT count(*) FROM events WHERE source = 'netflix'"
            ).fetchone()[0]
            con.execute("DELETE FROM events WHERE source = 'netflix'")
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


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
